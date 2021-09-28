import torch
import torch.nn as nn
import torch.optim as optim
from torchvision import datasets, transforms
from torch.utils.data import Dataset, DataLoader
from sklearn.decomposition import PCA
from scipy.stats import sem
from joblib import dump
import numpy as np
from copy import deepcopy
import argparse
import os
import sys
import time

# Prevent python from saving out .pyc files
sys.dont_write_bytecode = True

from architecture import *
from util import log

def check_path(path):
	if not os.path.exists(path):
		os.mkdir(path)

def train(args, encoder, class_out, conf_out, 
		device, train_loader, optimizer, epoch):
	# Create file for saving training progress
	train_prog_dir = './train_prog/'
	check_path(train_prog_dir)
	model_dir = train_prog_dir + 'run' + str(args.run) + '/'
	check_path(model_dir)
	train_prog_fname = model_dir + 'epoch_' + str(epoch) + '.txt'
	train_prog_f = open(train_prog_fname, 'w')
	train_prog_f.write('batch class_loss class_acc conf_loss conf\n')
	# Set to training mode
	encoder.train()
	class_out.train()
	conf_out.train()
	# Iterate over batches
	for batch_idx, (data, target) in enumerate(train_loader):
		# Batch start time
		start_time = time.time()
		# Load data
		x = data.to(device)
		y_targ = target.to(device)
		# Scale signal
		signal = ((torch.rand(x.shape[0]) * (args.signal_range[1] - args.signal_range[0])) + args.signal_range[0]).to(device)
		x = x * signal.view(-1, 1, 1, 1)
		# Scale to [-1, 1]
		x = (x - 0.5) / 0.5
		# Add noise
		noise = (torch.rand(x.shape[0]) * (args.noise_range[1] - args.noise_range[0])) + args.noise_range[0]
		noise = noise.view(-1, 1, 1, 1).repeat(1, 1, x.shape[2], x.shape[3])
		noise = (torch.randn(x.shape) * noise).to(device)
		x = x + noise
		# Threshold image
		x = nn.Hardtanh()(x)
		# Zero out gradients for optimizer 
		optimizer.zero_grad()
		# Get model predictions
		z = encoder(x, device)
		y_pred = class_out(z).squeeze()
		conf = conf_out(z).squeeze()
		# Classification loss
		class_loss_fn = torch.nn.BCELoss()
		class_loss = class_loss_fn(y_pred, y_targ.float())
		# Classification accuracy
		correct_preds = torch.eq(y_pred.round(), y_targ.float()).float()
		class_acc = correct_preds.mean().item() * 100.0
		# Confidence loss
		conf_loss_fn = torch.nn.BCELoss()
		conf_loss = conf_loss_fn(conf, correct_preds)
		# Combine losses and update model
		combined_loss = class_loss + conf_loss
		combined_loss.backward()
		optimizer.step()
		# Overall confidence
		avg_conf = conf.mean().item() * 100.0
		# Batch duration
		end_time = time.time()
		batch_dur = end_time - start_time
		# Report prgoress
		if batch_idx % args.log_interval == 0:
			log.info('[Epoch: ' + str(epoch) + '] ' + \
					 '[Batch: ' + str(batch_idx) + ' of ' + str(len(train_loader)) + '] ' + \
					 '[Class. Loss = ' + '{:.4f}'.format(class_loss.item()) + '] ' + \
					 '[Class. Acc. = ' + '{:.2f}'.format(class_acc) + '] ' + \
					 '[Conf. Loss = ' + '{:.4f}'.format(conf_loss.item()) + '] ' + \
					 '[Conf. = ' + '{:.2f}'.format(avg_conf) + '] ' + \
					 '[' + '{:.3f}'.format(batch_dur) + ' sec/batch]')
			# Save progress to file
			train_prog_f.write(str(batch_idx) + ' ' +\
				               '{:.4f}'.format(class_loss.item()) + ' ' + \
				               '{:.2f}'.format(class_acc) + ' ' + \
				               '{:.4f}'.format(conf_loss.item()) + ' ' + \
				               '{:.2f}'.format(avg_conf) + '\n')
	train_prog_f.close()

def test(args, encoder, class_out, conf_out, device, loader):
	# Set to evaluation mode
	encoder.eval()
	class_out.eval()
	conf_out.eval()
	# Iterate over batches
	all_z = []
	all_y_pred = []
	all_y_targ = []
	all_conf = []
	for batch_idx, (data, target) in enumerate(loader):
		# Load data
		x = data.to(device)
		y_targ = target.to(device)
		# Scale signal
		signal = ((torch.rand(x.shape[0]) * (args.signal_range[1] - args.signal_range[0])) + args.signal_range[0]).to(device)
		x = x * signal.view(-1, 1, 1, 1)
		# Scale to [-1, 1]
		x = (x - 0.5) / 0.5
		# Add noise
		noise = (torch.rand(x.shape[0]) * (args.noise_range[1] - args.noise_range[0])) + args.noise_range[0]
		noise = noise.view(-1, 1, 1, 1).repeat(1, 1, x.shape[2], x.shape[3])
		noise = (torch.randn(x.shape) * noise).to(device)
		x = x + noise
		# Threshold image
		x = nn.Hardtanh()(x)
		# Get model predictions
		z = encoder(x, device)
		y_pred = class_out(z).squeeze()
		conf = conf_out(z).squeeze()
		# Collect outputs
		all_z.append(z.detach().cpu().numpy())
		all_y_pred.append(y_pred.detach().cpu().numpy())
		all_y_targ.append(y_targ.detach().cpu().numpy())
		all_conf.append(conf.detach().cpu().numpy())
	# Concatenate batches
	all_z = np.concatenate(all_z)
	all_y_pred = np.concatenate(all_y_pred)
	all_y_targ = np.concatenate(all_y_targ)
	all_conf = np.concatenate(all_conf)
	return all_z, all_y_pred, all_y_targ, all_conf

def PE_test(args, encoder, class_out, conf_out, device, loader, pca, signal=1.0, noise=0.0):
	# Set to evaluation mode
	encoder.eval()
	class_out.eval()
	conf_out.eval()
	# Iterate over batches
	all_z_top2 = []
	all_y_pred = []
	all_y_targ = []
	all_conf = []
	for batch_idx, (data, target) in enumerate(loader):
		# Load data
		x = data.to(device)
		y_targ = target.to(device)
		all_y_targ.append(y_targ.detach().cpu().numpy())
		# Scale signal
		x = x * signal
		# Scale to [-1, 1]
		x = (x - 0.5) / 0.5
		# Add noise
		x = x + (torch.randn(x.shape) * noise).to(device)
		# Threshold image
		x = nn.Hardtanh()(x)
		# Get model predictions
		z = encoder(x, device)
		y_pred = class_out(z).squeeze()
		all_y_pred.append(y_pred.detach().cpu().numpy())
		conf = conf_out(z).squeeze()
		all_conf.append(conf.detach().cpu().numpy())
		# Apply PCA
		z_top2 = pca.transform(z.detach().cpu().numpy())[:,:2]
		all_z_top2.append(z_top2)
	# Concatenate batches
	all_z_top2 = np.concatenate(all_z_top2)
	all_y_pred = np.concatenate(all_y_pred)
	all_y_targ = np.concatenate(all_y_targ)
	all_conf = np.concatenate(all_conf)
	# Average accuracy and confidence
	avg_acc = (all_y_pred.round() == all_y_targ).astype(np.float).mean() * 100.0
	avg_conf = all_conf.mean() * 100.0
	# Report
	log.info('[Signal = ' + '{:.2f}'.format(signal) + '] ' + \
			 '[Noise = ' + '{:.2f}'.format(noise) + '] ' + \
			 '[Class. Acc. = ' + '{:.2f}'.format(avg_acc) + '] ' + \
			 '[Conf. = ' + '{:.2f}'.format(avg_conf) + ']')
	# Average recitified decision output
	y_pred_rect = np.abs(all_y_pred - 0.5) + 0.5
	# PCA outputs
	abs_pc1 = np.abs(all_z_top2[:,0]).mean()
	pc2 = z_top2[:,1].mean()
	pc1_mn_diff = np.abs(all_z_top2[all_y_targ==0,0].mean() - all_z_top2[all_y_targ==1,0].mean())
	pc1_var = np.mean([sem(all_z_top2[all_y_targ==0,0]), sem(all_z_top2[all_y_targ==1,0])])
	return avg_acc, avg_conf, abs_pc1, pc2, y_pred_rect, pc1_mn_diff, pc1_var, all_z_top2[:,0], all_y_pred, all_y_targ

def s1s2_test(args, encoder, class_out, conf_out, device, s1_loader, s2_loader, pca, s1_signal=1.0, s2_signal=1.0, noise=0.0):
	# Set to evaluation mode
	encoder.eval()
	class_out.eval()
	conf_out.eval()
	# Iterate over batches
	all_acc = []
	all_conf = []
	all_y_pred = []
	all_y_pred_rect = []
	all_pc1 = []
	all_pc2 = []
	all_abs_pc1 = []
	for batch_idx, ((data_s1, target_s1), (data_s2, target_s2)) in enumerate(zip(s1_loader, s2_loader)):
		# Load data
		x_s1 = data_s1.to(device)
		x_s2 = data_s2.to(device)
		# Generate targets (based on relative value of s1/s2 signal)
		if s2_signal > s1_signal:
			y_targ = torch.ones(args.test_batch_size).to(device)
		else:
			y_targ = torch.zeros(args.test_batch_size).to(device)
		# Apply contrast scaling and sumperimpose images
		x_s1 = x_s1 * s1_signal
		x_s2 = x_s2 * s2_signal
		x, _ = torch.stack([x_s1, x_s2],0).max(0)
		# Scale to [-1, 1]
		x = (x - 0.5) / 0.5
		# Add noise
		x = x + (torch.randn(x.shape) * noise).to(device)
		# Threshold image
		x = nn.Hardtanh()(x)
		# Get model predictions
		z = encoder(x, device)
		y_pred = class_out(z).squeeze()
		conf = conf_out(z).squeeze()
		# Classification accuracy
		correct_preds = torch.eq(y_pred.round(), y_targ).float()
		class_acc = correct_preds.mean().item() * 100.0
		all_acc.append(class_acc)
		# Overall confidence
		avg_conf = conf.mean().item() * 100.0
		all_conf.append(avg_conf)
		# Average class prediction
		avg_y_pred = y_pred.mean().item()
		all_y_pred.append(avg_y_pred)
		# Average recitifed class prediction
		avg_y_pred_rect = (torch.abs(y_pred - 0.5) + 0.5).mean().item()
		all_y_pred_rect.append(avg_y_pred_rect)
		# PCA
		z_top2 = pca.transform(z.detach().cpu().numpy())[:,:2]
		pc1 = z_top2[:,0]
		all_pc1.append(pc1)
		pc2 = z_top2[:,1]
		all_pc2.append(pc2)
		abs_pc1 = np.abs(pc1)
		all_abs_pc1.append(abs_pc1)
	# Average across batches
	acc = np.mean(all_acc)
	conf = np.mean(all_conf)
	y_pred = np.mean(all_y_pred)
	y_pred_rect = np.mean(all_y_pred_rect)
	avg_pc1 = np.concatenate(all_pc1).mean()
	avg_pc2 = np.concatenate(all_pc2).mean()
	avg_abs_pc1 = np.concatenate(all_abs_pc1).mean()
	return acc, conf, y_pred, y_pred_rect, avg_pc1, avg_pc2, avg_abs_pc1

def main():

	# Settings
	parser = argparse.ArgumentParser()
	parser.add_argument('--train-batch-size', type=int, default=32)
	parser.add_argument('--test-batch-size', type=int, default=100)
	parser.add_argument('--signal_range', type=list, default=[0.1,1.0])
	parser.add_argument('--signal_range_PE_test', type=list, default=[0.0,1.0])
	parser.add_argument('--signal_N_PE_test', type=int, default=500)
	parser.add_argument('--signal_N_s1s2_test', type=int, default=100)
	parser.add_argument('--noise_range', type=list, default=[1.0, 2.0])
	parser.add_argument('--noise_N_PE_test', type=int, default=2)
	parser.add_argument('--noise_s1s2_test', type=float, default=1.5)
	parser.add_argument('--img_size', type=int, default=32)
	parser.add_argument('--latent_dim', type=int, default=100)
	parser.add_argument('--epochs', type=int, default=5)
	parser.add_argument('--lr', type=float, default=5e-4)
	parser.add_argument('--no-cuda', action='store_true', default=False)
	parser.add_argument('--log_interval', type=int, default=10)
	parser.add_argument('--device', type=int, default=0)
	parser.add_argument('--run', type=str, default='1')
	args = parser.parse_args()
		
	# Set up cuda	
	use_cuda = not args.no_cuda and torch.cuda.is_available()
	device = torch.device("cuda:" + str(args.device) if use_cuda else "cpu")
	kwargs = {'num_workers': 1, 'pin_memory': True} if use_cuda else {}

	# Create training/test sets from MNIST
	log.info('Loading MNIST training/test sets...')
	# Set up directory to download datasets to
	dset_dir = './datasets'
	check_path(dset_dir)
	# Transforms
	transforms_to_apply = transforms.Compose([transforms.Resize(args.img_size), transforms.ToTensor()])
	# Training/test sets
	train_set = datasets.MNIST(dset_dir, train=True, download=True, transform=transforms_to_apply)
	test_set = datasets.MNIST(dset_dir, train=False, download=True, transform=transforms_to_apply)
	# Randomly select N digit classes
	log.info('Randomly selecting 2 out of 10 digit classes...')
	all_digit_classes = np.arange(10)
	np.random.shuffle(all_digit_classes)
	selected_classes = all_digit_classes[:2]
	log.info('Selected classes:')
	log.info(str(selected_classes))
	# Subset dataset to only these classes
	log.info('Subsetting...')
	# Training set
	s1 = train_set.targets == selected_classes[0]
	s2 = train_set.targets == selected_classes[1]
	train_set.targets = train_set.targets[torch.logical_or(s1,s2)]
	train_set.data = train_set.data[torch.logical_or(s1,s2)]
	# Convert targets to 0/1 (for binary training)
	s1 = train_set.targets == selected_classes[0]
	s2 = train_set.targets == selected_classes[1]
	train_set.targets[s1] = train_set.targets[s1] * 0
	train_set.targets[s2] = (train_set.targets[s2] * 0) + 1
	# Test sets (separate datasets for s1/s2)
	s1 = test_set.targets == selected_classes[0]
	s1_test_set = deepcopy(test_set)
	s1_test_set.targets = s1_test_set.targets[s1]
	s1_test_set.data = s1_test_set.data[s1]
	s2 = test_set.targets == selected_classes[1]
	s2_test_set = deepcopy(test_set)
	s2_test_set.targets = s2_test_set.targets[s2]
	s2_test_set.data = s2_test_set.data[s2]
	# Combined test set
	test_set.targets = test_set.targets[torch.logical_or(s1,s2)]
	test_set.data = test_set.data[torch.logical_or(s1,s2)]
	# Convert targets to 0/1 (for binary training)
	s1 = test_set.targets == selected_classes[0]
	s2 = test_set.targets == selected_classes[1]
	test_set.targets[s1] = test_set.targets[s1] * 0
	test_set.targets[s2] = (test_set.targets[s2] * 0) + 1
	# Convert to PyTorch DataLoaders
	log.info('Converting to DataLoaders...')
	train_loader = DataLoader(train_set, batch_size=args.train_batch_size, shuffle=True, drop_last=True)
	test_loader = DataLoader(test_set, batch_size=args.test_batch_size, shuffle=True, drop_last=True)
	s1_test_loader = DataLoader(s1_test_set, batch_size=args.test_batch_size, shuffle=True, drop_last=True)
	s2_test_loader = DataLoader(s2_test_set, batch_size=args.test_batch_size, shuffle=True, drop_last=True)

	# Build model
	log.info('Building model...')
	encoder = Encoder(args).to(device)
	class_out = Class_out(args).to(device)
	conf_out = Conf_out(args).to(device)
	all_modules = nn.ModuleList([encoder, class_out, conf_out])

	# Create optimizer
	log.info('Setting up optimizer...')
	optimizer = optim.Adam(all_modules.parameters(), lr=args.lr)

	# Train
	log.info('Training begins...')
	for epoch in range(1, args.epochs + 1):
		# Training loop
		train(args, encoder, class_out, conf_out, device, train_loader, optimizer, epoch)

	# Evaluate on test set
	log.info('Evaluating on test set...')
	all_z, all_y_pred, all_y_targ, all_conf = test(args, encoder, class_out, conf_out, device, test_loader)

	# Perform PCA on learned low dimensional representations (z)
	log.info('Performing PCA...')
	pca = PCA()
	pca.fit(all_z)
	# Save PCA
	test_dir = './test/'
	check_path(test_dir)
	model_dir = test_dir + 'run' + str(args.run) + '/'
	check_path(model_dir)
	dump(pca, model_dir + 'pca.joblib')
	# Save PCA results
	z_top2 = pca.transform(all_z)[:,:2]
	np.savez(model_dir + 'PCA_results.npz',
			 z_top2=z_top2,
			 y_pred=all_y_pred,
			 y_targ=all_y_targ,
			 conf=all_conf)

	# PE bias test
	log.info('PE bias test...')
	# Signal and noise values for test
	signal_test_vals = np.linspace(args.signal_range_PE_test[0], args.signal_range_PE_test[1], args.signal_N_PE_test)
	noise_test_vals = np.linspace(args.noise_range[0], args.noise_range[1], args.noise_N_PE_test)
	all_acc = []
	all_conf = []
	all_abs_pc1 = []
	all_pc2 = []
	all_y_pred_rect = []
	all_pc1_mn_diff = []
	all_pc1_var = []
	all_pc1 = []
	all_y_pred = []
	all_y_targ = []
	for n in range(noise_test_vals.shape[0]):
		all_signal_acc = []
		all_signal_conf = []
		all_signal_abs_pc1 = []
		all_signal_pc2 = []
		all_signal_y_pred_rect = []
		all_signal_pc1_mn_diff = []
		all_signal_pc1_var = []
		all_signal_pc1 = []
		all_signal_y_pred = []
		all_signal_y_targ = []
		for s in range(signal_test_vals.shape[0]):
			acc, conf, abs_pc1, pc2, y_pred_rect, pc1_mn_diff, pc1_var, pc1, y_pred, y_targ = PE_test(args, encoder, class_out, conf_out, device, test_loader, pca, signal=signal_test_vals[s], noise=noise_test_vals[n])
			all_signal_acc.append(acc)
			all_signal_conf.append(conf)
			all_signal_abs_pc1.append(abs_pc1)
			all_signal_pc2.append(pc2)
			all_signal_y_pred_rect.append(y_pred_rect)
			all_signal_pc1_mn_diff.append(pc1_mn_diff)
			all_signal_pc1_var.append(pc1_var)
			all_signal_pc1.append(pc1)
			all_signal_y_pred.append(y_pred)
			all_signal_y_targ.append(y_targ)
		all_acc.append(all_signal_acc)
		all_conf.append(all_signal_conf)
		all_abs_pc1.append(all_signal_abs_pc1)
		all_pc2.append(all_signal_pc2)
		all_pc1_mn_diff.append(all_signal_pc1_mn_diff)
		all_pc1_var.append(all_signal_pc1_var)
		all_pc1.append(all_signal_pc1)
		all_y_pred.append(all_signal_y_pred)
		all_y_targ.append(all_signal_y_targ)
	# Save test results
	np.savez(model_dir + 'PE_test_results.npz',
			 signal_test_vals=signal_test_vals,
			 noise_test_vals=noise_test_vals,
			 all_acc=np.array(all_acc),
			 all_conf=np.array(all_conf),
			 all_abs_pc1=np.array(all_abs_pc1),
			 all_pc2=np.array(all_pc2),
			 all_pc1_mn_diff=np.array(all_pc1_mn_diff),
			 all_pc1_var=np.array(all_pc1_var),
			 all_pc1=np.array(all_pc1),
			 all_y_pred=np.array(all_y_pred),
			 all_y_targ=np.array(all_y_targ))

	# s1/s2 test
	log.info('Evaluating on images of superimposed digits...')
	signal_test_vals = np.linspace(args.signal_range[0], args.signal_range[1], args.signal_N_s1s2_test)
	all_acc = []
	all_conf = []
	all_y_pred = []
	all_y_pred_rect = []
	all_pc1 = []
	all_pc2 = []
	all_abs_pc1 = []
	for s1 in range(signal_test_vals.shape[0]):
		all_s1_acc = []
		all_s1_conf = []
		all_s1_y_pred = []
		all_s1_y_pred_rect = []
		all_s1_pc1 = []
		all_s1_pc2 = []
		all_s1_abs_pc1 = []
		for s2 in range(signal_test_vals.shape[0]):
			acc, conf, y_pred, y_pred_rect, pc1, pc2, abs_pc1 = s1s2_test(args, encoder, class_out, conf_out, device, s1_test_loader, s2_test_loader, pca, 
				s1_signal=signal_test_vals[s1], s2_signal=signal_test_vals[s2], noise=args.noise_s1s2_test)
			log.info('[s1 signal = ' + '{:.2f}'.format(signal_test_vals[s1]) + '] ' + \
					 '[s2 signal = ' + '{:.2f}'.format(signal_test_vals[s2]) + '] ' + \
			 	 	 '[Class acc. = ' + '{:.2f}'.format(acc) + '] ' + \
			 	 	 '[Conf. = ' + '{:.2f}'.format(conf) + '] ' + \
			 	 	 '[predicted class = ' + '{:.4f}'.format(y_pred) + '] ' + \
			 	 	 '[probability of predicted class = ' + '{:.4f}'.format(y_pred_rect) + ']')
			all_s1_acc.append(acc)
			all_s1_conf.append(conf)
			all_s1_y_pred.append(y_pred)
			all_s1_y_pred_rect.append(y_pred_rect)
			all_s1_pc1.append(pc1)
			all_s1_pc2.append(pc2)
			all_s1_abs_pc1.append(abs_pc1)
		all_acc.append(all_s1_acc)
		all_conf.append(all_s1_conf)
		all_y_pred.append(all_s1_y_pred)
		all_y_pred_rect.append(all_s1_y_pred_rect)
		all_pc1.append(all_s1_pc1)
		all_pc2.append(all_s1_pc2)
		all_abs_pc1.append(all_s1_abs_pc1)
	# Save results
	np.savez(model_dir + 's1s2_test_results.npz',
			 signal_test_vals=signal_test_vals,
			 test_noise=args.noise_s1s2_test,
			 all_acc=np.array(all_acc),
			 all_conf=np.array(all_conf),
			 all_y_pred=np.array(all_y_pred),
			 all_y_pred_rect=np.array(all_y_pred_rect),
			 all_pc1=np.array(all_pc1),
			 all_pc2=np.array(all_pc2),
			 all_abs_pc1=np.array(all_abs_pc1))

if __name__ == '__main__':
	main()