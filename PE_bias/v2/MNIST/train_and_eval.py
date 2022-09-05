import torch
import torch.nn as nn
import torch.optim as optim
from torchvision import datasets, transforms
from torch.utils.data import Dataset, DataLoader
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

def train(args, encoder, class_out, conf_out, device, train_loader, optimizer, epoch):
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

def s1s2_test(args, encoder, class_out, conf_out, device, s1_loader, s2_loader, targ_signal=1.0, nontarg_signal=1.0, noise=0.0):
	# Set to evaluation mode
	encoder.eval()
	class_out.eval()
	conf_out.eval()
	# Iterate over batches
	all_test_correct_preds = []
	all_test_conf = []
	for batch_idx, ((data_s1, target_s1), (data_s2, target_s2)) in enumerate(zip(s1_loader, s2_loader)):
		# Load data
		x_s1 = data_s1.to(device)
		x_s2 = data_s2.to(device)
		# Sample targets and set signal level for s1/s2
		y_targ = torch.rand(args.test_batch_size).round().to(device)
		targ_signal = torch.ones(args.test_batch_size).to(device) * targ_signal
		nontarg_signal = torch.ones(args.test_batch_size).to(device) * nontarg_signal
		s1_signal = (targ_signal * torch.logical_not(y_targ).float()) + (nontarg_signal * y_targ)
		s2_signal = (targ_signal * y_targ) + (nontarg_signal * torch.logical_not(y_targ).float())
		# Apply contrast scaling and sumperimpose images
		x_s1 = x_s1 * s1_signal.view(-1,1,1,1)
		x_s2 = x_s2 * s2_signal.view(-1,1,1,1)
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
		# Collect responses
		# Correct predictions
		correct_preds = torch.eq(y_pred.round(), y_targ.float()).float()
		all_test_correct_preds.append(correct_preds.detach().cpu().numpy())
		# Confidence
		all_test_conf.append(conf.detach().cpu().numpy())
	# Average test accuracy and confidence
	all_test_correct_preds = np.concatenate(all_test_correct_preds)
	all_test_conf = np.concatenate(all_test_conf)
	avg_test_acc = np.mean(all_test_correct_preds) * 100.0
	avg_test_conf = np.mean(all_test_conf) * 100.0
	avg_test_conf_correct = np.mean(all_test_conf[all_test_correct_preds==1]) * 100.0
	avg_test_conf_incorrect = np.mean(all_test_conf[all_test_correct_preds==0]) * 100.0
	return avg_test_acc, avg_test_conf, avg_test_conf_correct, avg_test_conf_incorrect

def main():

	# Settings
	parser = argparse.ArgumentParser()
	parser.add_argument('--train-batch-size', type=int, default=32)
	parser.add_argument('--test-batch-size', type=int, default=100)
	parser.add_argument('--signal_range', type=list, default=[0.1, 1.0])
	parser.add_argument('--signal_test_vals', type=list, default=[0.5, 1.0])
	parser.add_argument('--signal_N_test', type=int, default=500)
	parser.add_argument('--noise_range', type=list, default=[1.0, 2.0])
	parser.add_argument('--noise_test', type=float, default=1.5)
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

	selected_classes = np.array([4,6])

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
	# Convert to PyTorch DataLoaders
	log.info('Converting to DataLoaders...')
	train_loader = DataLoader(train_set, batch_size=args.train_batch_size, shuffle=True, drop_last=True)
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

	# Test
	log.info('Test...')
	# Signal values for test
	low_PE_targ_signal = args.signal_test_vals[0]
	high_PE_targ_signal = args.signal_test_vals[1]
	nontarg_signal_test_vals = np.linspace(args.signal_range[0], args.signal_range[1], args.signal_N_test)
	# Low PE
	log.info('Low PE...')
	low_PE_test_acc = []
	low_PE_test_conf = []
	low_PE_test_conf_correct = []
	low_PE_test_conf_incorrect = []
	for s in range(nontarg_signal_test_vals.shape[0]):
		if nontarg_signal_test_vals[s] < low_PE_targ_signal:
			test_acc, test_conf, test_conf_correct, test_conf_incorrect = s1s2_test(args, encoder, class_out, conf_out, device, s1_test_loader, s2_test_loader, targ_signal=low_PE_targ_signal, nontarg_signal=nontarg_signal_test_vals[s], noise=args.noise_test)
			low_PE_test_acc.append(test_acc)
			low_PE_test_conf.append(test_conf)
			low_PE_test_conf_correct.append(test_conf_correct)
			low_PE_test_conf_incorrect.append(test_conf_incorrect)
	# High PE
	log.info('High PE...')
	high_PE_test_acc = []
	high_PE_test_conf = []
	high_PE_test_conf_correct = []
	high_PE_test_conf_incorrect = []
	for s in range(nontarg_signal_test_vals.shape[0]):
		if nontarg_signal_test_vals[s] < high_PE_targ_signal:
			test_acc, test_conf, test_conf_correct, test_conf_incorrect = s1s2_test(args, encoder, class_out, conf_out, device, s1_test_loader, s2_test_loader, targ_signal=high_PE_targ_signal, nontarg_signal=nontarg_signal_test_vals[s], noise=args.noise_test)
			high_PE_test_acc.append(test_acc)
			high_PE_test_conf.append(test_conf)
			high_PE_test_conf_correct.append(test_conf_correct)
			high_PE_test_conf_incorrect.append(test_conf_incorrect)
	# Save test results
	test_dir = './test/'
	check_path(test_dir)
	model_dir = test_dir + 'run' + str(args.run) + '/'
	check_path(model_dir)
	np.savez(model_dir + 'test_results.npz',
		low_PE_targ_signal=low_PE_targ_signal,
		high_PE_targ_signal=high_PE_targ_signal,
		nontarg_signal_test_vals=nontarg_signal_test_vals,
		low_PE_test_acc=low_PE_test_acc,
		low_PE_test_conf=low_PE_test_conf,
		low_PE_test_conf_correct=low_PE_test_conf_correct,
		low_PE_test_conf_incorrect=low_PE_test_conf_incorrect,
		high_PE_test_acc=high_PE_test_acc,
		high_PE_test_conf=high_PE_test_conf,
		high_PE_test_conf_correct=high_PE_test_conf_correct,
		high_PE_test_conf_incorrect=high_PE_test_conf_incorrect)

if __name__ == '__main__':
	main()