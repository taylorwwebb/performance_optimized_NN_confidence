import torch
import torch.nn as nn
import torch.optim as optim
from torchvision import datasets, transforms
from torch.utils.data import Dataset, DataLoader
import numpy as np
from sklearn.decomposition import PCA
from scipy.stats import norm
import argparse
import os
import sys
import time

# Prevent python from saving out .pyc files
sys.dont_write_bytecode = True

from architecture import *
from fit_meta_d_MLE import *
from fit_rs_meta_d_MLE import *
from util import log

def check_path(path):
	if not os.path.exists(path):
		os.mkdir(path)

def test(args, encoder, class_out, conf_out, 
		device, loader, s1_signal=-1, s2_signal=-1, noise_level=-1):
	# Set to evaluation mode
	encoder.eval()
	class_out.eval()
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
		if s1_signal == -1:
			# Using range
			signal = ((torch.rand(x.shape[0]) * (args.signal_range[1] - args.signal_range[0])) + args.signal_range[0]).to(device)
			x = x * signal.view(-1, 1, 1, 1)
		else:
			# Scale signal (independently for each stimulus)
			x[y_targ==0,:,:,:] = x[y_targ==0,:,:,:] * s1_signal
			x[y_targ==1,:,:,:] = x[y_targ==1,:,:,:] * s2_signal
		# Scale to [-1, 1]
		x = (x - 0.5) / 0.5
		# Add noise
		if noise_level == -1:
			# Using range
			noise = (torch.rand(x.shape[0]) * (args.noise_range[1] - args.noise_range[0])) + args.noise_range[0]
			noise = noise.view(-1, 1, 1, 1).repeat(1, 1, x.shape[2], x.shape[3])
			noise = (torch.randn(x.shape) * noise).to(device)
			x = x + noise
		else:
			# Using specifiied value
			x = x + (torch.randn(x.shape) * noise_level).to(device)
		# Threshold image
		x = nn.Hardtanh()(x)
		# Get model predictions
		z = encoder(x, device)
		y_pred = class_out(z).squeeze().round()
		conf = conf_out(z).squeeze()
		# Collect data for d'/meta-d' calculation
		all_z.append(z.detach().cpu().numpy())
		all_y_pred.append(y_pred.cpu().detach().numpy())
		all_y_targ.append(y_targ.cpu().numpy())
		all_conf.append(conf.cpu().detach().numpy())
	# Data for d'/meta-d' calculation
	all_z = np.concatenate(all_z)
	all_y_pred = np.concatenate(all_y_pred)
	all_y_targ = np.concatenate(all_y_targ)
	all_conf = np.concatenate(all_conf)
	return all_z, all_y_pred, all_y_targ, all_conf

def load_params(args, encoder, class_out, conf_out, epoch):
	# Directory
	test_dir = './test/'
	check_path(test_dir)
	run_dir = test_dir + 'run' + args.run + '/'
	check_path(run_dir)
	# Load parameters
	encoder_fname = run_dir + 'epoch' + str(epoch) + '_encoder_params.pt'
	encoder.load_state_dict(torch.load(encoder_fname))
	class_out_fname = run_dir + 'epoch' + str(epoch) + '_class_out_params.pt'
	class_out.load_state_dict(torch.load(class_out_fname))
	conf_out_fname = run_dir + 'epoch' + str(epoch) + '_conf_out_params.pt'
	conf_out.load_state_dict(torch.load(conf_out_fname))
	# Load selected classes
	selected_classes = np.load(run_dir + 'selected_classes.npz')['classes']
	return encoder, class_out, conf_out, selected_classes

def main():

	# Settings
	parser = argparse.ArgumentParser()
	parser.add_argument('--test-batch-size', type=int, default=100)
	parser.add_argument('--signal_range', type=list, default=[0.1, 1.0])
	parser.add_argument('--noise_range', type=list, default=[1.0, 2.0])
	parser.add_argument('--noise_val_test', type=float, default=2.0)
	parser.add_argument('--img_size', type=int, default=32)
	parser.add_argument('--latent_dim', type=int, default=100)
	parser.add_argument('--epochs', type=int, default=5)
	parser.add_argument('--no-cuda', action='store_true', default=False)
	parser.add_argument('--device', type=int, default=0)
	parser.add_argument('--run', type=str, default='1')
	args = parser.parse_args()
		
	# Set up cuda	
	use_cuda = not args.no_cuda and torch.cuda.is_available()
	device = torch.device("cuda:" + str(args.device) if use_cuda else "cpu")
	kwargs = {'num_workers': 1, 'pin_memory': True} if use_cuda else {}

	# Build model
	log.info('Building model...')
	encoder = Encoder(args).to(device)
	class_out = Class_out(args).to(device)
	conf_out = Conf_out(args).to(device)

	# Load pretrained parameters
	encoder, class_out, conf_out, selected_classes = load_params(args, encoder, class_out, conf_out, args.epochs)

	# Load s1/s2 signal values
	s1_signal = np.load('./test/i3_fit.npz')['estimated_mu'].item()
	i1245_signal = np.load('./test/i1245_fit.npz')['estimated_mu']
	s2_signal = np.insert(i1245_signal, 2, s1_signal)

	# Create test set from MNIST
	log.info('Loading MNIST test set...')
	# Set up directory to download datasets to
	dset_dir = './datasets'
	check_path(dset_dir)
	# Transforms
	transforms_to_apply = transforms.Compose([transforms.Resize(args.img_size), transforms.ToTensor()])
	# Test set
	test_set = datasets.MNIST(dset_dir, train=False, download=True, transform=transforms_to_apply)
	# Selected digit classes
	log.info('Selected classes:')
	log.info(str(selected_classes))
	# Subset dataset to only these classes
	log.info('Subsetting...')
	s1 = test_set.targets == selected_classes[0]
	s2 = test_set.targets == selected_classes[1]
	test_set.targets = test_set.targets[torch.logical_or(s1,s2)]
	test_set.data = test_set.data[torch.logical_or(s1,s2)]
	# Convert targets to 0/1 (for binary training)
	s1 = test_set.targets == selected_classes[0]
	s2 = test_set.targets == selected_classes[1]
	test_set.targets[s1] = test_set.targets[s1] * 0
	test_set.targets[s2] = (test_set.targets[s2] * 0) + 1
	# Convert to PyTorch DataLoaders
	log.info('Converting to DataLoaders...')
	test_loader = DataLoader(test_set, batch_size=args.test_batch_size, shuffle=True, drop_last=True)

	# Evaluate on test set
	log.info('Evaluating on test set...')
	z, y_pred, y_targ, conf = test(args, encoder, class_out, conf_out, device, test_loader)

	# Perform PCA
	log.info('Performing PCA...')
	pca = PCA()
	pca.fit(z)

	# Test
	log.info('Test...')
	all_z_top2 = []
	all_y_pred = []
	all_y_targ = []
	all_conf = []
	for s in range(s2_signal.shape[0]):
		z, y_pred, y_targ, conf = test(args, encoder, class_out, conf_out, device, test_loader, s1_signal=s1_signal, s2_signal=s2_signal[s], noise_level=args.noise_val_test)
		z_top2 = pca.transform(z)[:,:2]
		all_z_top2.append(z_top2)
		all_y_pred.append(y_pred)
		all_y_targ.append(y_targ)
		all_conf.append(conf)
	# Save test results
	test_dir = './test/'
	check_path(test_dir)
	model_dir = test_dir + 'run' + str(args.run) + '/'
	check_path(model_dir)
	np.savez(model_dir + 'PCA_test_results.npz',
			 z_top2=np.array(all_z_top2),
			 all_y_pred=np.array(all_y_pred),
			 all_y_targ=np.array(all_y_targ),
			 all_conf=np.array(all_conf))

if __name__ == '__main__':
	main()