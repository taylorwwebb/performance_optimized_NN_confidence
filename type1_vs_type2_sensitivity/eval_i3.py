import torch
import torch.nn as nn
import torch.optim as optim
from torchvision import datasets, transforms
from torch.utils.data import Dataset, DataLoader
import numpy as np
from scipy.stats import norm
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

def test(args, encoder, class_out, 
		device, loader, signal=1.0, noise=0.0):
	# Set to evaluation mode
	encoder.eval()
	class_out.eval()
	# Iterate over batches
	all_y_pred = []
	all_y_targ = []
	for batch_idx, (data, target) in enumerate(loader):
		# Load data
		x = data.to(device)
		y_targ = target.to(device)
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
		y_pred = class_out(z).squeeze().round()
		# Collect data for d'/meta-d' calculation
		all_y_pred.append(y_pred.cpu().detach().numpy())
		all_y_targ.append(y_targ.cpu().numpy())
	# Data for d'/meta-d' calculation
	all_y_pred = np.concatenate(all_y_pred)
	all_y_targ = np.concatenate(all_y_targ)
	return all_y_pred, all_y_targ

def calc_d_prime(y_pred, y_targ):
	# Hit rate and false alarm rate
	hr = np.logical_and(y_pred==1,y_targ==1).sum() / (y_targ==1).sum()
	if hr == 0:
		hr = 0.5 / (y_targ==1).sum()
	if hr == 1:
		hr = ((y_targ==1).sum() - 0.5) / (y_targ==1).sum()
	far = np.logical_and(y_pred==1,y_targ==0).sum() / (y_targ==0).sum()
	if far == 0:
		far = 0.5 / (y_targ==0).sum()
	if far == 1:
		far = ((y_targ==0).sum() - 0.5) / (y_targ==0).sum()
	# Calculate d-prime
	d_prime = norm.ppf(hr) - norm.ppf(far)
	return d_prime

def load_params(args, encoder, class_out, epoch):
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
	# Load selected classes
	selected_classes = np.load(run_dir + 'selected_classes.npz')['classes']
	return encoder, class_out, selected_classes

def main():

	# Settings
	parser = argparse.ArgumentParser()
	parser.add_argument('--test-batch-size', type=int, default=100)
	parser.add_argument('--signal_range_test', type=list, default=[0.25,0.45])
	parser.add_argument('--signal_N_test', type=int, default=200)
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

	# Load pretrained parameters
	encoder, class_out, selected_classes = load_params(args, encoder, class_out, args.epochs)

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

	# Test
	log.info('Test...')
	signal_test_vals = np.linspace(args.signal_range_test[0], args.signal_range_test[1], args.signal_N_test)
	all_d = []
	for s in range(signal_test_vals.shape[0]):
		y_pred, y_targ = test(args, encoder, class_out, device, test_loader, signal=signal_test_vals[s], noise=args.noise_val_test)
		d = calc_d_prime(y_pred, y_targ)
		log.info('[Signal = ' + '{:.2f}'.format(signal_test_vals[s]) + '] ' + \
			 	 '[d-prime = ' + '{:.2f}'.format(d) + ']')
		all_d.append(d)
	# Save test results
	test_dir = './test/'
	check_path(test_dir)
	model_dir = test_dir + 'run' + str(args.run) + '/'
	check_path(model_dir)
	np.savez(model_dir + 'i3_d_prime_results.npz',
			 signal_test_vals=signal_test_vals,
			 noise_test_vals=np.array([args.noise_val_test]),
			 all_d=np.array(all_d))

if __name__ == '__main__':
	main()