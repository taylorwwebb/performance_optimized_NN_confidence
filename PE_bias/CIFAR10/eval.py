import torch
import torch.nn as nn
import torch.optim as optim
from torchvision import datasets, transforms
import numpy as np
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

def test(args, model, device, loader, signal=1.0, noise=0.0):
	# Set to evaluation mode
	model.eval()
	# Iterate over batches
	all_test_acc = []
	all_test_conf = []
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
		y_pred_linear, y_pred, conf = model(x)
		# Classification accuracy
		correct_preds = torch.eq(y_pred, y_targ).type(torch.float)
		class_acc = correct_preds.mean().item() * 100.0
		all_test_acc.append(class_acc)
		# Overall confidence
		avg_conf = conf.mean().item() * 100.0
		all_test_conf.append(avg_conf)
	# Average test accuracy and confidence
	avg_test_acc = np.mean(all_test_acc)
	avg_test_conf = np.mean(all_test_conf)
	# Report
	log.info('[Signal = ' + '{:.2f}'.format(signal) + '] ' + \
			 '[Noise = ' + '{:.2f}'.format(noise) + '] ' + \
			 '[Class. Acc. = ' + '{:.2f}'.format(avg_test_acc) + '] ' + \
			 '[Conf. = ' + '{:.2f}'.format(avg_test_conf) + ']')
	return avg_test_acc, avg_test_conf

def load_params(args, model, epoch):
	# Directory
	params_dir = './trained_models/'
	check_path(params_dir)
	run_dir = params_dir + 'run' + args.run + '/'
	check_path(run_dir)
	# Load parameters
	params_fname = run_dir + 'epoch' + str(epoch) + '_params.pt'
	model.load_state_dict(torch.load(params_fname))
	return model

def main():

	# Settings
	parser = argparse.ArgumentParser()
	parser.add_argument('--N_res_blocks', type=int, default=9)
	parser.add_argument('--kaiming_init', action='store_true', default=True)
	parser.add_argument('--test-batch-size', type=int, default=128)
	parser.add_argument('--signal_range_test', type=list, default=[0.0, 0.2])
	parser.add_argument('--signal_N_test', type=int, default=500)
	parser.add_argument('--noise_range', type=list, default=[0.1, 0.2])
	parser.add_argument('--noise_N_test', type=int, default=2)
	parser.add_argument('--data_aug', action='store_true', default=True)
	parser.add_argument('--epochs', type=int, default=164)	
	parser.add_argument('--optim', type=str, default='sgd')
	parser.add_argument('--lr', type=float, default=5e-4)
	parser.add_argument('--lr_schedule', action='store_true', default=True)
	parser.add_argument('--no-cuda', action='store_true', default=False)
	parser.add_argument('--device', type=int, default=0)
	parser.add_argument('--run', type=str, default='1')
	args = parser.parse_args()
		
	# Set up cuda	
	use_cuda = not args.no_cuda and torch.cuda.is_available()
	device = torch.device("cuda:" + str(args.device) if use_cuda else "cpu")
	kwargs = {'num_workers': 1, 'pin_memory': True} if use_cuda else {}

	# Create training/test sets from CIFAR10
	log.info('Loading CIFAR10 test set...')
	# Set up directory to download datasets to
	dset_dir = './datasets'
	check_path(dset_dir)
	# Transforms
	test_transforms = []
	test_transforms.append(transforms.ToTensor())
	# Dataset
	test_set = datasets.CIFAR10(dset_dir, train=False, transform=transforms.Compose(test_transforms))
	# Data loader
	test_loader = torch.utils.data.DataLoader(test_set, batch_size=args.test_batch_size, shuffle=True, **kwargs)

	# Build model
	log.info('Building model...')
	model = Model(args).to(device)

	# Load pretrained model
	model = load_params(args, model, args.epochs)

	# Evaluate model over range of signal and noise values
	signal_test_vals = np.linspace(args.signal_range_test[0], args.signal_range_test[1], args.signal_N_test)
	noise_test_vals = np.linspace(args.noise_range[0], args.noise_range[1], args.noise_N_test)
	all_test_acc = []
	all_test_conf = []
	for n in range(noise_test_vals.shape[0]):
		all_signal_test_acc = []
		all_signal_test_conf = []
		for s in range(signal_test_vals.shape[0]):
			test_acc, test_conf = test(args, model, device, test_loader, signal=signal_test_vals[s], noise=noise_test_vals[n])
			all_signal_test_acc.append(test_acc)
			all_signal_test_conf.append(test_conf)
		all_test_acc.append(all_signal_test_acc)
		all_test_conf.append(all_signal_test_conf)
	# Save test results
	test_dir = './test/'
	check_path(test_dir)
	run_dir = test_dir + 'run' + args.run + '/'
	check_path(run_dir)
	np.savez(run_dir + 'test_results.npz',
			 signal_test_vals=signal_test_vals,
			 noise_test_vals=noise_test_vals,
			 all_test_acc=np.array(all_test_acc),
			 all_test_conf=np.array(all_test_conf))

if __name__ == '__main__':
	main()