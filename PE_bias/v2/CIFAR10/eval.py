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

def s1s2_test(args, model, device, s1_loader, s2_loader, targ_signal=1.0, nontarg_signal=1.0, noise=0.0):
	# Set to evaluation mode
	model.eval()
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
		y_pred, conf = model(x)
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
	parser.add_argument('--signal_range', type=list, default=[0.1, 1.0])
	parser.add_argument('--signal_test_vals', type=list, default=[0.5, 1.0])
	parser.add_argument('--signal_N_test', type=int, default=500)
	parser.add_argument('--noise_test', type=float, default=0.6)
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
	# Load selected classes
	params_dir = './trained_models/'
	run_dir = params_dir + 'run' + args.run + '/'
	classes_fname = run_dir + 'epoch' + str(args.epochs) + '_classes.npz'
	selected_classes = np.load(classes_fname)['selected_classes']
	# Test sets (separate datasets for s1/s2)
	s1 = test_set.targets == selected_classes[0]
	s1_test_set = deepcopy(test_set)
	s1_test_set.targets = np.array(s1_test_set.targets)[s1]
	s1_test_set.data = np.array(s1_test_set.data)[s1]
	s2 = test_set.targets == selected_classes[1]
	s2_test_set = deepcopy(test_set)
	s2_test_set.targets = np.array(s2_test_set.targets)[s2]
	s2_test_set.data = np.array(s2_test_set.data)[s2]
	# Convert to PyTorch DataLoaders
	log.info('Converting to DataLoaders...')
	s1_test_loader = DataLoader(s1_test_set, batch_size=args.test_batch_size, shuffle=True, drop_last=True)
	s2_test_loader = DataLoader(s2_test_set, batch_size=args.test_batch_size, shuffle=True, drop_last=True)	

	# Build model
	log.info('Building model...')
	model = Model(args).to(device)

	# Load pretrained model
	model = load_params(args, model, args.epochs)

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
			test_acc, test_conf, test_conf_correct, test_conf_incorrect = s1s2_test(args, model, device, s1_test_loader, s2_test_loader, targ_signal=low_PE_targ_signal, nontarg_signal=nontarg_signal_test_vals[s], noise=args.noise_test)
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
			test_acc, test_conf, test_conf_correct, test_conf_incorrect = s1s2_test(args, model, device, s1_test_loader, s2_test_loader, targ_signal=high_PE_targ_signal, nontarg_signal=nontarg_signal_test_vals[s], noise=args.noise_test)
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