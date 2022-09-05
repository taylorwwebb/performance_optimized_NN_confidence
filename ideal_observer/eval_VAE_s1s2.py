import torch
import torch.nn as nn
from torchvision import datasets, transforms
from torch.utils.data import DataLoader
import numpy as np
from scipy.stats import multivariate_normal
import matplotlib.pyplot as plt
import argparse
from copy import deepcopy
import os
import sys

# Prevent python from saving out .pyc files
sys.dont_write_bytecode = True

from architecture import *
from util import log

def check_path(path):
	if not os.path.exists(path):
		os.mkdir(path)

def s1s2_test(args, encoder, pos, train_s1_dist, train_s2_dist, train_p_correct, device, s1_loader, s2_loader, s1_signal_val=1.0, s2_signal_val=1.0, noise_val=0.0):
	# Set to evaluation mode
	encoder.eval()
	# Iterate over batches
	all_z = []
	all_y_targ = []
	for batch_idx, ((data_s1, target_s1), (data_s2, target_s2)) in enumerate(zip(s1_loader, s2_loader)):
		# Load data
		x_s1 = data_s1.to(device)
		x_s2 = data_s2.to(device)
		# Generate targets (based on relative value of s1/s2 signal)
		if s2_signal_val > s1_signal_val:
			y_targ = torch.ones(args.test_batch_size).to(device)
		else:
			y_targ = torch.zeros(args.test_batch_size).to(device)
		# Apply contrast scaling and sumperimpose images
		x_s1 = x_s1 * s1_signal_val
		x_s2 = x_s2 * s2_signal_val
		x, _ = torch.stack([x_s1, x_s2],0).max(0)
		# Scale to [-1, 1]
		x = (x - 0.5) / 0.5
		# Add noise
		x = x + (torch.randn(x.shape) * noise_val).to(device)
		# Threshold image
		x = nn.Hardtanh()(x)
		# Get latent representation
		z, _ = encoder(x)
		all_z.append(z.cpu().detach().numpy())
		all_y_targ.append(y_targ.cpu().detach().numpy())
	# Concatenate
	all_z = np.concatenate(all_z)
	all_y_targ = np.concatenate(all_y_targ)
	# Estimate test distribution
	z_mn = all_z.mean(0)
	z_cov = np.cov(all_z, rowvar=False)
	z_dist = multivariate_normal(z_mn, z_cov)
	z_pdf = z_dist.pdf(pos)	
	# Accuracy
	z_s1_dist_pdf = train_s1_dist.pdf(all_z)
	z_s2_dist_pdf = train_s2_dist.pdf(all_z)
	y_pred = (((z_s2_dist_pdf * 0.5) / ((z_s1_dist_pdf * 0.5) + (z_s2_dist_pdf * 0.5))) > 0.5).astype(float)
	acc = (y_pred == all_y_targ).astype(float).mean()
	# Confidence: p(correct|x) using training distribution
	conf = (train_p_correct.flatten() * z_pdf.flatten()).sum() / z_pdf.flatten().sum()
	# Report
	log.info('[s1 signal = ' + '{:.2f}'.format(s1_signal_val) + '] ' + \
			 '[s2 signal = ' + '{:.2f}'.format(s2_signal_val) + '] ' + \
			 '[Class. Acc. = ' + '{:.2f}'.format(acc) + '] ' + \
			 '[Conf. = ' + '{:.2f}'.format(conf) + ']')
	return acc, conf

def main():

	# Settings
	parser = argparse.ArgumentParser()
	parser.add_argument('--test-batch-size', type=int, default=100)
	parser.add_argument('--train_regime', type=str, default='standard', help="{'standard', fixed_mu'}")
	parser.add_argument('--signal_range', type=list, default=[0.1, 1.0])
	parser.add_argument('--signal_N_test', type=int, default=100)
	parser.add_argument('--noise_test', type=float, default=1.5)
	parser.add_argument('--img_size', type=int, default=32)
	parser.add_argument('--latent_dim', type=int, default=2)
	parser.add_argument('--epochs', type=int, default=20)
	parser.add_argument('--no-cuda', action='store_true', default=False)
	parser.add_argument('--device', type=int, default=0)
	parser.add_argument('--run', type=str, default='1')
	args = parser.parse_args()
		
	# Set up cuda	
	use_cuda = not args.no_cuda and torch.cuda.is_available()
	device = torch.device("cuda:" + str(args.device) if use_cuda else "cpu")
	kwargs = {'num_workers': 1, 'pin_memory': True} if use_cuda else {}

	# Directory with saved model
	params_dir = './trained_models/'
	check_path(params_dir)
	params_dir += args.train_regime + '_training/'
	check_path(params_dir)
	run_dir = params_dir + 'run' + str(args.run) + '/'
	check_path(run_dir)

	# Create training/test sets from MNIST
	log.info('Loading MNIST training/test sets...')
	# Set up directory to download datasets to
	dset_dir = './datasets'
	check_path(dset_dir)
	# Transforms
	transforms_to_apply = transforms.Compose([transforms.Resize(args.img_size), transforms.ToTensor()])
	# Training/test sets
	test_set = datasets.MNIST(dset_dir, train=False, download=True, transform=transforms_to_apply)
	# Load digit classes
	selected_classes_fname = run_dir + 'selected_classes.npz'
	selected_classes = np.load(selected_classes_fname)['selected_classes']
	log.info('Selected classes:')
	log.info(str(selected_classes))
	# Subset dataset to only these classes
	log.info('Subsetting...')
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
	s1_test_loader = DataLoader(s1_test_set, batch_size=args.test_batch_size, shuffle=True, drop_last=True)
	s2_test_loader = DataLoader(s2_test_set, batch_size=args.test_batch_size, shuffle=True, drop_last=True)

	# Build encoder
	log.info('Building model...')
	encoder = Encoder(args).to(device)
	# Load pre-trained parameters
	encoder_fname = run_dir + 'epoch' + str(args.epochs) + '_encoder.pt'
	encoder.load_state_dict(torch.load(encoder_fname))

	# Load training distribution parameters
	training_dist_param_fname = run_dir + 'training_dist_params.npz'
	training_dist_params = np.load(training_dist_param_fname)
	# Generate training distributions
	train_s1_dist = multivariate_normal(training_dist_params['z_s1_mn'], training_dist_params['z_s1_cov'])
	train_s2_dist = multivariate_normal(training_dist_params['z_s2_mn'], training_dist_params['z_s2_cov'])
	# Evaluation range
	x_min = -4
	x_max = 4
	x_int = 0.01
	x_range = np.arange(x_min,x_max+x_int,x_int)
	x, y = np.mgrid[x_min:x_max:x_int, x_min:x_max:x_int]
	pos = np.dstack((x, y))
	# Calculate p(correct|x) using training data distributions
	train_s1_pdf = train_s1_dist.pdf(pos)
	train_s2_pdf = train_s2_dist.pdf(pos)
	train_p_s1 = (train_s1_pdf * 0.5) / ((train_s1_pdf * 0.5) + (train_s2_pdf * 0.5))
	train_p_s2 = (train_s2_pdf * 0.5) / ((train_s1_pdf * 0.5) + (train_s2_pdf * 0.5))
	train_p_correct = np.stack([train_p_s1, train_p_s2],0).max(0)

	# Test
	log.info('Test...')
	# Signal values for test
	signal_test_vals = np.linspace(args.signal_range[0], args.signal_range[1], args.signal_N_test)
	# Evaluate
	all_acc = []
	all_conf = []
	for s1 in range(signal_test_vals.shape[0]):
		all_s1_acc = []
		all_s1_conf = []
		for s2 in range(signal_test_vals.shape[0]):
			test_acc, test_conf = s1s2_test(args, encoder, pos, train_s1_dist, train_s2_dist, train_p_correct, device, s1_test_loader, s2_test_loader, 
				s1_signal_val=signal_test_vals[s1], s2_signal_val=signal_test_vals[s2], noise_val=args.noise_test)
			all_s1_acc.append(test_acc)
			all_s1_conf.append(test_conf)
		all_acc.append(all_s1_acc)
		all_conf.append(all_s1_conf)
	# Save test results
	test_dir = './test/'
	check_path(test_dir)
	model_dir = test_dir + args.train_regime + '_training/'
	check_path(model_dir)
	model_dir += 'run' + str(args.run) + '/'
	check_path(model_dir)
	np.savez(model_dir + 's1s2_results.npz',
		signal_test_vals=signal_test_vals,
		all_acc=all_acc,
		all_conf=all_conf)

if __name__ == '__main__':
	main()