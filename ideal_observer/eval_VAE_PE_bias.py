import torch
import torch.nn as nn
from torchvision import datasets, transforms
from torch.utils.data import DataLoader
import numpy as np
from scipy.stats import multivariate_normal
import matplotlib.pyplot as plt
import argparse
import os
import sys

# Prevent python from saving out .pyc files
sys.dont_write_bytecode = True

from architecture import *
from util import log

def check_path(path):
	if not os.path.exists(path):
		os.mkdir(path)

def test(args, encoder, pos, train_p_correct, device, loader, signal=1.0, noise=0.0):
	# Set to evaluation mode
	encoder.eval()
	# Iterate over batches
	all_z = []
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
		# Get latent representation
		z, _ = encoder(x)
		all_z.append(z.cpu().detach().numpy())
		all_y_targ.append(y_targ.cpu().detach().numpy())
	# Concatenate
	all_z = np.concatenate(all_z)
	all_y_targ = np.concatenate(all_y_targ)
	# Estimate test distribution
	z_s1 = all_z[all_y_targ==0]
	z_s1_mn = z_s1.mean(0)
	z_s1_cov = np.cov(z_s1, rowvar=False)
	z_s1_dist = multivariate_normal(z_s1_mn, z_s1_cov)
	z_s2 = all_z[all_y_targ==1]
	z_s2_mn = z_s2.mean(0)
	z_s2_cov = np.cov(z_s2, rowvar=False)
	z_s2_dist = multivariate_normal(z_s2_mn, z_s2_cov)
	# Evaluate test distributions and calculate p(correct)
	test_s1_pdf = z_s1_dist.pdf(pos)
	test_s2_pdf = z_s2_dist.pdf(pos)
	test_p_s1 = (test_s1_pdf * 0.5) / ((test_s1_pdf * 0.5) + (test_s2_pdf * 0.5))
	test_p_s2 = (test_s2_pdf * 0.5) / ((test_s1_pdf * 0.5) + (test_s2_pdf * 0.5))
	test_p_correct = np.stack([test_p_s1, test_p_s2],0).max(0)
	# Replace nan's with 0's (shouldn't matter, since these are regions of the test distribution with pdf=0)
	test_p_correct = np.nan_to_num(test_p_correct)
	# Accuracy: p(correct|x) using test distribution
	s1_acc = (test_p_correct.flatten() * test_s1_pdf.flatten()).sum() / test_s1_pdf.flatten().sum()
	s2_acc = (test_p_correct.flatten() * test_s2_pdf.flatten()).sum() / test_s2_pdf.flatten().sum()
	acc = np.mean([s1_acc,s2_acc])
	# Confidence: p(correct|x) using training distribution
	s1_conf = (train_p_correct.flatten() * test_s1_pdf.flatten()).sum() / test_s1_pdf.flatten().sum()
	s2_conf = (train_p_correct.flatten() * test_s2_pdf.flatten()).sum() / test_s2_pdf.flatten().sum()
	conf = np.mean([s1_conf,s2_conf])
	s1_conf_correct = (train_p_correct.flatten() * test_s1_pdf.flatten())[(test_p_s1 > 0.5).flatten()].sum() / test_s1_pdf.flatten()[(test_p_s1 > 0.5).flatten()].sum()
	s1_conf_incorrect = (train_p_correct.flatten() * test_s1_pdf.flatten())[(test_p_s1 < 0.5).flatten()].sum() / test_s1_pdf.flatten()[(test_p_s1 < 0.5).flatten()].sum()
	s2_conf_correct = (train_p_correct.flatten() * test_s2_pdf.flatten())[(test_p_s2 > 0.5).flatten()].sum() / test_s2_pdf.flatten()[(test_p_s2 > 0.5).flatten()].sum()
	s2_conf_incorrect = (train_p_correct.flatten() * test_s2_pdf.flatten())[(test_p_s2 < 0.5).flatten()].sum() / test_s2_pdf.flatten()[(test_p_s2 < 0.5).flatten()].sum()
	conf_correct = np.mean([s1_conf_correct,s2_conf_correct])
	conf_incorrect = np.mean([s1_conf_incorrect,s2_conf_incorrect])
	# Report
	log.info('[Signal = ' + '{:.2f}'.format(signal) + '] ' + \
			 '[Noise = ' + '{:.2f}'.format(noise) + '] ' + \
			 '[Class. Acc. = ' + '{:.2f}'.format(acc) + '] ' + \
			 '[Conf. = ' + '{:.2f}'.format(conf) + '] ' + \
			 '[Conf. (correct) = ' + '{:.2f}'.format(conf_correct) + '] ' + \
			 '[Conf. (incorrect) = ' + '{:.2f}'.format(conf_incorrect) + ']')
	return acc, conf, conf_correct, conf_incorrect

def main():

	# Settings
	parser = argparse.ArgumentParser()
	parser.add_argument('--test-batch-size', type=int, default=100)
	parser.add_argument('--train_regime', type=str, default='standard', help="{'standard', 'fixed_mu'}")
	parser.add_argument('--signal_range_test', type=list, default=[0.0, 1.0])
	parser.add_argument('--signal_N_test', type=int, default=500)
	parser.add_argument('--noise_range', type=list, default=[1.0, 2.0])
	parser.add_argument('--noise_N_test', type=int, default=2)
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
	# Test set
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
	# Signal and noise values for test
	signal_test_vals = np.linspace(args.signal_range_test[0], args.signal_range_test[1], args.signal_N_test)
	noise_test_vals = np.linspace(args.noise_range[0], args.noise_range[1], args.noise_N_test)
	all_test_acc = []
	all_test_conf = []
	all_test_conf_correct = []
	all_test_conf_incorrect = []
	for n in range(noise_test_vals.shape[0]):
		all_signal_test_acc = []
		all_signal_test_conf = []
		all_signal_test_conf_correct = []
		all_signal_test_conf_incorrect = []
		for s in range(signal_test_vals.shape[0]):
			test_acc, test_conf, test_conf_correct, test_conf_incorrect = test(args, encoder, pos, train_p_correct, device, test_loader, signal=signal_test_vals[s], noise=noise_test_vals[n])
			all_signal_test_acc.append(test_acc)
			all_signal_test_conf.append(test_conf)
			all_signal_test_conf_correct.append(test_conf_correct)
			all_signal_test_conf_incorrect.append(test_conf_incorrect)
		all_test_acc.append(all_signal_test_acc)
		all_test_conf.append(all_signal_test_conf)
		all_test_conf_correct.append(all_signal_test_conf_correct)
		all_test_conf_incorrect.append(all_signal_test_conf_incorrect)
	# Save test results
	test_dir = './test/'
	check_path(test_dir)
	model_dir = test_dir + args.train_regime + '_training/'
	check_path(model_dir)
	model_dir += 'run' + str(args.run) + '/'
	check_path(model_dir)
	np.savez(model_dir + 'PE_bias_results.npz',
			 signal_test_vals=signal_test_vals,
			 noise_test_vals=noise_test_vals,
			 all_test_acc=np.array(all_test_acc),
			 all_test_conf=np.array(all_test_conf),
			 all_test_conf_correct=np.array(all_test_conf_correct),
			 all_test_conf_incorrect=np.array(all_test_conf_incorrect))

if __name__ == '__main__':
	main()