import torch
import torch.nn as nn
from torchvision import datasets, transforms
from torch.utils.data import DataLoader
import numpy as np
from scipy.stats import multivariate_normal, norm
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

def test(args, encoder, pos, train_s1_dist, train_s2_dist, device, loader, s1_signal=0.5, s2_signal=1.0, noise=0.0):
	# Set to evaluation mode
	encoder.eval()
	# Iterate over batches
	all_z = []
	all_y_targ = []
	for batch_idx, (data, target) in enumerate(loader):
		# Load data
		x = data.to(device)
		y_targ = target.to(device)
		# Scale signal (independently for each stimulus)
		x[y_targ==0,:,:,:] = x[y_targ==0,:,:,:] * s1_signal
		x[y_targ==1,:,:,:] = x[y_targ==1,:,:,:] * s2_signal
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
	# Convert latent representations to classification predictions
	z_s1_s1_dist_pdf = train_s1_dist.pdf(z_s1)
	z_s1_s2_dist_pdf = train_s2_dist.pdf(z_s1)
	z_s1_y_pred = (((z_s1_s2_dist_pdf * 0.5) / ((z_s1_s1_dist_pdf * 0.5) + (z_s1_s2_dist_pdf * 0.5))) > 0.5).astype(float)
	z_s2_s1_dist_pdf = train_s1_dist.pdf(z_s2)
	z_s2_s2_dist_pdf = train_s2_dist.pdf(z_s2)
	z_s2_y_pred = (((z_s2_s2_dist_pdf * 0.5) / ((z_s2_s1_dist_pdf * 0.5) + (z_s2_s2_dist_pdf * 0.5))) > 0.5).astype(float)
	y_pred = np.concatenate([z_s1_y_pred, z_s2_y_pred])
	y_targ = np.concatenate([np.zeros(z_s1.shape[0]), np.ones(z_s2.shape[0])])
	# Calculate d' and acc
	d_prime = calc_d_prime(y_pred, y_targ)
	# Report
	log.info('[Signal = ' + '{:.2f}'.format(s1_signal) + '] ' + \
			 "[d' = " + '{:.2f}'.format(d_prime) + ']')
	return d_prime

def main():

	# Settings
	parser = argparse.ArgumentParser()
	parser.add_argument('--test-batch-size', type=int, default=100)
	parser.add_argument('--train_regime', type=str, default='standard', help="{'standard', 'fixed_mu'}")
	parser.add_argument('--signal_range_test', type=list, default=[0.25,0.45])
	parser.add_argument('--signal_N_test', type=int, default=200)
	parser.add_argument('--noise_val_test', type=float, default=2.0)
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

	# Create test set from MNIST
	log.info('Loading MNIST test set...')
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

	# Test
	log.info('Test...')
	signal_test_vals = np.linspace(args.signal_range_test[0], args.signal_range_test[1], args.signal_N_test)
	all_d = []
	for s in range(signal_test_vals.shape[0]):
		d = test(args, encoder, pos, train_s1_dist, train_s2_dist, device, test_loader, s1_signal=signal_test_vals[s], s2_signal=signal_test_vals[s], noise=args.noise_val_test)
		all_d.append(d)
	# Save test results
	test_dir = './test/'
	check_path(test_dir)
	model_dir = test_dir + args.train_regime + '_training/'
	check_path(model_dir)
	model_dir += 'run' + str(args.run) + '/'
	check_path(model_dir)
	np.savez(model_dir + 'i3_dprime_results.npz',
			 signal_test_vals=signal_test_vals,
			 noise_test_vals=np.array([args.noise_val_test]),
			 all_d=np.array(all_d))

if __name__ == '__main__':
	main()