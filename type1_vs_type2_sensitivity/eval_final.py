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
from fit_meta_d_MLE import *
from fit_rs_meta_d_MLE import *
from util import log

def check_path(path):
	if not os.path.exists(path):
		os.mkdir(path)

def test(args, encoder, class_out, conf_out, 
		device, loader, s1_signal=1.0, s2_signal=1.0, noise=0.0):
	# Set to evaluation mode
	encoder.eval()
	class_out.eval()
	# Iterate over batches
	all_y_pred = []
	all_y_targ = []
	all_conf = []
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
		# Get model predictions
		z = encoder(x, device)
		y_pred = class_out(z).squeeze().round()
		conf = conf_out(z).squeeze()
		# Collect data for d'/meta-d' calculation
		all_y_pred.append(y_pred.cpu().detach().numpy())
		all_y_targ.append(y_targ.cpu().numpy())
		all_conf.append(conf.cpu().detach().numpy())
	# Data for d'/meta-d' calculation
	all_y_pred = np.concatenate(all_y_pred)
	all_y_targ = np.concatenate(all_y_targ)
	all_conf = np.concatenate(all_conf)
	return all_y_pred, all_y_targ, all_conf

def compute_sensitivity(y_pred, y_targ, conf):
	# Sort trials based on confidence thresholds
	conf2_thresh = 0.5 + (0.5/4)
	conf3_thresh = 0.5 + 2*(0.5/4)
	conf4_thresh = 0.5 + 3*(0.5/4)
	conf1_trials = conf < conf2_thresh
	conf2_trials = np.logical_and(conf >= conf2_thresh, conf < conf3_thresh) 
	conf3_trials = np.logical_and(conf >= conf3_thresh, conf < conf4_thresh) 
	conf4_trials = conf >= conf4_thresh
	# Sort trials based on y target and prediction
	s1 = y_targ == 0
	s2 = y_targ == 1
	resp_s1 = y_pred == 0
	resp_s2 = y_pred == 1
	s1_resp_s1 = np.logical_and(s1, resp_s1)
	s1_resp_s2 = np.logical_and(s1, resp_s2)
	s2_resp_s1 = np.logical_and(s2, resp_s1)
	s2_resp_s2 = np.logical_and(s2, resp_s2)
	# Combine confidence rating and y target/response
	s1_resp_s1_conf1 = np.logical_and(s1_resp_s1, conf1_trials) 
	s1_resp_s1_conf2 = np.logical_and(s1_resp_s1, conf2_trials)
	s1_resp_s1_conf3 = np.logical_and(s1_resp_s1, conf3_trials)
	s1_resp_s1_conf4 = np.logical_and(s1_resp_s1, conf4_trials)
	s1_resp_s2_conf1 = np.logical_and(s1_resp_s2, conf1_trials)
	s1_resp_s2_conf2 = np.logical_and(s1_resp_s2, conf2_trials)
	s1_resp_s2_conf3 = np.logical_and(s1_resp_s2, conf3_trials)
	s1_resp_s2_conf4 = np.logical_and(s1_resp_s2, conf4_trials)
	s2_resp_s1_conf1 = np.logical_and(s2_resp_s1, conf1_trials)
	s2_resp_s1_conf2 = np.logical_and(s2_resp_s1, conf2_trials)
	s2_resp_s1_conf3 = np.logical_and(s2_resp_s1, conf3_trials)
	s2_resp_s1_conf4 = np.logical_and(s2_resp_s1, conf4_trials)
	s2_resp_s2_conf1 = np.logical_and(s2_resp_s2, conf1_trials)
	s2_resp_s2_conf2 = np.logical_and(s2_resp_s2, conf2_trials)
	s2_resp_s2_conf3 = np.logical_and(s2_resp_s2, conf3_trials)
	s2_resp_s2_conf4 = np.logical_and(s2_resp_s2, conf4_trials)
	nR_s1 = [s1_resp_s1_conf4.sum() + 1/8, s1_resp_s1_conf3.sum() + 1/8, s1_resp_s1_conf2.sum() + 1/8, s1_resp_s1_conf1.sum() + 1/8,
			 s1_resp_s2_conf1.sum() + 1/8, s1_resp_s2_conf2.sum() + 1/8, s1_resp_s2_conf3.sum() + 1/8, s1_resp_s2_conf4.sum() + 1/8]
	nR_s2 = [s2_resp_s1_conf4.sum() + 1/8, s2_resp_s1_conf3.sum() + 1/8, s2_resp_s1_conf2.sum() + 1/8, s2_resp_s1_conf1.sum() + 1/8,
			 s2_resp_s2_conf1.sum() + 1/8, s2_resp_s2_conf2.sum() + 1/8, s2_resp_s2_conf3.sum() + 1/8, s2_resp_s2_conf4.sum() + 1/8]
	# Compute overall d' and meta-d'
	fit = fit_meta_d_MLE(nR_s1, nR_s2)
	d = fit['da']
	meta_d = fit['meta_da']
	# Compute response-specific meta-d'
	fit_rs = fit_rs_meta_d_MLE(nR_s1, nR_s2)
	meta_d_rS1 = fit_rs['meta_da_rS1']
	meta_d_rS2 = fit_rs['meta_da_rS2']
	return d, meta_d, meta_d_rS1, meta_d_rS2

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

	# Test
	log.info('Test...')
	all_d = []
	all_meta_d = []
	all_meta_d_rS1 = []
	all_meta_d_rS2 = []
	all_y_pred = []
	all_y_targ = []
	all_conf = []
	for s in range(s2_signal.shape[0]):
		y_pred, y_targ, conf = test(args, encoder, class_out, conf_out, device, test_loader, s1_signal=s1_signal, s2_signal=s2_signal[s], noise=args.noise_val_test)
		d, meta_d, meta_d_rS1, meta_d_rS2 = compute_sensitivity(y_pred, y_targ, conf)
		log.info('[S1 signal = ' + '{:.2f}'.format(s1_signal) + '] ' + \
				 '[S2 signal = ' + '{:.2f}'.format(s2_signal[s]) + '] ' + \
			 	 '[d-prime = ' + '{:.2f}'.format(d) + '] ' + \
			 	 '[meta-d-prime = ' + '{:.2f}'.format(meta_d) + '] ' + \
			 	 '[rS1 meta-d-prime = ' + '{:.2f}'.format(meta_d_rS1) + '] ' + \
			 	 '[rS2 meta-d-prime = ' + '{:.2f}'.format(meta_d_rS2) + ']')
		all_d.append(d)
		all_meta_d.append(meta_d)
		all_meta_d_rS1.append(meta_d_rS1)
		all_meta_d_rS2.append(meta_d_rS2)
		all_y_pred.append(y_pred)
		all_y_targ.append(y_targ)
		all_conf.append(conf)
	# Save test results
	test_dir = './test/'
	check_path(test_dir)
	model_dir = test_dir + 'run' + str(args.run) + '/'
	check_path(model_dir)
	np.savez(model_dir + 'test_results.npz',
			 s1_signal=s1_signal,
			 s2_signal=s2_signal,
			 noise_test_val=args.noise_val_test,
			 all_d=np.array(all_d),
			 all_meta_d=np.array(all_meta_d),
			 all_meta_d_rS1=np.array(all_meta_d_rS1),
			 all_meta_d_rS2=np.array(all_meta_d_rS2),
			 all_y_pred=np.array(all_y_pred),
			 all_y_targ=np.array(all_y_targ),
			 all_conf=np.array(all_conf))

if __name__ == '__main__':
	main()