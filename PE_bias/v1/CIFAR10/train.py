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

def train(args, model, device, train_loader, optimizer, epoch):
	# Create file for saving training progress
	train_prog_dir = './train_prog/'
	check_path(train_prog_dir)
	run_dir = train_prog_dir + 'run' + args.run + '/'
	check_path(run_dir)
	train_prog_fname = run_dir + 'epoch_' + str(epoch) + '.txt'
	train_prog_f = open(train_prog_fname, 'w')
	train_prog_f.write('batch class_loss class_acc conf_loss conf\n')
	# Set to training mode
	model.train()
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
		noise = noise.view(-1, 1, 1, 1).repeat(1, x.shape[1], x.shape[2], x.shape[3])
		noise = (torch.randn(x.shape) * noise).to(device)
		x = x + noise
		# Threshold image
		x = nn.Hardtanh()(x)
		# Zero out gradients for optimizer 
		optimizer.zero_grad()
		# Get model predictions
		y_pred_linear, y_pred, conf = model(x)
		# Classification loss
		class_loss_fn = torch.nn.CrossEntropyLoss()
		class_loss = class_loss_fn(y_pred_linear, y_targ)
		# Classification accuracy
		correct_preds = torch.eq(y_pred, y_targ).type(torch.float)
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

def save_params(args, model, epoch):
	# Directory
	params_dir = './trained_models/'
	check_path(params_dir)
	run_dir = params_dir + 'run' + args.run + '/'
	check_path(run_dir)
	# Save parameters
	params_fname = run_dir + 'epoch' + str(epoch) + '_params.pt'
	torch.save(model.state_dict(), params_fname)

def main():

	# Settings
	parser = argparse.ArgumentParser()
	parser.add_argument('--N_res_blocks', type=int, default=9)
	parser.add_argument('--kaiming_init', action='store_true', default=True)
	parser.add_argument('--train-batch-size', type=int, default=128)
	parser.add_argument('--signal_range', type=list, default=[0.1, 1.0])
	parser.add_argument('--noise_range', type=list, default=[0.1, 0.2])
	parser.add_argument('--data_aug', action='store_true', default=True)
	parser.add_argument('--epochs', type=int, default=164)
	parser.add_argument('--optim', type=str, default='sgd')
	parser.add_argument('--lr', type=float, default=5e-4)
	parser.add_argument('--lr_schedule', action='store_true', default=True)
	parser.add_argument('--no-cuda', action='store_true', default=False)
	parser.add_argument('--log_interval', type=int, default=10)
	parser.add_argument('--device', type=int, default=0) 
	parser.add_argument('--run', type=str, default='1')
	args = parser.parse_args()
		
	# Set up cuda	
	use_cuda = not args.no_cuda and torch.cuda.is_available()
	device = torch.device("cuda:" + str(args.device) if use_cuda else "cpu")
	kwargs = {'num_workers': 1, 'pin_memory': True} if use_cuda else {}

	# Create training/test sets from CIFAR10
	log.info('Loading CIFAR10 training and test sets...')
	# Set up directory to download datasets to
	dset_dir = './datasets'
	check_path(dset_dir)
	# Transforms
	train_transforms = []
	train_transforms.append(transforms.RandomCrop(32, padding=4))
	train_transforms.append(transforms.RandomHorizontalFlip())
	train_transforms.append(transforms.ToTensor())
	# Dataset
	train_set = datasets.CIFAR10(dset_dir, train=True, download=True, transform=transforms.Compose(train_transforms))
	# Data loader
	train_loader = torch.utils.data.DataLoader(train_set, batch_size=args.train_batch_size, shuffle=True, **kwargs)

	# Build model
	log.info('Building model...')
	model = Model(args).to(device)

	# Create optimizer
	if args.lr_schedule:
		args.lr = 0.1
	log.info('Setting up optimizer...')
	if args.optim == 'adam':
		optimizer = optim.Adam(model.parameters(), lr=args.lr)
	elif args.optim == 'sgd':
		optimizer = optim.SGD(model.parameters(), lr=args.lr, weight_decay=1e-4, momentum=0.9)

	# Train
	log.info('Training begins...')
	all_test_epoch = []
	all_test_acc = []
	all_test_conf = []
	for epoch in range(1, args.epochs + 1):
		# Training loop
		train(args, model, device, train_loader, optimizer, epoch)
		# Update learning rate
		if args.lr_schedule:
			if epoch == 82:
				optimizer.param_groups[0]['lr'] = 0.01
			elif epoch == 123:
				optimizer.param_groups[0]['lr'] = 0.001

	# Save model
	save_params(args, model, epoch)

if __name__ == '__main__':
	main()