import torch
import torch.nn as nn
import torch.optim as optim
from torchvision import datasets, transforms
from torch.utils.data import DataLoader
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

def train(args, encoder, decoder, 
		device, train_loader, optimizer, epoch):
	# Create file for saving training progress
	train_prog_dir = './train_prog/'
	check_path(train_prog_dir)
	model_dir = train_prog_dir + args.train_regime + '_training/'
	check_path(model_dir)
	model_dir += 'run' + str(args.run) + '/'
	check_path(model_dir)
	train_prog_fname = model_dir + 'epoch_' + str(epoch) + '.txt'
	train_prog_f = open(train_prog_fname, 'w')
	train_prog_f.write('batch recon_loss KL_div\n')
	# Set to training mode
	encoder.train()
	decoder.train()
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
		x_noise = x + noise
		# Threshold image
		x_noise = nn.Hardtanh()(x_noise)
		# Zero out gradients for optimizer 
		optimizer.zero_grad()
		# Get model predictions
		z_mn, z_logvar = encoder(x_noise)
		z_std = torch.sqrt(z_logvar.exp())
		z = z_mn + z_std * torch.normal(mean=torch.zeros(z_mn.shape), std=torch.ones(z_mn.shape)).to(device)
		x_pred = decoder(z)
		# Reconstruction loss
		recon_loss = nn.MSELoss(reduction='sum')(x_pred, x) / x.shape[0]
		# VAE loss
		KL_div = (-0.5 * (1 + z_logvar - z_mn.pow(2) - z_logvar.exp())).sum(1).mean(0)
		loss = recon_loss + KL_div
		# Update model
		loss.backward()
		optimizer.step()
		# Batch duration
		end_time = time.time()
		batch_dur = end_time - start_time
		# Report progress
		if batch_idx % args.log_interval == 0:
			log.info('[Epoch: ' + str(epoch) + '] ' + \
					 '[Batch: ' + str(batch_idx) + ' of ' + str(len(train_loader)) + '] ' + \
					 '[Reconstruction loss = ' + '{:.4f}'.format(recon_loss.item()) + '] ' + \
					 '[KL divergence = ' + '{:.4f}'.format(KL_div.item()) + '] ' + \
					 '[' + '{:.3f}'.format(batch_dur) + ' sec/batch]')
			# Save progress to file
			train_prog_f.write(str(batch_idx) + ' ' +\
				               '{:.4f}'.format(recon_loss.item()) + ' ' + \
				               '{:.4f}'.format(KL_div.item()) + '\n')
	train_prog_f.close()

def vis_z(args, encoder, decoder, 
		device, test_loader, epoch):
	# Create file for saving reconstructions
	recon_dir = './vis_z/'
	check_path(recon_dir)
	model_dir = recon_dir + args.train_regime + '_training/'
	check_path(model_dir)
	model_dir += 'run' + str(args.run) + '/'
	check_path(model_dir)
	# Iterate over batches
	log.info('Visualizing latent space...')
	all_y_targ = []
	all_z = []
	all_z_mn = []
	for batch_idx, (data, target) in enumerate(test_loader):
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
		x_noise = x + noise
		# Threshold image
		x_noise = nn.Hardtanh()(x_noise)
		# Get latent representation
		# Real digits
		z_mn, z_logvar = encoder(x_noise)
		z_std = torch.sqrt(z_logvar.exp())
		z = z_mn + z_std * torch.normal(mean=torch.zeros(z_mn.shape), std=torch.ones(z_mn.shape)).to(device)
		# Collect data
		all_z.append(z.detach().cpu().numpy())
		all_z_mn.append(z_mn.detach().cpu().numpy())
		all_y_targ.append(y_targ.detach().cpu().numpy())
	# Concatenate batches
	all_z = np.concatenate(all_z)
	all_z_mn = np.concatenate(all_z_mn)
	all_y_targ = np.concatenate(all_y_targ)
	# Save results
	save_fname = model_dir + 'epoch_' + str(epoch) + '.npz'
	np.savez(save_fname, z=all_z, y_targ=all_y_targ, z_mn=all_z_mn)

def save_params(args, encoder, decoder, selected_classes, epoch):
	# Directory
	params_dir = './trained_models/'
	check_path(params_dir)
	model_dir = params_dir + args.train_regime + '_training/'
	check_path(model_dir)
	run_dir = model_dir + 'run' + str(args.run) + '/'
	check_path(run_dir)
	# Save parameters
	# Encoder
	encoder_fname = run_dir + 'epoch' + str(epoch) + '_encoder.pt'
	torch.save(encoder.state_dict(), encoder_fname)
	# Decoder
	decoder_fname = run_dir + 'epoch' + str(epoch) + '_decoder.pt'
	torch.save(decoder.state_dict(), decoder_fname)
	# Save selected classes
	selected_classes_fname = run_dir + 'selected_classes.npz'
	np.savez(selected_classes_fname, selected_classes=selected_classes)

def main():

	# Settings
	parser = argparse.ArgumentParser()
	parser.add_argument('--train-batch-size', type=int, default=32)
	parser.add_argument('--test-batch-size', type=int, default=100)
	parser.add_argument('--train_regime', type=str, default='standard', help="{'standard', 'fixed_mu'}")
	parser.add_argument('--signal_range', type=list, default=[0.1,1.0])
	parser.add_argument('--noise_range', type=list, default=[1.0, 2.0])
	parser.add_argument('--img_size', type=int, default=32)
	parser.add_argument('--latent_dim', type=int, default=2)
	parser.add_argument('--epochs', type=int, default=20)
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
	train_loader = DataLoader(train_set, batch_size=args.train_batch_size, shuffle=True, drop_last=True)
	test_loader = DataLoader(test_set, batch_size=args.test_batch_size, shuffle=True, drop_last=True)

	# Training regime
	if args.train_regime == 'fixed_mu':
		args.signal_range = [np.mean(args.signal_range), np.mean(args.signal_range)]

	# Build model
	log.info('Building model...')
	encoder = Encoder(args).to(device)
	decoder = Decoder(args).to(device)
	all_modules = nn.ModuleList([encoder, decoder])

	# Create optimizer
	log.info('Setting up optimizer...')
	optimizer = optim.Adam(all_modules.parameters(), lr=args.lr)

	# Train
	log.info('Training begins...')
	for epoch in range(1, args.epochs + 1):
		# Training loop
		train(args, encoder, decoder, device, train_loader, optimizer, epoch)

	# Visualize latent space
	vis_z(args, encoder, decoder, device, test_loader, epoch)

	# Save model
	save_params(args, encoder, decoder, selected_classes, epoch)

if __name__ == '__main__':
	main()