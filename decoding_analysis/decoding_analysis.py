import torch
import torch.nn as nn
import torch.optim as optim
from torchvision import datasets, transforms
from torch.utils.data import Dataset, DataLoader
import numpy as np
from sklearn import metrics
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

def train_decoder(args, encoder, decoder,
		device, train_loader, optimizer, epoch, 
		signal=1.0, img_noise=0.0, decoder_noise=0.0):
	# Create file for saving training progress
	train_prog_dir = './decoder_train_prog/'
	check_path(train_prog_dir)
	train_prog_dir += args.decoder_input + '/'
	check_path(train_prog_dir)
	train_prog_dir += 'noise=' + str(args.decoder_noise) + '/'
	check_path(train_prog_dir)
	model_dir = train_prog_dir + 'run' + str(args.run) + '/'
	check_path(model_dir)
	train_prog_fname = model_dir + 'epoch_' + str(epoch) + '.txt'
	train_prog_f = open(train_prog_fname, 'w')
	train_prog_f.write('batch loss acc\n')
	# Set model to evaluation mode
	encoder.eval()
	# Set decoder to training mode
	decoder.train()
	# Iterate over batches
	for batch_idx, (data, target) in enumerate(train_loader):
		# Batch start time
		start_time = time.time()
		# Load data
		x = data.to(device)
		y_targ = target.to(device)
		# Scale signal
		x = x * signal
		# Scale to [-1, 1]
		x = (x - 0.5) / 0.5
		# Add noise
		x = x + (torch.randn(x.shape) * img_noise).to(device)
		# Threshold image
		x = nn.Hardtanh()(x)
		# Get model state
		z, all_layers = encoder(x, device)
		# Get decoder inputs
		if args.decoder_input == 'whole_network':
			decoder_in = torch.cat([all_layers['conv1'], all_layers['conv2'], all_layers['conv3'],
									all_layers['fc1'], all_layers['fc2'], all_layers['z']], 1).detach()
		elif args.decoder_input == 'z':
			decoder_in = z.detach()
		# Add decoder noise
		decoder_in = decoder_in + (torch.randn(decoder_in.shape) * decoder_noise).to(device)
		# Zero out gradients for optimizer 
		optimizer.zero_grad()
		# Get decoder predictions
		y_pred, _ = decoder(decoder_in)
		y_pred = y_pred.squeeze()
		# Classification loss
		class_loss_fn = torch.nn.BCELoss()
		class_loss = class_loss_fn(y_pred, y_targ.float())
		# Classification accuracy
		correct_preds = torch.eq(y_pred.round(), y_targ.float()).float()
		class_acc = correct_preds.mean().item() * 100.0
		# Update decoder
		class_loss.backward()
		optimizer.step()
		# Batch duration
		end_time = time.time()
		batch_dur = end_time - start_time
		# Report prgoress
		if batch_idx % args.decoder_log_interval == 0:
			log.info('[Epoch: ' + str(epoch) + '] ' + \
					 '[Batch: ' + str(batch_idx) + ' of ' + str(len(train_loader)) + '] ' + \
					 '[Loss = ' + '{:.4f}'.format(class_loss.item()) + '] ' + \
					 '[Acc. = ' + '{:.2f}'.format(class_acc) + '] ' + \
					 '[' + '{:.3f}'.format(batch_dur) + ' sec/batch]')
			# Save progress to file
			train_prog_f.write(str(batch_idx) + ' ' +\
				               '{:.4f}'.format(class_loss.item()) + ' ' + \
				               '{:.2f}'.format(class_acc) + '\n')
	train_prog_f.close()

def ROC_AUC_analysis(args, encoder, class_out, conf_out, decoder_w, device, test_loader, 
					 signal=1.0, img_noise=0.0, decoder_noise=0.0):
	# Set model to evaluation mode
	encoder.eval()
	class_out.eval()
	conf_out.eval()
	# Iterate over batches of test set
	all_choice_s1 = []
	all_choice_s2 = []
	all_high_conf = []
	all_BE = []
	all_s1_evidence = []
	all_s2_evidence = []
	all_DCE = []
	all_conf_BE = []
	for batch_idx, (data, target) in enumerate(test_loader):
		# Load data
		x = data.to(device)
		y_targ = target.to(device)
		# Scale signal
		x = x * signal
		# Scale to [-1, 1]
		x = (x - 0.5) / 0.5
		# Add noise
		x = x + (torch.randn(x.shape) * img_noise).to(device)
		# Threshold image
		x = nn.Hardtanh()(x)
		# Get model predictions
		z, all_layers = encoder(x, device)
		y_pred = class_out(z).squeeze().round().float().detach().cpu().numpy()
		high_conf = (conf_out(z).squeeze() > 0.75).int().detach().cpu().numpy()
		all_high_conf.append(high_conf)
		# Choice s1 vs. s2 trials
		choice_s1 = (y_pred == 0).astype(np.int)
		all_choice_s1.append(choice_s1)
		choice_s2 = (y_pred == 1).astype(np.int)
		all_choice_s2.append(choice_s2)
		# Get decoder inputs
		if args.decoder_input == 'whole_network':
			decoder_in = torch.cat([all_layers['conv1'], all_layers['conv2'], all_layers['conv3'],
									all_layers['fc1'], all_layers['fc2'], all_layers['z']], 1).detach()
		elif args.decoder_input == 'z':
			decoder_in = z.detach()
		# Add decoder noise
		decoder_in = decoder_in + (torch.randn(decoder_in.shape) * decoder_noise).to(device)
		# Extract stimulus specific evidence
		s1_dims = (decoder_w < 0).squeeze()
		s1_evidence = ((torch.abs(decoder_w) * decoder_in)[:, s1_dims].sum(1) / s1_dims.sum()).detach().cpu().numpy()
		all_s1_evidence.append(s1_evidence)
		s2_dims = (decoder_w > 0).squeeze()
		s2_evidence = ((torch.abs(decoder_w) * decoder_in)[:, s2_dims].sum(1) / s2_dims.sum()).detach().cpu().numpy()
		all_s2_evidence.append(s2_evidence)
		# Balance-of-evidence rule for predicting choice
		BE = s2_evidence - s1_evidence
		all_BE.append(BE)
		# Decision-congruent vs. decision-incongruent evidence
		stacked_evidence = np.stack([s1_evidence, s2_evidence],1)
		decision = np.expand_dims(y_pred.astype(np.int),1)
		DCE = np.take_along_axis(stacked_evidence, decision, axis=1).squeeze()
		all_DCE.append(DCE)
		DIE = np.take_along_axis(stacked_evidence, np.abs(decision-1), axis=1).squeeze()
		conf_BE = DCE - DIE
		all_conf_BE.append(conf_BE)
	# Convert to arrays
	all_choice_s1 = np.concatenate(all_choice_s1)
	all_choice_s2 = np.concatenate(all_choice_s2)
	all_high_conf = np.concatenate(all_high_conf)
	all_BE = np.concatenate(all_BE)
	all_s1_evidence = np.concatenate(all_s1_evidence)
	all_s2_evidence = np.concatenate(all_s2_evidence)
	all_DCE = np.concatenate(all_DCE)
	all_conf_BE = np.concatenate(all_conf_BE)
	# ROC / AUC analysis
	N_criteria = 1000
	standard_far = np.linspace(0,1,N_criteria)
	# Choice balance-of-evidence 
	far, hr, _ = metrics.roc_curve(all_choice_s2, all_BE)
	choice_BE_hr = np.interp(standard_far,far,hr)
	choice_BE_AUC = metrics.auc(standard_far, choice_BE_hr)
	# Choice decision-congruent-evidence
	far, hr, _ = metrics.roc_curve(all_choice_s1, all_s1_evidence)
	s1_hr = np.interp(standard_far,far,hr)
	far, hr, _ = metrics.roc_curve(all_choice_s2, all_s2_evidence)
	s2_hr = np.interp(standard_far,far,hr)
	choice_DCE_hr = np.stack([s1_hr,s2_hr],0).mean(0)
	choice_DCE_AUC = metrics.auc(standard_far, choice_DCE_hr)
	# Confidence balance-of-evidence
	far, hr, _ = metrics.roc_curve(all_high_conf, all_conf_BE)
	conf_BE_hr = np.interp(standard_far,far,hr)
	conf_BE_AUC = metrics.auc(standard_far, conf_BE_hr)
	# Confidence decision-congruent-evidence
	far, hr, _ = metrics.roc_curve(all_high_conf, all_DCE)
	conf_DCE_hr = np.interp(standard_far,far,hr)
	conf_DCE_AUC = metrics.auc(standard_far, conf_DCE_hr)
	return standard_far, choice_BE_hr, choice_BE_AUC, choice_DCE_hr, choice_DCE_AUC, conf_BE_hr, conf_BE_AUC, conf_DCE_hr, conf_DCE_AUC

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
	parser.add_argument('--train-batch-size', type=int, default=32)
	parser.add_argument('--test-batch-size', type=int, default=100)
	parser.add_argument('--img_noise', type=float, default=2.0)
	parser.add_argument('--img_size', type=int, default=32)
	parser.add_argument('--latent_dim', type=int, default=100)
	parser.add_argument('--model_epochs', type=int, default=5)
	parser.add_argument('--no-cuda', action='store_true', default=False)
	parser.add_argument('--decoder_noise', type=float, default=0.0)
	parser.add_argument('--decoder_training_epochs', type=int, default=5)
	parser.add_argument('--decoder_lr', type=float, default=5e-4)
	parser.add_argument('--decoder_log_interval', type=int, default=10)
	parser.add_argument('--decoder_input', type=str, default='whole_network', help="{'whole_network', 'z'}")
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
	encoder, class_out, conf_out, selected_classes = load_params(args, encoder, class_out, conf_out, args.model_epochs)

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
	# Selected digit classes
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

	# Determine input size for decoder
	(data, target) = next(iter(train_loader))
	x = data.to(device)
	_, all_layers = encoder(x, device)
	if args.decoder_input == 'whole_network':
		decoder_input_size = all_layers['conv1'].shape[1] + all_layers['conv2'].shape[1] + all_layers['conv3'].shape[1] + \
							 all_layers['fc1'].shape[1] + all_layers['fc2'].shape[1] + all_layers['z'].shape[1]
	elif args.decoder_input == 'z':
		decoder_input_size = all_layers['z'].shape[1]

	# Build decoder
	log.info('Building decoder...')
	decoder = Decoder(decoder_input_size).to(device)

	# Create optimizer
	log.info('Setting up optimizer...')
	optimizer = optim.Adam(decoder.parameters(), lr=args.decoder_lr)

	# Load threshold signal level
	threshold_signal = np.load('./test/threshold_fit.npz')['estimated_mu'].item()

	# Train decoder
	log.info('Training decoder...')
	for epoch in range(1, args.decoder_training_epochs + 1):
		# Training loop
		train_decoder(args, encoder, decoder, device, train_loader, optimizer, epoch, 
					  signal=threshold_signal, img_noise=args.img_noise, decoder_noise=args.decoder_noise)

	# Get decoder weights
	_, decoder_w = next(iter(decoder.named_parameters()))

	# ROC / AUC analysis
	log.info('ROC / AUC analysis...')
	far, choice_BE_hr, choice_BE_AUC, choice_DCE_hr, choice_DCE_AUC, conf_BE_hr, conf_BE_AUC, conf_DCE_hr, conf_DCE_AUC = \
		ROC_AUC_analysis(args, encoder, class_out, conf_out, decoder_w, device, test_loader, 
					 	 signal=threshold_signal, img_noise=args.img_noise, decoder_noise=args.decoder_noise)
	# Save results
	test_dir = './decoder_test/'
	check_path(test_dir)
	test_dir += args.decoder_input + '/'
	check_path(test_dir)
	test_dir += 'noise=' + str(args.decoder_noise) + '/'
	check_path(test_dir)
	model_dir = test_dir + 'run' + str(args.run) + '/'
	check_path(model_dir)
	np.savez(model_dir + 'ROC_AUC_results.npz',
			 far=far,
			 choice_BE_hr=choice_BE_hr,
			 choice_DCE_hr=choice_DCE_hr,
			 conf_BE_hr=conf_BE_hr,
			 conf_DCE_hr=conf_DCE_hr,
			 choice_BE_AUC=choice_BE_AUC,
			 choice_DCE_AUC=choice_DCE_AUC,
			 conf_BE_AUC=conf_BE_AUC,
			 conf_DCE_AUC=conf_DCE_AUC)

if __name__ == '__main__':
	main()