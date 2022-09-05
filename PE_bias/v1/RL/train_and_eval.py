import torch
import torch.nn as nn
import torch.optim as optim
from torchvision import datasets, transforms
from PIL import Image, ImageOps
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

def train(args, encoder, actor, critic, device, gabors, optimizer):
	# Create file for saving training progress
	train_prog_dir = './train_prog/'
	check_path(train_prog_dir)
	model_dir = train_prog_dir + 'run' + str(args.run) + '/'
	check_path(model_dir)
	train_prog_fname = model_dir + 'train_prog.txt'
	train_prog_f = open(train_prog_fname, 'w')
	train_prog_f.write('batch actor_loss critic_loss class_acc opt_out_rate\n')
	# Set to training mode
	encoder.train()
	actor.train()
	critic.train()
	# Model accuracy (for titrating opt-out rate)
	current_acc = 0.5
	# Iterate over batches
	for batch_idx in range(args.N_train_batches):
		# Batch start time
		start_time = time.time()
		# Generate batch
		y = torch.rand(args.train_batch_size).round().long().to(device)
		x = gabors[y].unsqueeze(1)
		# Scale signal
		signal = ((torch.rand(x.shape[0]) * (args.signal_range[1] - args.signal_range[0])) + args.signal_range[0]).to(device)
		x = x * signal.view(-1, 1, 1, 1)
		# Scale to [-1, 1]
		x = (x - 0.5) / 0.5
		# Add noise
		noise = (torch.rand(x.shape[0]) * (args.noise_range[1] - args.noise_range[0])) + args.noise_range[0]
		noise = noise.view(-1, 1, 1, 1).repeat(1, 1, x.shape[2], x.shape[3])
		noise = (torch.randn(x.shape) * noise).to(device)
		x = x + noise
		# Threshold image
		x = nn.Hardtanh()(x)
		# Zero out gradients for optimizer 
		optimizer.zero_grad()
		# Get model outputs
		z = encoder(x, device)
		p_a = actor(z)
		v = critic(z).squeeze()
		# Sample actions
		actions = p_a.sample()
		# Generate rewards
		# Initialize
		all_rewards = torch.zeros(args.train_batch_size).to(device)
		# Opt-out trials
		opt_out_reward = np.min([args.max_opt_out, current_acc])
		opt_out_trials = actions == 2
		all_rewards[opt_out_trials] += opt_out_reward
		# Corret trials
		correct_trials = actions == y
		all_rewards[correct_trials] += 1
		# Loss
		# Actor
		log_prob = p_a.log_prob(actions)
		rpe = all_rewards - v
		actor_loss = (-log_prob * rpe).sum()
		# Critic
		critic_loss = nn.SmoothL1Loss(reduction='sum')(v,all_rewards)
		# Combined
		loss = actor_loss + critic_loss
		# Update model
		loss.backward()
		optimizer.step()
		# Accuracy and opt-out rate
		acc = (correct_trials.sum() / (args.train_batch_size - opt_out_trials.sum())).item() * 100.0
		if not np.isnan(acc):
			current_acc = acc / 100.0	# Only update accuracy if opt-out rate < 100% (otherwise accuracy = nan)
		opt_out_rate = (opt_out_trials.sum() / args.train_batch_size).item() * 100.0
		# Batch duration
		end_time = time.time()
		batch_dur = end_time - start_time
		# Report progress
		if batch_idx % args.log_interval == 0:
			log.info('[Batch: ' + str(batch_idx) + ' of ' + str(args.N_train_batches) + '] ' + \
					 '[Actor Loss = ' + '{:.4f}'.format(actor_loss.item()) + '] ' + \
					 '[Critic Loss = ' + '{:.4f}'.format(critic_loss.item()) + '] ' + \
					 '[Class. Acc. = ' + '{:.2f}'.format(acc) + '] ' + \
					 '[Opt-out rate = ' + '{:.2f}'.format(opt_out_rate) + '] ' + \
					 '[' + '{:.3f}'.format(batch_dur) + ' sec/batch]')
			# Save progress to file
			train_prog_f.write(str(batch_idx) + ' ' +\
				               '{:.4f}'.format(actor_loss.item()) + ' ' + \
				               '{:.4f}'.format(critic_loss.item()) + ' ' + \
				               '{:.2f}'.format(acc) + ' ' + \
				               '{:.2f}'.format(opt_out_rate) + '\n')
	train_prog_f.close()

def test(args, encoder, actor, critic, device, gabors, signal=1.0, noise=0.0):
	# Set to evaluation mode
	encoder.eval()
	actor.eval()
	critic.eval()
	# Iterate over batches
	all_test_correct_preds = []
	all_test_opt_out = []
	for batch_idx in range(args.N_test_batches):
		# Generate batch
		y = torch.rand(args.test_batch_size).round().long().to(device)
		x = gabors[y].unsqueeze(1)
		# Scale signal
		x = x * signal
		# Scale to [-1, 1]
		x = (x - 0.5) / 0.5
		# Add noise
		x = x + (torch.randn(x.shape) * noise).to(device)
		# Threshold image
		x = nn.Hardtanh()(x)
		# Get model outputs
		z = encoder(x, device)
		p_a = actor(z)
		v = critic(z).squeeze()
		# Collect responses
		# Correct predictions
		correct_preds = (p_a.probs[:,:2].argmax(1) == y).float()
		all_test_correct_preds.append(correct_preds.detach().cpu().numpy())
		# Opt-out rate
		opt_out = (p_a.probs.argmax(1) == 2).float()
		all_test_opt_out.append(opt_out.detach().cpu().numpy())
	# Overall test accuracy and opt-out-rate
	all_test_correct_preds = np.concatenate(all_test_correct_preds)
	all_test_opt_out = np.concatenate(all_test_opt_out)
	avg_test_acc = np.mean(all_test_correct_preds) * 100.0
	avg_test_opt_out_rate = np.mean(all_test_opt_out) * 100.0
	avg_test_opt_out_rate_correct = np.mean(all_test_opt_out[all_test_correct_preds==1]) * 100.0
	avg_test_opt_out_rate_incorrect = np.mean(all_test_opt_out[all_test_correct_preds==0]) * 100.0
	# Report
	log.info('[Signal = ' + '{:.2f}'.format(signal) + '] ' + \
			 '[Noise = ' + '{:.2f}'.format(noise) + '] ' + \
			 '[Class. Acc. = ' + '{:.2f}'.format(avg_test_acc) + '] ' + \
			 '[Opt-out rate = ' + '{:.2f}'.format(avg_test_opt_out_rate) + '] ' + \
			 '[Opt-out rate (correct) = ' + '{:.2f}'.format(avg_test_opt_out_rate_correct) + '] ' + \
			 '[Opt-out rate (incorrect) = ' + '{:.2f}'.format(avg_test_opt_out_rate_incorrect) + ']')
	return avg_test_acc, avg_test_opt_out_rate, avg_test_opt_out_rate_correct, avg_test_opt_out_rate_incorrect

def main():

	# Settings
	parser = argparse.ArgumentParser()
	parser.add_argument('--train-batch-size', type=int, default=32)
	parser.add_argument('--test-batch-size', type=int, default=100)
	parser.add_argument('--signal_range', type=list, default=[0.1, 1.0])
	parser.add_argument('--signal_range_test', type=list, default=[0.0, 1.0])
	parser.add_argument('--signal_N_test', type=int, default=500)
	parser.add_argument('--noise_range', type=list, default=[0.5, 1.0])
	parser.add_argument('--noise_N_test', type=int, default=2)
	parser.add_argument('--img_size', type=int, default=32)
	parser.add_argument('--latent_dim', type=int, default=100)
	parser.add_argument('--max_opt_out', type=float, default=0.75)
	parser.add_argument('--N_train_batches', type=int, default=5000)
	parser.add_argument('--N_test_batches', type=int, default=100)
	parser.add_argument('--lr', type=float, default=1e-3)
	parser.add_argument('--no-cuda', action='store_true', default=False)
	parser.add_argument('--log_interval', type=int, default=10)
	parser.add_argument('--device', type=int, default=0)
	parser.add_argument('--run', type=str, default='1')
	args = parser.parse_args()
		
	# Set up cuda	
	use_cuda = not args.no_cuda and torch.cuda.is_available()
	device = torch.device("cuda:" + str(args.device) if use_cuda else "cpu")
	kwargs = {'num_workers': 1, 'pin_memory': True} if use_cuda else {}

	# Load images
	gabor_r = torch.Tensor(np.array(ImageOps.grayscale(Image.open('./gabor_r.png')).resize((args.img_size,args.img_size)))) / 255.
	gabor_l = torch.Tensor(np.array(ImageOps.grayscale(Image.open('./gabor_l.png')).resize((args.img_size,args.img_size)))) / 255.
	gabors = torch.stack([gabor_r, gabor_l]).to(device)

	# Build model
	log.info('Building model...')
	encoder = Encoder(args).to(device)
	actor = Actor(args).to(device)
	critic = Critic(args).to(device)
	all_modules = nn.ModuleList([encoder, actor, critic])

	# Create optimizer
	log.info('Setting up optimizer...')
	optimizer = optim.Adam(all_modules.parameters(), lr=args.lr)

	# Train
	log.info('Training begins...')
	# Training loop
	train(args, encoder, actor, critic, device, gabors, optimizer)

	# Test
	log.info('Test...')
	# Evaluate without noise
	test_acc_noiseless, _, __, ___ = test(args, encoder, actor, critic, device, gabors, signal=1, noise=0)
	# Signal and noise values for test
	signal_test_vals = np.linspace(args.signal_range_test[0], args.signal_range_test[1], args.signal_N_test)
	noise_test_vals = np.linspace(args.noise_range[0], args.noise_range[1], args.noise_N_test)
	all_test_acc = []
	all_test_opt_out_rate = []
	all_test_opt_out_rate_correct = []
	all_test_opt_out_rate_incorrect = []
	for n in range(noise_test_vals.shape[0]):
		all_signal_test_acc = []
		all_signal_test_opt_out_rate = []
		all_signal_test_opt_out_rate_correct = []
		all_signal_test_opt_out_rate_incorrect = []
		for s in range(signal_test_vals.shape[0]):
			test_acc, test_opt_out_rate, test_opt_out_rate_correct, test_opt_out_rate_incorrect = test(args, encoder, actor, critic, device, gabors, signal=signal_test_vals[s], noise=noise_test_vals[n])
			all_signal_test_acc.append(test_acc)
			all_signal_test_opt_out_rate.append(test_opt_out_rate)
			all_signal_test_opt_out_rate_correct.append(test_opt_out_rate_correct)
			all_signal_test_opt_out_rate_incorrect.append(test_opt_out_rate_incorrect)
		all_test_acc.append(all_signal_test_acc)
		all_test_opt_out_rate.append(all_signal_test_opt_out_rate)
		all_test_opt_out_rate_correct.append(all_signal_test_opt_out_rate_correct)
		all_test_opt_out_rate_incorrect.append(all_signal_test_opt_out_rate_incorrect)
	# Save test results
	test_dir = './test/'
	check_path(test_dir)
	model_dir = test_dir + 'run' + str(args.run) + '/'
	check_path(model_dir)
	np.savez(model_dir + 'test_results.npz',
			 signal_test_vals=signal_test_vals,
			 noise_test_vals=noise_test_vals,
			 test_acc_noiseless=test_acc_noiseless,
			 all_test_acc=np.array(all_test_acc),
			 all_test_opt_out_rate=np.array(all_test_opt_out_rate),
			 all_test_opt_out_rate_correct=np.array(all_test_opt_out_rate_correct),
			 all_test_opt_out_rate_incorrect=np.array(all_test_opt_out_rate_incorrect))

if __name__ == '__main__':
	main()