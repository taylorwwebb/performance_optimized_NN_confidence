import torch
import torch.nn as nn
from util import log

class resnet_block(nn.Module):
	def __init__(self, args, in_channels, out_channels, kernel_size=3, in_stride=1, padding=1):
		super(resnet_block, self).__init__()
		# Define convolutional layers
		self.conv1 = nn.Conv2d(in_channels, out_channels, kernel_size, stride=in_stride, padding=padding, bias=False)
		self.conv2 = nn.Conv2d(out_channels, out_channels, kernel_size, stride=1, padding=padding, bias=False)
		# Define batch norm layers
		self.BN1 = nn.BatchNorm2d(out_channels)
		self.BN2 = nn.BatchNorm2d(out_channels)
		# Shortcut convolution and batch norm (if in_stride > 1, or in_channels != out_channels)
		if (in_stride > 1) or (in_channels != out_channels):
			self.use_shortcut = True
			self.conv_shortcut = nn.Conv2d(in_channels, out_channels, 1, stride=in_stride, padding=0, bias=False)
			self.BN_shortcut = nn.BatchNorm2d(out_channels)
		else:
			self.use_shortcut = False
		# Nonlinearity
		self.relu = nn.ReLU()
		# Initialize parameters
		if args.kaiming_init:
			for name, param in self.named_parameters():
				if not ('BN' in name):
					if 'bias' in name:
						nn.init.constant_(param, 0.0)
					elif 'weight' in name:
						nn.init.kaiming_normal_(param, nonlinearity='relu')
	def forward(self, block_in):
		layer1_out = self.relu(self.BN1(self.conv1(block_in)))
		if self.use_shortcut:
			block_in = self.BN_shortcut(self.conv_shortcut(block_in))
		layer2_out = self.relu(self.BN2(self.conv2(layer1_out)) + block_in)
		return layer2_out

class resnet_stack(nn.Module):
	def __init__(self, args, in_channels, out_channels, N_blocks=3, kernel_size=3, out_stride=2, padding=1):
		super(resnet_stack, self).__init__()
		all_blocks = []
		for block in range(N_blocks):
			if block == 0:
				all_blocks.append(resnet_block(args, in_channels, out_channels, kernel_size=kernel_size, padding=padding))
			elif block == N_blocks - 1:
				all_blocks.append(resnet_block(args, out_channels, out_channels, kernel_size=kernel_size, in_stride=out_stride, padding=padding))
			else:
				all_blocks.append(resnet_block(args, out_channels, out_channels, kernel_size=kernel_size, padding=padding))
		self.all_blocks = nn.Sequential(*all_blocks)
	def forward(self, stack_in):
		stack_out = self.all_blocks(stack_in)
		return stack_out

class Model(nn.Module):
	def __init__(self, args):
		super(Model, self).__init__()
		# Initial convolutional layer
		log.info('Initial convolutional layer...')
		self.init_conv = nn.Conv2d(3, 16, 3, stride=1, padding=1, bias=False)
		self.init_BN = nn.BatchNorm2d(16)
		# Residual stack 1
		log.info('Residual stack #1...')
		self.stack1 = resnet_stack(args, 16, 16, N_blocks=args.N_res_blocks, kernel_size=3, out_stride=2, padding=1)
		# Residual stack 2
		log.info('Residual stack #2...')
		self.stack2 = resnet_stack(args, 16, 32, N_blocks=args.N_res_blocks, kernel_size=3, out_stride=2, padding=1)
		# Residual stack 3
		log.info('Residual stack #3...')
		self.stack3 = resnet_stack(args, 32, 64, N_blocks=args.N_res_blocks, kernel_size=3, out_stride=1, padding=1)
		# Average pooling operation
		log.info('Average pooling operation...')
		self.avg_pool = nn.AvgPool2d(8)
		# Output layers
		log.info('Output layers...')
		self.class_out = nn.Linear(64, 10)
		self.conf_out = nn.Linear(64, 1)
		# Nonlinearities
		self.relu = nn.ReLU()
		self.sigmoid = nn.Sigmoid()
		# Initialize parameters
		if args.kaiming_init:
			for name, param in self.named_parameters():
				if not ('BN' in name) and not ('blocks' in name):
					if 'bias' in name:
						nn.init.constant_(param, 0.0)
					elif 'weight' in name:
						if 'out' in name:
							nn.init.xavier_normal_(param)
						else:
							nn.init.kaiming_normal_(param, nonlinearity='relu')
	def forward(self, x):
		# Initial convolutional layer
		init_conv_out = self.relu(self.init_BN(self.init_conv(x)))
		# Residual stack 1
		stack1_out = self.stack1(init_conv_out)
		# Residual stack 2
		stack2_out = self.stack2(stack1_out)
		# Residual stack 3
		stack3_out = self.stack3(stack2_out)
		# Average pooling
		avg_pool_out = self.avg_pool(stack3_out).squeeze()
		# Output layers
		class_out_linear = self.class_out(avg_pool_out)
		class_pred = class_out_linear.argmax(1)
		conf_out = self.sigmoid(self.conf_out(avg_pool_out)).squeeze()
		return class_out_linear, class_pred, conf_out