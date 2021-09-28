import torch
import torch.nn as nn
from util import log

class Encoder(nn.Module):
	def __init__(self, args):
		super(Encoder, self).__init__()
		log.info('Building encoder...')
		# Convolutional encoder
		log.info('Conv. layers...')
		self.conv1 = nn.Conv2d(1, 32, 3, stride=2, padding=1)
		self.conv1_BN = nn.BatchNorm2d(32)
		self.conv2 = nn.Conv2d(32, 32, 3, stride=2, padding=1)
		self.conv2_BN = nn.BatchNorm2d(32)
		self.conv3 = nn.Conv2d(32, 32, 3, stride=2, padding=1)
		self.conv3_BN = nn.BatchNorm2d(32)
		# MLP encoder
		log.info('MLP...')
		self.fc1 = nn.Linear(512, 256)
		self.fc1_BN = nn.BatchNorm1d(256)
		self.fc2 = nn.Linear(256, 128)
		self.fc2_BN = nn.BatchNorm1d(128)
		self.latent_dim = args.latent_dim
		self.z_out = nn.Linear(128, self.latent_dim)
		# Nonlinearities
		self.leaky_relu = nn.LeakyReLU()
	def forward(self, x, device):
		# Convolutional encoder
		conv1_out = self.leaky_relu(self.conv1_BN(self.conv1(x)))
		conv2_out = self.leaky_relu(self.conv2_BN(self.conv2(conv1_out)))
		conv3_out = self.leaky_relu(self.conv3_BN(self.conv3(conv2_out)))
		# Flatten output of conv. net
		conv3_out_flat = torch.flatten(conv3_out, 1)
		# MLP encoder
		fc1_out = self.leaky_relu(self.fc1_BN(self.fc1(conv3_out_flat)))
		fc2_out = self.leaky_relu(self.fc2_BN(self.fc2(fc1_out)))
		z = self.z_out(fc2_out)
		# Create dict with flattened states (for decoding analyses)
		all_layers = {'conv1': torch.flatten(conv1_out, 1),
					  'conv2': torch.flatten(conv2_out, 1),
					  'conv3': conv3_out_flat,
					  'fc1': fc1_out,
					  'fc2': fc2_out,
					  'z': z}
		return z, all_layers

class Class_out(nn.Module):
	def __init__(self, args):
		super(Class_out, self).__init__()
		log.info('Building classifier output layer...')
		# Feedforward layer
		self.fc = nn.Linear(args.latent_dim, 1)
		# Nonlinearities
		self.sigmoid = nn.Sigmoid()
	def forward(self, z):
		y = self.sigmoid(self.fc(z))
		return y

class Conf_out(nn.Module):
	def __init__(self, args):
		super(Conf_out, self).__init__()
		log.info('Building confidence output layer...')
		# Feedforward layer
		self.fc = nn.Linear(args.latent_dim, 1)
		# Nonlinearities
		self.sigmoid = nn.Sigmoid()
	def forward(self, z):
		conf = self.sigmoid(self.fc(z))
		return conf

class Decoder(nn.Module):
	def __init__(self, input_size):
		super(Decoder, self).__init__()
		log.info('Building decoder...')
		# Feedforward layer
		self.fc = nn.Linear(input_size, 1, bias=False)
		# Nonlinearities
		self.sigmoid = nn.Sigmoid()
	def forward(self, x):
		y_linear = self.fc(x)
		y = self.sigmoid(y_linear)
		return y, y_linear