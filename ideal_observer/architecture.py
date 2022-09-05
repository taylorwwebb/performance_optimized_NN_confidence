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
		# Latent space
		self.z_mn_out = nn.Linear(128, self.latent_dim)
		self.z_logvar_out = nn.Linear(128, self.latent_dim)
		# Nonlinearities
		self.leaky_relu = nn.LeakyReLU()
	def forward(self, x):
		# Convolutional encoder
		conv1_out = self.leaky_relu(self.conv1_BN(self.conv1(x)))
		conv2_out = self.leaky_relu(self.conv2_BN(self.conv2(conv1_out)))
		conv3_out = self.leaky_relu(self.conv3_BN(self.conv3(conv2_out)))
		# Flatten output of conv. net
		conv3_out_flat = torch.flatten(conv3_out, 1)
		# MLP encoder
		fc1_out = self.leaky_relu(self.fc1_BN(self.fc1(conv3_out_flat)))
		fc2_out = self.leaky_relu(self.fc2_BN(self.fc2(fc1_out)))
		# Latent space
		z_mn = self.z_mn_out(fc2_out)
		z_logvar = self.z_logvar_out(fc2_out)
		return z_mn, z_logvar

class Decoder(nn.Module):
	def __init__(self, args):
		super(Decoder, self).__init__()
		log.info('Building decoder...')
		# MLP decoder
		log.info('MLP...')
		self.latent_dim = args.latent_dim
		self.fc1 = nn.Linear(self.latent_dim, 128)
		self.fc1_BN = nn.BatchNorm1d(128)
		self.fc2 = nn.Linear(128, 256)
		self.fc2_BN = nn.BatchNorm1d(256)
		self.fc3 = nn.Linear(256, 512)
		self.fc3_BN = nn.BatchNorm1d(512)
		# Convolutional decoder
		log.info('Conv. layers...')
		self.conv1 = nn.ConvTranspose2d(32, 32, 4, stride=2, padding=1)
		self.conv1_BN = nn.BatchNorm2d(32)
		self.conv2 = nn.ConvTranspose2d(32, 32, 4, stride=2, padding=1)
		self.conv2_BN = nn.BatchNorm2d(32)
		self.conv3 = nn.ConvTranspose2d(32, 1, 4, stride=2, padding=1)
		self.conv3_BN = nn.BatchNorm2d(1)
		# Nonlinearities
		self.leaky_relu = nn.LeakyReLU()
		self.tanh = nn.Tanh()
	def forward(self, z):
		# MLP decoder
		fc1_out = self.leaky_relu(self.fc1_BN(self.fc1(z)))
		fc2_out = self.leaky_relu(self.fc2_BN(self.fc2(fc1_out)))
		fc3_out = self.leaky_relu(self.fc3_BN(self.fc3(fc2_out)))
		# Reshape for convolutional layers
		fc3_out_reshaped = fc3_out.reshape(-1, 32, 4, 4)
		# Convolutional decoder
		conv1_out = self.leaky_relu(self.conv1_BN(self.conv1(fc3_out_reshaped)))
		conv2_out = self.leaky_relu(self.conv2_BN(self.conv2(conv1_out)))
		# x_pred = self.tanh(self.conv3_BN(self.conv3(conv2_out)))
		x_pred = self.tanh(self.conv3(conv2_out))
		return x_pred

