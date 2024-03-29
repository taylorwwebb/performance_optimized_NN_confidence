import numpy as np
import argparse
import glob
import pdb

def main():

	# Settings
	parser = argparse.ArgumentParser()
	parser.add_argument('--train_regime', type=str, default='standard', help="{'standard', 'fixed_mu'}")
	args = parser.parse_args()

	# Model directory
	model_dir = './' + args.train_regime + '_training/'

	# Collect error terms
	fnames = glob.glob(model_dir + 'meta_d_mse*')
	N_conf_noise = len(fnames)
	all_mse = []
	all_conf_noise = []
	for i in range(N_conf_noise):
		# Load results
		fname = fnames[i]
		results = np.load(fname)
		mse = results['mse']
		all_mse.append(mse)
		conf_noise = results['conf_noise']
		all_conf_noise.append(conf_noise)
	# Convert to arrays
	all_mse = np.array(all_mse)
	all_conf_noise = np.array(all_conf_noise)
	# Select confidence noise level
	min_mse = all_mse.argmin()
	conf_noise_fit = all_conf_noise[min_mse]
	# Save results
	np.savez(model_dir + 'conf_noise_fit.npz', conf_noise=conf_noise_fit)

if __name__ == '__main__':
	main()
