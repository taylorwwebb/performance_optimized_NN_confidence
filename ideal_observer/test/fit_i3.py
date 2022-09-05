import numpy as np
import matplotlib.pyplot as plt
import argparse

def plot(args):
	# Model directory
	model_dir = './' + args.train_regime + '_training/'
	# Get test results
	all_runs_d_prime = []
	for r in range(args.N_runs):
		# Run directory
		run_dir = model_dir + 'run' + str(r+1) + '/'
		# Load test results
		test_results = np.load(run_dir + 'i3_dprime_results.npz')
		all_d_prime = test_results['all_d']
		signal_test_vals = test_results['signal_test_vals']
		# Collect results for all runs
		all_runs_d_prime.append(all_d_prime)
	# Convert to arrays
	all_runs_d_prime = np.array(all_runs_d_prime)
	# Average
	d_prime_mn = all_runs_d_prime.mean(0)
	# Target d'
	targ_d_prime = np.load('./behavioral_data.npz')['d_mn'][2]
	# Interpolate to estimate signal corresponding to desired d'
	targ_signal = np.interp(targ_d_prime, d_prime_mn, signal_test_vals)
	# Plot signal vs. d'
	ax = plt.subplot(111)
	plt.plot(signal_test_vals, d_prime_mn, color='black')
	plt.plot([signal_test_vals.min(), targ_signal], [targ_d_prime, targ_d_prime], color='black', linestyle='dashed')
	plt.plot([targ_signal, targ_signal], [d_prime_mn.min(), targ_d_prime], color='black', linestyle=':')
	plt.ylabel("d'")
	plt.xlabel(r'$\mu$')
	ax.spines['right'].set_visible(False)
	ax.spines['top'].set_visible(False)
	ax.yaxis.set_ticks_position('left')
	ax.xaxis.set_ticks_position('bottom')
	plt.legend(['data',"target d'",'estimated ' + r'$\mu$'])
	plt.savefig(model_dir + 'i3_mu_vs_d_prime.png')
	plt.close()
	# Save target signal
	np.savez(model_dir + 'i3_fit.npz', targ_d_prime=targ_d_prime, estimated_mu=targ_signal)

def main():

	# Settings
	parser = argparse.ArgumentParser()
	parser.add_argument('--train_regime', type=str, default='standard', help="{'standard', 'fixed_mu'}")
	parser.add_argument('--N_runs', type=int, default=100)
	args = parser.parse_args()

	# Plot 
	plot(args)

if __name__ == '__main__':
	main()