import numpy as np
import matplotlib.pyplot as plt
import argparse

def plot(args):
	all_runs_d_prime = []
	for r in range(args.N_runs):
		# Run directory
		run_dir = './run' + str(r+1) + '/'
		# Load test results
		test_results = np.load(run_dir + 'i1245_d_prime_results.npz')
		all_d_prime = test_results['all_d']
		signal_test_vals = test_results['signal_test_vals']
		# Collect results for all runs
		all_runs_d_prime.append(all_d_prime)
	# Convert to arrays
	all_runs_d_prime = np.array(all_runs_d_prime)
	# Average
	d_prime_mn = all_runs_d_prime.mean(0)
	# Target d-prime
	targ_d_prime = np.load('./behavioral_data.npz')['d_mn'][[0,1,3,4]]
	# Interpolate to estimate signal corresponding to desired d'
	all_targ_signal = []
	for i in range(4):
		targ_signal = np.interp(targ_d_prime[i], d_prime_mn, signal_test_vals)
		all_targ_signal.append(targ_signal)
	# Plot signal vs. d'
	ax = plt.subplot(111)
	plt.plot(signal_test_vals, d_prime_mn, color='black')
	for i in range(4):
		plt.plot([signal_test_vals.min(), all_targ_signal[i]], [targ_d_prime[i], targ_d_prime[i]], color='black', linestyle='dashed')
		plt.plot([all_targ_signal[i], all_targ_signal[i]], [d_prime_mn.min(), targ_d_prime[i]], color='black', linestyle=':')
	plt.ylabel("d'")
	plt.xlabel(r'$\mu$')
	ax.spines['right'].set_visible(False)
	ax.spines['top'].set_visible(False)
	ax.yaxis.set_ticks_position('left')
	ax.xaxis.set_ticks_position('bottom')
	plt.legend(['data',"target d'",'estimated ' + r'$\mu$'])
	plt.savefig('./i1245_mu_vs_d_prime.png')
	plt.close()
	# Save target signal
	np.savez('./i1245_fit.npz', targ_d_prime=targ_d_prime, estimated_mu=np.array(all_targ_signal))

def main():

	# Settings
	parser = argparse.ArgumentParser()
	parser.add_argument('--N_runs', type=int, default=100)
	args = parser.parse_args()

	# Plot
	plot(args)

if __name__ == '__main__':
	main()