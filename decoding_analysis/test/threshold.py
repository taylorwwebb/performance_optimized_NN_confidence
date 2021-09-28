import numpy as np
import matplotlib.pyplot as plt
from scipy.stats import sem
import argparse

def plot(args):
	all_runs_acc = []
	for r in range(args.N_runs):
		# Run directory
		run_dir = './run' + str(r+1) + '/'
		# Load threshold results
		results = np.load(run_dir + 'threshold_results.npz')
		acc = results['all_test_acc'] / 100.0
		signal_test_vals = results['signal_test_vals']
		all_runs_acc.append(acc)
	# Convert to array
	all_runs_acc = np.array(all_runs_acc)
	# Compute summary statistics
	acc_mn = all_runs_acc.mean(0)
	acc_se = sem(all_runs_acc,0)
	# Find threshold
	targ_acc = 0.75
	threshold_cond = np.argmin(np.abs(acc_mn - targ_acc))
	threshold_signal = signal_test_vals[threshold_cond]
	# Plot
	ax = plt.subplot(111)
	plt.plot([signal_test_vals.min(), threshold_signal], [targ_acc, targ_acc], color='black', linestyle='dashed')
	plt.plot([threshold_signal, threshold_signal], [0.5, targ_acc], color='black', linestyle=':')
	plt.errorbar(signal_test_vals, acc_mn, yerr=acc_se, color='black')
	plt.ylabel('P(Correct)')
	plt.ylim([0.5,1.0])
	plt.xlabel(r'$\mu$')
	plt.xlim([signal_test_vals.min(),signal_test_vals.max()])
	ax.spines['right'].set_visible(False)
	ax.spines['top'].set_visible(False)
	ax.yaxis.set_ticks_position('left')
	ax.xaxis.set_ticks_position('bottom')
	plt.legend(['target accuracy','estimated ' + r'$\mu$','data'])
	plt.savefig('./threshold_fit.png')
	plt.close()
	# Save target signal
	np.savez('./threshold_fit.npz', targ_acc=targ_acc, estimated_mu=threshold_signal)

def main():

	# Settings
	parser = argparse.ArgumentParser()
	parser.add_argument('--N_runs', type=int, default=100)
	args = parser.parse_args()

	# Plot 
	plot(args)

if __name__ == '__main__':
	main()