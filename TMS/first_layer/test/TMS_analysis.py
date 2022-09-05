import numpy as np
import matplotlib.pyplot as plt
from scipy.stats import sem
import argparse

def main():

	# Training settings
	parser = argparse.ArgumentParser()
	parser.add_argument('--N_runs', type=int, default=100)
	args = parser.parse_args()

	# Load data for all runs
	all_conf = []
	all_d_prime = []
	for r in range(args.N_runs):
		# Load data
		results_fname = './run' + str(r+1) + '/results.npz'
		results = np.load(results_fname)
		# Collect results
		TMS_vals = results['TMS_vals']
		all_conf.append(results['all_test_conf'])
		all_d_prime.append(results['all_test_d'])
	# Convert to arrays
	all_conf = np.array(all_conf)
	all_d_prime = np.array(all_d_prime)
	# Summary stats (average across runs)
	conf_mn = all_conf.mean(0)
	conf_se = sem(all_conf,0)
	d_prime_mn = all_d_prime.mean(0)
	d_prime_se = sem(all_d_prime,0)

	# TMS condition colors
	min_TMS_color = [0, 0.2, 0.4]
	max_TMS_color = [0.8, 0.9, 1]
	N_TMS_conditions = TMS_vals.shape[0] - 1
	all_TMS_colors = np.linspace(np.array(min_TMS_color),np.array(max_TMS_color),N_TMS_conditions)

	# Plot d' vs. confidence
	axis_fontsize = 18
	tick_fontsize = 16
	legend_fontsize = 16
	title_fontsize = 18
	ax = plt.subplot(111)
	plt.errorbar(d_prime_mn[0,:], conf_mn[0,:], xerr=d_prime_se[0,:], yerr=conf_se[0,:], color='black')
	plt.errorbar(d_prime_mn[1,:], conf_mn[1,:], xerr=d_prime_se[1,:], yerr=conf_se[1,:], color=all_TMS_colors[0,:])
	plt.errorbar(d_prime_mn[-1,:], conf_mn[-1,:], xerr=d_prime_se[-1,:], yerr=conf_se[-1,:], color=all_TMS_colors[-1,:])
	for c in range(1,TMS_vals.shape[0]):
		plt.errorbar(d_prime_mn[c,:], conf_mn[c,:], xerr=d_prime_se[c,:], yerr=conf_se[c,:], color=all_TMS_colors[c-1,:])
	plt.xlabel("d'", fontsize=axis_fontsize)
	# plt.xticks([0,0.5,1,1.5,2,2.5,3,3.5],['0','0.5','1','1.5','2','2.5','3','3.5'],fontsize=tick_fontsize)
	plt.xlim([0,4])
	plt.xticks([0,1,2,3,4],['0','1','2','3','4'],fontsize=tick_fontsize)
	plt.ylabel('Confidence', fontsize=axis_fontsize)
	# plt.yticks([0.65,0.7,0.75,0.8,0.85,0.9,0.95,1],['0.65','0.7','0.75','0.8','0.85','0.9','0.95','1'],fontsize=tick_fontsize)
	plt.ylim([0.6,1])
	plt.yticks([0.6,0.7,0.8,0.9,1],['0.6','0.7','0.8','0.9','1'],fontsize=tick_fontsize)
	ax.spines['right'].set_visible(False)
	ax.spines['top'].set_visible(False)
	ax.yaxis.set_ticks_position('left')
	ax.xaxis.set_ticks_position('bottom')
	plt.title('Effect of simulated V1 TMS', fontsize=title_fontsize)
	plt.legend(['Control', 'TMS ($\\xi$=1)', 'TMS ($\\xi$=5)'], frameon=False, fontsize=legend_fontsize)
	plt.savefig('./TMS_d_prime_vs_confidence.png', bbox_inches='tight', dpi=300)
	plt.close()

if __name__ == '__main__':
	main()