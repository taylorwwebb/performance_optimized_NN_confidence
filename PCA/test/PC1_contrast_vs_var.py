import numpy as np
import matplotlib.pyplot as plt
from scipy.stats import ttest_rel, sem
import argparse

def plot(args):
	all_runs_pc1_var = []
	for r in range(args.N_runs):
		# Run directory
		run_dir = './run' + str(r+1) + '/'
		# Load test results
		test_results = np.load(run_dir + 'PE_test_results.npz')
		# Collect results
		all_runs_pc1_var.append(test_results['all_pc1_var'])
	# Convert to arrays
	all_runs_pc1_var = np.array(all_runs_pc1_var)
	# Summary statistics
	all_runs_pc1_var_mn = all_runs_pc1_var.mean(0)
	all_runs_pc1_var_se = sem(all_runs_pc1_var,0)
	# Limit to two conditions (low vs. high contrast)
	all_runs_pc1_var = all_runs_pc1_var[:,1,[np.abs(test_results['signal_test_vals'] - 0.1).argmin(), -1]]
	all_runs_pc1_var_mn = all_runs_pc1_var_mn[1,[np.abs(test_results['signal_test_vals'] - 0.1).argmin(), -1]]
	all_runs_pc1_var_se = all_runs_pc1_var_se[1,[np.abs(test_results['signal_test_vals'] - 0.1).argmin(), -1]]
	# Font sizes
	axis_fontsize = 18
	ticks_fontsize = 16
	significance_fontsize = 16
	bar_width = 0.8
	# Plot
	ax1 = plt.subplot(111)
	ax1.bar([0,1],all_runs_pc1_var_mn,yerr=all_runs_pc1_var_se,width=bar_width,color='black')
	ax1.set_ylabel('PC 1 variance', fontsize=axis_fontsize)
	plt.ylim([0.04,0.16])
	plt.yticks([0.04,0.08,0.12,0.16],['0.04','0.08','0.12','0.16'], fontsize=ticks_fontsize)
	plt.xticks([0,1],['Low','High'], fontsize=ticks_fontsize)
	ax1.set_xlabel('Contrast',fontsize=axis_fontsize)
	ax1.spines['top'].set_visible(False)
	ax1.spines['right'].set_visible(False)
	ax1.set_aspect(aspect=25)
	# Stats
	pc1_var_t, pc1_var_p = ttest_rel(all_runs_pc1_var[:,0], all_runs_pc1_var[:,1])
	if pc1_var_p >= 0.05: pc1_var_p_symb = 'ns'
	if pc1_var_p < 0.05: pc1_var_p_symb = '*'
	if pc1_var_p < 0.01: pc1_var_p_symb = '**'
	if pc1_var_p < 0.001: pc1_var_p_symb = '***'
	if pc1_var_p < 0.0001: pc1_var_p_symb = '****'
	max_y_val = np.max([all_runs_pc1_var_mn[0] + all_runs_pc1_var_se[0], all_runs_pc1_var_mn[1] + all_runs_pc1_var_se[1]])
	y_start = max_y_val + 0.003
	y_end = max_y_val + 0.0045
	ax1.plot([0,0,1,1],[y_start,y_end,y_end,y_start],color='black')
	ax1.text(0.5,y_end+0.0025,pc1_var_p_symb,fontsize=significance_fontsize,horizontalalignment='center')
	# Save plot
	plot_fname = './pc1_var_vs_contrast.png'
	plt.savefig(plot_fname, bbox_inches='tight', dpi=300)
	plt.close()

def main():

	# Settings
	parser = argparse.ArgumentParser()
	parser.add_argument('--N_runs', type=int, default=100)
	args = parser.parse_args()

	# Plot
	plot(args)

if __name__ == '__main__':
	main()