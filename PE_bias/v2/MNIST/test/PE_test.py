import numpy as np
import matplotlib.pyplot as plt
from scipy.stats import sem
from scipy.stats import ttest_rel
import argparse

def plot(args):
	all_runs_low_PE_test_acc = []
	all_runs_low_PE_test_conf = []
	all_runs_high_PE_test_acc = []
	all_runs_high_PE_test_conf = []
	for r in range(args.N_runs):
		# Run directory
		run_dir = './run' + str(r+1) + '/'
		# Load test results
		test_results = np.load(run_dir + 'test_results.npz')
		low_PE_test_acc = test_results['low_PE_test_acc'] / 100
		low_PE_test_conf = test_results['low_PE_test_conf'] / 100
		high_PE_test_acc = test_results['high_PE_test_acc'] / 100
		high_PE_test_conf = test_results['high_PE_test_conf'] / 100
		# Collect results for all runs
		all_runs_low_PE_test_acc.append(low_PE_test_acc)
		all_runs_low_PE_test_conf.append(low_PE_test_conf)
		all_runs_high_PE_test_acc.append(high_PE_test_acc)
		all_runs_high_PE_test_conf.append(high_PE_test_conf)
	# Convert to arrays
	all_runs_low_PE_test_acc = np.array(all_runs_low_PE_test_acc)
	all_runs_low_PE_test_conf = np.array(all_runs_low_PE_test_conf)
	all_runs_high_PE_test_acc = np.array(all_runs_high_PE_test_acc)
	all_runs_high_PE_test_conf = np.array(all_runs_high_PE_test_conf)
	# Summary statistics
	all_runs_low_PE_test_acc_mean = all_runs_low_PE_test_acc.mean(0)
	all_runs_low_PE_test_acc_se = sem(all_runs_low_PE_test_acc,0)
	all_runs_low_PE_test_conf_mean = all_runs_low_PE_test_conf.mean(0)
	all_runs_low_PE_test_conf_se = sem(all_runs_low_PE_test_conf,0)
	all_runs_high_PE_test_acc_mean = all_runs_high_PE_test_acc.mean(0)
	all_runs_high_PE_test_acc_se = sem(all_runs_high_PE_test_acc,0)
	all_runs_high_PE_test_conf_mean = all_runs_high_PE_test_conf.mean(0)
	all_runs_high_PE_test_conf_se = sem(all_runs_high_PE_test_conf,0)
	# Find PE conditions
	target_acc = 0.75
	# Low PE
	low_PE_ind = np.abs(all_runs_low_PE_test_acc_mean - target_acc).argmin()
	low_PE_test_acc = all_runs_low_PE_test_acc[:,low_PE_ind]
	low_PE_test_acc_mean = all_runs_low_PE_test_acc_mean[low_PE_ind]
	low_PE_test_acc_se = all_runs_low_PE_test_acc_se[low_PE_ind]
	low_PE_test_conf = all_runs_low_PE_test_conf[:,low_PE_ind]
	low_PE_test_conf_mean = all_runs_low_PE_test_conf_mean[low_PE_ind]
	low_PE_test_conf_se = all_runs_low_PE_test_conf_se[low_PE_ind]
	# High PE
	high_PE_ind = np.abs(all_runs_high_PE_test_acc_mean - target_acc).argmin()
	high_PE_test_acc = all_runs_high_PE_test_acc[:,high_PE_ind]
	high_PE_test_acc_mean = all_runs_high_PE_test_acc_mean[high_PE_ind]
	high_PE_test_acc_se = all_runs_high_PE_test_acc_se[high_PE_ind]
	high_PE_test_conf = all_runs_high_PE_test_conf[:,high_PE_ind]
	high_PE_test_conf_mean = all_runs_high_PE_test_conf_mean[high_PE_ind]
	high_PE_test_conf_se = all_runs_high_PE_test_conf_se[high_PE_ind]

	# Stimulus parameters
	low_PE_targ_signal = test_results['low_PE_targ_signal']
	low_PE_nontarg_signal = test_results['nontarg_signal_test_vals'][low_PE_ind]
	high_PE_targ_signal = test_results['high_PE_targ_signal']
	high_PE_nontarg_signal = test_results['nontarg_signal_test_vals'][high_PE_ind]
	# Write to file
	params_fname = './stim_params.txt'
	fid = open(params_fname,'w')
	fid.write('Low PE:\n')
	fid.write('    target signal = ' + str(low_PE_targ_signal) + '\n')
	fid.write('    non-target signal = ' + str(low_PE_nontarg_signal) + '\n')
	fid.write('High PE:\n')
	fid.write('    target signal = ' + str(high_PE_targ_signal) + '\n')
	fid.write('    non-target signal = ' + str(high_PE_nontarg_signal))
	fid.close()

	# Stats
	# Open file
	stats_fname = './PE_stats.txt'
	fid = open(stats_fname,'w')
	# Accuracy difference
	acc_diff = high_PE_test_acc_mean - low_PE_test_acc_mean
	fid.write('Acc. diff. = ' + str(acc_diff) + '\n')
	# T-test for accuracy
	acc_t, acc_p = ttest_rel(high_PE_test_acc,low_PE_test_acc)
	fid.write('Accuracy t-test: t = ' + str(acc_t) + ', p = ' + str(acc_p) + '\n')
	# Confidence difference
	conf_diff = high_PE_test_conf_mean - low_PE_test_conf_mean
	fid.write('Conf. diff. = ' + str(conf_diff) + '\n')
	# T-test for confidence
	conf_t, conf_p = ttest_rel(high_PE_test_conf,low_PE_test_conf)
	fid.write('t-test: t = ' + str(conf_t) + ', p = ' + str(conf_p))
	# Close file
	fid.close()
	# Significance symbols
	if conf_p >= 0.05: conf_p_symb = 'ns'
	if conf_p < 0.05: conf_p_symb = '*'
	if conf_p < 0.01: conf_p_symb = '**'
	if conf_p < 0.001: conf_p_symb = '***'
	if conf_p < 0.0001: conf_p_symb = '****'
	if acc_p >= 0.05: acc_p_symb = 'ns'
	if acc_p < 0.05: acc_p_symb = '*'
	if acc_p < 0.01: acc_p_symb = '**'
	if acc_p < 0.001: acc_p_symb = '***'
	if acc_p < 0.0001: acc_p_symb = '****'

	# Font sizes
	axis_label_font_size = 22
	tick_font_size = 20
	significance_font_size = 20
	title_font_size = 30
	
	# Combined plot
	ax1 = plt.subplot(111)
	ax1.bar([0,1],[low_PE_test_acc_mean,high_PE_test_acc_mean],yerr=[low_PE_test_acc_se,high_PE_test_acc_se],width=0.8,color='gray')
	ax1.set_ylabel('P(Correct)', fontsize=axis_label_font_size)
	plt.ylim([0.7,1])
	plt.xticks([0,1,2.5,3.5],['Low','High','Low','High'], fontsize=tick_font_size)
	ax1.set_xlabel('Positive evidence',fontsize=axis_label_font_size)
	plt.yticks([0.7,0.8,0.9,1],['0.7','0.8','0.9','1'], fontsize=tick_font_size)
	ax1.spines['top'].set_visible(False)
	ax2 = ax1.twinx()
	ax2.bar([2.5,3.5],[low_PE_test_conf_mean,high_PE_test_conf_mean],yerr=[low_PE_test_conf_se,high_PE_test_conf_se],width=0.8,color='black')
	ax2.set_ylabel('Confidence', fontsize=axis_label_font_size)
	plt.ylim([0.7,1])
	plt.yticks([0.7,0.8,0.9,1],['0.7','0.8','0.9','1'], fontsize=tick_font_size)
	ax2.spines['top'].set_visible(False)
	# Significance
	max_y_val = np.max([low_PE_test_acc_mean + low_PE_test_acc_se, high_PE_test_acc_mean + high_PE_test_acc_se])
	y_start = max_y_val + 0.01
	y_end = max_y_val + 0.015
	ax1.plot([0,0,1,1],[y_start,y_end,y_end,y_start],color='black')
	ax1.text(0.5,y_end+0.005,acc_p_symb,fontsize=significance_font_size,horizontalalignment='center')
	max_y_val = np.max([low_PE_test_conf_mean + low_PE_test_conf_se, high_PE_test_conf_mean + high_PE_test_conf_se])
	y_start = max_y_val + 0.01
	y_end = max_y_val + 0.015
	ax2.plot([2.5,2.5,3.5,3.5],[y_start,y_end,y_end,y_start],color='black')
	ax2.text(3,y_end+0.005,conf_p_symb,fontsize=significance_font_size,horizontalalignment='center')
	# Title
	plt.title('MNIST',fontsize=title_font_size)
	# Save plot
	plot_fname = './PE_bias_v2_MNIST.png'
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