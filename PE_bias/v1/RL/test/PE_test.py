import numpy as np
import matplotlib.pyplot as plt
from scipy.stats import sem
from scipy.stats import ttest_rel
import argparse

def plot(args):
	all_runs_test_acc = []
	all_runs_test_opt_out_rate = []
	for r in range(args.N_runs):
		# Run directory
		run_dir = './run' + str(r+1) + '/'
		# Load test results
		test_results = np.load(run_dir + 'test_results.npz')
		signal_test_vals = test_results['signal_test_vals']
		noise_test_vals = test_results['noise_test_vals']
		all_test_acc = test_results['all_test_acc'] / 100
		all_test_opt_out_rate = test_results['all_test_opt_out_rate'] / 100
		# Collect results for all runs
		all_runs_test_acc.append(all_test_acc)
		all_runs_test_opt_out_rate.append(all_test_opt_out_rate)
	# Convert to arrays
	all_runs_test_acc = np.array(all_runs_test_acc)
	all_runs_test_opt_out_rate = np.array(all_runs_test_opt_out_rate)
	# Summary statistics
	all_runs_test_acc_mean = all_runs_test_acc.mean(0)
	all_runs_test_acc_se = sem(all_runs_test_acc,0)
	all_runs_test_opt_out_rate_mean = all_runs_test_opt_out_rate.mean(0)
	all_runs_test_opt_out_rate_se = sem(all_runs_test_opt_out_rate,0)
	# Find PE conditions
	target_acc = 0.75
	# Low PE
	low_PE_ind = np.abs(all_runs_test_acc_mean[0,:] - target_acc).argmin()
	low_PE_signal = signal_test_vals[low_PE_ind]
	low_PE_test_acc = all_runs_test_acc[:,0,low_PE_ind]
	low_PE_test_acc_mean = all_runs_test_acc_mean[0,low_PE_ind]
	low_PE_test_acc_se = all_runs_test_acc_se[0,low_PE_ind]
	low_PE_test_opt_out_rate = all_runs_test_opt_out_rate[:,0,low_PE_ind]
	low_PE_test_opt_out_rate_mean = all_runs_test_opt_out_rate_mean[0,low_PE_ind]
	low_PE_test_opt_out_rate_se = all_runs_test_opt_out_rate_se[0,low_PE_ind]
	# High PE
	high_PE_ind = np.abs(all_runs_test_acc_mean[1,:] - target_acc).argmin()
	high_PE_signal = signal_test_vals[high_PE_ind]
	high_PE_test_acc = all_runs_test_acc[:,1,high_PE_ind]
	high_PE_test_acc_mean = all_runs_test_acc_mean[1,high_PE_ind]
	high_PE_test_acc_se = all_runs_test_acc_se[1,high_PE_ind]
	high_PE_test_opt_out_rate = all_runs_test_opt_out_rate[:,1,high_PE_ind]
	high_PE_test_opt_out_rate_mean = all_runs_test_opt_out_rate_mean[1,high_PE_ind]
	high_PE_test_opt_out_rate_se = all_runs_test_opt_out_rate_se[1,high_PE_ind]

	# Stats
	# Open file
	stats_fname = './PE_stats.txt'
	fid = open(stats_fname,'w')
	# Low and high PE conditions
	fid.write('Low PE condition: sigma = ' + str(noise_test_vals[0]) + ', mu = ' + str(low_PE_signal) + '\n')
	fid.write('High PE condition: sigma = ' + str(noise_test_vals[1]) + ', mu = ' + str(high_PE_signal) + '\n')
	# Accuracy difference
	acc_diff = high_PE_test_acc_mean - low_PE_test_acc_mean
	fid.write('Acc. diff. = ' + str(acc_diff) + '\n')
	# T-test for accuracy
	acc_t, acc_p = ttest_rel(high_PE_test_acc,low_PE_test_acc)
	fid.write('Accuracy t-test: t = ' + str(acc_t) + ', p = ' + str(acc_p) + '\n')
	# Opt-out rate difference
	opt_out_rate_diff = high_PE_test_opt_out_rate_mean - low_PE_test_opt_out_rate_mean
	fid.write('Opt-out rate diff. = ' + str(opt_out_rate_diff) + '\n')
	# T-test for opt-out rate
	opt_out_rate_t, opt_out_rate_p = ttest_rel(high_PE_test_opt_out_rate,low_PE_test_opt_out_rate)
	fid.write('t-test: t = ' + str(opt_out_rate_t) + ', p = ' + str(opt_out_rate_p))
	# Close file
	fid.close()
	# Significance symbols
	if opt_out_rate_p >= 0.05: opt_out_rate_p_symb = 'ns'
	if opt_out_rate_p < 0.05: opt_out_rate_p_symb = '*'
	if opt_out_rate_p < 0.01: opt_out_rate_p_symb = '**'
	if opt_out_rate_p < 0.001: opt_out_rate_p_symb = '***'
	if opt_out_rate_p < 0.0001: opt_out_rate_p_symb = '****'
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
	plt.ylim([0.65,0.85])
	plt.xticks([0,1,2.5,3.5],['Low','High','Low','High'], fontsize=tick_font_size)
	ax1.set_xlabel('Positive evidence',fontsize=axis_label_font_size)
	plt.yticks([0.65,0.7,0.75,0.8,0.85],['0.65','0.7','0.75','0.8','0.85'], fontsize=tick_font_size)
	ax1.spines['top'].set_visible(False)
	ax2 = ax1.twinx()
	ax2.bar([2.5,3.5],[low_PE_test_opt_out_rate_mean,high_PE_test_opt_out_rate_mean],yerr=[low_PE_test_opt_out_rate_se,high_PE_test_opt_out_rate_se],width=0.8,color='black')
	ax2.set_ylabel('P(Opt-out)', fontsize=axis_label_font_size)
	plt.ylim([0.4,0.65])
	plt.yticks([0.4,0.45,0.5,0.55,0.6,0.65],['0.4','0.45','0.5','0.55','0.6','0.65'], fontsize=tick_font_size)
	ax2.spines['top'].set_visible(False)
	# Significance
	max_y_val = np.max([low_PE_test_acc_mean + low_PE_test_acc_se, high_PE_test_acc_mean + high_PE_test_acc_se])
	y_start = max_y_val + 0.01
	y_end = max_y_val + 0.015
	ax1.plot([0,0,1,1],[y_start,y_end,y_end,y_start],color='black')
	ax1.text(0.5,y_end+0.005,acc_p_symb,fontsize=significance_font_size,horizontalalignment='center')
	max_y_val = np.max([low_PE_test_opt_out_rate_mean + low_PE_test_opt_out_rate_se, high_PE_test_opt_out_rate_mean + high_PE_test_opt_out_rate_se])
	y_start = max_y_val + 0.01
	y_end = max_y_val + 0.015
	ax2.plot([2.5,2.5,3.5,3.5],[y_start,y_end,y_end,y_start],color='black')
	ax2.text(3,y_end+0.005,opt_out_rate_p_symb,fontsize=significance_font_size,horizontalalignment='center')
	# Title
	plt.title('RL',fontsize=title_font_size)
	# Save plot
	plot_fname = './PE_bias_RL.png'
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