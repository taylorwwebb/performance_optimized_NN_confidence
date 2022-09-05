import numpy as np
import matplotlib.pyplot as plt
from scipy.stats import sem
from scipy.stats import ttest_rel
import argparse

def plot(args):
	# Model directory
	model_dir = './' + args.train_regime + '_training/'
	# Get test results
	all_runs_test_acc = []
	all_runs_test_conf = []
	for r in range(args.N_runs):
		# Run directory
		run_dir = model_dir + 'run' + str(r+1) + '/'
		# Load test results
		test_results = np.load(run_dir + 'PE_bias_results.npz')
		signal_test_vals = test_results['signal_test_vals']
		noise_test_vals = test_results['noise_test_vals']
		all_test_acc = test_results['all_test_acc']
		all_test_conf = test_results['all_test_conf']
		# Collect results for all runs
		all_runs_test_acc.append(all_test_acc)
		all_runs_test_conf.append(all_test_conf)
	# Convert to arrays
	all_runs_test_acc = np.array(all_runs_test_acc)
	all_runs_test_conf = np.array(all_runs_test_conf)
	# Summary statistics
	all_runs_test_acc_mean = all_runs_test_acc.mean(0)
	all_runs_test_acc_se = sem(all_runs_test_acc,0)
	all_runs_test_conf_mean = all_runs_test_conf.mean(0)
	all_runs_test_conf_se = sem(all_runs_test_conf,0)
	# Find PE conditions
	target_acc = 0.75
	# Low PE
	low_PE_ind = np.abs(all_runs_test_acc_mean[0,:] - target_acc).argmin()
	low_PE_test_acc = all_runs_test_acc[:,0,low_PE_ind]
	low_PE_test_acc_mean = all_runs_test_acc_mean[0,low_PE_ind]
	low_PE_test_acc_se = all_runs_test_acc_se[0,low_PE_ind]
	low_PE_test_conf = all_runs_test_conf[:,0,low_PE_ind]
	low_PE_test_conf_mean = all_runs_test_conf_mean[0,low_PE_ind]
	low_PE_test_conf_se = all_runs_test_conf_se[0,low_PE_ind]
	# High PE
	high_PE_ind = np.abs(all_runs_test_acc_mean[1,:] - target_acc).argmin()
	high_PE_test_acc = all_runs_test_acc[:,1,high_PE_ind]
	high_PE_test_acc_mean = all_runs_test_acc_mean[1,high_PE_ind]
	high_PE_test_acc_se = all_runs_test_acc_se[1,high_PE_ind]
	high_PE_test_conf = all_runs_test_conf[:,1,high_PE_ind]
	high_PE_test_conf_mean = all_runs_test_conf_mean[1,high_PE_ind]
	high_PE_test_conf_se = all_runs_test_conf_se[1,high_PE_ind]

	# Stimulus parameters
	low_PE_signal = signal_test_vals[low_PE_ind]
	low_PE_noise = noise_test_vals[0]
	high_PE_signal = signal_test_vals[high_PE_ind]
	high_PE_noise = noise_test_vals[1]
	# Write to file
	params_fname = model_dir + 'PE_bias_stim_params.txt'
	fid = open(params_fname,'w')
	fid.write('Low PE:\n')
	fid.write('    signal = ' + str(low_PE_signal) + '\n')
	fid.write('    noise = ' + str(low_PE_noise) + '\n')
	fid.write('High PE:\n')
	fid.write('    signal = ' + str(high_PE_signal) + '\n')
	fid.write('    noise = ' + str(high_PE_noise))
	fid.close()

	# Stats
	# Open file
	stats_fname = model_dir + 'PE_bias_stats.txt'
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

	# Plot for comparison across datasets
	# Font sizes
	axis_label_font_size = 22
	tick_font_size = 20
	significance_font_size = 20
	title_font_size = 30
	# Combined plot
	ax1 = plt.subplot(111)
	ax1.bar([0,1],[low_PE_test_acc_mean,high_PE_test_acc_mean],yerr=[low_PE_test_acc_se,high_PE_test_acc_se],width=0.8,color='gray')
	ax1.set_ylabel('P(Correct)', fontsize=axis_label_font_size)
	plt.ylim([0.6,0.9])
	plt.xticks([0,1,2.5,3.5],['Low','High','Low','High'], fontsize=tick_font_size)
	ax1.set_xlabel('Positive evidence',fontsize=axis_label_font_size)
	plt.yticks([0.6,0.7,0.8,0.9],['0.6','0.7','0.8','0.9'], fontsize=tick_font_size)
	ax1.spines['top'].set_visible(False)
	ax2 = ax1.twinx()
	ax2.bar([2.5,3.5],[low_PE_test_conf_mean,high_PE_test_conf_mean],yerr=[low_PE_test_conf_se,high_PE_test_conf_se],width=0.8,color='black')
	ax2.set_ylabel('Confidence', fontsize=axis_label_font_size)
	plt.ylim([0.6,0.9])
	plt.yticks([0.6,0.7,0.8,0.9],['0.6','0.7','0.8','0.9'], fontsize=tick_font_size)
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
	plt.title('PE bias (v1)', fontsize=title_font_size)
	# Save plot
	plot_fname = model_dir + 'ideal_observer_PE_bias.png'
	plt.savefig(plot_fname, bbox_inches='tight', dpi=300)
	plt.close()

def main():

	# Training settings
	parser = argparse.ArgumentParser()
	parser.add_argument('--train_regime', type=str, default='standard', help="{'standard', 'fixed_mu'}")
	parser.add_argument('--N_runs', type=int, default=100)
	args = parser.parse_args()

	# Plot training progress
	plot(args)

if __name__ == '__main__':
	main()