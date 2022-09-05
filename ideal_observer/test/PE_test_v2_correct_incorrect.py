import numpy as np
import matplotlib.pyplot as plt
from scipy.stats import sem
from scipy.stats import ttest_rel
import argparse

def plot(args):
	# Model directory
	model_dir = './' + args.train_regime + '_training/'
	# Get test results
	all_runs_low_PE_test_acc = []
	all_runs_low_PE_test_conf_correct = []
	all_runs_low_PE_test_conf_incorrect = []
	all_runs_high_PE_test_acc = []
	all_runs_high_PE_test_conf_correct = []
	all_runs_high_PE_test_conf_incorrect = []
	for r in range(args.N_runs):
		# Run directory
		run_dir = model_dir + 'run' + str(r+1) + '/'
		# Load test results
		test_results = np.load(run_dir + 'PE_bias_v2_results.npz')
		low_PE_test_acc = test_results['low_PE_test_acc']
		low_PE_test_conf_correct = test_results['low_PE_test_conf_correct'] 
		low_PE_test_conf_incorrect = test_results['low_PE_test_conf_incorrect']
		high_PE_test_acc = test_results['high_PE_test_acc'] 
		high_PE_test_conf_correct = test_results['high_PE_test_conf_correct']
		high_PE_test_conf_incorrect = test_results['high_PE_test_conf_incorrect'] 
		# Collect results for all runs
		all_runs_low_PE_test_acc.append(low_PE_test_acc)
		all_runs_low_PE_test_conf_correct.append(low_PE_test_conf_correct)
		all_runs_low_PE_test_conf_incorrect.append(low_PE_test_conf_incorrect)
		all_runs_high_PE_test_acc.append(high_PE_test_acc)
		all_runs_high_PE_test_conf_correct.append(high_PE_test_conf_correct)
		all_runs_high_PE_test_conf_incorrect.append(high_PE_test_conf_incorrect)
	# Convert to arrays
	all_runs_low_PE_test_acc = np.array(all_runs_low_PE_test_acc)
	all_runs_low_PE_test_conf_correct = np.array(all_runs_low_PE_test_conf_correct)
	all_runs_low_PE_test_conf_incorrect = np.array(all_runs_low_PE_test_conf_incorrect)
	all_runs_high_PE_test_acc = np.array(all_runs_high_PE_test_acc)
	all_runs_high_PE_test_conf_correct = np.array(all_runs_high_PE_test_conf_correct)
	all_runs_high_PE_test_conf_incorrect = np.array(all_runs_high_PE_test_conf_incorrect)
	# Summary statistics
	all_runs_low_PE_test_acc_mean = all_runs_low_PE_test_acc.mean(0)
	all_runs_low_PE_test_conf_correct_mean = all_runs_low_PE_test_conf_correct.mean(0)
	all_runs_low_PE_test_conf_correct_se = sem(all_runs_low_PE_test_conf_correct,0)
	all_runs_low_PE_test_conf_incorrect_mean = all_runs_low_PE_test_conf_incorrect.mean(0)
	all_runs_low_PE_test_conf_incorrect_se = sem(all_runs_low_PE_test_conf_incorrect,0)
	all_runs_high_PE_test_acc_mean = all_runs_high_PE_test_acc.mean(0)
	all_runs_high_PE_test_conf_correct_mean = all_runs_high_PE_test_conf_correct.mean(0)
	all_runs_high_PE_test_conf_correct_se = sem(all_runs_high_PE_test_conf_correct,0)
	all_runs_high_PE_test_conf_incorrect_mean = all_runs_high_PE_test_conf_incorrect.mean(0)
	all_runs_high_PE_test_conf_incorrect_se = sem(all_runs_high_PE_test_conf_incorrect,0)
	# Find PE conditions
	target_acc = 0.75
	# Low PE
	low_PE_ind = np.abs(all_runs_low_PE_test_acc_mean - target_acc).argmin()
	low_PE_test_conf_correct = all_runs_low_PE_test_conf_correct[:,low_PE_ind]
	low_PE_test_conf_correct_mean = all_runs_low_PE_test_conf_correct_mean[low_PE_ind]
	low_PE_test_conf_correct_se = all_runs_low_PE_test_conf_correct_se[low_PE_ind]
	low_PE_test_conf_incorrect = all_runs_low_PE_test_conf_incorrect[:,low_PE_ind]
	low_PE_test_conf_incorrect_mean = all_runs_low_PE_test_conf_incorrect_mean[low_PE_ind]
	low_PE_test_conf_cinorrect_se = all_runs_low_PE_test_conf_incorrect_se[low_PE_ind]
	# High PE
	high_PE_ind = np.abs(all_runs_high_PE_test_acc_mean - target_acc).argmin()
	high_PE_test_conf_correct = all_runs_high_PE_test_conf_correct[:,high_PE_ind]
	high_PE_test_conf_correct_mean = all_runs_high_PE_test_conf_correct_mean[high_PE_ind]
	high_PE_test_conf_correct_se = all_runs_high_PE_test_conf_correct_se[high_PE_ind]
	high_PE_test_conf_incorrect = all_runs_high_PE_test_conf_incorrect[:,high_PE_ind]
	high_PE_test_conf_incorrect_mean = all_runs_high_PE_test_conf_incorrect_mean[high_PE_ind]
	high_PE_test_conf_cinorrect_se = all_runs_high_PE_test_conf_incorrect_se[high_PE_ind]

	# Stats
	# Open file
	stats_fname = model_dir + 'PE_stats_correct_incorrect.txt'
	fid = open(stats_fname,'w')
	# Confidence difference -- correct trials
	conf_correct_diff = high_PE_test_conf_correct_mean - low_PE_test_conf_correct_mean
	fid.write('Conf. diff. = ' + str(conf_correct_diff) + '\n')
	# T-test for confidence -- correct_trials
	conf_correct_t, conf_correct_p = ttest_rel(high_PE_test_conf_correct,low_PE_test_conf_correct)
	fid.write('t-test: t = ' + str(conf_correct_t) + ', p = ' + str(conf_correct_p) + '\n')
	# Confidence difference -- incorrect trials
	conf_incorrect_diff = high_PE_test_conf_incorrect_mean - low_PE_test_conf_incorrect_mean
	fid.write('Conf. diff. = ' + str(conf_incorrect_diff) + '\n')
	# T-test for confidence -- incorrect_trials
	conf_incorrect_t, conf_incorrect_p = ttest_rel(high_PE_test_conf_incorrect,low_PE_test_conf_incorrect)
	fid.write('t-test: t = ' + str(conf_incorrect_t) + ', p = ' + str(conf_incorrect_p))
	# Close file
	fid.close()
	# Significance symbols
	if conf_correct_p >= 0.05: conf_correct_p_symb = 'ns'
	if conf_correct_p < 0.05: conf_correct_p_symb = '*'
	if conf_correct_p < 0.01: conf_correct_p_symb = '**'
	if conf_correct_p < 0.001: conf_correct_p_symb = '***'
	if conf_correct_p < 0.0001: conf_correct_p_symb = '****'
	if conf_incorrect_p >= 0.05: conf_incorrect_p_symb = 'ns'
	if conf_incorrect_p < 0.05: conf_incorrect_p_symb = '*'
	if conf_incorrect_p < 0.01: conf_incorrect_p_symb = '**'
	if conf_incorrect_p < 0.001: conf_incorrect_p_symb = '***'
	if conf_incorrect_p < 0.0001: conf_incorrect_p_symb = '****'

	# Font sizes
	axis_label_font_size = 22
	tick_font_size = 20
	significance_font_size = 20
	title_font_size = 30

	# Violin plot
	ax = plt.subplot(111)
	# Dummy figures for legend
	plt.fill([100,101],[100,101],color='lightgreen',alpha=0.5)
	plt.fill([100,101],[100,101],color='slateblue',alpha=0.5)
	# Confidence -- correct
	conf_correct_data = np.stack([low_PE_test_conf_correct, high_PE_test_conf_correct],1)
	parts = ax.violinplot(conf_correct_data, positions=[0,1], widths=0.8)
	for pc in parts['bodies']:
		pc.set_facecolor('lightgreen')
		pc.set_alpha(0.5)
	parts['cmins'].set_edgecolor('lightgreen')
	parts['cmaxes'].set_edgecolor('lightgreen')
	parts['cbars'].set_edgecolor('lightgreen')
	plt.scatter([0,1],[low_PE_test_conf_correct_mean, high_PE_test_conf_correct_mean], color='lightgreen')
	# Confidence -- incorrect
	conf_incorrect_data = np.stack([low_PE_test_conf_incorrect, high_PE_test_conf_incorrect],1)
	parts = ax.violinplot(conf_incorrect_data, positions=[2.5,3.5], widths=0.8)
	for pc in parts['bodies']:
		pc.set_facecolor('slateblue')
		pc.set_alpha(0.5)
	parts['cmins'].set_edgecolor('slateblue')
	parts['cmaxes'].set_edgecolor('slateblue')
	parts['cbars'].set_edgecolor('slateblue')
	ax.scatter([2.5,3.5],[low_PE_test_conf_incorrect_mean, high_PE_test_conf_incorrect_mean], color='slateblue')
	# Axis labels
	plt.xlim([-0.5,4])
	plt.xticks([0,1,2.5,3.5],['Low','High','Low','High'], fontsize=tick_font_size)
	ax.set_xlabel('Positive evidence',fontsize=axis_label_font_size)
	plt.ylim([0.42,1])
	plt.yticks([0.5,0.6,0.7,0.8,0.9,1],['0.5','0.6','0.7','0.8','0.9','1'], fontsize=tick_font_size)
	ax.set_ylabel('Confidence', fontsize=axis_label_font_size)
	ax.spines['top'].set_visible(False)
	ax.spines['right'].set_visible(False)
	# Legend
	plt.legend(['Correct','Incorrect'],frameon=False,fontsize=axis_label_font_size,bbox_to_anchor=(0.52,0.32))
	# Significance
	max_y_val = np.max([low_PE_test_conf_correct.max(),high_PE_test_conf_correct.max()])
	y_start = max_y_val + 0.01
	y_end = max_y_val + 0.015
	ax.plot([0,0,1,1],[y_start,y_end,y_end,y_start],color='black')
	ax.text(0.5,y_end,conf_correct_p_symb,fontsize=significance_font_size,horizontalalignment='center')
	max_y_val = np.max([low_PE_test_conf_incorrect.max(),high_PE_test_conf_incorrect.max()])
	y_start = max_y_val + 0.01
	y_end = max_y_val + 0.015
	ax.plot([2.5,2.5,3.5,3.5],[y_start,y_end,y_end,y_start],color='black')
	ax.text(3,y_end,conf_incorrect_p_symb,fontsize=significance_font_size,horizontalalignment='center')
	# Title
	plt.title('PE bias (v2)',fontsize=title_font_size)
	# Save plot
	plt.tight_layout()
	plot_fname = model_dir + 'ideal_observer_PE_bias_v2_correct_incorrect.png'
	plt.savefig(plot_fname, bbox_inches='tight', dpi=300)
	plt.close()

def main():

	# Training settings
	parser = argparse.ArgumentParser()
	parser.add_argument('--train_regime', type=str, default='standard', help="{'standard', 'fixed_mu'}")
	parser.add_argument('--N_runs', type=int, default=100)
	args = parser.parse_args()

	# Plot
	plot(args)

if __name__ == '__main__':
	main()