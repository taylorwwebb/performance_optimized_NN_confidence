import numpy as np
import matplotlib.pyplot as plt
from scipy.stats import sem
from scipy.stats import ttest_rel
import argparse

def plot(args):
	all_runs_test_acc = []
	all_runs_test_opt_out_rate_correct = []
	all_runs_test_opt_out_rate_incorrect = []
	for r in range(args.N_runs):
		# Run directory
		run_dir = './run' + str(r+1) + '/'
		# Load test results
		test_results = np.load(run_dir + 'test_results.npz')
		all_test_acc = test_results['all_test_acc'] / 100
		all_test_opt_out_rate_correct = test_results['all_test_opt_out_rate_correct'] / 100
		all_test_opt_out_rate_incorrect = test_results['all_test_opt_out_rate_incorrect'] / 100
		# Collect results for all runs
		all_runs_test_acc.append(all_test_acc)
		all_runs_test_opt_out_rate_correct.append(all_test_opt_out_rate_correct)
		all_runs_test_opt_out_rate_incorrect.append(all_test_opt_out_rate_incorrect)
	# Convert to arrays
	all_runs_test_acc = np.array(all_runs_test_acc)
	all_runs_test_opt_out_rate_correct = np.array(all_runs_test_opt_out_rate_correct)
	all_runs_test_opt_out_rate_incorrect = np.array(all_runs_test_opt_out_rate_incorrect)
	# Summary statistics
	all_runs_test_acc_mean = all_runs_test_acc.mean(0)
	all_runs_test_opt_out_rate_correct_mean = all_runs_test_opt_out_rate_correct.mean(0)
	all_runs_test_opt_out_rate_correct_se = sem(all_runs_test_opt_out_rate_correct,0)
	all_runs_test_opt_out_rate_incorrect_mean = all_runs_test_opt_out_rate_incorrect.mean(0)
	all_runs_test_opt_out_rate_incorrect_se = sem(all_runs_test_opt_out_rate_incorrect,0)
	# Find PE conditions
	target_acc = 0.75
	# Low PE
	low_PE_ind = np.abs(all_runs_test_acc_mean[0,:] - target_acc).argmin()
	low_PE_test_opt_out_rate_correct = all_runs_test_opt_out_rate_correct[:,0,low_PE_ind]
	low_PE_test_opt_out_rate_correct_mean = all_runs_test_opt_out_rate_correct_mean[0,low_PE_ind]
	low_PE_test_opt_out_rate_correct_se = all_runs_test_opt_out_rate_correct_se[0,low_PE_ind]
	low_PE_test_opt_out_rate_incorrect = all_runs_test_opt_out_rate_incorrect[:,0,low_PE_ind]
	low_PE_test_opt_out_rate_incorrect_mean = all_runs_test_opt_out_rate_incorrect_mean[0,low_PE_ind]
	low_PE_test_opt_out_rate_incorrect_se = all_runs_test_opt_out_rate_incorrect_se[0,low_PE_ind]
	# High PE
	high_PE_ind = np.abs(all_runs_test_acc_mean[1,:] - target_acc).argmin()
	high_PE_test_opt_out_rate_correct = all_runs_test_opt_out_rate_correct[:,1,high_PE_ind]
	high_PE_test_opt_out_rate_correct_mean = all_runs_test_opt_out_rate_correct_mean[1,high_PE_ind]
	high_PE_test_opt_out_rate_correct_se = all_runs_test_opt_out_rate_correct_se[1,high_PE_ind]
	high_PE_test_opt_out_rate_incorrect = all_runs_test_opt_out_rate_incorrect[:,1,high_PE_ind]
	high_PE_test_opt_out_rate_incorrect_mean = all_runs_test_opt_out_rate_incorrect_mean[1,high_PE_ind]
	high_PE_test_opt_out_rate_incorrect_se = all_runs_test_opt_out_rate_incorrect_se[1,high_PE_ind]

	# Stats
	# Open file
	stats_fname = './PE_stats_correct_incorrect.txt'
	fid = open(stats_fname,'w')
	# Opt-out rate difference -- correct trials
	opt_out_rate_correct_diff = high_PE_test_opt_out_rate_correct_mean - low_PE_test_opt_out_rate_correct_mean
	fid.write('Opt-out rate (correct) diff. = ' + str(opt_out_rate_correct_diff) + '\n')
	# T-test for opt-out rate -- correct_trials
	opt_out_rate_correct_t, opt_out_rate_correct_p = ttest_rel(high_PE_test_opt_out_rate_correct,low_PE_test_opt_out_rate_correct)
	fid.write('t-test: t = ' + str(opt_out_rate_correct_t) + ', p = ' + str(opt_out_rate_correct_p) + '\n')
	# Opt-out rate difference -- incorrect trials
	opt_out_rate_incorrect_diff = high_PE_test_opt_out_rate_incorrect_mean - low_PE_test_opt_out_rate_incorrect_mean
	fid.write('Opt-out rate (incorrect) diff. = ' + str(opt_out_rate_incorrect_diff) + '\n')
	# T-test for opt-out rate -- incorrect_trials
	opt_out_rate_incorrect_t, opt_out_rate_incorrect_p = ttest_rel(high_PE_test_opt_out_rate_incorrect,low_PE_test_opt_out_rate_incorrect)
	fid.write('t-test: t = ' + str(opt_out_rate_incorrect_t) + ', p = ' + str(opt_out_rate_incorrect_p))
	# Close file
	fid.close()
	# Significance symbols
	if opt_out_rate_correct_p >= 0.05: opt_out_rate_correct_p_symb = 'ns'
	if opt_out_rate_correct_p < 0.05: opt_out_rate_correct_p_symb = '*'
	if opt_out_rate_correct_p < 0.01: opt_out_rate_correct_p_symb = '**'
	if opt_out_rate_correct_p < 0.001: opt_out_rate_correct_p_symb = '***'
	if opt_out_rate_correct_p < 0.0001: opt_out_rate_correct_p_symb = '****'
	if opt_out_rate_incorrect_p >= 0.05: opt_out_rate_incorrect_p_symb = 'ns'
	if opt_out_rate_incorrect_p < 0.05: opt_out_rate_incorrect_p_symb = '*'
	if opt_out_rate_incorrect_p < 0.01: opt_out_rate_incorrect_p_symb = '**'
	if opt_out_rate_incorrect_p < 0.001: opt_out_rate_incorrect_p_symb = '***'
	if opt_out_rate_incorrect_p < 0.0001: opt_out_rate_incorrect_p_symb = '****'

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
	# Opt-out rate -- correct
	opt_out_rate_correct_data = np.stack([low_PE_test_opt_out_rate_correct, high_PE_test_opt_out_rate_correct],1)
	parts = ax.violinplot(opt_out_rate_correct_data, positions=[0,1], widths=0.8)
	for pc in parts['bodies']:
		pc.set_facecolor('lightgreen')
		pc.set_alpha(0.5)
	parts['cmins'].set_edgecolor('lightgreen')
	parts['cmaxes'].set_edgecolor('lightgreen')
	parts['cbars'].set_edgecolor('lightgreen')
	plt.scatter([0,1],[low_PE_test_opt_out_rate_correct_mean, high_PE_test_opt_out_rate_correct_mean], color='lightgreen')
	# Opt-out rate -- incorrect
	opt_out_rate_incorrect_data = np.stack([low_PE_test_opt_out_rate_incorrect, high_PE_test_opt_out_rate_incorrect],1)
	parts = ax.violinplot(opt_out_rate_incorrect_data, positions=[2.5,3.5], widths=0.8)
	for pc in parts['bodies']:
		pc.set_facecolor('slateblue')
		pc.set_alpha(0.5)
	parts['cmins'].set_edgecolor('slateblue')
	parts['cmaxes'].set_edgecolor('slateblue')
	parts['cbars'].set_edgecolor('slateblue')
	ax.scatter([2.5,3.5],[low_PE_test_opt_out_rate_incorrect_mean, high_PE_test_opt_out_rate_incorrect_mean], color='slateblue')
	# Axis labels
	plt.xlim([-0.5,4])
	plt.xticks([0,1,2.5,3.5],['Low','High','Low','High'], fontsize=tick_font_size)
	ax.set_xlabel('Positive evidence',fontsize=axis_label_font_size)
	plt.ylim([0,1])
	plt.yticks([0,0.2,0.4,0.6,0.8,1],['0','0.2','0.4','0.6','0.8','1'], fontsize=tick_font_size)
	ax.set_ylabel('P(Opt-out)', fontsize=axis_label_font_size)
	ax.spines['top'].set_visible(False)
	ax.spines['right'].set_visible(False)
	# Legend
	plt.legend(['Correct','Incorrect'],frameon=False,fontsize=axis_label_font_size, bbox_to_anchor=(0.5, 0.33))
	# Significance
	min_y_val = np.min([low_PE_test_opt_out_rate_correct.min(),high_PE_test_opt_out_rate_correct.min()])
	y_start = min_y_val - 0.015
	y_end = min_y_val - 0.03
	ax.plot([0,0,1,1],[y_start,y_end,y_end,y_start],color='black')
	ax.text(0.5,y_end-0.08,opt_out_rate_correct_p_symb,fontsize=significance_font_size,horizontalalignment='center')
	min_y_val = np.min([low_PE_test_opt_out_rate_incorrect.min(),high_PE_test_opt_out_rate_incorrect.min()])
	y_start = min_y_val - 0.015
	y_end = min_y_val - 0.03
	ax.plot([2.5,2.5,3.5,3.5],[y_start,y_end,y_end,y_start],color='black')
	ax.text(3,y_end-0.08,opt_out_rate_incorrect_p_symb,fontsize=significance_font_size,horizontalalignment='center')
	# Title
	plt.title('PE bias (v1) - RL',fontsize=title_font_size)
	# Save plot
	plt.tight_layout()
	plot_fname = './PE_bias_RL_correct_incorrect.png'
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