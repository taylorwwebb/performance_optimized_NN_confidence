import numpy as np
import matplotlib.pyplot as plt
from scipy.stats import sem
from scipy.stats import gaussian_kde
import argparse

def plot(args):
	all_runs_control_d = []
	all_runs_control_meta_d = []
	all_runs_control_trial_conf = []
	all_runs_control_correct_pred = []
	all_runs_lesion_d = []
	all_runs_lesion_meta_d = []
	all_runs_lesion_trial_conf = []
	all_runs_lesion_correct_pred = []
	for r in range(args.N_runs):
		# Control results
		results_dir = './lesion_' + str(args.lesion) +'/run' + str(r+1) + '/'
		results = np.load(results_dir + 'control_results.npz')
		all_runs_control_d.append(results['all_test_d'])
		all_runs_control_meta_d.append(results['all_test_meta_d'])
		all_runs_control_trial_conf.append(results['all_trial_conf'].flatten())
		all_runs_control_correct_pred.append(results['all_trial_y_targ'].flatten() == results['all_trial_y_pred'].flatten().round())
		# Lesion results
		results_dir = './lesion_' + str(args.lesion) +'/run' + str(r+1) + '/'
		results = np.load(results_dir + 'lesion_results.npz')
		all_runs_lesion_d.append(results['all_test_d'])
		all_runs_lesion_meta_d.append(results['all_test_meta_d'])
		all_runs_lesion_trial_conf.append(results['all_trial_conf'].flatten())
		all_runs_lesion_correct_pred.append(results['all_trial_y_targ'].flatten() == results['all_trial_y_pred'].flatten().round())
	# Convert to arrays
	all_runs_control_d = np.array(all_runs_control_d)
	all_runs_control_meta_d = np.array(all_runs_control_meta_d)
	all_runs_control_trial_conf = np.concatenate(all_runs_control_trial_conf)
	all_runs_control_correct_pred = np.concatenate(all_runs_control_correct_pred)
	all_runs_lesion_d = np.array(all_runs_lesion_d)
	all_runs_lesion_meta_d = np.array(all_runs_lesion_meta_d)
	all_runs_lesion_trial_conf = np.concatenate(all_runs_lesion_trial_conf)
	all_runs_lesion_correct_pred = np.concatenate(all_runs_lesion_correct_pred)
	# Summary statistics
	control_d_mn = all_runs_control_d.mean(0)
	control_d_se = sem(all_runs_control_d,0)
	control_meta_d_mn = all_runs_control_meta_d.mean(0)
	control_meta_d_se = sem(all_runs_control_meta_d,0)
	lesion_d_mn = all_runs_lesion_d.mean(0)
	lesion_d_se = sem(all_runs_lesion_d,0)
	lesion_meta_d_mn = all_runs_lesion_meta_d.mean(0)
	lesion_meta_d_se = sem(all_runs_lesion_meta_d,0)
	# Plot d' vs. meta-d'
	axis_fontsize = 18
	tick_fontsize = 16
	legend_fontsize = 16
	title_fontsize = 18
	ax = plt.subplot(111)
	ax.plot([0,2.5],[0,2.5],color='gray',linestyle='dashed',alpha=0.5)
	ax.plot([0,2.5],[0,0],color='black',linestyle=':',alpha=0.5)
	ax.errorbar(control_d_mn, control_meta_d_mn, xerr=control_d_se, yerr=control_meta_d_se, color='black')
	ax.errorbar(lesion_d_mn, lesion_meta_d_mn, xerr=lesion_d_se, yerr=lesion_meta_d_se, color='red')
	plt.legend(["meta-d'=d'", "meta-d'=0", 'Control', 'Lesion'], frameon=False, fontsize=legend_fontsize, loc=2)
	plt.xlim([0.5,2])
	plt.ylim([-0.25,2])
	plt.xticks([0.5,1,1.5,2],['0.5','1','1.5','2'],fontsize=tick_fontsize)
	plt.yticks([0,0.5,1,1.5,2],['0','0.5','1','1.5','2'],fontsize=tick_fontsize)
	plt.xlabel("d'",fontsize=axis_fontsize)
	plt.ylabel("meta-d'",fontsize=axis_fontsize)
	ax.spines['right'].set_visible(False)
	ax.spines['top'].set_visible(False)
	ax.yaxis.set_ticks_position('left')
	ax.xaxis.set_ticks_position('bottom')
	plt.title('Effect of simulated V1 lesion', fontsize=title_fontsize)
	plot_fname = './lesion_' + str(args.lesion) + '/blindsight_d_vs_meta_d.png'
	plt.savefig(plot_fname, bbox_inches='tight', dpi=300)
	plt.close()
	# Plot confidence distributions for correct vs. incorrect trials
	axis_fontsize = 14
	tick_fontsize = 12
	legend_fontsize = 12
	title_fontsize = 14
	# X range
	min_x = .45
	max_x = 1
	x_range = np.linspace(min_x,max_x,1000)
	# Control
	control_conf_correct = all_runs_control_trial_conf[all_runs_control_correct_pred]
	control_conf_correct_kernel = gaussian_kde(control_conf_correct)
	control_conf_correct_dist = control_conf_correct_kernel(x_range)
	control_conf_incorrect = all_runs_control_trial_conf[np.logical_not(all_runs_control_correct_pred)]
	control_conf_incorrect_kernel = gaussian_kde(control_conf_incorrect)
	control_conf_incorrect_dist = control_conf_incorrect_kernel(x_range)
	ax = plt.subplot(2,1,1)
	plt.fill(np.append(x_range, [x_range[-1], x_range[0]]),np.append(control_conf_correct_dist, [0, control_conf_correct_dist[0]]),color='turquoise',alpha=0.5,edgecolor=None)
	plt.fill(np.append(x_range, [x_range[-1], x_range[0]]),np.append(control_conf_incorrect_dist, [0, control_conf_incorrect_dist[0]]),color='salmon',alpha=0.5,edgecolor=None)
	plt.xlabel('Confidence', fontsize=axis_fontsize)
	plt.ylabel('Density estimate', fontsize=axis_fontsize)
	plt.xlim([min_x,max_x])
	plt.xticks([.5,.6,.7,.8,.9,1],['0.5','0.6','0.7','0.8','0.9','1'], fontsize=tick_fontsize)
	plt.yticks(fontsize=tick_fontsize)
	plt.title('Control', fontsize=title_fontsize)
	ax.spines['right'].set_visible(False)
	ax.spines['top'].set_visible(False)
	ax.yaxis.set_ticks_position('left')
	ax.xaxis.set_ticks_position('bottom')
	plt.tight_layout()
	# Lesion
	lesion_conf_correct = all_runs_lesion_trial_conf[all_runs_lesion_correct_pred]
	lesion_conf_correct_kernel = gaussian_kde(lesion_conf_correct)
	lesion_conf_correct_dist = lesion_conf_correct_kernel(x_range)
	lesion_conf_incorrect = all_runs_lesion_trial_conf[np.logical_not(all_runs_lesion_correct_pred)]
	lesion_conf_incorrect_kernel = gaussian_kde(lesion_conf_incorrect)
	lesion_conf_incorrect_dist = lesion_conf_incorrect_kernel(x_range)
	ax = plt.subplot(2,1,2)
	plt.fill(np.append(x_range, [x_range[-1], x_range[0]]),np.append(lesion_conf_correct_dist, [0, lesion_conf_correct_dist[0]]),color='turquoise',alpha=0.5,edgecolor=None)
	plt.fill(np.append(x_range, [x_range[-1], x_range[0]]),np.append(lesion_conf_incorrect_dist, [0, lesion_conf_incorrect_dist[0]]),color='salmon',alpha=0.5,edgecolor=None)
	plt.xlabel('Confidence', fontsize=axis_fontsize)
	plt.ylabel('Density estimate', fontsize=axis_fontsize)
	plt.xlim([min_x,max_x])
	plt.xticks([.5,.6,.7,.8,.9,1],['0.5','0.6','0.7','0.8','0.9','1'], fontsize=tick_fontsize)
	plt.yticks(fontsize=tick_fontsize)
	plt.title('Lesion', fontsize=title_fontsize)
	ax.spines['right'].set_visible(False)
	ax.spines['top'].set_visible(False)
	ax.yaxis.set_ticks_position('left')
	ax.xaxis.set_ticks_position('bottom')
	plt.tight_layout()
	plt.legend(['Correct', 'Incorrect'], frameon=False, loc=1)
	# Save
	plot_fname = './lesion_' + str(args.lesion) + '/blindsight_conf_dist.png'
	plt.savefig(plot_fname, bbox_inches='tight', dpi=300)
	plt.close()

def main():

	# Settings
	parser = argparse.ArgumentParser()
	parser.add_argument('--N_runs', type=int, default=100)
	parser.add_argument('--lesion', type=float, default=0.01)
	args = parser.parse_args()

	# Plot
	plot(args)

if __name__ == '__main__':
	main()