import numpy as np
import matplotlib.pyplot as plt
from scipy.stats import sem
import argparse
import sys

def plot(args):
	# Load behavioral results
	if args.model_behavior == 'behavior':
		behavioral_data = np.load('./behavioral_data.npz')
		d_mn = behavioral_data['d_mn']
		d_se = behavioral_data['d_se']
		meta_d_mn = behavioral_data['meta_d_mn']
		meta_d_se = behavioral_data['meta_d_se']
		meta_d_rS1_mn = behavioral_data['meta_d_s1_mn']
		meta_d_rS1_se = behavioral_data['meta_d_s1_se']
		meta_d_rS2_mn = behavioral_data['meta_d_s2_mn']
		meta_d_rS2_se = behavioral_data['meta_d_s2_se']
	# Load model results
	elif args.model_behavior == 'model' or args.model_behavior == 'model_conf_noise_fit':
		# Filename 
		if args.model_behavior == 'model':
			if args.conf_noise == 0.0:
				fname = 'test_results.npz'
			else:
				fname = 'test_results_conf_noise_' + str(args.conf_noise) + '.npz'
		elif args.model_behavior == 'model_conf_noise_fit':
			conf_noise_fit = np.load('./conf_noise_fit.npz')['conf_noise']
			fname = 'test_results_conf_noise_' + str(conf_noise_fit) + '.npz'
		# Collect data from all runs	
		all_runs_d = []
		all_runs_meta_d = []
		all_runs_meta_d_rS1 = []
		all_runs_meta_d_rS2 = []
		for r in range(args.N_runs):
			# Run directory
			run_dir = './run' + str(r+1) + '/'
			# Load test results
			test_results = np.load(run_dir + fname)
			d = test_results['all_d']
			meta_d = test_results['all_meta_d']
			meta_d_rS1 = test_results['all_meta_d_rS1']
			meta_d_rS2 = test_results['all_meta_d_rS2']
			# Collect results for all runs
			all_runs_d.append(d)
			all_runs_meta_d.append(meta_d)
			all_runs_meta_d_rS1.append(meta_d_rS1)
			all_runs_meta_d_rS2.append(meta_d_rS2)
		# Convert to arrays
		all_runs_d = np.array(all_runs_d)
		all_runs_meta_d = np.array(all_runs_meta_d)
		all_runs_meta_d_rS1 = np.array(all_runs_meta_d_rS1)
		all_runs_meta_d_rS2 = np.array(all_runs_meta_d_rS2)
		# Summary statistics
		d_mn = all_runs_d.mean(0)
		d_se = sem(all_runs_d, 0)
		meta_d_mn = all_runs_meta_d.mean(0)
		meta_d_se = sem(all_runs_meta_d, 0)
		meta_d_rS1_mn = all_runs_meta_d_rS1.mean(0)
		meta_d_rS1_se = sem(all_runs_meta_d_rS1, 0)
		meta_d_rS2_mn = all_runs_meta_d_rS2.mean(0)
		meta_d_rS2_se = sem(all_runs_meta_d_rS2, 0)
	# Plot d' vs. meta-d'
	# Font sizes
	axis_label_fontsize = 18
	tick_fontsize = 14
	legend_fontsize = 14
	title_fontsize = 20
	# Axis limits / ticks
	x_min = 0.8
	x_max = 2.5
	x_ticks = [1,1.5,2,2.5]
	y_min = 0
	if args.model_behavior == 'behavior' or args.model_behavior == 'model_conf_noise_fit':
		y_max = 2.5
		y_ticks = [0,0.5,1,1.5,2,2.5]
	else:
		y_max = 3.5
		y_ticks = [0,1,2,3]
	# Generate plot
	ax = plt.subplot(111)
	ax.plot([x_min, x_max], [x_min, x_max], linestyle='dashed', color='gray')
	ax.errorbar(d_mn, meta_d_mn, xerr=d_se, yerr=meta_d_se, color='black')
	ax.errorbar(d_mn, meta_d_rS1_mn, xerr=d_se, yerr=meta_d_rS1_se, color='red')
	ax.errorbar(d_mn, meta_d_rS2_mn, xerr=d_se, yerr=meta_d_rS2_se, color='blue')
	plt.xlabel("d'", fontsize=axis_label_fontsize)
	plt.ylabel("meta-d'", fontsize=axis_label_fontsize)
	ax.spines['right'].set_visible(False)
	ax.spines['top'].set_visible(False)
	ax.yaxis.set_ticks_position('left')
	ax.xaxis.set_ticks_position('bottom')
	plt.xlim([x_min, x_max])
	plt.ylim([y_min, y_max])
	plt.xticks(x_ticks, np.array(x_ticks).astype(str), fontsize=tick_fontsize)
	plt.yticks(y_ticks, np.array(y_ticks).astype(str), fontsize=tick_fontsize)
	ax.legend(["meta-d' = d'", 'combined', 'response = s1', 'response = s2'], frameon=False, fontsize=legend_fontsize)
	if args.model_behavior == 'behavior':
		plt.title('Behavior', fontsize=title_fontsize)
		plot_fname = './d_vs_meta_d_behavior.png'
	elif args.model_behavior == 'model':
		plt.title('Model, ' + r'$\xi$' + ' = ' + str(args.conf_noise), fontsize=title_fontsize)
		plot_fname = './d_vs_meta_d_conf_noise_' + str(args.conf_noise) + '.png'
	elif args.model_behavior == 'model_conf_noise_fit':
		plt.title('Model', fontsize=title_fontsize)
		plot_fname = './d_vs_meta_d_model_conf_noise_fit.png'
	plt.savefig(plot_fname, bbox_inches='tight',dpi=300)
	plt.close()

def main():

	# Settings
	parser = argparse.ArgumentParser()
	parser.add_argument('--N_runs', type=int, default=100)
	parser.add_argument('--conf_noise', type=float, default=0.0)
	parser.add_argument('--model_behavior', type=str, default='model', help="{'model', 'behavior', 'model_conf_noise_fit'}")
	args = parser.parse_args()

	# Plot
	plot(args)

if __name__ == '__main__':
	main()