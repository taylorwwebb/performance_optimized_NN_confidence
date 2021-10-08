import numpy as np
import matplotlib.pyplot as plt
from scipy.stats import sem
from scipy.stats import gaussian_kde
import argparse

def plot(args):
	all_runs_control_d = []
	all_runs_control_meta_d = []
	all_runs_lesion_d = []
	all_runs_lesion_meta_d = []
	for r in range(args.N_runs):
		# Control results
		results_dir = './lesion_' + str(args.lesion) +'/run' + str(r+1) + '/'
		results = np.load(results_dir + 'control_results.npz')
		all_runs_control_d.append(results['all_test_d'])
		all_runs_control_meta_d.append(results['all_test_meta_d'])
		# Lesion results
		results_dir = './lesion_' + str(args.lesion) +'/run' + str(r+1) + '/'
		results = np.load(results_dir + 'lesion_results.npz')
		all_runs_lesion_d.append(results['all_test_d'])
		all_runs_lesion_meta_d.append(results['all_test_meta_d'])
	# Convert to arrays
	all_runs_control_d = np.array(all_runs_control_d)
	all_runs_control_meta_d = np.array(all_runs_control_meta_d)
	all_runs_lesion_d = np.array(all_runs_lesion_d)
	all_runs_lesion_meta_d = np.array(all_runs_lesion_meta_d)
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
	ax.errorbar(control_d_mn, control_meta_d_mn, xerr=control_d_se, yerr=control_meta_d_se, color='black')
	ax.errorbar(lesion_d_mn, lesion_meta_d_mn, xerr=lesion_d_se, yerr=lesion_meta_d_se, color='red')
	plt.legend(["meta-d'=d'", 'Control', 'Lesion'], frameon=False, fontsize=legend_fontsize, loc=2)
	plt.xlim([0.5,2.5])
	plt.ylim([0.5,2.5])
	plt.xticks([0.5,1,1.5,2,2.5],['0.5','1','1.5','2','2.5'],fontsize=tick_fontsize)
	plt.yticks([0.5,1,1.5,2,2.5],['0.5','1','1.5','2','2.5'],fontsize=tick_fontsize)
	plt.xlabel("d'",fontsize=axis_fontsize)
	plt.ylabel("meta-d'",fontsize=axis_fontsize)
	ax.spines['right'].set_visible(False)
	ax.spines['top'].set_visible(False)
	ax.yaxis.set_ticks_position('left')
	ax.xaxis.set_ticks_position('bottom')
	plt.title('Effect of simulated lesion to z layer', fontsize=title_fontsize)
	plot_fname = './lesion_' + str(args.lesion) + '/z_lesion_d_vs_meta_d.png'
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