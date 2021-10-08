import numpy as np
import matplotlib.pyplot as plt
from scipy.stats import sem
from scipy.stats import gaussian_kde
import argparse

def plot(args):
	all_runs_d = []
	all_runs_meta_d = []
	for r in range(args.N_runs):
		# Results
		results_dir = './run' + str(r+1) + '/'
		results = np.load(results_dir + 'results.npz')
		TMS_vals = results['TMS_vals']
		all_runs_d.append(results['all_test_d'])
		all_runs_meta_d.append(results['all_test_meta_d'])
	# Convert to arrays
	all_runs_d = np.array(all_runs_d)
	all_runs_meta_d = np.array(all_runs_meta_d)
	# Summary statistics
	d_mn = all_runs_d.mean(0)
	d_se = sem(all_runs_d,0)
	meta_d_mn = all_runs_meta_d.mean(0)
	meta_d_se = sem(all_runs_meta_d,0)
	# TMS condition colors
	min_TMS_color = [0, 0.2, 0.4]
	max_TMS_color = [0.4, 0.55, 0.7]
	N_TMS_conditions = d_mn.shape[0] - 1
	all_TMS_colors = np.linspace(np.array(min_TMS_color),np.array(max_TMS_color),N_TMS_conditions)
	# Plot d' vs. meta-d'
	axis_fontsize = 18
	tick_fontsize = 16
	legend_fontsize = 16
	title_fontsize = 18
	ax = plt.subplot(111)
	ax.plot([0,3],[0,3],color='gray',linestyle='dashed',alpha=0.5)
	ax.errorbar(d_mn[0], meta_d_mn[0], xerr=d_se[0], yerr=meta_d_se[0], color='black')
	ax.errorbar(d_mn[1], meta_d_mn[1], xerr=d_se[1], yerr=meta_d_se[1], color=all_TMS_colors[0,:])
	ax.errorbar(d_mn[-1], meta_d_mn[-1], xerr=d_se[-1], yerr=meta_d_se[-1], color=all_TMS_colors[-1,:])
	ax.errorbar(d_mn, meta_d_mn, xerr=0, yerr=0, color='gray', alpha=0.2, linestyle=':')
	for t in range(1,d_mn.shape[0]):
		ax.errorbar(d_mn[t], meta_d_mn[t], xerr=d_se[t], yerr=meta_d_se[t], color=all_TMS_colors[t-1,:])
	plt.xlim([0,3])
	plt.ylim([0,3])
	plt.xticks([0,1,2,3],['0','1','2','3'],fontsize=tick_fontsize)
	plt.yticks([0,1,2,3],['0','1','2','3'],fontsize=tick_fontsize)
	plt.xlabel("d'",fontsize=axis_fontsize)
	plt.ylabel("meta-d'",fontsize=axis_fontsize)
	ax.spines['right'].set_visible(False)
	ax.spines['top'].set_visible(False)
	ax.yaxis.set_ticks_position('left')
	ax.xaxis.set_ticks_position('bottom')
	plt.legend(["meta-d'=d'", 'Control', 'TMS (' + r'$\xi$'  + '=' + str(int(TMS_vals[1])) + ')', 'TMS (' + r'$\xi$'  + '=' + str(int(TMS_vals[-1])) + ')'], frameon=False, fontsize=legend_fontsize, loc=2)
	plt.title('Effect of simulated TMS to z layer', fontsize=title_fontsize)
	plot_fname = './TMS_d_meta_d.png'
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