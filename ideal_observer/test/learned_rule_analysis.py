import numpy as np
import matplotlib.pyplot as plt
import argparse

def plot(args):
	# Model directory
	model_dir = './' + args.train_regime + '_training/'
	# Get results
	all_runs_acc = []
	all_runs_conf = []
	for r in range(args.N_runs):
		# Run directory
		run_dir = model_dir + 'run' + str(r+1) + '/'
		# Load test results
		test_results = np.load(run_dir + 's1s2_results.npz')
		signal_vals = test_results['signal_test_vals']
		# Collect results
		all_runs_acc.append(test_results['all_acc'])
		all_runs_conf.append(test_results['all_conf'])
	# Convert to arrays
	all_runs_acc = np.array(all_runs_acc)
	all_runs_conf = np.array(all_runs_conf)
	# Summary statistics
	all_runs_acc_mn = all_runs_acc.mean(0)
	all_runs_conf_mn = all_runs_conf.mean(0)
	# Axis ticks indices
	signal_vals_plot = np.array([0.2,0.4,0.6,0.8,1.0])
	signal_vals_plot_ind = np.abs(np.expand_dims(signal_vals,0)-np.expand_dims(signal_vals_plot,1)).argmin(1)
	# Font sizes
	cbar_label_fontsize = 18
	cbar_tick_fontsize = 16
	axis_label_fontsize = 18
	axis_tick_fontsize = 16
	title_fontsize = 22
	# Plot evidence for s1/s2 vs. accuracy
	acc_min = 0.5
	acc_max = 1.0
	acc_ticks = [0.5,0.6,0.7,0.8,0.9,1.0]
	ax = plt.subplot(111)
	plt.imshow(all_runs_acc_mn, origin='lower', vmin=acc_min, vmax=acc_max)
	cbar = plt.colorbar(ticks=acc_ticks)
	cbar.ax.set_yticklabels(np.array(acc_ticks).astype(np.str))
	cbar.ax.tick_params(labelsize=cbar_tick_fontsize)
	cbar.set_label('P(Correct)', fontsize=cbar_label_fontsize)
	plt.xticks(signal_vals_plot_ind, signal_vals_plot.astype(np.str), fontsize=axis_tick_fontsize)
	plt.yticks(signal_vals_plot_ind, signal_vals_plot.astype(np.str), fontsize=axis_tick_fontsize)
	plt.xlabel('$\mu_{s1}$', fontsize=axis_label_fontsize)
	plt.ylabel('$\mu_{s2}$', fontsize=axis_label_fontsize)
	plt.title('Ideal observer accuracy\nvs. stimulus evidence', fontsize=title_fontsize)
	plot_fname = model_dir + 's1s2_acc.png'
	plt.savefig(plot_fname, bbox_inches='tight', dpi=300)
	plt.close()
	# Plot evidence for s1/s2 vs. confidence
	conf_min = 0.6
	conf_max = 0.95
	conf_ticks = [0.6,0.7,0.8,0.9]
	ax = plt.subplot(111)
	plt.imshow(all_runs_conf_mn, origin='lower', vmin=conf_min, vmax=conf_max)
	cbar = plt.colorbar(ticks=conf_ticks)
	cbar.ax.set_yticklabels(np.array(conf_ticks).astype(np.str))
	cbar.ax.tick_params(labelsize=cbar_tick_fontsize)
	cbar.set_label('Confidence', fontsize=cbar_label_fontsize)
	plt.xticks(signal_vals_plot_ind, signal_vals_plot.astype(np.str), fontsize=axis_tick_fontsize)
	plt.yticks(signal_vals_plot_ind, signal_vals_plot.astype(np.str), fontsize=axis_tick_fontsize)
	plt.xlabel('$\mu_{s1}$', fontsize=axis_label_fontsize)
	plt.ylabel('$\mu_{s2}$', fontsize=axis_label_fontsize)
	plt.title('Ideal observer confidence\nvs. stimulus evidence', fontsize=title_fontsize)
	plot_fname = model_dir + 's1s2_conf.png'
	plt.savefig(plot_fname, bbox_inches='tight', dpi=300)
	plt.close()

	# Save accuracy and confidence functions
	np.savez(model_dir + 'ideal_observer_conf.npz', conf=all_runs_conf_mn)
	np.savez(model_dir + 'ideal_observer_acc.npz', acc=all_runs_acc_mn)

def main():

	# Settings
	parser = argparse.ArgumentParser()
	parser.add_argument('--train_regime', type=str, default='standard', help="{'standard', 'fixed_mu'}")
	parser.add_argument('--N_runs', type=int, default=100)
	args = parser.parse_args()

	# Plot
	plot(args)

if __name__ == '__main__':
	main()