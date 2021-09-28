import numpy as np
import matplotlib.pyplot as plt
from scipy.stats import ttest_rel, sem
from scipy.special import logit
from scipy.special import expit
from sklearn.linear_model import LinearRegression
import argparse

def plot(args):
	all_runs_acc = []
	all_runs_conf = []
	all_runs_y_pred = []
	all_runs_y_pred_rect = []
	all_runs_pc1 = []
	all_runs_pc2 = []
	all_runs_abs_pc1 = []
	all_BE_r2 = []
	all_BE_pred = []
	all_RCE_r2 = []
	all_RCE_pred = []
	all_multi_beta = []
	all_multi_r2 = []
	all_multi_pred = []
	for r in range(args.N_runs):
		# Run directory
		run_dir = './run' + str(r+1) + '/'
		# Load test results
		test_results = np.load(run_dir + 's1s2_test_results.npz')
		signal_vals = test_results['signal_test_vals']
		# Collect results
		all_runs_acc.append(test_results['all_acc'] / 100.0)
		all_runs_conf.append(test_results['all_conf'] / 100.0)
		all_runs_y_pred.append(test_results['all_y_pred'])
		all_runs_y_pred_rect.append(test_results['all_y_pred_rect'])
		all_runs_pc1.append(test_results['all_pc1'])
		all_runs_pc2.append(test_results['all_pc2'])
		all_runs_abs_pc1.append(test_results['all_abs_pc1'])
		# Regression models
		# S1/S2
		s1 = np.tile(np.expand_dims(signal_vals,1),[1,100])
		s2 = np.tile(np.expand_dims(signal_vals,0),[100,1])
		# Convert confidence using logit transform
		logit_conf = logit(test_results['all_conf'] / 100.0)
		# Balance-of-evidence
		BE = np.abs(s1 - s2)
		BE_reg = LinearRegression().fit(np.expand_dims(BE.flatten(),1), logit_conf.flatten())
		BE_r2 = BE_reg.score(np.expand_dims(BE.flatten(),1), logit_conf.flatten())
		all_BE_r2.append(BE_r2)
		BE_pred = expit(BE_reg.predict(np.expand_dims(BE.flatten(),1)).reshape(100,100))
		all_BE_pred.append(BE_pred)
		# Response-congruent-evidence
		RCE = ((s1.flatten()*(s1.flatten()>s2.flatten()).astype(np.int)) + (s2.flatten()*(s2.flatten()>=s1.flatten()).astype(np.int))).reshape(100,100)
		RCE_reg = LinearRegression().fit(np.expand_dims(RCE.flatten(),1), logit_conf.flatten())
		RCE_r2 = RCE_reg.score(np.expand_dims(RCE.flatten(),1), logit_conf.flatten())
		all_RCE_r2.append(RCE_r2)
		RCE_pred = expit(RCE_reg.predict(np.expand_dims(RCE.flatten(),1)).reshape(100,100))
		all_RCE_pred.append(RCE_pred)
		# Multiple regression (BE + RCE)
		multi_reg = LinearRegression().fit(np.stack([BE.flatten(), RCE.flatten()],1), logit_conf.flatten())
		multi_reg_beta = multi_reg.coef_
		all_multi_beta.append(multi_reg_beta)
		multi_r2 = multi_reg.score(np.stack([BE.flatten(), RCE.flatten()],1), logit_conf.flatten())
		all_multi_r2.append(multi_r2)
		multi_pred = expit(multi_reg.predict(np.stack([BE.flatten(), RCE.flatten()],1)).reshape(100,100))
		all_multi_pred.append(multi_pred)
	# Convert to arrays
	all_runs_acc = np.array(all_runs_acc)
	all_runs_conf = np.array(all_runs_conf)
	all_runs_y_pred = np.array(all_runs_y_pred)
	all_runs_y_pred_rect = np.array(all_runs_y_pred_rect)
	all_runs_pc1 = np.array(all_runs_pc1)
	all_runs_pc2 = np.array(all_runs_pc2)
	all_runs_abs_pc1 = np.array(all_runs_abs_pc1)
	all_BE_r2 = np.array(all_BE_r2)
	all_BE_pred = np.array(all_BE_pred)
	all_RCE_r2 = np.array(all_RCE_r2)
	all_RCE_pred = np.array(all_RCE_pred)
	all_multi_beta = np.array(all_multi_beta)
	all_multi_r2 = np.array(all_multi_r2)
	all_multi_pred = np.array(all_multi_pred)
	# Summary statistics
	all_runs_acc_mn = all_runs_acc.mean(0)
	all_runs_conf_mn = all_runs_conf.mean(0)
	all_runs_y_pred_mn = all_runs_y_pred.mean(0)
	all_runs_y_pred_rect_mn = all_runs_y_pred_rect.mean(0)
	all_runs_pc1_mn = all_runs_pc1.mean(0)
	all_runs_pc2_mn = all_runs_pc2.mean(0)
	all_runs_abs_pc1_mn = all_runs_abs_pc1.mean(0)
	all_BE_r2_mn = all_BE_r2.mean(0)
	all_BE_r2_se = sem(all_BE_r2,0)
	all_RCE_r2_mn = all_RCE_r2.mean(0)
	all_RCE_r2_se = sem(all_RCE_r2,0)
	all_multi_r2_mn = all_multi_r2.mean(0)
	all_multi_r2_se = sem(all_multi_r2,0)
	all_multi_beta_mn = all_multi_beta.mean(0)
	all_BE_pred_mn = all_BE_pred.mean(0)
	all_RCE_pred_mn = all_RCE_pred.mean(0)
	all_multi_pred_mn = all_multi_pred.mean(0)
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
	plt.title('Decision accuracy\nvs. stimulus evidence', fontsize=title_fontsize)
	plot_fname = './s1_vs_s2_PENE_acc.png'
	plt.savefig(plot_fname, bbox_inches='tight', dpi=300)
	plt.close()
	# Plot evidence for s1/s2 vs. confidence
	conf_min = 0.65
	conf_max = 1.0
	conf_ticks = [0.7,0.8,0.9,1.0]
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
	plt.title('Confidence\nvs. stimulus evidence', fontsize=title_fontsize)
	plot_fname = './s1_vs_s2_PENE_conf.png'
	plt.savefig(plot_fname, bbox_inches='tight', dpi=300)
	plt.close()
	# Plot evidence for s1/s2 vs. predicted class
	y_pred_min = 0.0
	y_pred_max = 1.0
	y_pred_ticks = [0.0,0.2,0.4,0.6,0.8,1.0]
	ax = plt.subplot(111)
	plt.imshow(all_runs_y_pred_mn, origin='lower', vmin=y_pred_min, vmax=y_pred_max)
	cbar = plt.colorbar(ticks=y_pred_ticks)
	cbar.ax.set_yticklabels(np.array(y_pred_ticks).astype(np.str))
	cbar.ax.tick_params(labelsize=cbar_tick_fontsize)
	cbar.set_label('Decision output', fontsize=cbar_label_fontsize)
	plt.xticks(signal_vals_plot_ind, signal_vals_plot.astype(np.str), fontsize=axis_tick_fontsize)
	plt.yticks(signal_vals_plot_ind, signal_vals_plot.astype(np.str), fontsize=axis_tick_fontsize)
	plt.xlabel('$\mu_{s1}$', fontsize=axis_label_fontsize)
	plt.ylabel('$\mu_{s2}$', fontsize=axis_label_fontsize)
	plt.title('Neural network decision output\nvs. stimulus evidence', fontsize=title_fontsize)
	plot_fname = './s1_vs_s2_PENE_y_pred.png'
	plt.savefig(plot_fname, bbox_inches='tight', dpi=300)
	plt.close()
	# Plot evidence for s1/s2 vs. probability of chosen class
	ax = plt.subplot(111)
	plt.imshow(all_runs_y_pred_rect_mn, origin='lower', vmin=conf_min, vmax=conf_max)
	cbar = plt.colorbar(ticks=conf_ticks)
	cbar.ax.set_yticklabels(np.array(conf_ticks).astype(np.str))
	cbar.ax.tick_params(labelsize=cbar_tick_fontsize)
	cbar.set_label('Rectified decision output', fontsize=cbar_label_fontsize)
	plt.xticks(signal_vals_plot_ind, signal_vals_plot.astype(np.str), fontsize=axis_tick_fontsize)
	plt.yticks(signal_vals_plot_ind, signal_vals_plot.astype(np.str), fontsize=axis_tick_fontsize)
	plt.xlabel('$\mu_{s1}$', fontsize=axis_label_fontsize)
	plt.ylabel('$\mu_{s2}$', fontsize=axis_label_fontsize)
	plt.title('Rectified decision output\nvs. stimulus evidence', fontsize=title_fontsize)
	plot_fname = './s1_vs_s2_PENE_y_pred_rect.png'
	plt.savefig(plot_fname, bbox_inches='tight', dpi=300)
	plt.close()
	# Plot evidence for s1/s2 vs. BE regression model
	ax = plt.subplot(111)
	plt.imshow(all_BE_pred_mn, origin='lower')
	cbar = plt.colorbar()
	cbar.ax.tick_params(labelsize=cbar_tick_fontsize)
	cbar.set_label('Confidence', fontsize=cbar_label_fontsize)
	plt.xticks(signal_vals_plot_ind, signal_vals_plot.astype(np.str), fontsize=axis_tick_fontsize)
	plt.yticks(signal_vals_plot_ind, signal_vals_plot.astype(np.str), fontsize=axis_tick_fontsize)
	plt.xlabel('$\mu_{s1}$', fontsize=axis_label_fontsize)
	plt.ylabel('$\mu_{s2}$', fontsize=axis_label_fontsize)
	plt.title('BE regression model\nvs. stimulus evidence', fontsize=title_fontsize)
	plot_fname = './s1_vs_s2_PENE_BE_pred.png'
	plt.savefig(plot_fname, bbox_inches='tight', dpi=300)
	plt.close()
	# Plot evidence for s1/s2 vs. RCE regression model
	ax = plt.subplot(111)
	plt.imshow(all_RCE_pred_mn, origin='lower', vmin=conf_min, vmax=conf_max)
	cbar = plt.colorbar(ticks=conf_ticks)
	cbar.ax.set_yticklabels(np.array(conf_ticks).astype(np.str))
	cbar.ax.tick_params(labelsize=cbar_tick_fontsize)
	cbar.set_label('Confidence', fontsize=cbar_label_fontsize)
	plt.xticks(signal_vals_plot_ind, signal_vals_plot.astype(np.str), fontsize=axis_tick_fontsize)
	plt.yticks(signal_vals_plot_ind, signal_vals_plot.astype(np.str), fontsize=axis_tick_fontsize)
	plt.xlabel('$\mu_{s1}$', fontsize=axis_label_fontsize)
	plt.ylabel('$\mu_{s2}$', fontsize=axis_label_fontsize)
	plt.title('RCE regression model\nvs. stimulus evidence', fontsize=title_fontsize)
	plot_fname = './s1_vs_s2_PENE_RCE_pred.png'
	plt.savefig(plot_fname, bbox_inches='tight', dpi=300)
	plt.close()
	# Plot evidence for s1/s2 vs. multiple regression model
	ax = plt.subplot(111)
	plt.imshow(all_multi_pred_mn, origin='lower', vmin=conf_min, vmax=conf_max)
	cbar = plt.colorbar(ticks=conf_ticks)
	cbar.ax.set_yticklabels(np.array(conf_ticks).astype(np.str))
	cbar.ax.tick_params(labelsize=cbar_tick_fontsize)
	cbar.set_label('Confidence', fontsize=cbar_label_fontsize)
	plt.xticks(signal_vals_plot_ind, signal_vals_plot.astype(np.str), fontsize=axis_tick_fontsize)
	plt.yticks(signal_vals_plot_ind, signal_vals_plot.astype(np.str), fontsize=axis_tick_fontsize)
	plt.xlabel('$\mu_{s1}$', fontsize=axis_label_fontsize)
	plt.ylabel('$\mu_{s2}$', fontsize=axis_label_fontsize)
	plt.title('RCE+BE regression model\nvs. stimulus evidence', fontsize=title_fontsize)
	plot_fname = './s1_vs_s2_PENE_multi_pred.png'
	plt.savefig(plot_fname, bbox_inches='tight', dpi=300)
	plt.close()
	# Plot evidence for s1/s2 vs. PC 1
	ax = plt.subplot(111)
	plt.imshow(all_runs_pc1_mn, origin='lower')
	cbar = plt.colorbar()
	cbar.ax.tick_params(labelsize=cbar_tick_fontsize)
	cbar.set_label('Principal component 1', fontsize=cbar_label_fontsize)
	plt.xticks(signal_vals_plot_ind, signal_vals_plot.astype(np.str), fontsize=axis_tick_fontsize)
	plt.yticks(signal_vals_plot_ind, signal_vals_plot.astype(np.str), fontsize=axis_tick_fontsize)
	plt.xlabel('$\mu_{s1}$', fontsize=axis_label_fontsize)
	plt.ylabel('$\mu_{s2}$', fontsize=axis_label_fontsize)
	plt.title('Principal component 1\nvs. stimulus evidence', fontsize=title_fontsize)
	plot_fname = './s1_vs_s2_PENE_pc1.png'
	plt.savefig(plot_fname, bbox_inches='tight', dpi=300)
	plt.close()
	# Plot evidence for s1/s2 vs. PC 2
	ax = plt.subplot(111)
	plt.imshow(all_runs_pc2_mn, origin='lower')
	cbar = plt.colorbar()
	cbar.ax.tick_params(labelsize=cbar_tick_fontsize)
	cbar.set_label('Principal component 2', fontsize=cbar_label_fontsize)
	plt.xticks(signal_vals_plot_ind, signal_vals_plot.astype(np.str), fontsize=axis_tick_fontsize)
	plt.yticks(signal_vals_plot_ind, signal_vals_plot.astype(np.str), fontsize=axis_tick_fontsize)
	plt.xlabel('$\mu_{s1}$', fontsize=axis_label_fontsize)
	plt.ylabel('$\mu_{s2}$', fontsize=axis_label_fontsize)
	plt.title('Principal component 2\nvs. stimulus evidence', fontsize=title_fontsize)
	plot_fname = './s1_vs_s2_PENE_pc2.png'
	plt.savefig(plot_fname, bbox_inches='tight', dpi=300)
	plt.close()
	# Plot evidence for s1/s2 vs. |PC 1|
	ax = plt.subplot(111)
	plt.imshow(all_runs_abs_pc1_mn, origin='lower', vmin=2)
	cbar = plt.colorbar()
	cbar.ax.tick_params(labelsize=cbar_tick_fontsize)
	cbar.set_label('|Principal component 1|', fontsize=cbar_label_fontsize)
	plt.xticks(signal_vals_plot_ind, signal_vals_plot.astype(np.str), fontsize=axis_tick_fontsize)
	plt.yticks(signal_vals_plot_ind, signal_vals_plot.astype(np.str), fontsize=axis_tick_fontsize)
	plt.xlabel('$\mu_{s1}$', fontsize=axis_label_fontsize)
	plt.ylabel('$\mu_{s2}$', fontsize=axis_label_fontsize)
	plt.title('|Principal component 1|\nvs. stimulus evidence', fontsize=title_fontsize)
	plot_fname = './s1_vs_s2_PENE_abs_pc1.png'
	plt.savefig(plot_fname, bbox_inches='tight', dpi=300)
	plt.close()
	# Statistical comparison of regression models
	fid = open('./model_comparison.txt', 'w')
	fid.write('BE r^2 = ' + str(all_BE_r2_mn) + ' +- ' + str(all_BE_r2_se) + '\n')
	fid.write('RCE r^2 = ' + str(all_RCE_r2_mn) + ' +- ' + str(all_RCE_r2_se)  + '\n')
	fid.write('BE+RCE r^2 = ' + str(all_multi_r2_mn) + ' +- ' + str(all_multi_r2_se)  + '\n')
	fid.write(' \n')
	# t-tests
	fid.write('pairwise r^2 comparisons (t-tests):\n')
	t, p = ttest_rel(all_RCE_r2, all_BE_r2)
	fid.write('RCE vs. BE: t = ' + str(t) + ', p = ' + str(p) + '\n')
	t, p = ttest_rel(all_multi_r2, all_BE_r2)
	fid.write('BE+RCE vs. BE: t = ' + str(t) + ', p = ' + str(p) + '\n')
	t, p = ttest_rel(all_multi_r2, all_RCE_r2)
	fid.write('BE+RCE vs. RCE: t = ' + str(t) + ', p = ' + str(p) + '\n')
	fid.write(' \n')
	# Multiple regression model
	fid.write('multiple regression model:\n')
	fid.write('BE beta = ' + str(all_multi_beta_mn[0]) + '\n')
	fid.write('RCE beta = ' + str(all_multi_beta_mn[1]) + '\n')
	t, p = ttest_rel(all_multi_beta[:,1], all_multi_beta[:,0])
	fid.write('RCE beta vs. BE beta: t = ' + str(t) + ', p = ' + str(p))
	fid.close()

def main():

	# Settings
	parser = argparse.ArgumentParser()
	parser.add_argument('--N_runs', type=int, default=100)
	args = parser.parse_args()

	# Plot
	plot(args)

if __name__ == '__main__':
	main()