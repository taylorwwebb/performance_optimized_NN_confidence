import numpy as np
import matplotlib.pyplot as plt
from scipy.stats import ttest_rel, gaussian_kde, sem
import argparse

def plot(args):
	all_acc = []
	all_conf = []
	all_abs_pc1 = []
	all_pc2 = []
	all_y_pred_rect = []
	all_pc1_mn_diff = []
	all_pc1_var = []
	for r in range(args.N_runs):
		# Run directory
		run_dir = './run' + str(r+1) + '/'
		# Load PE results
		results = np.load(run_dir + 'PE_test_results.npz')
		# Rectify decision output at 0.5
		y_pred_rect = (np.abs(results['all_y_pred'] - 0.5) + 0.5).mean(2)
		# Collect results
		all_acc.append(results['all_acc'] / 100.0)
		all_conf.append(results['all_conf'] / 100.0)
		all_abs_pc1.append(results['all_abs_pc1'])
		all_pc2.append(results['all_pc2'])
		all_y_pred_rect.append(y_pred_rect)
		all_pc1_mn_diff.append(results['all_pc1_mn_diff'])
		all_pc1_var.append(results['all_pc1_var'])
	# Concatenate results from all networks
	all_acc = np.array(all_acc)
	all_conf = np.array(all_conf)
	all_abs_pc1 = np.array(all_abs_pc1)
	all_pc2 = np.array(all_pc2)
	all_y_pred_rect = np.array(all_y_pred_rect)
	all_pc1_mn_diff = np.array(all_pc1_mn_diff)
	all_pc1_var = np.array(all_pc1_var)
	# Summary stats
	acc_mn = all_acc.mean(0)
	acc_se = sem(all_acc,0)
	conf_mn = all_conf.mean(0)
	conf_se = sem(all_conf,0)
	abs_pc1_mn = all_abs_pc1.mean(0)
	abs_pc1_se = sem(all_abs_pc1,0)
	pc2_mn = all_pc2.mean(0)
	pc2_se = sem(all_pc2,0)
	y_pred_rect_mn = all_y_pred_rect.mean(0)
	y_pred_rect_se = sem(all_y_pred_rect,0)
	pc1_mn_diff_mn = all_pc1_mn_diff.mean(0)
	pc1_mn_diff_se = sem(all_pc1_mn_diff,0)
	pc1_var_mn = all_pc1_var.mean(0)
	pc1_var_se = sem(all_pc1_var,0)
	# Identify low and high PE conditions
	targ_acc = 0.75
	low_PE_ind = np.abs(acc_mn - targ_acc).argmin(1)[0]
	high_PE_ind = np.abs(acc_mn - targ_acc).argmin(1)[1]
	# Open file for recording stats
	fid = open('./PE_PCA_stats.txt', 'w')

	# Accuracy and confidence
	# Plot
	ax1 = plt.subplot(111)
	ax1.bar([0,1],[acc_mn[0,low_PE_ind],acc_mn[1,high_PE_ind]],yerr=[acc_se[0,low_PE_ind],acc_se[1,high_PE_ind]],width=args.bar_width,color='gray')
	ax1.set_ylabel('P(Correct)', fontsize=args.axis_fontsize)
	plt.ylim([0.65,0.85])
	plt.xticks([0,1,2.5,3.5],['Low','High','Low','High'], fontsize=args.ticks_fontsize)
	ax1.set_xlabel('Positive evidence',fontsize=args.axis_fontsize)
	plt.yticks([0.65,0.7,0.75,0.8,0.85],['0.65','0.7','0.75','0.8','0.85'], fontsize=args.ticks_fontsize)
	ax1.spines['top'].set_visible(False)
	ax2 = ax1.twinx()
	ax2.bar([2.5,3.5],[conf_mn[0,low_PE_ind],conf_mn[1,high_PE_ind]],yerr=[conf_se[0,low_PE_ind],conf_se[1,high_PE_ind]],width=args.bar_width,color='black')
	ax2.set_ylabel('Confidence', fontsize=args.axis_fontsize)
	plt.ylim([0.65,0.85])
	plt.yticks([0.65,0.7,0.75,0.8,0.85],['0.65','0.7','0.75','0.8','0.85'], fontsize=args.ticks_fontsize)
	ax2.spines['top'].set_visible(False)
	# Stats
	acc_t, acc_p = ttest_rel(all_acc[:,1,high_PE_ind],all_acc[:,0,low_PE_ind])
	fid.write('Accuracy t-test: t = ' + str(acc_t) + ', p = ' + str(acc_p) + '\n')
	conf_t, conf_p = ttest_rel(all_conf[:,1,high_PE_ind],all_conf[:,0,low_PE_ind])
	fid.write('Confidence t-test: t = ' + str(conf_t) + ', p = ' + str(conf_p) + '\n')
	fid.write(' \n')
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
	max_y_val = np.max([acc_mn[0,low_PE_ind] + acc_se[0,low_PE_ind], acc_mn[1,high_PE_ind] + acc_se[1,high_PE_ind]])
	y_start = max_y_val + 0.01
	y_end = max_y_val + 0.015
	ax1.plot([0,0,1,1],[y_start,y_end,y_end,y_start],color='black')
	ax1.text(0.5,y_end+0.005,acc_p_symb,fontsize=args.significance_fontsize,horizontalalignment='center')
	max_y_val = np.max([conf_mn[0,low_PE_ind] + conf_se[0,low_PE_ind], conf_mn[1,high_PE_ind] + conf_se[1,high_PE_ind]])
	y_start = max_y_val + 0.01
	y_end = max_y_val + 0.015
	ax2.plot([2.5,2.5,3.5,3.5],[y_start,y_end,y_end,y_start],color='black')
	ax2.text(3,y_end+0.005,conf_p_symb,fontsize=args.significance_fontsize,horizontalalignment='center')
	# Save plot
	plot_fname = './PE_bias_acc_conf.png'
	plt.savefig(plot_fname, bbox_inches='tight', dpi=300)
	plt.close()

	# |Principal component 1|
	ax1 = plt.subplot(111)
	ax1.bar([0,1],[abs_pc1_mn[0,low_PE_ind],abs_pc1_mn[1,high_PE_ind]],yerr=[abs_pc1_se[0,low_PE_ind],abs_pc1_se[1,high_PE_ind]],width=args.bar_width,color='black')
	ax1.set_ylabel('|Principal component 1|', fontsize=args.axis_fontsize)
	plt.ylim([1,3.5])
	plt.yticks([1,1.5,2,2.5,3,3.5],['1','1.5','2','2.5','3','3.5'], fontsize=args.ticks_fontsize)
	plt.xticks([0,1],['Low','High'], fontsize=args.ticks_fontsize)
	ax1.set_xlabel('Positive evidence',fontsize=args.axis_fontsize)
	ax1.spines['top'].set_visible(False)
	ax1.spines['right'].set_visible(False)
	ax1.set_aspect(aspect=1.5)
	# Stats
	abs_pc1_t, abs_pc1_p = ttest_rel(all_abs_pc1[:,1,high_PE_ind],all_abs_pc1[:,0,low_PE_ind])
	fid.write('|PC 1| t-test: t = ' + str(abs_pc1_t) + ', p = ' + str(abs_pc1_p) + '\n')
	fid.write(' \n')
	if abs_pc1_p >= 0.05: abs_pc1_p_symb = 'ns'
	if abs_pc1_p < 0.05: abs_pc1_p_symb = '*'
	if abs_pc1_p < 0.01: abs_pc1_p_symb = '**'
	if abs_pc1_p < 0.001: abs_pc1_p_symb = '***'
	if abs_pc1_p < 0.0001: abs_pc1_p_symb = '****'
	max_y_val = np.max([abs_pc1_mn[0,low_PE_ind] + abs_pc1_se[0,low_PE_ind], abs_pc1_mn[1,high_PE_ind] + abs_pc1_se[1,high_PE_ind]])
	y_start = max_y_val + 0.1
	y_end = max_y_val + 0.15
	ax1.plot([0,0,1,1],[y_start,y_end,y_end,y_start],color='black')
	ax1.text(0.5,y_end+0.05,abs_pc1_p_symb,fontsize=args.significance_fontsize,horizontalalignment='center')
	# Save plot
	plot_fname = './PE_bias_abs_pc1.png'
	plt.savefig(plot_fname, bbox_inches='tight', dpi=300)
	plt.close()

	# Principal component 2
	ax1 = plt.subplot(111)
	ax1.bar([0,1],[pc2_mn[0,low_PE_ind],pc2_mn[1,high_PE_ind]],yerr=[pc2_se[0,low_PE_ind],pc2_se[1,high_PE_ind]],width=args.bar_width,color='black')
	ax1.set_ylabel('Principal component 2', fontsize=args.axis_fontsize)
	plt.ylim([-4.5,-1])
	plt.yticks([-4,-3,-2,-1],['-4','-3','-2','-1'], fontsize=args.ticks_fontsize)
	plt.xticks([0,1],['Low','High'], fontsize=args.ticks_fontsize)
	ax1.set_xlabel('Positive evidence',fontsize=args.axis_fontsize)
	ax1.spines['top'].set_visible(False)
	ax1.spines['right'].set_visible(False)
	ax1.set_aspect(aspect=1.05)
	# Stats
	pc2_t, pc2_p = ttest_rel(all_pc2[:,1,high_PE_ind],all_pc2[:,0,low_PE_ind])
	fid.write('PC 2 t-test: t = ' + str(pc2_t) + ', p = ' + str(pc2_p) + '\n')
	fid.write(' \n')
	if pc2_p >= 0.05: pc2_p_symb = 'ns'
	if pc2_p < 0.05: pc2_p_symb = '*'
	if pc2_p < 0.01: pc2_p_symb = '**'
	if pc2_p < 0.001: pc2_p_symb = '***'
	if pc2_p < 0.0001: pc2_p_symb = '****'
	min_y_val = np.min([pc2_mn[0,low_PE_ind] - pc2_se[0,low_PE_ind], pc2_mn[1,high_PE_ind] - pc2_se[1,high_PE_ind]])
	y_start = min_y_val - 0.1
	y_end = min_y_val - 0.15
	ax1.plot([0,0,1,1],[y_start,y_end,y_end,y_start],color='black')
	ax1.text(0.5,y_end-0.4,pc2_p_symb,fontsize=args.significance_fontsize,horizontalalignment='center')
	# Save plot
	plot_fname = './PE_bias_pc2.png'
	plt.savefig(plot_fname, bbox_inches='tight', dpi=300)
	plt.close()

	# Rectified decision output
	ax1 = plt.subplot(111)
	ax1.bar([0,1],[y_pred_rect_mn[0,low_PE_ind],y_pred_rect_mn[1,high_PE_ind]],yerr=[y_pred_rect_se[0,low_PE_ind],y_pred_rect_se[1,high_PE_ind]],width=args.bar_width,color='black')
	ax1.set_ylabel('Rectified decision output', fontsize=args.axis_fontsize)
	plt.ylim([0.65,0.85])
	plt.yticks([0.65,0.7,0.75,0.8,0.85],['0.65','0.7','0.75','0.8','0.85'], fontsize=args.ticks_fontsize)
	plt.xticks([0,1],['Low','High'], fontsize=args.ticks_fontsize)
	ax1.set_xlabel('Positive evidence',fontsize=args.axis_fontsize)
	ax1.spines['top'].set_visible(False)
	ax1.spines['right'].set_visible(False)
	ax1.set_aspect(aspect=19)
	# Stats
	y_pred_rect_t, y_pred_rect_p = ttest_rel(all_y_pred_rect[:,1,high_PE_ind],all_y_pred_rect[:,0,low_PE_ind])
	fid.write('Rectified decision output t-test: t = ' + str(y_pred_rect_t) + ', p = ' + str(y_pred_rect_p) + '\n')
	fid.write(' \n')
	if y_pred_rect_p >= 0.05: y_pred_rect_p_symb = 'ns'
	if y_pred_rect_p < 0.05: y_pred_rect_p_symb = '*'
	if y_pred_rect_p < 0.01: y_pred_rect_p_symb = '**'
	if y_pred_rect_p < 0.001: y_pred_rect_p_symb = '***'
	if y_pred_rect_p < 0.0001: y_pred_rect_p_symb = '****'
	max_y_val = np.max([y_pred_rect_mn[0,low_PE_ind] + y_pred_rect_se[0,low_PE_ind], y_pred_rect_mn[1,high_PE_ind] + y_pred_rect_se[1,high_PE_ind]])
	y_start = max_y_val + 0.01
	y_end = max_y_val + 0.015
	ax1.plot([0,0,1,1],[y_start,y_end,y_end,y_start],color='black')
	ax1.text(0.5,y_end+0.005,y_pred_rect_p_symb,fontsize=args.significance_fontsize,horizontalalignment='center')
	# Save plot
	plot_fname = './PE_bias_y_pred_rect.png'
	plt.savefig(plot_fname, bbox_inches='tight', dpi=300)
	plt.close()

	# Mean difference and variance along PC 1
	# Plot
	ax1 = plt.subplot(111)
	ax1.bar([0,1],[pc1_mn_diff_mn[0,low_PE_ind],pc1_mn_diff_mn[1,high_PE_ind]],yerr=[pc1_mn_diff_se[0,low_PE_ind],pc1_mn_diff_se[1,high_PE_ind]],width=args.bar_width,color='gray')
	ax1.set_ylabel('PC 1 mean difference', fontsize=args.axis_fontsize)
	plt.xticks([0,1,2.5,3.5],['Low','High','Low','High'], fontsize=args.ticks_fontsize)
	ax1.set_xlabel('Positive evidence',fontsize=args.axis_fontsize)
	plt.ylim([0,4.5])
	plt.yticks([0,1,2,3,4],['0','1','2','3','4'], fontsize=args.ticks_fontsize)
	ax1.spines['top'].set_visible(False)
	ax2 = ax1.twinx()
	ax2.bar([2.5,3.5],[pc1_var_mn[0,low_PE_ind],pc1_var_mn[1,high_PE_ind]],yerr=[pc1_var_se[0,low_PE_ind],pc1_var_se[1,high_PE_ind]],width=args.bar_width,color='black')
	ax2.set_ylabel('PC 1 variance', fontsize=args.axis_fontsize)
	plt.ylim([0,0.12])
	plt.yticks([0,0.04,0.08,0.12],['0','0.04','0.08','0.12'], fontsize=args.ticks_fontsize)
	ax2.spines['top'].set_visible(False)
	# Stats
	pc1_mn_diff_t, pc1_mn_diff_p = ttest_rel(all_pc1_mn_diff[:,1,high_PE_ind],all_pc1_mn_diff[:,0,low_PE_ind])
	fid.write('PC 1 mean difference t-test: t = ' + str(pc1_mn_diff_t) + ', p = ' + str(pc1_mn_diff_p) + '\n')
	pc1_var_t, pc1_var_p = ttest_rel(all_pc1_var[:,1,high_PE_ind],all_pc1_var[:,0,low_PE_ind])
	fid.write('PC 1 variance t-test: t = ' + str(pc1_var_t) + ', p = ' + str(pc1_var_p))
	fid.write(' \n')
	if pc1_var_p >= 0.05: pc1_var_p_symb = 'ns'
	if pc1_var_p < 0.05: pc1_var_p_symb = '*'
	if pc1_var_p < 0.01: pc1_var_p_symb = '**'
	if pc1_var_p < 0.001: pc1_var_p_symb = '***'
	if pc1_var_p < 0.0001: pc1_var_p_symb = '****'
	if pc1_mn_diff_p >= 0.05: pc1_mn_diff_p_symb = 'ns'
	if pc1_mn_diff_p < 0.05: pc1_mn_diff_p_symb = '*'
	if pc1_mn_diff_p < 0.01: pc1_mn_diff_p_symb = '**'
	if pc1_mn_diff_p < 0.001: pc1_mn_diff_p_symb = '***'
	if pc1_mn_diff_p < 0.0001: pc1_mn_diff_p_symb = '****'
	max_y_val = np.max([pc1_mn_diff_mn[0,low_PE_ind] + pc1_mn_diff_se[0,low_PE_ind], pc1_mn_diff_mn[1,high_PE_ind] + pc1_mn_diff_se[1,high_PE_ind]])
	y_start = max_y_val + 0.1
	y_end = max_y_val + 0.15
	ax1.plot([0,0,1,1],[y_start,y_end,y_end,y_start],color='black')
	ax1.text(0.5,y_end+0.05,pc1_mn_diff_p_symb,fontsize=args.significance_fontsize,horizontalalignment='center')
	max_y_val = np.max([pc1_var_mn[0,low_PE_ind] + pc1_var_se[0,low_PE_ind], pc1_var_mn[1,high_PE_ind] + pc1_var_se[1,high_PE_ind]])
	y_start = max_y_val + 0.003
	y_end = max_y_val + 0.0045
	ax2.plot([2.5,2.5,3.5,3.5],[y_start,y_end,y_end,y_start],color='black')
	ax2.text(3,y_end+0.0025,pc1_var_p_symb,fontsize=args.significance_fontsize,horizontalalignment='center')
	# Save plot
	plot_fname = './PE_bias_pc1_mn_diff_var.png'
	plt.savefig(plot_fname, bbox_inches='tight', dpi=300)
	plt.close()

	# Close stats file
	fid.close()

	# Loop back through runs and plot s1/s2 distributions along PC 1 for low vs. high PE
	axis_fontsize = 12
	ticks_fontsize = 12
	title_fontsize = 14
	legend_fontsize = 12
	for r in range(args.N_runs):
		# Run directory
		run_dir = './run' + str(r+1) + '/'
		# Load PE results
		results = np.load(run_dir + 'PE_test_results.npz')
		pc1 = results['all_pc1']
		y_targ = results['all_y_targ']
		# Low and high PE conditions
		low_PE_pc1 = pc1[0,low_PE_ind,:]
		high_PE_pc1 = pc1[1,high_PE_ind,:]
		low_PE_y_targ = y_targ[0,low_PE_ind,:]
		high_PE_y_targ = y_targ[1,high_PE_ind,:]
		# Plot 
		# Low PE
		low_PE_pc1_s1 = low_PE_pc1[low_PE_y_targ == 0]
		low_PE_pc1_s2 = low_PE_pc1[low_PE_y_targ == 1]
		x_range = np.linspace(-30,30,1000)
		low_PE_pc1_s1_kernel = gaussian_kde(low_PE_pc1_s1)
		low_PE_pc1_s1_dist = low_PE_pc1_s1_kernel(x_range)
		low_PE_pc1_s2_kernel = gaussian_kde(low_PE_pc1_s2)
		low_PE_pc1_s2_dist = low_PE_pc1_s2_kernel(x_range)
		ax = plt.subplot(211)
		plt.fill(np.append(np.insert(x_range, 0, x_range[0]), x_range[-1]), np.append(np.insert(low_PE_pc1_s1_dist, 0, 0), 0), color='turquoise', alpha=0.5)
		plt.fill(np.append(np.insert(x_range, 0, x_range[0]), x_range[-1]), np.append(np.insert(low_PE_pc1_s2_dist, 0, 0), 0), color='salmon', alpha=0.5)
		plt.xlabel('Principal component 1', fontsize=axis_fontsize)
		plt.xlim([-10,10])
		plt.xticks([-10,-5,0,5,10], ['-10','-5','0','5','10',], fontsize=ticks_fontsize)
		plt.ylabel('Density estimate', fontsize=axis_fontsize)
		plt.ylim([0,0.3])
		plt.yticks([0,0.1,0.2,0.3],['0','0.1','0.2','0.3'],fontsize=ticks_fontsize)
		ax.spines['right'].set_visible(False)
		ax.spines['top'].set_visible(False)
		ax.yaxis.set_ticks_position('left')
		ax.xaxis.set_ticks_position('bottom')
		plt.legend(['s1','s2'],frameon=False,fontsize=legend_fontsize)
		plt.title('Low positive evidence', fontsize=title_fontsize)
		# High PE
		high_PE_pc1_s1 = high_PE_pc1[high_PE_y_targ == 0]
		high_PE_pc1_s2 = high_PE_pc1[high_PE_y_targ == 1]
		x_range = np.linspace(-30,30,1000)
		high_PE_pc1_s1_kernel = gaussian_kde(high_PE_pc1_s1)
		high_PE_pc1_s1_dist = high_PE_pc1_s1_kernel(x_range)
		high_PE_pc1_s2_kernel = gaussian_kde(high_PE_pc1_s2)
		high_PE_pc1_s2_dist = high_PE_pc1_s2_kernel(x_range)
		ax = plt.subplot(212)
		plt.fill(np.append(np.insert(x_range, 0, x_range[0]), x_range[-1]), np.append(np.insert(high_PE_pc1_s1_dist, 0, 0), 0), color='turquoise', alpha=0.5)
		plt.fill(np.append(np.insert(x_range, 0, x_range[0]), x_range[-1]), np.append(np.insert(high_PE_pc1_s2_dist, 0, 0), 0), color='salmon', alpha=0.5)
		plt.xlabel('Principal component 1', fontsize=axis_fontsize)
		plt.xlim([-10,10])
		plt.xticks([-10,-5,0,5,10], ['-10','-5','0','5','10',], fontsize=ticks_fontsize)
		plt.ylabel('Density estimate', fontsize=axis_fontsize)
		plt.ylim([0,0.3])
		plt.yticks([0,0.1,0.2,0.3],['0','0.1','0.2','0.3'],fontsize=ticks_fontsize)
		ax.spines['right'].set_visible(False)
		ax.spines['top'].set_visible(False)
		ax.yaxis.set_ticks_position('left')
		ax.xaxis.set_ticks_position('bottom')
		plt.legend(['s1','s2'],frameon=False,fontsize=legend_fontsize)
		plt.title('High positive evidence', fontsize=title_fontsize)
		plt.tight_layout()
		fig_name = run_dir + './PE_pc1_vs_s1s2_dist.png'
		plt.savefig(fig_name, bbox_inches='tight', dpi=300)
		plt.close()

def main():

	# Settings
	parser = argparse.ArgumentParser()
	parser.add_argument('--N_runs', type=int, default=100)
	parser.add_argument('--bar_width', type=float, default=0.8)
	parser.add_argument('--axis_fontsize', type=int, default=22)
	parser.add_argument('--ticks_fontsize', type=int, default=20)
	parser.add_argument('--significance_fontsize', type=int, default=20)
	parser.add_argument('--title_fontsize', type=int, default=30)
	args = parser.parse_args()

	# Plot
	plot(args)

if __name__ == '__main__':
	main()