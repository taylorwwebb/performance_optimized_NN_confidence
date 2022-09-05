import numpy as np
import matplotlib.pyplot as plt
from scipy.stats import ttest_rel, gaussian_kde, sem
from sklearn.linear_model import LogisticRegression, LinearRegression
from scipy.special import logit, expit
import argparse
from joblib import load

def plot(args):
	all_var_explained = []
	all_acc = []
	all_pc1_y_targ_acc = []
	all_pc2_correct_incorrect_acc = []
	all_pc1_y_pred_r2 = []
	all_pc2_conf_r2 = []
	all_abs_pc1_pc2_r2 = []
	for r in range(args.N_runs):
		# Run directory
		run_dir = './run' + str(r+1) + '/'
		# Load PCA model
		pca = load(run_dir + 'pca.joblib')
		all_var_explained.append(pca.explained_variance_ratio_)
		# Load PCA results
		results = np.load(run_dir + 'PCA_results.npz')
		pc1 = results['z_top2'][:,0]
		pc2 = results['z_top2'][:,1]
		y_pred = results['y_pred']
		y_targ = results['y_targ']
		conf = results['conf']
		# Model accuracy
		correct_preds = (y_pred.round()==y_targ).astype(np.float)
		acc = correct_preds.mean()
		all_acc.append(acc)
		# Plot PC 1 vs. s1/s2
		pc1_s1 = pc1[y_targ==0]
		pc1_s2 = pc1[y_targ==1]
		x_range = np.linspace(-30,30,1000)
		pc1_s1_kernel = gaussian_kde(pc1_s1)
		pc1_s1_dist = pc1_s1_kernel(x_range)
		pc1_s2_kernel = gaussian_kde(pc1_s2)
		pc1_s2_dist = pc1_s2_kernel(x_range)
		ax = plt.subplot(111)
		plt.fill(np.append(np.insert(x_range, 0, x_range[0]), x_range[-1]), np.append(np.insert(pc1_s1_dist, 0, 0), 0), color='turquoise', alpha=0.5)
		plt.fill(np.append(np.insert(x_range, 0, x_range[0]), x_range[-1]), np.append(np.insert(pc1_s2_dist, 0, 0), 0), color='salmon', alpha=0.5)
		plt.xlabel('Principal component 1', fontsize=args.axis_fontsize)
		plt.xlim([-25,25])
		plt.xticks([-20,-10,0,10,20], ['-20','-10','0','10','20'], fontsize=args.ticks_fontsize)
		plt.ylabel('Density estimate', fontsize=args.axis_fontsize)
		plt.ylim([0,0.1])
		plt.yticks([0,0.02,0.04,0.06,0.08,0.1],['0','0.02','0.04','0.06','0.08','0.1'],fontsize=args.ticks_fontsize)
		ax.spines['right'].set_visible(False)
		ax.spines['top'].set_visible(False)
		ax.yaxis.set_ticks_position('left')
		ax.xaxis.set_ticks_position('bottom')
		plt.legend(['s1','s2'],frameon=False,fontsize=args.legend_fontsize)
		fig_name = run_dir + './pc1_vs_s1s2_dist.png'
		plt.savefig(fig_name, bbox_inches='tight', dpi=300)
		plt.close()
		# Regression: PC 1 -> stimulus class
		clf = LogisticRegression().fit(np.expand_dims(pc1,1), y_targ)
		pc1_y_targ_acc = clf.score(np.expand_dims(pc1,1), y_targ)
		all_pc1_y_targ_acc.append(pc1_y_targ_acc)
		# Plot PC 2 vs. correct/incorrect
		pc2_correct = pc2[y_pred.round()==y_targ]
		pc2_incorrect = pc2[y_pred.round()!=y_targ]
		x_range = np.linspace(-10,17,1000)
		pc2_correct_kernel = gaussian_kde(pc2_correct)
		pc2_correct_dist = pc2_correct_kernel(x_range)
		pc2_incorrect_kernel = gaussian_kde(pc2_incorrect)
		pc2_incorrect_dist = pc2_incorrect_kernel(x_range)
		ax = plt.subplot(111)
		plt.fill(np.append(np.insert(x_range, 0, x_range[0]), x_range[-1]), np.append(np.insert(pc2_correct_dist, 0, 0), 0), color='lightgreen', alpha=0.5)
		plt.fill(np.append(np.insert(x_range, 0, x_range[0]), x_range[-1]), np.append(np.insert(pc2_incorrect_dist, 0, 0), 0), color='slateblue', alpha=0.5)
		plt.xlabel('Principal component 2', fontsize=args.axis_fontsize)
		plt.xlim([-8,17])
		plt.xticks([-5,0,5,10,15], ['-5','0','5','10','15'], fontsize=args.ticks_fontsize)
		plt.ylabel('Density estimate', fontsize=args.axis_fontsize)
		plt.ylim([0,0.4])
		plt.yticks([0,0.1,0.2,0.3,0.4,0.5],['0','0.1','0.2','0.3','0.4','0.5'],fontsize=args.ticks_fontsize)
		ax.spines['right'].set_visible(False)
		ax.spines['top'].set_visible(False)
		ax.yaxis.set_ticks_position('left')
		ax.xaxis.set_ticks_position('bottom')
		plt.legend(['Correct','Incorrect'],frameon=False,fontsize=args.legend_fontsize)
		fig_name = run_dir + 'pc2_vs_correct_incorrect_dist.png'
		plt.savefig(fig_name, bbox_inches='tight', dpi=300)
		plt.close()
		# Regression: PC 2 -> correct/incorrect
		clf = LogisticRegression().fit(np.expand_dims(pc2,1), correct_preds)
		pc2_correct_incorrect_acc = clf.score(np.expand_dims(pc2,1), correct_preds)
		all_pc2_correct_incorrect_acc.append(pc2_correct_incorrect_acc)
		# Plot PC 1 vs. decision output
		ax = plt.subplot(111)
		plt.scatter(pc1, y_pred, color='gray')
		plt.xlabel('Principal component 1', fontsize=args.axis_fontsize)
		plt.xlim([-25,25])
		plt.xticks([-20,-10,0,10,20], ['-20','-10','0','10','20'], fontsize=args.ticks_fontsize)
		plt.xticks(fontsize=args.ticks_fontsize)
		plt.ylabel('Decision output', fontsize=args.axis_fontsize)
		plt.ylim([0,1])
		plt.yticks([0,0.2,0.4,0.6,0.8,1],['0','0.2','0.4','0.6','0.8','1'],fontsize=args.ticks_fontsize)
		ax.spines['right'].set_visible(False)
		ax.spines['top'].set_visible(False)
		ax.yaxis.set_ticks_position('left')
		ax.xaxis.set_ticks_position('bottom')
		fig_name = run_dir + './pc1_vs_y_pred.png'
		plt.savefig(fig_name, bbox_inches='tight', dpi=300)
		plt.close()
		# Regression: PC 1 -> logit(decision output)
		eps = 1e-8
		y_pred = np.stack([y_pred, np.ones(y_pred.shape) * eps]).max(0)
		y_pred = np.stack([y_pred, np.ones(y_pred.shape) * (1-eps)]).min(0)
		logit_y_pred = logit(y_pred)
		clf = LinearRegression().fit(np.expand_dims(pc1,1), logit_y_pred)
		pc1_y_pred_r2 = clf.score(np.expand_dims(pc1,1), logit_y_pred)
		all_pc1_y_pred_r2.append(pc1_y_pred_r2)
		# Plot PC 2 vs. confidence
		ax = plt.subplot(111)
		plt.scatter(pc2, conf, color='gray')
		plt.xlabel('Principal component 2', fontsize=args.axis_fontsize)
		plt.xlim([-8,17])
		plt.xticks([-5,0,5,10,15], ['-5','0','5','10','15'], fontsize=args.ticks_fontsize)
		plt.xticks(fontsize=args.ticks_fontsize)
		plt.ylabel('Confidence', fontsize=args.axis_fontsize)
		plt.ylim([0.5,1])
		plt.yticks([0.5,0.6,0.7,0.8,0.9,1],['0.5','0.6','0.7','0.8','0.9','1'],fontsize=args.ticks_fontsize)
		ax.spines['right'].set_visible(False)
		ax.spines['top'].set_visible(False)
		ax.yaxis.set_ticks_position('left')
		ax.xaxis.set_ticks_position('bottom')
		fig_name = run_dir + './pc2_vs_conf.png'
		plt.savefig(fig_name, bbox_inches='tight', dpi=300)
		plt.close()
		# Regression: PC 2 -> logit(confidence)
		conf = np.stack([conf, np.ones(conf.shape) * eps]).max(0)
		conf = np.stack([conf, np.ones(conf.shape) * (1-eps)]).min(0)
		logit_conf = logit(conf)
		clf = LinearRegression().fit(np.expand_dims(pc2,1), logit_conf)
		pc2_conf_r2 = clf.score(np.expand_dims(pc2,1), logit_conf)
		all_pc2_conf_r2.append(pc2_conf_r2)
		# Plot PC 1 vs. PC 2
		ax = plt.subplot(111)
		plt.scatter(pc1, pc2, color='gray')
		plt.xlabel('Principal component 1', fontsize=args.axis_fontsize)
		plt.ylabel('Principal component 2', fontsize=args.axis_fontsize)
		plt.xlim([-25,25])
		plt.xticks([-20,-10,0,10,20], ['-20','-10','0','10','20'], fontsize=args.ticks_fontsize)
		plt.ylim([-7,18])
		plt.yticks([-5,0,5,10,15], ['-5','0','5','10','15'], fontsize=args.ticks_fontsize)
		ax.spines['right'].set_visible(False)
		ax.spines['top'].set_visible(False)
		ax.yaxis.set_ticks_position('left')
		ax.xaxis.set_ticks_position('bottom')
		fig_name = run_dir + 'pc1_vs_pc2.png'
		plt.savefig(fig_name, bbox_inches='tight', dpi=300)
		plt.close()
		# Regression: |PC 1| -> PC 2
		abs_pc1 = np.abs(pc1)
		clf = LinearRegression().fit(np.expand_dims(abs_pc1,1), pc2)
		abs_pc1_pc2_r2 = clf.score(np.expand_dims(abs_pc1,1), pc2)
		all_abs_pc1_pc2_r2.append(abs_pc1_pc2_r2)
	# Concatenate results from all networks
	all_var_explained = np.array(all_var_explained)
	all_acc = np.array(all_acc)
	all_pc1_y_targ_acc = np.array(all_pc1_y_targ_acc)
	all_pc2_correct_incorrect_acc = np.array(all_pc2_correct_incorrect_acc)
	all_pc1_y_pred_r2 = np.array(all_pc1_y_pred_r2)
	all_pc2_conf_r2 = np.array(all_pc2_conf_r2)
	all_abs_pc1_pc2_r2 = np.array(all_abs_pc1_pc2_r2)
	# Summary stats
	var_explained_mn = all_var_explained.mean(0)
	var_explained_se = sem(all_var_explained,0)
	acc_mn = all_acc.mean(0)
	acc_se = sem(all_acc,0)
	pc1_y_targ_acc_mn = all_pc1_y_targ_acc.mean(0)
	pc1_y_targ_acc_se = sem(all_pc1_y_targ_acc,0)
	pc2_correct_incorrect_acc_mn = all_pc2_correct_incorrect_acc.mean(0)
	pc2_correct_incorrect_acc_se = sem(all_pc2_correct_incorrect_acc,0)
	pc1_y_pred_r2_mn = all_pc1_y_pred_r2.mean(0)
	pc1_y_pred_r2_se = sem(all_pc1_y_pred_r2,0)
	pc2_conf_r2_mn = all_pc2_conf_r2.mean(0)
	pc2_conf_r2_se = sem(all_pc2_conf_r2,0)
	abs_pc1_pc2_r2_mn = all_abs_pc1_pc2_r2.mean(0)
	abs_pc1_pc2_r2_se = sem(all_abs_pc1_pc2_r2,0)
	# Plot variance explained by top N principal components
	ax = plt.subplot(111)
	plt.bar(np.arange(args.N_PCs_plot)+1, var_explained_mn[:args.N_PCs_plot], yerr=var_explained_se[:args.N_PCs_plot], color='black', width=args.bar_width)
	plt.xlabel('Principal component', fontsize=args.axis_fontsize)
	plt.xticks(np.arange(args.N_PCs_plot)+1, (np.arange(args.N_PCs_plot)+1).astype(np.str), fontsize=args.ticks_fontsize)
	plt.ylabel('Variance explained', fontsize=args.axis_fontsize)
	plt.ylim([0,1])
	plt.yticks([0,0.2,0.4,0.6,0.8,1], ['0','0.2','0.4','0.6','0.8','1'], fontsize=args.ticks_fontsize)
	ax.spines['right'].set_visible(False)
	ax.spines['top'].set_visible(False)
	ax.yaxis.set_ticks_position('left')
	ax.xaxis.set_ticks_position('bottom')
	fig_name = './PCA_var_explained.png'
	plt.savefig(fig_name, bbox_inches='tight', dpi=300)
	plt.close()
	# Save regression statistics
	fid = open('./pca_regression_stats.txt', 'w')
	fid.write('Average decision accuracy (neural network) = ' + str(acc_mn) + ' +/- ' + str(acc_se) + '\n')
	fid.write(' \n')
	fid.write('PC 1 -> stimulus class: accuracy = ' + str(pc1_y_targ_acc_mn) + ' +/- ' + str(pc1_y_targ_acc_se) + '\n')
	fid.write(' \n')
	fid.write('PC 2 -> correct/incorrect: accuracy = ' + str(pc2_correct_incorrect_acc_mn) + ' +/- ' + str(pc2_correct_incorrect_acc_se) + '\n')
	fid.write(' \n')
	fid.write('PC 1 -> decision output: R^2 = ' + str(pc1_y_pred_r2_mn) + ' +/- ' + str(pc1_y_pred_r2_se) + '\n')
	fid.write(' \n')
	fid.write('PC 2 -> confidence: R^2 = ' + str(pc2_conf_r2_mn) + ' +/- ' + str(pc2_conf_r2_se) + '\n')
	fid.write(' \n')
	fid.write('|PC 1| -> PC 2: R^2 = ' + str(abs_pc1_pc2_r2_mn) + ' +/- ' + str(abs_pc1_pc2_r2_se) + '\n')
	fid.write(' \n')
	fid.close()

def main():

	# Settings
	parser = argparse.ArgumentParser()
	parser.add_argument('--N_runs', type=int, default=100)
	parser.add_argument('--N_PCs_plot', type=int, default=5)
	parser.add_argument('--bar_width', type=float, default=0.8)
	parser.add_argument('--axis_fontsize', type=int, default=18)
	parser.add_argument('--ticks_fontsize', type=int, default=16)
	parser.add_argument('--legend_fontsize', type=int, default=16)
	args = parser.parse_args()

	# Plot
	plot(args)

if __name__ == '__main__':
	main()