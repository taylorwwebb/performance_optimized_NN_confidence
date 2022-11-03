import numpy as np
import matplotlib.pyplot as plt
from scipy.stats import ttest_rel, ttest_ind, gaussian_kde, sem
from sklearn.linear_model import LinearRegression
from scipy.special import logit
import argparse
from joblib import load

def plot(args):
	for r in range(args.N_runs):
		# Run directory
		run_dir = './run' + str(r+1) + '/'
		# Load PCA model
		pca = load(run_dir + 'pca.joblib')
		# Load results
		results = np.load(run_dir + 'PCA_results.npz')
		all_z = results['all_z']
		y_pred = results['all_p_a'][:,1]
		opt_out = results['all_p_a'][:,-1]
		s1_trials = results['all_p_a'][:,:2].argmax(1) == 0
		s2_trials = results['all_p_a'][:,:2].argmax(1) == 1
		opt_out_trials = results['all_p_a'].argmax(1) == 2
		# Visualize each neuron
		all_T_in_act_mn = []
		all_T_in_act_se = []
		all_T_opp_act_mn = []
		all_T_opp_act_se = []
		all_T_sure_act_mn = []
		all_T_sure_act_se = []
		all_T_in_vs_T_sure_significant = []
		all_T_opp_vs_T_sure_significant = []
		all_y_pred_r2 = []
		all_opt_out_r2 = []
		all_theta = []
		for n in range(all_z.shape[1]):
			# Activations
			z_n = all_z[:,n]
			# Angle of projction on to top 2 PCs
			z_n_theta = np.degrees(np.arccos(np.dot(pca.components_[:2,n], np.array([1,0])) / (np.linalg.norm(pca.components_[:2,n]) * np.linalg.norm(np.array([1,0])))))
			if pca.components_[1,n] < 0:
				z_n_theta = 360 - z_n_theta
			all_theta.append(z_n_theta)
			# Regressions
			# Decision output
			eps = 1e-8
			y_pred = np.stack([y_pred, np.ones(y_pred.shape) * eps]).max(0)
			y_pred = np.stack([y_pred, np.ones(y_pred.shape) * (1-eps)]).min(0)
			logit_y_pred = logit(y_pred)
			clf = LinearRegression().fit(np.expand_dims(z_n,1), logit_y_pred)
			y_pred_r2 = clf.score(np.expand_dims(z_n,1), logit_y_pred)
			all_y_pred_r2.append(y_pred_r2)
			# Opt-out
			opt_out = np.stack([opt_out, np.ones(opt_out.shape) * eps]).max(0)
			opt_out = np.stack([opt_out, np.ones(opt_out.shape) * (1-eps)]).min(0)
			logit_opt_out = logit(opt_out)
			clf = LinearRegression().fit(np.expand_dims(z_n,1), logit_opt_out)
			opt_out_r2 = clf.score(np.expand_dims(z_n,1), logit_opt_out)
			all_opt_out_r2.append(opt_out_r2)
			# Kiani & Shadlen analysis
			if y_pred_r2 > opt_out_r2:
				# Normalize activations
				norm_act = (z_n - z_n.min()) / all_z.std()**2
				# Preferred stimulus 
				if norm_act[s2_trials].mean() > norm_act[s1_trials].mean():
					T_in_act = norm_act[s2_trials]
					T_opp_act = norm_act[s1_trials]
				else:
					T_in_act = norm_act[s1_trials]
					T_opp_act = norm_act[s2_trials]
				# T_in
				all_T_in_act_mn.append(T_in_act.mean())
				all_T_in_act_se.append(sem(T_in_act))
				# T_opp
				all_T_opp_act_mn.append(T_opp_act.mean())
				all_T_opp_act_se.append(sem(T_opp_act))
				# Opt-out
				T_sure_act = norm_act[opt_out_trials]
				all_T_sure_act_mn.append(T_sure_act.mean())
				all_T_sure_act_se.append(sem(T_sure_act))
				# Compute statistical significance
				# T_in vs. T_sure
				_, p = ttest_ind(T_in_act, T_sure_act)
				if T_in_act.mean() - T_sure_act.mean() > 0 and p < 0.05:
					all_T_in_vs_T_sure_significant.append(True)
				else:
					all_T_in_vs_T_sure_significant.append(False)
				# T_opp vs. T_sure
				_, p = ttest_ind(T_opp_act, T_sure_act)
				if T_sure_act.mean() - T_opp_act.mean() > 0 and p < 0.05:
					all_T_opp_vs_T_sure_significant.append(True)
				else:
					all_T_opp_vs_T_sure_significant.append(False)
			# Visualize individual neurons
			if args.vis_ind_neuron:
				# Create figure
				fig, axs = plt.subplots(2,1)
				# Decision output
				ax = axs[0]
				axs[0].scatter(z_n, y_pred, color='gray', s=9)
				axs[0].set_xlabel('Activation', fontsize=args.axis_fontsize)
				axs[0].tick_params(axis='x', labelsize=args.ticks_fontsize)
				axs[0].tick_params(axis='y', labelsize=args.ticks_fontsize)
				axs[0].set_ylabel('Decision output', fontsize=args.axis_fontsize)
				axs[0].set_ylim([-0.05,1.05])
				axs[0].set_yticks([0,0.5,1])
				axs[0].set_yticklabels(['0','0.5','1'])
				axs[0].spines['right'].set_visible(False)
				axs[0].spines['top'].set_visible(False)
				# Plot PC 2 vs. opt-out
				ax = axs[1]
				axs[1].scatter(z_n, opt_out, color='gray', s=8)
				axs[1].set_xlabel('Activation', fontsize=args.axis_fontsize)
				axs[1].tick_params(axis='x', labelsize=args.ticks_fontsize)
				axs[1].tick_params(axis='y', labelsize=args.ticks_fontsize)
				axs[1].set_ylabel('Opt-out', fontsize=args.axis_fontsize)
				axs[1].set_ylim([-0.05,1.05])
				axs[1].set_yticks([0,0.5,1])
				axs[1].set_yticklabels(['0','0.5','1'])
				axs[1].spines['right'].set_visible(False)
				axs[1].spines['top'].set_visible(False)
				# Save
				plt.tight_layout()
				fig_name = run_dir + './z' + str(n) + '.png'
				plt.savefig(fig_name, bbox_inches='tight', dpi=300)
				plt.close()
		# Convert to arrays
		all_T_in_act_mn = np.array(all_T_in_act_mn)
		all_T_in_act_se = np.array(all_T_in_act_se)
		all_T_opp_act_mn = np.array(all_T_opp_act_mn)
		all_T_opp_act_se = np.array(all_T_opp_act_se)
		all_T_sure_act_mn = np.array(all_T_sure_act_mn)
		all_T_sure_act_se = np.array(all_T_sure_act_se)
		all_y_pred_r2 = np.array(all_y_pred_r2)
		all_opt_out_r2 = np.array(all_opt_out_r2)
		all_T_in_vs_T_sure_significant = np.array(all_T_in_vs_T_sure_significant)
		all_T_opp_vs_T_sure_significant = np.array(all_T_opp_vs_T_sure_significant)
		all_theta = np.array(all_theta)
		# Classify neurons
		decision_neurons = all_y_pred_r2 > all_opt_out_r2
		confidence_neurons = all_opt_out_r2 > all_y_pred_r2
		# Plot distribution of decision_r2 - opt_out_r2 (delta R^2)
		delta_r2 = all_y_pred_r2 - all_opt_out_r2
		x_range = np.linspace(-1,1,1000)
		delta_r2_kernel = gaussian_kde(delta_r2)
		delta_r2_dist = delta_r2_kernel(x_range)
		ax = plt.subplot(111)
		plt.fill(np.append(np.insert(x_range, 0, x_range[0]), x_range[-1]), np.append(np.insert(delta_r2_dist, 0, 0), 0), color='black', alpha=0.5)
		plt.text(-0.5,0.65,'Confidence\nneurons',fontsize=args.axis_fontsize,horizontalalignment='center',verticalalignment='center')
		plt.text(0.5,0.65,'Decision\nneurons',fontsize=args.axis_fontsize,horizontalalignment='center',verticalalignment='center')
		plt.plot([0,0],[0,0.7],color='black',linestyle='dashed')
		plt.xlabel(r'$\Delta R^{2}$', fontsize=args.axis_fontsize)
		plt.xticks([-1,-0.5,0,0.5,1], ['-1','-0.5','0','0.5','1'], fontsize=args.ticks_fontsize)
		plt.ylabel('Density estimate', fontsize=args.axis_fontsize)
		plt.yticks([0,0.2,0.4,0.6],['0','0.2','0.4','0.6'],fontsize=args.ticks_fontsize)
		plt.xlim([-1,1])
		plt.ylim([0,0.7])
		ax.spines['right'].set_visible(False)
		ax.spines['top'].set_visible(False)
		ax.yaxis.set_ticks_position('left')
		ax.xaxis.set_ticks_position('bottom')
		plt.tight_layout()
		fig_name = run_dir + './delta_r2_pdf.png'
		plt.savefig(fig_name, bbox_inches='tight', dpi=300)
		plt.close()
		# Plot theta vs. delta R^2
		ax = plt.subplot(111)
		plt.scatter(all_theta, delta_r2, color='black')
		plt.plot([0,360],[0,0],color='black',linestyle='dashed')
		plt.xlabel(r'$\theta$', fontsize=args.axis_fontsize)
		plt.xticks([0,90,180,270,360], [r'$0^{\circ}$',r'$90^{\circ}$',r'$180^{\circ}$',r'$270^{\circ}$',r'$360^{\circ}$'], fontsize=args.ticks_fontsize)
		plt.ylabel(r'$\Delta R^{2}$', fontsize=args.axis_fontsize)
		plt.yticks([-1,-0.5,0,0.5,1], ['-1','-0.5','0','0.5','1'], fontsize=args.ticks_fontsize)
		plt.xlim([0,360])
		plt.ylim([-1,1])
		ax.spines['right'].set_visible(False)
		ax.spines['top'].set_visible(False)
		ax.yaxis.set_ticks_position('left')
		ax.xaxis.set_ticks_position('bottom')
		plt.tight_layout()
		fig_name = run_dir + './theta_vs_delta_r2.png'
		plt.savefig(fig_name, bbox_inches='tight', dpi=300)
		plt.close()
		# Kiani & Shadlen analysis
		ticks_fontsize = 20
		axis_fontsize = 20
		# T_in vs. T_sure
		ax = plt.subplot(111)
		plt.errorbar(all_T_sure_act_mn, all_T_in_act_mn, xerr=all_T_sure_act_se, yerr=all_T_in_act_se, color='black', fmt="o")
		plt.plot([0,5.5],[0,5.5],color='gray')
		plt.xticks([0,2,4],['0','2','4'],fontsize=ticks_fontsize)
		plt.xlabel(r'$T_{S}$ choice (norm. activity)', fontsize=axis_fontsize)
		plt.yticks([0,2,4],['0','2','4'],fontsize=ticks_fontsize)
		plt.ylabel(r'$T_{in}$ choice (norm. activity)', fontsize=axis_fontsize)
		ax.spines['right'].set_visible(False)
		ax.spines['top'].set_visible(False)
		ax.yaxis.set_ticks_position('left')
		ax.xaxis.set_ticks_position('bottom')
		ax.axis('square')
		plt.xlim([0,7.5])
		plt.ylim([0,7.5])
		plt.tight_layout()
		fig_name = run_dir + './T_in_vs_T_sure.png'
		plt.savefig(fig_name, bbox_inches='tight', dpi=300)
		plt.close()
		# T_opp vs. T_sure
		ax = plt.subplot(111)
		plt.errorbar(all_T_sure_act_mn, all_T_opp_act_mn, xerr=all_T_sure_act_se, yerr=all_T_opp_act_se, color='black', fmt="o")
		plt.plot([0,5.5],[0,5.5],color='gray')
		plt.xticks([0,2,4],['0','2','4'],fontsize=ticks_fontsize)
		plt.xlabel(r'$T_{S}$ choice (norm. activity)', fontsize=axis_fontsize)
		plt.yticks([0,2,4],['0','2','4'],fontsize=ticks_fontsize)
		plt.ylabel(r'$T_{opp}$ choice (norm. activity)', fontsize=axis_fontsize)
		ax.spines['right'].set_visible(False)
		ax.spines['top'].set_visible(False)
		ax.yaxis.set_ticks_position('left')
		ax.xaxis.set_ticks_position('bottom')
		ax.axis('square')
		plt.xlim([0,7.5])
		plt.ylim([0,7.5])
		plt.tight_layout()
		fig_name = run_dir + './T_opp_vs_T_sure.png'
		plt.savefig(fig_name, bbox_inches='tight', dpi=300)
		plt.close()
		# Stats
		print('run ' + str(r))
		t,p = ttest_rel(all_T_in_act_mn, all_T_sure_act_mn)
		print('T_in vs. T_sure: t=' + str(np.around(t,4)) + ', p=' + str(np.around(p,4)))
		t,p = ttest_rel(all_T_opp_act_mn, all_T_sure_act_mn)
		print('T_opp vs. T_sure: t=' + str(np.around(t,4)) + ', p=' + str(np.around(p,4)))
		# Histograms
		# T_in vs. T_sure
		T_in_vs_T_sure = all_T_sure_act_mn - all_T_in_act_mn
		ax = plt.subplot(111)
		counts, bins, patches = plt.hist(T_in_vs_T_sure, color='white', edgecolor='black', linewidth=3)
		max_count = max(counts)
		counts, bins, patches = plt.hist(T_in_vs_T_sure[all_T_in_vs_T_sure_significant], color='gray', edgecolor='black', bins=bins, linewidth=3)
		plt.arrow(T_in_vs_T_sure.mean(),max_count+1,0,-0.5,width=0.025,head_width=0.1,head_length=0.4,color='black')
		plt.xlabel(r'$T_{S} - T_{in}$', fontsize=args.axis_fontsize)
		plt.xlim([-3,3])
		plt.xticks([-3,-2,-1,0,1,2,3], ['-3','-2','-1','0','1','2','3'], fontsize=args.ticks_fontsize)
		plt.ylabel('Count', fontsize=args.axis_fontsize)
		plt.yticks(fontsize=args.ticks_fontsize)
		plt.ylim([0,max_count+1.1])
		ax.spines['right'].set_visible(False)
		ax.spines['top'].set_visible(False)
		ax.yaxis.set_ticks_position('left')
		ax.xaxis.set_ticks_position('bottom')
		ax.set_aspect(0.3)
		plt.tight_layout()
		fig_name = run_dir + './T_in_vs_T_sure_hist.png'
		plt.savefig(fig_name, bbox_inches='tight', dpi=300)
		plt.close()
		# T_opp vs. T_sure
		T_opp_vs_T_sure = all_T_sure_act_mn - all_T_opp_act_mn
		ax = plt.subplot(111)
		counts, bins, patches = plt.hist(T_opp_vs_T_sure, color='white', edgecolor='black', linewidth=3)
		max_count = max(counts)
		counts, bins, patches = plt.hist(T_opp_vs_T_sure[all_T_opp_vs_T_sure_significant], color='gray', edgecolor='black', bins=bins, linewidth=3)
		plt.arrow(T_opp_vs_T_sure.mean(),max_count+1,0,-0.5,width=0.025,head_width=0.1,head_length=0.4,color='black')
		plt.xlabel(r'$T_{S} - T_{opp}$', fontsize=args.axis_fontsize)
		plt.xlim([-3,3])
		plt.xticks([-3,-2,-1,0,1,2,3], ['-3','-2','-1','0','1','2','3'], fontsize=args.ticks_fontsize)
		plt.ylabel('Count', fontsize=args.axis_fontsize)
		plt.yticks(fontsize=args.ticks_fontsize)
		plt.ylim([0,max_count+1.1])
		ax.spines['right'].set_visible(False)
		ax.spines['top'].set_visible(False)
		ax.yaxis.set_ticks_position('left')
		ax.xaxis.set_ticks_position('bottom')
		ax.set_aspect(0.3)
		plt.tight_layout()
		fig_name = run_dir + './T_opp_vs_T_sure_hist.png'
		plt.savefig(fig_name, bbox_inches='tight', dpi=300)
		plt.close()

def main():

	# Settings
	parser = argparse.ArgumentParser()
	parser.add_argument('--N_runs', type=int, default=100)
	parser.add_argument('--vis_ind_neuron', action='store_true', default=False)
	parser.add_argument('--bar_width', type=float, default=0.8)
	parser.add_argument('--title_fontsize', type=int, default=24)
	parser.add_argument('--axis_fontsize', type=int, default=20)
	parser.add_argument('--ticks_fontsize', type=int, default=18)
	parser.add_argument('--legend_fontsize', type=int, default=16)
	args = parser.parse_args()

	# Plot
	plot(args)

if __name__ == '__main__':
	main()