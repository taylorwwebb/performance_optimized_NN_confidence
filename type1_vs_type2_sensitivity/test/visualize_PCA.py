import numpy as np
import matplotlib.pyplot as plt
from scipy.stats import gaussian_kde
import argparse

def plot(args):
	for r in range(args.N_runs):
		# Run directory
		run_dir = './run' + str(r+1) + '/'
		# Load PCA results
		results = np.load(run_dir + 'PCA_test_results.npz')
		pc1_all_conds = results['z_top2'][:,:,0]
		pc2_all_conds = results['z_top2'][:,:,1]
		y_pred_all_conds = results['all_y_pred']
		y_targ_all_conds = results['all_y_targ']
		conf_all_conds = results['all_conf']
		# Only plot conditions with lowest and highest s2 contrast
		N_cond = pc1_all_conds.shape[0]
		plot_cond = [0, N_cond-1]
		for i in plot_cond:
			# Get results for this condition
			pc1 = pc1_all_conds[i]
			pc2 = pc2_all_conds[i]
			y_pred = y_pred_all_conds[i]
			y_targ = y_targ_all_conds[i]
			conf = conf_all_conds[i]
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
			plt.ylim([0,0.5])
			plt.yticks([0,0.1,0.2,0.3,0.4,0.5],['0','0.1','0.2','0.3','0.4','0.5'],fontsize=args.ticks_fontsize)
			ax.spines['right'].set_visible(False)
			ax.spines['top'].set_visible(False)
			ax.yaxis.set_ticks_position('left')
			ax.xaxis.set_ticks_position('bottom')
			plt.legend(['s1','s2'],frameon=False,fontsize=args.legend_fontsize)
			if i == 0:
				plt.title('s2 low contrast', fontsize=args.axis_fontsize)
				fig_name = run_dir + './pc1_vs_s1s2_dist_s2low.png'
			elif i == (N_cond - 1):
				plt.title('s2 high contrast', fontsize=args.axis_fontsize)
				fig_name = run_dir + './pc1_vs_s1s2_dist_s2high.png'
			plt.savefig(fig_name, bbox_inches='tight', dpi=300)
			plt.close()
			# Plot PC 2 vs. correct/incorrect (response-specific)
			# Response = s1
			resp_s1 = y_pred < 0.5
			pc2_correct = pc2[np.logical_and(y_pred.round()==y_targ,resp_s1)]
			pc2_incorrect = pc2[np.logical_and(y_pred.round()!=y_targ,resp_s1)]
			x_range = np.linspace(-10,17,1000)
			pc2_correct_kernel = gaussian_kde(pc2_correct)
			pc2_correct_dist = pc2_correct_kernel(x_range)
			pc2_incorrect_kernel = gaussian_kde(pc2_incorrect)
			pc2_incorrect_dist = pc2_incorrect_kernel(x_range)
			ax = plt.subplot(111)
			plt.fill(np.append(np.insert(x_range, 0, x_range[0]), x_range[-1]), np.append(np.insert(pc2_correct_dist, 0, 0), 0), color='turquoise', alpha=0.5)
			plt.fill(np.append(np.insert(x_range, 0, x_range[0]), x_range[-1]), np.append(np.insert(pc2_incorrect_dist, 0, 0), 0), color='salmon', alpha=0.5)
			plt.xlabel('Principal component 2', fontsize=args.axis_fontsize)
			plt.xlim([-8,17])
			plt.xticks([-5,0,5,10,15], ['-5','0','5','10','15'], fontsize=args.ticks_fontsize)
			plt.ylabel('Density estimate', fontsize=args.axis_fontsize)
			plt.ylim([0,0.9])
			plt.yticks([0,0.2,0.4,0.6,0.8],['0','0.2','0.4','0.6','0.8'],fontsize=args.ticks_fontsize)
			ax.spines['right'].set_visible(False)
			ax.spines['top'].set_visible(False)
			ax.yaxis.set_ticks_position('left')
			ax.xaxis.set_ticks_position('bottom')
			plt.legend(['Correct','Incorrect'],frameon=False,fontsize=args.legend_fontsize)
			if i == 0:
				plt.title('s2 low contrast\nresponse = s1', fontsize=args.axis_fontsize)
				fig_name = run_dir + 'pc2_vs_correct_incorrect_dist_resp_s1_s2low.png'
			elif i == (N_cond - 1):
				plt.title('s2 high contrast\nresponse = s1', fontsize=args.axis_fontsize)
				fig_name = run_dir + 'pc2_vs_correct_incorrect_dist_resp_s1_s2high.png'
			plt.savefig(fig_name, bbox_inches='tight', dpi=300)
			plt.close()
			# Response = s2
			resp_s2 = y_pred >= 0.5
			pc2_correct = pc2[np.logical_and(y_pred.round()==y_targ,resp_s2)]
			pc2_incorrect = pc2[np.logical_and(y_pred.round()!=y_targ,resp_s2)]
			x_range = np.linspace(-10,17,1000)
			pc2_correct_kernel = gaussian_kde(pc2_correct)
			pc2_correct_dist = pc2_correct_kernel(x_range)
			pc2_incorrect_kernel = gaussian_kde(pc2_incorrect)
			pc2_incorrect_dist = pc2_incorrect_kernel(x_range)
			ax = plt.subplot(111)
			plt.fill(np.append(np.insert(x_range, 0, x_range[0]), x_range[-1]), np.append(np.insert(pc2_correct_dist, 0, 0), 0), color='turquoise', alpha=0.5)
			plt.fill(np.append(np.insert(x_range, 0, x_range[0]), x_range[-1]), np.append(np.insert(pc2_incorrect_dist, 0, 0), 0), color='salmon', alpha=0.5)
			plt.xlabel('Principal component 2', fontsize=args.axis_fontsize)
			plt.xlim([-8,17])
			plt.xticks([-5,0,5,10,15], ['-5','0','5','10','15'], fontsize=args.ticks_fontsize)
			plt.ylabel('Density estimate', fontsize=args.axis_fontsize)
			plt.ylim([0,0.9])
			plt.yticks([0,0.2,0.4,0.6,0.8],['0','0.2','0.4','0.6','0.8'],fontsize=args.ticks_fontsize)
			ax.spines['right'].set_visible(False)
			ax.spines['top'].set_visible(False)
			ax.yaxis.set_ticks_position('left')
			ax.xaxis.set_ticks_position('bottom')
			plt.legend(['Correct','Incorrect'],frameon=False,fontsize=args.legend_fontsize)
			if i == 0:
				plt.title('s2 low contrast\nresponse = s2', fontsize=args.axis_fontsize)
				fig_name = run_dir + 'pc2_vs_correct_incorrect_dist_resp_s2_s2low.png'
			elif i == (N_cond - 1):
				plt.title('s2 high contrast\nresponse = s2', fontsize=args.axis_fontsize)
				fig_name = run_dir + 'pc2_vs_correct_incorrect_dist_resp_s2_s2high.png'
			plt.savefig(fig_name, bbox_inches='tight', dpi=300)
			plt.close()
			# Same as above, but for |PC 1|
			abs_pc1 = np.abs(pc1)
			# Response = s1
			abs_pc1_correct = abs_pc1[np.logical_and(y_pred.round()==y_targ,resp_s1)]
			abs_pc1_incorrect = abs_pc1[np.logical_and(y_pred.round()!=y_targ,resp_s1)]
			x_range = np.linspace(-10,17,1000)
			abs_pc1_correct_kernel = gaussian_kde(abs_pc1_correct)
			abs_pc1_correct_dist = abs_pc1_correct_kernel(x_range)
			abs_pc1_incorrect_kernel = gaussian_kde(abs_pc1_incorrect)
			abs_pc1_incorrect_dist = abs_pc1_incorrect_kernel(x_range)
			ax = plt.subplot(111)
			plt.fill(np.append(np.insert(x_range, 0, x_range[0]), x_range[-1]), np.append(np.insert(abs_pc1_correct_dist, 0, 0), 0), color='turquoise', alpha=0.5)
			plt.fill(np.append(np.insert(x_range, 0, x_range[0]), x_range[-1]), np.append(np.insert(abs_pc1_incorrect_dist, 0, 0), 0), color='salmon', alpha=0.5)
			plt.xlabel('|Principal component 1|', fontsize=args.axis_fontsize)
			plt.xlim([0,17])
			plt.xticks([0,5,10,15], ['0','5','10','15'], fontsize=args.ticks_fontsize)
			plt.ylabel('Density estimate', fontsize=args.axis_fontsize)
			plt.ylim([0,1.1])
			plt.yticks([0,0.2,0.4,0.6,0.8,1],['0','0.2','0.4','0.6','0.8','1'],fontsize=args.ticks_fontsize)
			ax.spines['right'].set_visible(False)
			ax.spines['top'].set_visible(False)
			ax.yaxis.set_ticks_position('left')
			ax.xaxis.set_ticks_position('bottom')
			plt.legend(['Correct','Incorrect'],frameon=False,fontsize=args.legend_fontsize)
			if i == 0:
				plt.title('s2 low contrast\nresponse = s1', fontsize=args.axis_fontsize)
				fig_name = run_dir + 'abs_pc1_vs_correct_incorrect_dist_resp_s1_s2low.png'
			elif i == (N_cond - 1):
				plt.title('s2 high contrast\nresponse = s1', fontsize=args.axis_fontsize)
				fig_name = run_dir + 'abs_pc1_vs_correct_incorrect_dist_resp_s1_s2high.png'
			plt.savefig(fig_name, bbox_inches='tight', dpi=300)
			plt.close()
			# Response = s2
			resp_s2 = y_pred >= 0.5
			abs_pc1_correct = abs_pc1[np.logical_and(y_pred.round()==y_targ,resp_s2)]
			abs_pc1_incorrect = abs_pc1[np.logical_and(y_pred.round()!=y_targ,resp_s2)]
			x_range = np.linspace(-10,17,1000)
			abs_pc1_correct_kernel = gaussian_kde(abs_pc1_correct)
			abs_pc1_correct_dist = abs_pc1_correct_kernel(x_range)
			abs_pc1_incorrect_kernel = gaussian_kde(abs_pc1_incorrect)
			abs_pc1_incorrect_dist = abs_pc1_incorrect_kernel(x_range)
			ax = plt.subplot(111)
			plt.fill(np.append(np.insert(x_range, 0, x_range[0]), x_range[-1]), np.append(np.insert(abs_pc1_correct_dist, 0, 0), 0), color='turquoise', alpha=0.5)
			plt.fill(np.append(np.insert(x_range, 0, x_range[0]), x_range[-1]), np.append(np.insert(abs_pc1_incorrect_dist, 0, 0), 0), color='salmon', alpha=0.5)
			plt.xlabel('|Principal component 1|', fontsize=args.axis_fontsize)
			plt.xlim([0,17])
			plt.xticks([0,5,10,15], ['0','5','10','15'], fontsize=args.ticks_fontsize)
			plt.ylabel('Density estimate', fontsize=args.axis_fontsize)
			plt.ylim([0,1.1])
			plt.yticks([0,0.2,0.4,0.6,0.8,1],['0','0.2','0.4','0.6','0.8','1'],fontsize=args.ticks_fontsize)
			ax.spines['right'].set_visible(False)
			ax.spines['top'].set_visible(False)
			ax.yaxis.set_ticks_position('left')
			ax.xaxis.set_ticks_position('bottom')
			plt.legend(['Correct','Incorrect'],frameon=False,fontsize=args.legend_fontsize)
			if i == 0:
				plt.title('s2 low contrast\nresponse = s1', fontsize=args.axis_fontsize)
				fig_name = run_dir + 'abs_pc1_vs_correct_incorrect_dist_resp_s2_s2low.png'
			elif i == (N_cond - 1):
				plt.title('s2 high contrast\nresponse = s1', fontsize=args.axis_fontsize)
				fig_name = run_dir + 'abs_pc1_vs_correct_incorrect_dist_resp_s2_s2high.png'
			plt.savefig(fig_name, bbox_inches='tight', dpi=300)
			plt.close()

def main():

	# Settings
	parser = argparse.ArgumentParser()
	parser.add_argument('--N_runs', type=int, default=100)
	parser.add_argument('--axis_fontsize', type=int, default=18)
	parser.add_argument('--ticks_fontsize', type=int, default=16)
	parser.add_argument('--legend_fontsize', type=int, default=16)
	args = parser.parse_args()

	# Plot
	plot(args)

if __name__ == '__main__':
	main()