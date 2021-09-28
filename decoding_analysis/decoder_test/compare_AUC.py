import numpy as np
import matplotlib.pyplot as plt
from scipy.stats import sem
from scipy.stats import ttest_rel
import os
import argparse

def plot(args):
	all_whole_network_AUC_interaction = []
	all_z_AUC_interaction = []
	for r in range(args.N_runs):
		# Whole network results
		# Run directory
		run_dir = './whole_network/noise=' + str(args.decoder_noise) + '/run' + str(r+1) + '/'
		# Load results
		results = np.load(run_dir + 'ROC_AUC_results.npz')
		choice_BE_AUC = results['choice_BE_AUC']
		choice_DCE_AUC = results['choice_DCE_AUC']
		conf_BE_AUC = results['conf_BE_AUC']
		conf_DCE_AUC = results['conf_DCE_AUC']
		AUC_interaction = (choice_BE_AUC - choice_DCE_AUC) - (conf_BE_AUC - conf_DCE_AUC)
		all_whole_network_AUC_interaction.append(AUC_interaction)
		# z results
		# Run directory
		run_dir = './z/noise=' + str(args.decoder_noise) + '/run' + str(r+1) + '/'
		# Load results
		results = np.load(run_dir + 'ROC_AUC_results.npz')
		choice_BE_AUC = results['choice_BE_AUC']
		choice_DCE_AUC = results['choice_DCE_AUC']
		conf_BE_AUC = results['conf_BE_AUC']
		conf_DCE_AUC = results['conf_DCE_AUC']
		AUC_interaction = (choice_BE_AUC - choice_DCE_AUC) - (conf_BE_AUC - conf_DCE_AUC)
		all_z_AUC_interaction.append(AUC_interaction)
	# Convert to arrays
	all_whole_network_AUC_interaction = np.array(all_whole_network_AUC_interaction)
	all_z_AUC_interaction = np.array(all_z_AUC_interaction)
	# Summary statistics
	whole_network_AUC_interaction_mn = all_whole_network_AUC_interaction.mean(0)
	whole_network_AUC_interaction_se = sem(all_whole_network_AUC_interaction,0)
	z_AUC_interaction_mn = all_z_AUC_interaction.mean(0)
	z_AUC_interaction_se = sem(all_z_AUC_interaction,0)
	# Plot
	# Statistics
	_, interaction_AUC_p = ttest_rel(all_whole_network_AUC_interaction,all_z_AUC_interaction)
	# Significance symbols
	if interaction_AUC_p >= 0.05: interaction_AUC_p_symb = 'ns'
	if interaction_AUC_p < 0.05: interaction_AUC_p_symb = '*'
	if interaction_AUC_p < 0.01: interaction_AUC_p_symb = '**'
	if interaction_AUC_p < 0.001: interaction_AUC_p_symb = '***'
	if interaction_AUC_p < 0.0001: interaction_AUC_p_symb = '****'
	# Plot AUC
	ax = plt.subplot(111)
	plt.bar([0,1], [whole_network_AUC_interaction_mn, z_AUC_interaction_mn], yerr=[whole_network_AUC_interaction_se, z_AUC_interaction_se], width=0.8, color='black')
	plt.plot([-0.5,1.5],[0,0],color='black')
	plt.xlim([-0.5,1.5])
	plt.xticks([0,1], ['whole network','z'], fontsize=18)
	plt.ylim([-0.02,0.08])
	plt.yticks([-0.02,0,0.02,0.04,0.06,0.08],['-0.02','0','0.02','0.04','0.06','0.08'], fontsize=18)
	plt.ylabel('AUC interaction', fontsize=18)
	ax.spines['right'].set_visible(False)
	ax.spines['top'].set_visible(False)
	ax.yaxis.set_ticks_position('left')
	ax.xaxis.set_ticks_position('bottom')
	# Significance
	max_y_val = np.array([whole_network_AUC_interaction_mn + whole_network_AUC_interaction_se, z_AUC_interaction_mn + z_AUC_interaction_se]).max()
	y_start = max_y_val + 0.005
	y_end = max_y_val + 0.0075
	plt.plot([0,0,1,1],[y_start,y_end,y_end,y_start],color='black', clip_on=False)
	plt.text(0.5,y_end+0.0025,interaction_AUC_p_symb,fontsize=18,horizontalalignment='center')
	# Save
	plot_fname = './noise=' + str(args.decoder_noise) + '_AUC_interaction_comparison.png'
	plt.savefig(plot_fname, bbox_inches='tight', dpi=300)
	plt.close()

def main():

	# Settings
	parser = argparse.ArgumentParser()
	parser.add_argument('--N_runs', type=int, default=100)
	parser.add_argument('--decoder_noise', type=float, default=0.0)
	args = parser.parse_args()

	# Plot 
	plot(args)

if __name__ == '__main__':
	main()