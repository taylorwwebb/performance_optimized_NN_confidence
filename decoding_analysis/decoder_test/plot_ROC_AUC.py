import numpy as np
import matplotlib.pyplot as plt
from scipy.stats import sem
from scipy.stats import ttest_rel
import os
import argparse

def plot(args):
	all_choice_BE_hr = []
	all_choice_DCE_hr = []
	all_conf_BE_hr = []
	all_conf_DCE_hr = []
	all_choice_BE_AUC = []
	all_choice_DCE_AUC = []
	all_conf_BE_AUC = []
	all_conf_DCE_AUC = []
	for r in range(args.N_runs):
		# Run directory
		run_dir = './' + args.decoder_input + '/noise=' + str(args.decoder_noise) + '/run' + str(r+1) + '/'
		# Load results
		results = np.load(run_dir + 'ROC_AUC_results.npz')
		far = results['far']
		all_choice_BE_hr.append(results['choice_BE_hr'])
		all_choice_DCE_hr.append(results['choice_DCE_hr'])
		all_conf_BE_hr.append(results['conf_BE_hr'])
		all_conf_DCE_hr.append(results['conf_DCE_hr'])
		all_choice_BE_AUC.append(results['choice_BE_AUC'])
		all_choice_DCE_AUC.append(results['choice_DCE_AUC'])
		all_conf_BE_AUC.append(results['conf_BE_AUC'])
		all_conf_DCE_AUC.append(results['conf_DCE_AUC'])
	# Convert to arrays
	all_choice_BE_hr = np.array(all_choice_BE_hr)
	all_choice_DCE_hr = np.array(all_choice_DCE_hr)
	all_conf_BE_hr = np.array(all_conf_BE_hr)
	all_conf_DCE_hr = np.array(all_conf_DCE_hr)
	all_choice_BE_AUC = np.array(all_choice_BE_AUC)
	all_choice_DCE_AUC = np.array(all_choice_DCE_AUC)
	all_conf_BE_AUC = np.array(all_conf_BE_AUC)
	all_conf_DCE_AUC = np.array(all_conf_DCE_AUC)
	# Summary statistics
	choice_BE_hr_mn = all_choice_BE_hr.mean(0)
	choice_BE_hr_se = sem(all_choice_BE_hr,0)
	choice_DCE_hr_mn = all_choice_DCE_hr.mean(0)
	choice_DCE_hr_se = sem(all_choice_DCE_hr,0)
	conf_BE_hr_mn = all_conf_BE_hr.mean(0)
	conf_BE_hr_se = sem(all_conf_BE_hr,0)
	conf_DCE_hr_mn = all_conf_DCE_hr.mean(0)
	conf_DCE_hr_se = sem(all_conf_DCE_hr,0)
	choice_BE_AUC_mn = all_choice_BE_AUC.mean(0)
	choice_BE_AUC_se = sem(all_choice_BE_AUC,0)
	choice_DCE_AUC_mn = all_choice_DCE_AUC.mean(0)
	choice_DCE_AUC_se = sem(all_choice_DCE_AUC,0)
	conf_BE_AUC_mn = all_conf_BE_AUC.mean(0)
	conf_BE_AUC_se = sem(all_conf_BE_AUC,0)
	conf_DCE_AUC_mn = all_conf_DCE_AUC.mean(0)
	conf_DCE_AUC_se = sem(all_conf_DCE_AUC,0)
	# Plot
	# ROC 
	fig = plt.figure()
	far_fill = np.concatenate([far, np.flip(far)])
	# Choice 
	ax = fig.add_subplot(1,2,1)
	plt.plot(far, choice_BE_hr_mn, color='black')
	plt.plot(far, choice_DCE_hr_mn, color='gray')
	choice_BE_fill_y = np.concatenate([choice_BE_hr_mn - choice_BE_hr_se,
									   np.flip(choice_BE_hr_mn + choice_BE_hr_se)])
	plt.fill(far_fill , choice_BE_fill_y, color='black', alpha=0.25)
	choice_DCE_fill_y = np.concatenate([choice_DCE_hr_mn - choice_DCE_hr_se,
									   np.flip(choice_DCE_hr_mn + choice_DCE_hr_se)])
	plt.fill(far_fill , choice_DCE_fill_y, color='gray', alpha=0.25)
	plt.plot([0,1], [0,1], linestyle='dashed', color='lightgray')
	plt.xlim([0,1])
	plt.xticks([0,0.5,1],['0','0.5','1'])
	plt.yticks([0,0.2,0.4,0.6,0.8,1],['0','0.2','0.4','0.6','0.8','1'])
	plt.ylim([0,1])
	ax.set_aspect('equal')
	plt.xlabel('False alarm rate')
	plt.ylabel('Hit rate')
	plt.title('Decision')
	ax.spines['right'].set_visible(False)
	ax.spines['top'].set_visible(False)
	ax.yaxis.set_ticks_position('left')
	ax.xaxis.set_ticks_position('bottom')
	lgd = ax.legend(['BE', 'RCE'], bbox_to_anchor=(0.55,1.4), frameon=False)
	# Confidence
	ax = fig.add_subplot(1,2,2)
	plt.plot(far, conf_BE_hr_mn, color='black')
	plt.plot(far, conf_DCE_hr_mn, color='gray')
	conf_BE_fill_y = np.concatenate([conf_BE_hr_mn - conf_BE_hr_se,
									   np.flip(conf_BE_hr_mn + conf_BE_hr_se)])
	plt.fill(far_fill , conf_BE_fill_y, color='black', alpha=0.25)
	conf_DCE_fill_y = np.concatenate([conf_DCE_hr_mn - conf_DCE_hr_se,
									   np.flip(conf_DCE_hr_mn + conf_DCE_hr_se)])
	plt.fill(far_fill , conf_DCE_fill_y, color='gray', alpha=0.25)
	plt.plot([0,1], [0,1], linestyle='dashed', color='lightgray')
	plt.xlim([0,1])
	plt.xticks([0,0.5,1],['0','0.5','1'])
	plt.yticks([0,0.2,0.4,0.6,0.8,1],['0','0.2','0.4','0.6','0.8','1'])
	plt.ylim([0,1])
	ax.set_aspect('equal')
	plt.xlabel('False alarm rate')
	plt.ylabel('Hit rate')
	plt.title('Confidence')
	ax.spines['right'].set_visible(False)
	ax.spines['top'].set_visible(False)
	ax.yaxis.set_ticks_position('left')
	ax.xaxis.set_ticks_position('bottom')
	plot_fname = './' + args.decoder_input + '/noise=' + str(args.decoder_noise) + '/decoding_ROC.png'
	fig.tight_layout()
	plt.savefig(plot_fname, dpi=300, bbox_inches='tight', bbox_extra_artists=(lgd,))
	plt.close()
	# AUC
	# Statistics
	_, choice_p = ttest_rel(all_choice_BE_AUC,all_choice_DCE_AUC)
	_, conf_p = ttest_rel(all_conf_BE_AUC,all_conf_DCE_AUC)
	_, diff_p = ttest_rel(all_choice_BE_AUC - all_choice_DCE_AUC, all_conf_BE_AUC - all_conf_DCE_AUC)
	# Significance symbols
	if choice_p >= 0.05: choice_p_symb = 'ns'
	if choice_p < 0.05: choice_p_symb = '*'
	if choice_p < 0.01: choice_p_symb = '**'
	if choice_p < 0.001: choice_p_symb = '***'
	if choice_p < 0.0001: choice_p_symb = '****'
	if conf_p >= 0.05: conf_p_symb = 'ns'
	if conf_p < 0.05: conf_p_symb = '*'
	if conf_p < 0.01: conf_p_symb = '**'
	if conf_p < 0.001: conf_p_symb = '***'
	if conf_p < 0.0001: conf_p_symb = '****'
	if diff_p >= 0.05: diff_p_symb = 'ns'
	if diff_p < 0.05: diff_p_symb = '*'
	if diff_p < 0.01: diff_p_symb = '**'
	if diff_p < 0.001: diff_p_symb = '***'
	if diff_p < 0.0001: diff_p_symb = '****'
	# Plot AUC
	ax = plt.subplot(111)
	x_points_BE = [-0.12,0.48]
	x_points_DCE = [0.12,0.72]
	ind_bar_width = 0.17
	plt.bar(x_points_BE, [choice_BE_AUC_mn, conf_BE_AUC_mn], yerr=[choice_BE_AUC_se, conf_BE_AUC_se], width=0.17, color='black')
	plt.bar(x_points_DCE, [choice_DCE_AUC_mn, conf_DCE_AUC_mn], yerr=[choice_DCE_AUC_se, conf_DCE_AUC_se], width=0.17, color='gray')
	plt.xlim([-0.3,0.9])
	plt.xticks([0,0.6], ['Decision','Confidence'], fontsize=18)
	plt.ylim([0,1])
	plt.yticks([0,0.2,0.4,0.6,0.8,1],['0','0.2','0.4','0.6','0.8','1'], fontsize=18)
	plt.ylabel('Choice probability', fontsize=18)
	ax.spines['right'].set_visible(False)
	ax.spines['top'].set_visible(False)
	ax.yaxis.set_ticks_position('left')
	ax.xaxis.set_ticks_position('bottom')
	lgd = ax.legend(['BE', 'RCE'], bbox_to_anchor=(0.4,1.55), fontsize=18, frameon=False)
	plt.plot([-0.3,1],[0.5,0.5],color='lightgray',linestyle='dashed',alpha=0.5)
	# Significance
	max_y_val = np.array([choice_BE_AUC_mn + choice_BE_AUC_se,
	 					  conf_BE_AUC_mn + conf_BE_AUC_se,
	 		  			  choice_DCE_AUC_mn + choice_DCE_AUC_se,
	 		  			  conf_DCE_AUC_mn + conf_DCE_AUC_se]).max()
	y_start = max_y_val + 0.02
	y_end = max_y_val + 0.05
	plt.plot([x_points_BE[0],x_points_BE[0],x_points_DCE[0],x_points_DCE[0]],[y_start,y_end,y_end,y_start],color='black', clip_on=False)
	plt.text(0,y_end+0.02,choice_p_symb,fontsize=18,horizontalalignment='center')
	plt.plot([x_points_BE[1],x_points_BE[1],x_points_DCE[1],x_points_DCE[1]],[y_start,y_end,y_end,y_start],color='black', clip_on=False)
	plt.text(0.6,y_end+0.02,conf_p_symb,fontsize=18,horizontalalignment='center')
	y_start = y_end + 0.1
	y_end = y_start + 0.03
	plt.plot([0,0,0.6,0.6],[y_start,y_end,y_end,y_start],color='black', clip_on=False)
	plt.text(0.3,y_end+0.02,diff_p_symb,fontsize=18,horizontalalignment='center')
	plt.text(0.3,y_end-0.07,'Interaction',fontsize=18,horizontalalignment='center')
	# Save
	plot_fname = './' + args.decoder_input + '/noise=' + str(args.decoder_noise) + '/decoding_AUC.png'
	plt.savefig(plot_fname, bbox_extra_artists=(lgd,), bbox_inches='tight', dpi=300)
	plt.close()

def main():

	# Settings
	parser = argparse.ArgumentParser()
	parser.add_argument('--N_runs', type=int, default=100)
	parser.add_argument('--decoder_input', type=str, default='whole_network', help="{'whole_network', 'z'}")
	parser.add_argument('--decoder_noise', type=float, default=0.0)
	args = parser.parse_args()

	# Plot 
	plot(args)

if __name__ == '__main__':
	main()