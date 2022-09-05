import numpy as np
from scipy.stats import sem, ttest_ind, ttest_1samp, multivariate_normal
import matplotlib.pyplot as plt
from matplotlib.patches import Ellipse
import math
import argparse

def main():

	# Settings
	parser = argparse.ArgumentParser()
	parser.add_argument('--N_runs', type=int, default=100)
	args = parser.parse_args()

	# Get parameters for standard training regime
	model_dir = './trained_models/standard_training/' 
	all_standard_theta_diff = []
	all_standard_sigma_ratio = []
	all_standard_width = []
	all_standard_height = []
	all_standard_mn_diff = []
	for r in range(args.N_runs):
		# Load file
		params_fname = model_dir + 'run' + str(r+1) + '/training_dist_params.npz'
		params = np.load(params_fname)
		# Angular difference between z_s1 and z_s2
		theta_diff = params['theta_diff']
		if theta_diff > 90:
			theta_diff = 90 - (theta_diff - 90)
		all_standard_theta_diff.append(theta_diff)
		# Ratio of variance in target / nontarget dimensions
		z_s1_sigma_ratio = params['z_s1_var_explained'][0] / params['z_s1_var_explained'][1]
		z_s2_sigma_ratio = params['z_s2_var_explained'][0] / params['z_s2_var_explained'][1]
		z_sigma_ratio = np.mean([z_s1_sigma_ratio, z_s2_sigma_ratio])
		all_standard_sigma_ratio.append(z_sigma_ratio)
		# Width, height, and mean differences
		all_standard_width.append(params['z_s1_width'])
		all_standard_width.append(params['z_s2_width'])
		all_standard_height.append(params['z_s1_height'])
		all_standard_height.append(params['z_s2_height'])
		all_standard_mn_diff.append(np.sqrt(((params['z_s2_mn'] - params['z_s1_mn'])**2).sum()))
	all_standard_theta_diff = np.array(all_standard_theta_diff)
	all_standard_sigma_ratio = np.array(all_standard_sigma_ratio)
	all_standard_width = np.array(all_standard_width)
	all_standard_height = np.array(all_standard_height)
	all_standard_mn_diff = np.array(all_standard_mn_diff)
	# Get parameters for fixed contrast training regime
	model_dir = './trained_models/fixed_mu_training/'
	all_fixed_mu_theta_diff = []
	all_fixed_mu_sigma_ratio = []
	all_fixed_mu_width = []
	all_fixed_mu_height = []
	all_fixed_mu_mn_diff = []
	for r in range(args.N_runs):
		# Load file
		params_fname = model_dir + 'run' + str(r+1) + '/training_dist_params.npz'
		params = np.load(params_fname)
		# Angular difference between z_s1 and z_s2
		theta_diff = params['theta_diff']
		if theta_diff > 90:
			theta_diff = 90 - (theta_diff - 90)
		all_fixed_mu_theta_diff.append(theta_diff)
		# Ratio of variance in target / nontarget dimensions
		z_s1_sigma_ratio = params['z_s1_var_explained'][0] / params['z_s1_var_explained'][1]
		z_s2_sigma_ratio = params['z_s2_var_explained'][0] / params['z_s2_var_explained'][1]
		z_sigma_ratio = np.mean([z_s1_sigma_ratio, z_s2_sigma_ratio])
		all_fixed_mu_sigma_ratio.append(z_sigma_ratio)
		# Width, height, and mean differences
		all_fixed_mu_width.append(params['z_s1_width'])
		all_fixed_mu_width.append(params['z_s2_width'])
		all_fixed_mu_height.append(params['z_s1_height'])
		all_fixed_mu_height.append(params['z_s2_height'])
		all_fixed_mu_mn_diff.append(np.sqrt(((params['z_s2_mn'] - params['z_s1_mn'])**2).sum()))
	all_fixed_mu_theta_diff = np.array(all_fixed_mu_theta_diff)
	all_fixed_mu_sigma_ratio = np.array(all_fixed_mu_sigma_ratio)
	all_fixed_mu_width = np.array(all_fixed_mu_width)
	all_fixed_mu_height = np.array(all_fixed_mu_height)
	all_fixed_mu_mn_diff = np.array(all_fixed_mu_mn_diff)

	# Stats
	theta_diff_t, theta_diff_p = ttest_ind(all_standard_theta_diff, all_fixed_mu_theta_diff)
	sigma_ratio_t, sigma_ratio_p = ttest_ind(all_standard_sigma_ratio, all_fixed_mu_sigma_ratio)
	standard_theta_t, standard_theta_p = ttest_1samp(all_standard_theta_diff, 0)
	fixed_mu_theta_t, fixed_mu_theta_p = ttest_1samp(all_fixed_mu_theta_diff, 0)
	standard_sigma_ratio_t, standard_sigma_ratio_p = ttest_1samp(all_standard_sigma_ratio, 1)
	fixed_mu_sigma_ratio_t, fixed_mu_sigma_ratio_p = ttest_1samp(all_fixed_mu_sigma_ratio, 1)
	# Save stats
	fname = './gaussian_fit_stats.txt'
	fid = open(fname, 'w')
	fid.write('standard, theta_s2 - theta_s1 = ' + str(np.mean(all_standard_theta_diff)) + ' +/- ' + str(sem(all_standard_theta_diff)) + '\n')
	fid.write('t-test (>0): t = ' + str(standard_theta_t) + ', p = ' + str(standard_theta_p) + '\n')
	fid.write('\n')
	fid.write('fixed-mu, theta_s2 - theta_s1 = ' + str(np.mean(all_fixed_mu_theta_diff)) + ' +/- ' + str(sem(all_fixed_mu_theta_diff)) + '\n')
	fid.write('t-test (>0): t = ' + str(fixed_mu_theta_t) + ', p = ' + str(fixed_mu_theta_p) + '\n')
	fid.write('\n')
	fid.write('standard, sigma_targ / sigma_nontarg = ' + str(np.mean(all_standard_sigma_ratio)) + ' +/- ' + str(sem(all_standard_sigma_ratio)) + '\n')
	fid.write('t-test (>1): t = ' + str(standard_sigma_ratio_t) + ', p = ' + str(standard_sigma_ratio_p) + '\n')
	fid.write('\n')
	fid.write('fixed-mu, sigma_targ / sigma_nontarg = ' + str(np.mean(all_fixed_mu_sigma_ratio)) + ' +/- ' + str(sem(all_fixed_mu_sigma_ratio)) + '\n')
	fid.write('t-test (>1): t = ' + str(fixed_mu_sigma_ratio_t) + ', p = ' + str(fixed_mu_sigma_ratio_p) + '\n')
	fid.write('\n')
	fid.write('standard vs. fixed-mu, theta_s2 - theta_s1:\n')
	fid.write('    t = ' + str(theta_diff_t) + ', p = ' + str(theta_diff_p) + '\n')
	fid.write('\n')
	fid.write('standard vs. fixed-mu, sigma_targ / sigma_nontarg:\n')
	fid.write('    t = ' + str(sigma_ratio_t) + ', p = ' + str(sigma_ratio_p) + '\n')
	fid.close()

	## Create summary visualization of ideal observer model for both standard and fixed mu regimes
	# Evaluation range
	x_min = -3
	x_max = 3
	x_int = 0.01
	x_range = np.arange(x_min,x_max+x_int,x_int)
	x, y = np.mgrid[x_min:x_max:x_int, x_min:x_max:x_int]
	pos = np.dstack((x, y))
	# Font sizes
	axis_label_fontsize = 18
	tick_fontsize = 16

	## Standard regime
	# Parameters
	targ_mu = all_standard_mn_diff.mean() / np.sqrt(2)
	targ_sigma = all_standard_width.mean()
	theta_diff = all_standard_theta_diff.mean()
	nontarg_mu = 0
	nontarg_sigma = all_standard_height.mean()
	# Generate stimulus distributions and p(correct|x)
	theta_s1 = math.radians(90 - (45 + theta_diff / 2))
	rotation_matrix_s1 = np.array([[np.cos(theta_s1), -1 * np.sin(theta_s1)], [np.sin(theta_s1), np.cos(theta_s1)]])
	cov_s1 = np.array([[targ_sigma, 0], [0, nontarg_sigma]])
	cov_s1 = rotation_matrix_s1 @ cov_s1 @ rotation_matrix_s1.T
	s1_dist = multivariate_normal(mean=[targ_mu, nontarg_mu], cov=cov_s1)
	s1_pdf = s1_dist.pdf(pos)
	theta_s2 = math.radians(-1 * (90 - (45 + theta_diff / 2)))
	rotation_matrix_s2 = np.array([[np.cos(theta_s2), -1 * np.sin(theta_s2)], [np.sin(theta_s2), np.cos(theta_s2)]])
	cov_s2 = np.array([[nontarg_sigma, 0], [0, targ_sigma]])
	cov_s2 = rotation_matrix_s2 @ cov_s2 @ rotation_matrix_s2.T
	s2_dist = multivariate_normal(mean=[nontarg_mu, targ_mu], cov=cov_s2)
	s2_pdf = s2_dist.pdf(pos)
	p_s1 = (s1_pdf * 0.5) / ((s1_pdf * 0.5) + (s2_pdf * 0.5))
	p_s2 = (s2_pdf * 0.5) / ((s1_pdf * 0.5) + (s2_pdf * 0.5))
	p_correct = np.stack([p_s1,p_s2],0).max(0)
	# Indices and labels
	axis_int = 1
	axis_ticks = np.arange(x_min,x_max+axis_int,axis_int)
	axis_ticks_ind = []
	for i in range(axis_ticks.shape[0]):
		axis_ticks_ind.append(np.argmin(np.abs(x_range-axis_ticks[i])))
	axis_tick_labels = axis_ticks.astype(str)
	min_ind = np.argmin(np.abs(x_range-x_min))
	max_ind = np.argmin(np.abs(x_range-x_max))
	xlim_min = -1.5
	xlim_max = 3
	xlim_min_ind = np.argmin(np.abs(x_range-xlim_min))
	xlim_max_ind = np.argmin(np.abs(x_range-xlim_max))
	zero_ind = np.argmin(np.abs(x_range))
	# Plot
	ax = plt.subplot(111)
	plt.imshow(p_correct, origin='lower', vmin=0.5, vmax=1)
	cbar = plt.colorbar()
	cbar.set_ticks([0.5,0.6,0.7,0.8,0.9,1])
	cbar.ax.set_yticklabels(['0.5','0.6','0.7','0.8','0.9','1'])
	cbar.ax.tick_params(labelsize=tick_fontsize)
	cbar.set_label('Confidence', fontsize=axis_label_fontsize)
	plt.xlabel('$z_{2}$', fontsize=axis_label_fontsize)
	plt.ylabel('$z_{1}$', fontsize=axis_label_fontsize)
	plt.xticks(axis_ticks_ind, axis_tick_labels, fontsize=tick_fontsize)
	plt.yticks(axis_ticks_ind, axis_tick_labels, fontsize=tick_fontsize)
	plt.xlim([xlim_min_ind, xlim_max_ind])
	plt.ylim([xlim_min_ind, xlim_max_ind])
	# Add stimulus distributions
	axis_norm_factor = (max_ind - min_ind) / (x_max - x_min)
	N_SD = 3
	for sd in range(N_SD):
		s1_ellipse = Ellipse(xy=(np.array([nontarg_mu, targ_mu]) * axis_norm_factor) + zero_ind,width=targ_sigma*(sd+1)*axis_norm_factor,height=nontarg_sigma*(sd+1)*axis_norm_factor,angle=45+(theta_diff/2),color='black',fill=False,linewidth=1)
		ax.add_patch(s1_ellipse)
	for sd in range(N_SD):
		s2_ellipse = Ellipse(xy=(np.array([targ_mu, nontarg_mu]) * axis_norm_factor) + zero_ind,width=targ_sigma*(sd+1)*axis_norm_factor,height=nontarg_sigma*(sd+1)*axis_norm_factor,angle=45-(theta_diff/2),color='black',fill=False,linewidth=1)
		ax.add_patch(s2_ellipse)
	# Label stimulus distributions
	s1_x_loc = -1
	s1_y_loc = 2
	ax.text((s1_x_loc * axis_norm_factor) + zero_ind, (s1_y_loc * axis_norm_factor) + zero_ind,'s1',fontsize=axis_label_fontsize,horizontalalignment='center', verticalalignment='center')
	s2_x_loc = 2
	s2_y_loc = -1
	ax.text((s2_x_loc * axis_norm_factor) + zero_ind, (s2_y_loc * axis_norm_factor) + zero_ind,'s2',fontsize=axis_label_fontsize,horizontalalignment='center', verticalalignment='center')
	# Title
	plt.title('Latent ideal observer', fontsize=20)
	# Save
	plt.tight_layout()
	plt.savefig('./ideal_observer_summary.png', dpi=300)
	plt.close()

	## Fixed-mu regime
	# Parameters
	targ_mu = all_fixed_mu_mn_diff.mean() / np.sqrt(2)
	targ_sigma = all_fixed_mu_width.mean()
	theta_diff = all_fixed_mu_theta_diff.mean()
	nontarg_mu = 0
	nontarg_sigma = all_fixed_mu_height.mean()
	# Generate stimulus distributions and p(correct|x)
	theta_s1 = math.radians(90 - (45 + theta_diff / 2))
	rotation_matrix_s1 = np.array([[np.cos(theta_s1), -1 * np.sin(theta_s1)], [np.sin(theta_s1), np.cos(theta_s1)]])
	cov_s1 = np.array([[targ_sigma, 0], [0, nontarg_sigma]])
	cov_s1 = rotation_matrix_s1 @ cov_s1 @ rotation_matrix_s1.T
	s1_dist = multivariate_normal(mean=[targ_mu, nontarg_mu], cov=cov_s1)
	s1_pdf = s1_dist.pdf(pos)
	theta_s2 = math.radians(-1 * (90 - (45 + theta_diff / 2)))
	rotation_matrix_s2 = np.array([[np.cos(theta_s2), -1 * np.sin(theta_s2)], [np.sin(theta_s2), np.cos(theta_s2)]])
	cov_s2 = np.array([[nontarg_sigma, 0], [0, targ_sigma]])
	cov_s2 = rotation_matrix_s2 @ cov_s2 @ rotation_matrix_s2.T
	s2_dist = multivariate_normal(mean=[nontarg_mu, targ_mu], cov=cov_s2)
	s2_pdf = s2_dist.pdf(pos)
	p_s1 = (s1_pdf * 0.5) / ((s1_pdf * 0.5) + (s2_pdf * 0.5))
	p_s2 = (s2_pdf * 0.5) / ((s1_pdf * 0.5) + (s2_pdf * 0.5))
	p_correct = np.stack([p_s1,p_s2],0).max(0)
	# Indices and labels
	axis_int = 1
	axis_ticks = np.arange(x_min,x_max+axis_int,axis_int)
	axis_ticks_ind = []
	for i in range(axis_ticks.shape[0]):
		axis_ticks_ind.append(np.argmin(np.abs(x_range-axis_ticks[i])))
	axis_tick_labels = axis_ticks.astype(str)
	min_ind = np.argmin(np.abs(x_range-x_min))
	max_ind = np.argmin(np.abs(x_range-x_max))
	xlim_min = -1.5
	xlim_max = 3
	xlim_min_ind = np.argmin(np.abs(x_range-xlim_min))
	xlim_max_ind = np.argmin(np.abs(x_range-xlim_max))
	zero_ind = np.argmin(np.abs(x_range))
	# Plot
	ax = plt.subplot(111)
	plt.imshow(p_correct, origin='lower', vmin=0.5, vmax=1)
	cbar = plt.colorbar()
	cbar.set_ticks([0.5,0.6,0.7,0.8,0.9,1])
	cbar.ax.set_yticklabels(['0.5','0.6','0.7','0.8','0.9','1'])
	cbar.ax.tick_params(labelsize=tick_fontsize)
	cbar.set_label('Confidence', fontsize=axis_label_fontsize)
	plt.xlabel('$z_{2}$', fontsize=axis_label_fontsize)
	plt.ylabel('$z_{1}$', fontsize=axis_label_fontsize)
	plt.xticks(axis_ticks_ind, axis_tick_labels, fontsize=tick_fontsize)
	plt.yticks(axis_ticks_ind, axis_tick_labels, fontsize=tick_fontsize)
	plt.xlim([xlim_min_ind, xlim_max_ind])
	plt.ylim([xlim_min_ind, xlim_max_ind])
	# Add stimulus distributions
	axis_norm_factor = (max_ind - min_ind) / (x_max - x_min)
	N_SD = 3
	for sd in range(N_SD):
		s1_ellipse = Ellipse(xy=(np.array([nontarg_mu, targ_mu]) * axis_norm_factor) + zero_ind,width=targ_sigma*(sd+1)*axis_norm_factor,height=nontarg_sigma*(sd+1)*axis_norm_factor,angle=45+(theta_diff/2),color='black',fill=False,linewidth=1)
		ax.add_patch(s1_ellipse)
	for sd in range(N_SD):
		s2_ellipse = Ellipse(xy=(np.array([targ_mu, nontarg_mu]) * axis_norm_factor) + zero_ind,width=targ_sigma*(sd+1)*axis_norm_factor,height=nontarg_sigma*(sd+1)*axis_norm_factor,angle=45-(theta_diff/2),color='black',fill=False,linewidth=1)
		ax.add_patch(s2_ellipse)
	# Label stimulus distributions
	s1_x_loc = -1
	s1_y_loc = 2
	ax.text((s1_x_loc * axis_norm_factor) + zero_ind, (s1_y_loc * axis_norm_factor) + zero_ind,'s1',fontsize=axis_label_fontsize,horizontalalignment='center', verticalalignment='center')
	s2_x_loc = 2
	s2_y_loc = -1
	ax.text((s2_x_loc * axis_norm_factor) + zero_ind, (s2_y_loc * axis_norm_factor) + zero_ind,'s2',fontsize=axis_label_fontsize,horizontalalignment='center', verticalalignment='center')
	# Title
	plt.title('Latent ideal observer\n(fixed ' + r'$\mu$' + ' training)', fontsize=20)
	# Save
	plt.tight_layout()
	plt.savefig('./fixed_mu_ideal_observer.png', dpi=300)
	plt.close()

if __name__ == '__main__':
	main()
