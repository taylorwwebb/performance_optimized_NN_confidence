import numpy as np
import matplotlib.pyplot as plt
from scipy.stats import sem
from scipy.stats import gaussian_kde
import imageio
import sys
import os

# Prevent python from saving out .pyc files
sys.dont_write_bytecode = True

from fit_meta_d_MLE import *

def check_path(path):
	if not os.path.exists(path):
		os.mkdir(path)

def compute_sdt_metrics(y_pred, y_targ, trial_conf):
	# Determine confidence thresholds based on actual distribution of confidence values
	trial_conf_sorted = np.sort(trial_conf)
	N_trials = trial_conf.shape[0]
	conf2_thresh = trial_conf_sorted[int(0.25*N_trials)]
	conf3_thresh = trial_conf_sorted[int(0.5*N_trials)]
	conf4_thresh = trial_conf_sorted[int(0.75*N_trials)]
	# Sort trials based on confidence thresholds
	conf1_trials = trial_conf < conf2_thresh
	conf2_trials = np.logical_and(trial_conf >= conf2_thresh, trial_conf < conf3_thresh) 
	conf3_trials = np.logical_and(trial_conf >= conf3_thresh, trial_conf < conf4_thresh) 
	conf4_trials = trial_conf >= conf4_thresh
	# Sort trials based on y target and prediction
	s1 = y_targ == 0
	s2 = y_targ == 1
	resp_s1 = y_pred == 0
	resp_s2 = y_pred == 1
	s1_resp_s1 = np.logical_and(s1, resp_s1)
	s1_resp_s2 = np.logical_and(s1, resp_s2)
	s2_resp_s1 = np.logical_and(s2, resp_s1)
	s2_resp_s2 = np.logical_and(s2, resp_s2)
	# Combine confidence rating and y target/response
	s1_resp_s1_conf1 = np.logical_and(s1_resp_s1, conf1_trials) 
	s1_resp_s1_conf2 = np.logical_and(s1_resp_s1, conf2_trials)
	s1_resp_s1_conf3 = np.logical_and(s1_resp_s1, conf3_trials)
	s1_resp_s1_conf4 = np.logical_and(s1_resp_s1, conf4_trials)
	s1_resp_s2_conf1 = np.logical_and(s1_resp_s2, conf1_trials)
	s1_resp_s2_conf2 = np.logical_and(s1_resp_s2, conf2_trials)
	s1_resp_s2_conf3 = np.logical_and(s1_resp_s2, conf3_trials)
	s1_resp_s2_conf4 = np.logical_and(s1_resp_s2, conf4_trials)
	s2_resp_s1_conf1 = np.logical_and(s2_resp_s1, conf1_trials)
	s2_resp_s1_conf2 = np.logical_and(s2_resp_s1, conf2_trials)
	s2_resp_s1_conf3 = np.logical_and(s2_resp_s1, conf3_trials)
	s2_resp_s1_conf4 = np.logical_and(s2_resp_s1, conf4_trials)
	s2_resp_s2_conf1 = np.logical_and(s2_resp_s2, conf1_trials)
	s2_resp_s2_conf2 = np.logical_and(s2_resp_s2, conf2_trials)
	s2_resp_s2_conf3 = np.logical_and(s2_resp_s2, conf3_trials)
	s2_resp_s2_conf4 = np.logical_and(s2_resp_s2, conf4_trials)
	nR_s1 = [s1_resp_s1_conf4.sum() + 1/8, s1_resp_s1_conf3.sum() + 1/8, s1_resp_s1_conf2.sum() + 1/8, s1_resp_s1_conf1.sum() + 1/8,
			 s1_resp_s2_conf1.sum() + 1/8, s1_resp_s2_conf2.sum() + 1/8, s1_resp_s2_conf3.sum() + 1/8, s1_resp_s2_conf4.sum() + 1/8]
	nR_s2 = [s2_resp_s1_conf4.sum() + 1/8, s2_resp_s1_conf3.sum() + 1/8, s2_resp_s1_conf2.sum() + 1/8, s2_resp_s1_conf1.sum() + 1/8,
			 s2_resp_s2_conf1.sum() + 1/8, s2_resp_s2_conf2.sum() + 1/8, s2_resp_s2_conf3.sum() + 1/8, s2_resp_s2_conf4.sum() + 1/8]
	# Compute overall d' and meta-d'
	fit = fit_meta_d_MLE(nR_s1, nR_s2)
	d = fit['da']
	meta_d = fit['meta_da']
	return d, meta_d

# Define stimulus distributions
s1_mn = -1
s2_mn = 1
mns = np.array([s1_mn, s2_mn])
var = 1

# Sample stimuli
N_samples = 100000
y_targ = np.random.rand(N_samples).round().astype(np.int)
x_mn = mns[y_targ]
x = x_mn + (np.random.normal(size=N_samples) * var)
y_pred = (x > 0).astype(np.int)
correct_preds = (y_pred==y_targ).astype(np.int)
abs_x = np.abs(x)

# Get d' and meta-d' across TMS conditions (TMS applied after rectification)
TMS_range = [0,1.5]
N_TMS = 6
TMS_vals = np.linspace(TMS_range[0],TMS_range[-1],N_TMS)
all_d = []
all_meta_d = []
for t in range(N_TMS):
	# Apply TMS
	TMS_noise_var = TMS_vals[t]
	TMS_noise = np.random.normal(size=N_samples) * TMS_noise_var
	x_TMS = x + TMS_noise
	y_pred_TMS = (x_TMS > 0).astype(np.int)
	abs_x_TMS = abs_x + TMS_noise
	d, meta_d = compute_sdt_metrics(y_pred_TMS, y_targ, abs_x_TMS)
	# Collect results
	all_d.append(d)
	all_meta_d.append(meta_d)

# TMS condition colors
min_TMS_color = [0, 0.2, 0.4]
max_TMS_color = [0.4, 0.55, 0.7]
N_TMS_conditions = N_TMS - 1
all_TMS_colors = np.linspace(np.array(min_TMS_color),np.array(max_TMS_color),N_TMS_conditions)
# Plot d' vs. meta-d'
axis_fontsize = 18
tick_fontsize = 16
legend_fontsize = 16
title_fontsize = 18
ax = plt.subplot(111)
ax.plot([0,3],[0,3],color='gray',linestyle='dashed',alpha=0.5, label="meta-d'=d'")
ax.scatter(all_d[0], all_meta_d[0], color='black', label='Control')
ax.scatter(all_d[1], all_meta_d[1], color=all_TMS_colors[0,:], label='TMS (' + r'$\xi$'  + '=' + str(TMS_vals[1]) + ')')
ax.scatter(all_d[-1], all_meta_d[-1], color=all_TMS_colors[-1,:], label='TMS (' + r'$\xi$'  + '=' + str(TMS_vals[-1]) + ')')
ax.plot(all_d, all_meta_d, color='gray', alpha=0.2, linestyle=':', label='_Hidden')
for t in range(1,N_TMS):
	ax.scatter(all_d[t], all_meta_d[t], color=all_TMS_colors[t-1,:], label='_Hidden')
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
plt.legend(frameon=False, fontsize=legend_fontsize, loc=2)
plt.title('SDT simulation of TMS', fontsize=title_fontsize)
output_dir = './SDT_sim_results/'
check_path(output_dir)
plot_fname = output_dir + 'SDT_TMS_d_meta_d.png'
plt.savefig(plot_fname, bbox_inches='tight', dpi=300)
plt.close()

# TMS
TMS_noise_var = 1.5
TMS_noise = np.random.normal(size=N_samples) * TMS_noise_var
x_TMS = x + TMS_noise
y_pred_TMS = (x_TMS > 0).astype(np.int)
abs_x_TMS = abs_x + TMS_noise

# Plot stimulus distributions vs. x
x_min = -8
x_max = 8
x_range = np.linspace(x_min,x_max,1000)
x_s1_kernel = gaussian_kde(x[y_targ==0])
x_s1_dist = x_s1_kernel(x_range)
x_s2_kernel = gaussian_kde(x[y_targ==1])
x_s2_dist = x_s2_kernel(x_range)
ax = plt.subplot(2,1,1)
plt.fill(np.append(x_range, x_range[0]),np.append(x_s1_dist, x_s1_dist[0]),color='turquoise',alpha=0.5,edgecolor=None)
plt.fill(np.append(x_range, x_range[0]),np.append(x_s2_dist, x_s2_dist[0]),color='salmon',alpha=0.5,edgecolor=None)
plt.xlabel('Decision variable',fontsize=axis_fontsize)
plt.ylabel('Density estimate',fontsize=axis_fontsize)
plt.title('Control', fontsize=title_fontsize)
plt.xlim([x_min, x_max])
plt.xticks([-8,-4,0,4,8],['-8','-4','0','4','8'],fontsize=tick_fontsize)
plt.ylim([0,0.5])
plt.yticks([0,0.2,0.4],['0','0.2','0.4'],fontsize=tick_fontsize)
ax.spines['right'].set_visible(False)
ax.spines['top'].set_visible(False)
ax.yaxis.set_ticks_position('left')
ax.xaxis.set_ticks_position('bottom')
plt.tight_layout()
# Plot stimulus distributions vs. TMS(x)
x_min = -8
x_max = 8
x_range = np.linspace(x_min,x_max,1000)
x_TMS_s1_kernel = gaussian_kde(x_TMS[y_targ==0])
x_TMS_s1_dist = x_TMS_s1_kernel(x_range)
x_TMS_s2_kernel = gaussian_kde(x_TMS[y_targ==1])
x_TMS_s2_dist = x_TMS_s2_kernel(x_range)
ax = plt.subplot(2,1,2)
plt.fill(np.append(x_range, x_range[0]),np.append(x_TMS_s1_dist, x_TMS_s1_dist[0]),color='turquoise',alpha=0.5,edgecolor=None)
plt.fill(np.append(x_range, x_range[0]),np.append(x_TMS_s2_dist, x_TMS_s2_dist[0]),color='salmon',alpha=0.5,edgecolor=None)
plt.xlabel('Decision variable',fontsize=axis_fontsize)
plt.ylabel('Density estimate',fontsize=axis_fontsize)
plt.title('TMS (' + r'$\xi$'  + '=' + str(TMS_noise_var) + ')', fontsize=title_fontsize)
plt.xlim([x_min, x_max])
plt.xticks([-8,-4,0,4,8],['-8','-4','0','4','8'],fontsize=tick_fontsize)
plt.ylim([0,0.5])
plt.yticks([0,0.2,0.4],['0','0.2','0.4'],fontsize=tick_fontsize)
ax.spines['right'].set_visible(False)
ax.spines['top'].set_visible(False)
ax.yaxis.set_ticks_position('left')
ax.xaxis.set_ticks_position('bottom')
plt.tight_layout()
plt.legend(['s1','s2'],frameon=False, fontsize=legend_fontsize)
plt.savefig(output_dir + 'x_vs_s1s2.png', bbox_inches='tight')
plt.close()

# Plot correct/incorrect vs. |x|
x_min = -8
x_max = 8
x_range = np.linspace(x_min,x_max,1000)
abs_x = np.abs(x)
abs_x_correct_kernel = gaussian_kde(abs_x[correct_preds==1])
abs_x_correct_dist = abs_x_correct_kernel(x_range)
abs_x_incorrect_kernel = gaussian_kde(abs_x[correct_preds==0])
abs_x_incorrect_dist = abs_x_incorrect_kernel(x_range)
ax = plt.subplot(2,1,1)
plt.fill(np.append(x_range, x_range[0]),np.append(abs_x_correct_dist, abs_x_correct_dist[0]),color='lightgreen',alpha=0.5,edgecolor=None)
plt.fill(np.append(x_range, x_range[0]),np.append(abs_x_incorrect_dist, abs_x_incorrect_dist[0]),color='slateblue',alpha=0.5,edgecolor=None)
plt.xlabel('|Decision variable|',fontsize=axis_fontsize)
plt.ylabel('Density estimate',fontsize=axis_fontsize)
plt.title('Control', fontsize=title_fontsize)
plt.xlim([x_min, x_max])
plt.xticks([-8,-4,0,4,8],['-8','-4','0','4','8'],fontsize=tick_fontsize)
plt.ylim([0,1.4])
plt.yticks([0,0.6,1.2],['0','0.6','1.2'],fontsize=tick_fontsize)
ax.spines['right'].set_visible(False)
ax.spines['top'].set_visible(False)
ax.yaxis.set_ticks_position('left')
ax.xaxis.set_ticks_position('bottom')
plt.tight_layout()
# Plot correct/incorrect vs. TMS(|x|)
x_min = -8
x_max = 8
x_range = np.linspace(x_min,x_max,1000)
abs_x_TMS_correct_kernel = gaussian_kde(abs_x_TMS[correct_preds==1])
abs_x_TMS_correct_dist = abs_x_TMS_correct_kernel(x_range)
abs_x_TMS_incorrect_kernel = gaussian_kde(abs_x_TMS[correct_preds==0])
abs_x_TMS_incorrect_dist = abs_x_TMS_incorrect_kernel(x_range)
ax = plt.subplot(2,1,2)
plt.fill(np.append(x_range, x_range[0]),np.append(abs_x_TMS_correct_dist, abs_x_TMS_correct_dist[0]),color='lightgreen',alpha=0.5,edgecolor=None)
plt.fill(np.append(x_range, x_range[0]),np.append(abs_x_TMS_incorrect_dist, abs_x_TMS_incorrect_dist[0]),color='slateblue',alpha=0.5,edgecolor=None)
plt.xlabel('|Decision variable|',fontsize=axis_fontsize)
plt.ylabel('Density estimate',fontsize=axis_fontsize)
plt.title('TMS (' + r'$\xi$'  + '=' + str(TMS_noise_var) + ')', fontsize=title_fontsize)
plt.xlim([x_min, x_max])
plt.xticks([-8,-4,0,4,8],['-8','-4','0','4','8'],fontsize=tick_fontsize)
plt.ylim([0,1.3])
plt.yticks([0,0.6,1.2],['0','0.6','1.2'],fontsize=tick_fontsize)
ax.spines['right'].set_visible(False)
ax.spines['top'].set_visible(False)
ax.yaxis.set_ticks_position('left')
ax.xaxis.set_ticks_position('bottom')
plt.tight_layout()
plt.legend(['Correct','Incorrect'],frameon=False, fontsize=legend_fontsize)
plt.savefig(output_dir + 'abs_x_vs_correct_incorrect.png', bbox_inches='tight')
plt.close()


