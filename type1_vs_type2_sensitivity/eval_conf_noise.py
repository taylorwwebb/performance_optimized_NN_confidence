import numpy as np
from scipy.special import logit
from scipy.special import expit
from sklearn.metrics import mean_squared_error
import argparse
import sys

# Prevent python from saving out .pyc files
sys.dont_write_bytecode = True

from fit_meta_d_MLE import *
from fit_rs_meta_d_MLE import *
from util import log

def compute_sensitivity(y_pred, y_targ, trial_conf):
	# Sort trials based on confidence thresholds
	conf2_thresh = 0.5 + (0.5/4)
	conf3_thresh = 0.5 + 2*(0.5/4)
	conf4_thresh = 0.5 + 3*(0.5/4)
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
	# Compute response-specific meta-d'
	fit_rs = fit_rs_meta_d_MLE(nR_s1, nR_s2)
	meta_d_rS1 = fit_rs['meta_da_rS1']
	meta_d_rS2 = fit_rs['meta_da_rS2']
	return d, meta_d, meta_d_rS1, meta_d_rS2

def plot(args):
	all_runs_d = []
	all_runs_meta_d = []
	all_runs_meta_d_rS1 = []
	all_runs_meta_d_rS2 = []
	for r in range(args.N_runs):
		print('run ' + str(r+1) + '...')
		# Run directory
		run_dir = './test/run' + str(r+1) + '/'
		# Load test results
		test_results = np.load(run_dir + 'test_results.npz')
		y_pred = test_results['all_y_pred']
		y_targ = test_results['all_y_targ']
		conf = test_results['all_conf']
		# Add noise to confidence
		eps = 1e-5
		conf[conf==1] -= eps
		conf[conf==0] += eps
		conf_pre_sigmoid = logit(conf)
		conf_pre_sigmoid += np.random.normal(loc=0.0,scale=args.conf_noise,size=conf_pre_sigmoid.shape)
		conf = expit(conf_pre_sigmoid)
		# Compute SDT metrics w/ confidence noise
		all_d = []
		all_meta_d = []
		all_meta_d_rS1 = []
		all_meta_d_rS2 = []
		for i in range(conf.shape[0]):
			d, meta_d, meta_d_rS1, meta_d_rS2 = compute_sensitivity(y_pred[i,:], y_targ[i,:], conf[i,:])
			log.info('[i = ' + str(i+1) + '] ' + \
			 	 '[d-prime = ' + '{:.2f}'.format(d) + '] ' + \
			 	 '[meta-d-prime = ' + '{:.2f}'.format(meta_d) + '] ' + \
			 	 '[rS1 meta-d-prime = ' + '{:.2f}'.format(meta_d_rS1) + '] ' + \
			 	 '[rS2 meta-d-prime = ' + '{:.2f}'.format(meta_d_rS2) + ']')
			all_d.append(d)
			all_meta_d.append(meta_d)
			all_meta_d_rS1.append(meta_d_rS1)
			all_meta_d_rS2.append(meta_d_rS2)
		# Save
		conf_noise_fname = run_dir + 'test_results_conf_noise_' + str(args.conf_noise) + '.npz'
		np.savez(conf_noise_fname,
				 all_d=np.array(all_d),
				 all_meta_d=np.array(all_meta_d),
				 all_meta_d_rS1=np.array(all_meta_d_rS1),
				 all_meta_d_rS2=np.array(all_meta_d_rS2))
		# Collect results for all runs
		all_runs_d.append(all_d)
		all_runs_meta_d.append(all_meta_d)
		all_runs_meta_d_rS1.append(all_meta_d_rS1)
		all_runs_meta_d_rS2.append(all_meta_d_rS2)
	# Meta-d' averages
	meta_d_mn = np.array(all_runs_meta_d).mean(0)
	meta_d_rS1_mn = np.array(all_runs_meta_d_rS1).mean(0)
	meta_d_rS2_mn = np.array(all_runs_meta_d_rS2).mean(0)
	# Compute meta-d' error
	behavioral_data = np.load('./test/behavioral_data.npz')
	meta_d_mn_behavior = behavioral_data['meta_d_mn']
	meta_d_rS1_mn_behavior = behavioral_data['meta_d_s1_mn']
	meta_d_rS2_mn_behavior = behavioral_data['meta_d_s2_mn']
	all_behavior_meta_d = np.concatenate([meta_d_mn_behavior, meta_d_rS1_mn_behavior, meta_d_rS2_mn_behavior])
	all_model_conf_noise_meta_d = np.concatenate([meta_d_mn, meta_d_rS1_mn, meta_d_rS2_mn])
	meta_d_error = mean_squared_error(all_behavior_meta_d, all_model_conf_noise_meta_d)
	mse_fname = './test/meta_d_mse_noise_' + str(args.conf_noise) + '.npz'
	np.savez(mse_fname, mse=meta_d_error, conf_noise=args.conf_noise)

def main():

	# Settings
	parser = argparse.ArgumentParser()
	parser.add_argument('--N_runs', type=int, default=100)
	parser.add_argument('--conf_noise', type=float, default=1.2)
	args = parser.parse_args()

	# Plot 
	plot(args)

if __name__ == '__main__':
	main()