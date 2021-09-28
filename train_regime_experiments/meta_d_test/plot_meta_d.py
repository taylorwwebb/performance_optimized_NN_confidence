import numpy as np
import matplotlib.pyplot as plt
from scipy.stats import sem
from scipy.stats import ttest_ind
import argparse

def plot(args):
	all_regimes = ['standard', 'fixed_sigma_mu_ratio', 'fixed_mu', 'fixed_sigma']
	all_meta_d = []
	for train_regime in all_regimes:
		meta_d = []
		for test_regime in all_regimes:
			all_runs_meta_d = []
			for r in range(args.N_runs):
				results_fname = './' + train_regime + '/run' + str(r+1) + '/' + 'all_test_regimes_meta_d.npz'
				results = np.load(results_fname)
				all_runs_meta_d.append(results[test_regime + '_meta_d'].item())
			meta_d.append(all_runs_meta_d)
		all_meta_d.append(meta_d)
	# Convert to array
	all_meta_d = np.array(all_meta_d)
	# Summary statistics
	all_meta_d_mn = all_meta_d.mean(2)
	all_meta_d_se = sem(all_meta_d,2)
	# Plot
	ax = plt.subplot(111)
	plt.bar([0,4+0.5,8+1,12+1.5],all_meta_d_mn[0,:],yerr=all_meta_d_se[0,:],width=0.8,color='red')
	plt.bar([1,5+0.5,9+1,13+1.5],all_meta_d_mn[1,:],yerr=all_meta_d_se[1,:],width=0.8,color='limegreen')
	plt.bar([2,6+0.5,10+1,14+1.5],all_meta_d_mn[2,:],yerr=all_meta_d_se[2,:],width=0.8,color='teal')
	plt.bar([3,7+0.5,11+1,15+1.5],all_meta_d_mn[3,:],yerr=all_meta_d_se[3,:],width=0.8,color='orange')
	plt.bar([0],all_meta_d_mn[0,0],yerr=all_meta_d_se[0,0],width=0.8,facecolor='None',hatch='//')
	plt.bar([5+0.5],all_meta_d_mn[1,1],yerr=all_meta_d_se[1,1],width=0.8,facecolor='None',hatch='//')
	plt.bar([10+1],all_meta_d_mn[2,2],yerr=all_meta_d_se[2,2],width=0.8,facecolor='None',hatch='//')
	plt.bar([15+1.5],all_meta_d_mn[3,3],yerr=all_meta_d_se[3,3],width=0.8,facecolor='None',hatch='//')
	plt.yticks([1.5,2,2.5,3],['1.5','2','2.5','3'], fontsize=16)
	plt.ylim([1.5,3])
	plt.xlabel('Test regime', fontsize=18)
	plt.xticks([1.5,5.5+0.5,9.5+1,13.5+1.5],['Standard','Fixed '+r'$\mu/\sigma$','Fixed '+r'$\mu$','Fixed '+r'$\sigma$'], fontsize=16)
	plt.ylabel("Meta-d'", fontsize=18)
	ax.spines['right'].set_visible(False)
	ax.spines['top'].set_visible(False)
	ax.yaxis.set_ticks_position('left')
	ax.xaxis.set_ticks_position('bottom')
	lgd = ax.legend(['Standard','Fixed '+r'$\mu/\sigma$','Fixed '+r'$\mu$','Fixed '+r'$\sigma$', 'Train=test'], bbox_to_anchor=(1.0,1), frameon=False, fontsize=16, title='Training regime', title_fontsize=16)
	plt.title('Training vs. test regime', fontsize=18)
	plt.savefig('./multi_paradigm_meta_d.png', bbox_extra_artists=(lgd,), bbox_inches='tight', dpi=300)
	plt.close()
	# T-tests
	fid = open('./multi_paradigm_meta_d_t_tests.txt', 'w')
	for test in range(len(all_regimes)):
		fid.write('Test regime: ' + all_regimes[test] + '\n')
		for train in range(len(all_regimes)):
			if train != test:
				fid.write('Train regime: ' + all_regimes[train] + '\n')
				t, p = ttest_ind(all_meta_d[test,test,:], all_meta_d[train,test,:])
				fid.write('t = ' + str(t) + '\n')
				fid.write('p = ' + str(p) + '\n')
		fid.write(' \n')
	fid.close()

def main():

	# Settings
	parser = argparse.ArgumentParser()
	parser.add_argument('--N_runs', type=int, default=200)
	args = parser.parse_args()

	# Plot
	plot(args)

if __name__ == '__main__':
	main()