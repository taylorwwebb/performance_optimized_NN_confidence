import numpy as np
from sklearn.decomposition import PCA
import argparse

def main():

	# Settings
	parser = argparse.ArgumentParser()
	parser.add_argument('--train_regime', type=str, default='standard', help="{'standard', 'fixed_mu'}")
	parser.add_argument('--epochs', type=int, default=20)
	parser.add_argument('--run', type=str, default='1')
	args = parser.parse_args()

	# Load latent representations
	vis_dir = './vis_z/' + args.train_regime + '_training/' +  'run' + str(args.run) + '/'
	latent_reps_fname = vis_dir + 'epoch_' + str(args.epochs) + '.npz'
	latent_reps = np.load(latent_reps_fname)
	z_s1 = latent_reps['z_mn'][latent_reps['y_targ']==0]
	z_s2 = latent_reps['z_mn'][latent_reps['y_targ']==1]

	# Fit gaussians
	# s1
	z_s1_mn = z_s1.mean(0)
	z_s1_cov = np.cov(z_s1, rowvar=False)
	# s2
	z_s2_mn = z_s2.mean(0)
	z_s2_cov = np.cov(z_s2, rowvar=False)

	# Calculate variance along major and minor axes, get angular difference between distributions
	# s1
	z_s1_pca = PCA()
	z_s1_pca.fit(z_s1)
	z_s1_var_explained = z_s1_pca.explained_variance_ratio_
	z_s1_pc = z_s1_pca.components_
	z_s1_width = z_s1_pca.transform(z_s1)[:,0].std()
	z_s1_height = z_s1_pca.transform(z_s1)[:,1].std()
	z_s1_theta = np.degrees(np.arccos(np.dot(z_s1_pc[0,:], np.array([1,0]))))
	if z_s1_pc[0,1] < 0:
		z_s1_theta = 360 - z_s1_theta
	# s2
	z_s2_pca = PCA()
	z_s2_pca.fit(z_s2)
	z_s2_var_explained = z_s2_pca.explained_variance_ratio_
	z_s2_pc = z_s2_pca.components_
	z_s2_width = z_s2_pca.transform(z_s2)[:,0].std()
	z_s2_height = z_s2_pca.transform(z_s2)[:,1].std()
	z_s2_theta = np.degrees(np.arccos(np.dot(z_s2_pc[0,:], np.array([1,0]))))
	if z_s2_pc[0,1] < 0:
		z_s2_theta = 360 - z_s2_theta
	# Angular difference between s1 and s2
	theta_diff = np.degrees(np.arccos(np.dot(z_s1_pc[0,:], z_s2_pc[0,:])))

	# Save distribution parameters
	dist_param_fname = 'trained_models/' + args.train_regime + '_training/' + 'run' + args.run + '/training_dist_params.npz'
	np.savez(dist_param_fname, z_s1_mn=z_s1_mn, z_s1_cov=z_s1_cov, z_s2_mn=z_s2_mn, z_s2_cov=z_s2_cov,
							   z_s1_width=z_s1_width, z_s1_height=z_s1_height, z_s2_width=z_s2_width, z_s2_height=z_s2_height,
							   z_s1_var_explained=z_s1_var_explained, z_s1_pc=z_s1_pc, z_s2_var_explained=z_s2_var_explained, z_s2_pc=z_s2_pc, theta_diff=theta_diff)

if __name__ == '__main__':
	main()