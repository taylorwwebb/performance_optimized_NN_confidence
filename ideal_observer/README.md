## Latent ideal observer

To train VAEs on the standard training regime, and fit latent distributions, run:
```
./train_all_VAE.sh
./fit_gaussians_all_VAE.sh
```
To train VAEs on the fixed contrast training regime, and fit latent distributions, run:
```
./train_all_VAE_fixed_mu.sh
./fit_gaussians_all_VAE_fixed_mu.sh
```
To plot a summary of the resulting distributions, and compare the standard vs. fixed contrast regimes, run:
```
python3 ./plot_gaussian_fit_params_summary.py
```
### PE bias

To evaluate the ideal observer for the PE bias (version 1), run:
```
./eval_PE_bias_all_VAE.sh
```
To analyze the results, navigate to `./test/` and run:
```
./PE_test.py
./PE_test_correct_incorrect.py
```
To evaluate the ideal observer for the PE bias (version 2), run:
```
./eval_PE_bias_v2_all_VAE.sh
```
To analyze the results, navigate to `./test/` and run:
```
./PE_test_v2.py
./PE_test_v2_correct_incorrect.py
```
### Dissociation between type-1 and type-2 sensitivity

To evaluate the ideal observer for fitting intermediate contrast value (i=3), run:
```
./eval_i3_dprime_all_VAE.sh
```
To fit intermediate contrast, navigate to `./test/` and run:
```
python3 ./fit_i3.py
```
Alternatively, fitted intermediate contrast value can be loaded from `./test/standard_training/i3_fit.npz`.

To evaluate the ideal observer for fitting other contrast values (i=[1,2,4,5]), run:
```
./eval_i1245_dprime_all_VAE.sh
```
To fit contrast values, navigate to `./test/` and run:
```
python3 ./fit_i1245.py
```
Alternatively, fitted contrast values can be loaded from `./test/standard_training/i1245_fit.npz`.

To evaluate the ideal observer using fitted contrast values, run:
```
./eval_type1_vs_type2_sensitivity_all_VAE.sh
```
To evaluate the ideal observer for fitting type-2 noise parameter, run:
```
./eval_type1_vs_type2_sensitivity_conf_noise.sh
```
To fit type-2 noise parameter, navigate to `./test/` and run:
```
python3 ./fit_conf_noise.py
```
Alternatively, fitted type-2 noise parameter can be loaded from `./test/standard_training/conf_noise_fit.npz`.

To analyze results, navigate to `./test/`. For model results without type-2 noise, run:
```
python3 ./plot_d_vs_meta_d.py --model_behavior model
```
For model results with fitted type-2 noise, run:
```
python3 ./plot_d_vs_meta_d.py --model_behavior model_conf_noise_fit
```
To plot original behavioral results, run:
```
python3 ./plot_d_vs_meta_d.py --model_behavior behavior
```
### Analysis of confidence as a function of sensory evidence space

To analyze ideal observer confidence as a function of sensory evidence space, run:
```
./eval_s1s2_all_VAE.sh
```
Navigate to `./test/` and run:
```
python3 ./learned_rule_analysis.py
```
Move the resulting file, `./standard_training/ideal_observer_conf.npz`, to `../PCA/test/` to perform comparison with BE and RCE models.
