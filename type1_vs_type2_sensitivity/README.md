### Testing for dissociation between type-1 and type-2 sensitivity

To train multiple networks on two-choice variant of MNIST, run:
```
./train.sh
```
To evaluate networks for fitting intermediate contrast value (i=3), run:
```
./eval_i3.sh
```
To fit intermediate contrast, navigate to `./test/` and run:
```
python3 ./fit_i3.py
```
Alternatively, fitted intermediate contrast value can be loaded from `./test/i3_fit.npz`.

To evaluate networks for fitting other contrast values (i=[1,2,4,5]), run:
```
./eval_i1245.sh
```
To fit contrast values, navigate to `./test/` and run:
```
python3 ./fit_i1245.py
```
Alternatively, fitted contrast values can be loaded from `./test/i1245_fit.npz`.

To evaluate networks using fitted contrast values, run:
```
./eval_final.sh
```
To evaluate networks for fitting type-2 noise parameter, run:
```
./eval_conf_noise.sh
```
To fit type-2 noise parameter, navigate to `./test/` and run:
```
python3 ./fit_conf_noise.py
```
Alternatively, fitted type-2 noise parameter can be loaded from `./test/conf_noise_fit.npz`.

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
To evaluate response of principal components to this task (analyses in supplementary material), first run:
```
python3 ./eval_final_PCA.py
```
then navigate to `./test/` and run:
```
python3 ./visualize_PCA.py
```
