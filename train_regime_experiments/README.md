### Testing for effect of training regime on PE bias

To train multiple networks on each of four different training regimes, and evaluate effect on PE bias, run:
```
./all_train_regimes_PE_test.sh
```
To analyze results, navigate to `./PE_test/` and run:
```
python3 ./PE_test.py
```

To train multiple networks on each of four different training regimes, and evaluate type-2 sensitivity, run:
```
./all_train_regimes_meta_d.sh
```
To analyze results, navigate to `./meta_d_test/` and run:
```
python3 ./plot_meta_d.py
```
