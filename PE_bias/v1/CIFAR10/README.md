### Testing for PE bias using CIFAR-10 dataset

To train multiple networks on the CIFAR-10 training set and evaluate over a range of contrast and noise values, run:
```
./train_and_eval.sh
```
To analyze results, navigate to `./test/` and run:
```
python3 ./PE_test.py
```
