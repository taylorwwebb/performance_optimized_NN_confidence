### Testing for PE bias using CIFAR-10 dataset

To train a single network on the CIFAR-10 training set, run:
```
python3 ./train.py
```
To evaluate a single trained network over a range of contrast and noise values, run:
```
python3 ./eval.py
```
To train and evaluate multiple networks, run:
```
./train_and_eval.sh
```
To analyze results, navigate to `./test/` and run:
```
python3 ./PE_test.py
```
