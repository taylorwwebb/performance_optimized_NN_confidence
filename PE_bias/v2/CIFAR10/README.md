### Testing for PE bias (version 2) using CIFAR-10 dataset

To train multiple networks on the CIFAR-10 training set and evaluate over a range of contrast values, run:
```
./train_and_eval.sh
```
To analyze results, navigate to `./test/` and run:
```
python3 ./PE_test.py
```
To analyze confidence in correct vs. incorrect trials, run:
```
python3 ./PE_test_correct_incorrect.py
```
