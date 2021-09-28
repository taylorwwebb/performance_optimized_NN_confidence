### Testing for PE bias using MNIST dataset

To train multiple networks on the MNIST training set and evaluate them over a range of contrast and noise values, run:
```
./train_and_eval.sh
```
To analyze results, navigate to `./test/` and run:
```
python3 ./PE_test.py
```
