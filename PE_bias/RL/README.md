### Testing for PE bias in RL setting

To train a single network on the orientation discrimination task using RL, and evaluate it over a range of contrast and noise values, run:
```
python3 ./train_and_eval.py
```
To train and evaluate multiple networks, run:
```
./train_and_eval.sh
```
To analyze results, navigate to `./test/` and run:
```
python3 ./PE_test.py
```