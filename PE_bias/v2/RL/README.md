### Testing for PE bias (version 2) in RL setting

To train multiple networks on the orientation discrimination task using RL, and evaluate over a range of contrast values, run:
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
