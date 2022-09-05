### Testing for PE bias in RL setting

To train multiple networks on the orientation discrimination task using RL, and evaluate over a range of contrast and noise values, run:
```
./train_and_eval.sh
```
To analyze results, navigate to `./test/` and run:
```
python3 ./PE_test.py
```
