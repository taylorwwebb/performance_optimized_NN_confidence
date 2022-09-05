### Simulating TMS to first layer 

To train multiple networks on two-choice variant of MNIST, and evaluate over range of TMS noise levels, run:
```
./train_and_eval.sh
```
To analyze results, navigate to `./test/` and run:
```
python3 ./TMS_analysis.py
```
