### Simulating TMS to penultimate (z) layer 

To train multiple networks on two-choice variant of MNIST, and evaluate over range of TMS noise levels, run:
```
./train_and_eval.sh
```
To analyze results, navigate to `./test/` and run:
```
python3 ./TMS_analysis.py
```
For signal detection theory model, run:
```
python3 ./SDT_sim.py
```
