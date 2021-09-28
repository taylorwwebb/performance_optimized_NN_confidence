### Testing for blindsight

To train multiple networks on two-choice variant of MNIST, and evaluate both with and without simulated lesion, run:
```
./train_and_eval.sh
```
To analyze results, navigate to `./test/` and run:
```
python3 ./lesion_analysis.py
```
