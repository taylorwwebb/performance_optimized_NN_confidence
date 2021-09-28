### Analysis of learned representations and decision rules

To train multiple networks, perform PCA, and evaluate learned representations and decision rules, run: 
```
./train_and_eval.sh
```
To analyze results, navigate to `./test/`. To visualize results of PCA, run:
```
python3 ./visualize_PCA.py
```
To analyze learned decision rules, run:
```
python3 ./learned_rule_analysis.py
```
To test for sensitivity of principal components to PE bias, run:
```
python3 ./PCA_PE_test.py
```
