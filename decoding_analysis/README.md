### Decoding analysis

To train multiple networks on two-choice variant of MNIST and evaluate over a range of contrast values, run:
```
./train_and_threshold.sh
```
To identify threshold-level contrast, navigate to `./test/` and run:
```
python3 ./threshold.py
```
To run decoding analysis using inputs from all layers in network, run:
```
./whole_network_decoding.sh
```
To analyze results, navigate to `./decoder_test/` and run:
```
python3 ./plot_ROC_AUC.py --decoder_input whole_network
```
To run decoding analysis using inputs from penultimate layer only, run:
```
./z_decoding.sh
```
To analyze results, navigate to `./decoder_test/` and run:
```
python3 ./plot_ROC_AUC.py --decoder_input z
```
To incorporate noise into decoding analysis, set `--decoder_noise` to desired value in both `./decoding_analysis.py` and `./test/plot_ROC_AUC.py`.

To compare results for decoding from whole network vs. penultimate layer only, navigate to `./decoder_test/` and run:
```
python3 ./compare_AUC.py
```
