#!/bin/bash

for r in {1..100}
do
	python3 ./train_VAE.py --train_regime fixed_mu --run $r
done