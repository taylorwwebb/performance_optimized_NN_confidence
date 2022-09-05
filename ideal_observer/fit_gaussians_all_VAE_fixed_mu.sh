#!/bin/bash

for r in {1..100}
do
	python3 ./fit_gaussians.py --train_regime fixed_mu --run $r
done