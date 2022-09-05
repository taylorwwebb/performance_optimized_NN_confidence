#!/bin/bash

for r in {1..100}
do
	python3 ./eval_VAE_PE_bias_v2.py --run $r
done