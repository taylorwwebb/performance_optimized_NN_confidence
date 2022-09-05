#!/bin/bash

for r in {1..100}
do
	python3 ./eval_VAE_PE_bias.py --run $r
done