#!/bin/bash

for r in {1..100}
do
	python3 ./eval_i3_dprime_VAE.py --run $r
done
