#!/bin/bash

for r in {1..100}
do
	python3 ./eval_VAE_s1s2.py --run $r
done
