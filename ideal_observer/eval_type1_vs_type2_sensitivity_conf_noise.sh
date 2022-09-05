#!/bin/bash

for xi in `seq 0.1 0.1 2.0`
do
	python3 eval_type1_vs_type2_sensitivity_conf_noise.py --conf_noise $xi
done
