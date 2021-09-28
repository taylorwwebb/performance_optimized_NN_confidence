#!/bin/bash

for xi in `seq 0.1 0.1 2.0`
do
	python3 eval_conf_noise.py --conf_noise $xi
done
