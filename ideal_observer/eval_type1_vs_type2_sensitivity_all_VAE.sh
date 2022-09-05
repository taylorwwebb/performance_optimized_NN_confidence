#!/bin/bash

for r in {1..100}
do
	python3 ./eval_type1_vs_type2_sensitivity.py --run $r
done
