#!/bin/bash

for r in {1..100}
do
	python3 ./eval_i3.py --run $r
done
