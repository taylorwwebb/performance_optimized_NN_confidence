#!/bin/bash

for r in {1..100}
do
	python3 ./eval_final.py --run $r
done
