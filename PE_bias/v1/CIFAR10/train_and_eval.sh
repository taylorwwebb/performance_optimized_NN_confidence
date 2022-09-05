#!/bin/bash

for r in {1..100}
do
	python3 ./train.py --run $r
	python3 ./eval.py --run $r
done
