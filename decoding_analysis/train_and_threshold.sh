#!/bin/bash

for r in {1..100}
do
	python3 ./train_and_threshold.py --run $r
done
