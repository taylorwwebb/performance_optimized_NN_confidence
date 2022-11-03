#!/bin/bash

for r in {1..100}
do
	python3 ./train.py --run $r
done
