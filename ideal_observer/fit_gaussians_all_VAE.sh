#!/bin/bash

for r in {1..100}
do
	python3 ./fit_gaussians.py --run $r
done