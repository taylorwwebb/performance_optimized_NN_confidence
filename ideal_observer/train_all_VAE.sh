#!/bin/bash

for r in {1..100}
do
	python3 ./train_VAE.py --run $r
done