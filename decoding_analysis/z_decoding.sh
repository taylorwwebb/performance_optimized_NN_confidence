#!/bin/bash

for r in {1..100}
do
	python3 ./decoding_analysis.py --run $r --decoder_input z
done