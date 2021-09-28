#!/bin/bash

for r in {1..100}
do
	python3 ./train_and_eval_PE_test.py --run $r --train_regime standard
	python3 ./train_and_eval_PE_test.py --run $r --train_regime fixed_sigma_mu_ratio
	python3 ./train_and_eval_PE_test.py --run $r --train_regime fixed_sigma
	python3 ./train_and_eval_PE_test.py --run $r --train_regime fixed_mu
done
