#!/usr/bin/env bash
d=$1
s=$2
m=$3
python3 run_add_noise_fair_models.py --data $d --sigma $s --seed $m  > result_${s}.out