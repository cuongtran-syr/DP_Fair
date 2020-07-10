#!/usr/bin/env bash
d=$1
f=$2
s=$3
c=$4
m=$5
python3 run_private_fair_models_v3.py --data $d --fair_choice $f --sigma $s --C $c --seed $m  > result_${s}.out