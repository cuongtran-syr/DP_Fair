#!/usr/bin/env bash
d=$1
s=$2
c=$3
m=$4
python3 run_old_private_fair_models.py --data $d --sigma $s --C $c --seed $m  > result_${s}.out