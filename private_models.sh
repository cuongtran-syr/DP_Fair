#!/usr/bin/env bash
d=$1
s=$2
python3 run_private_models.py --data $d --seed $s > result_${s}.out