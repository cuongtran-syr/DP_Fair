#!/usr/bin/env bash
c=$1
s=$2
d=$3
python3 simple_test_ICDM_INCOME.py --C $c --sigma $s --seed $d  > result_${C}_${s}_${d}.out