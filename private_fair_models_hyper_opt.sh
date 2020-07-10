#!/usr/bin/env bash
d=$1
s=$2
c=$3
python3 opt_hyper_private_fair.py --data $d --lr $s --mult_lr $c  > result_${s}.out