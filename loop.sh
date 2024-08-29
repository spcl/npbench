#!/bin/bash
hostname
for SZ in paper
do
  for FW in 'numpy' 'numba' 'pythran' 'dace_cpu'
  do
    python $CODE/run_benchmark.py -b $BM -f $FW -p $SZ -v 0 -r 1 -t 300
  done
done
