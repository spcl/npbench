#!/bin/bash
for SZ in paper
do
  for FM in 'numpy' 'numba' 'pythran' 'dace_cpu'
  do
    python $CODE/run_benchmark.py -b $BM -f $FM -p $SZ -v 0 -r 1 -t 100
  done
done
