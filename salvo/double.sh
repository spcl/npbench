#!/bin/bash

BNM=atax
source ../../conda-npb/bin/activate
python ../run_benchmark.py -p M -b $BNM -f numba &
python ../run_benchmark.py -p M -b $BNM -f numpy &
python ../run_benchmark.py -p M -b $BNM -f pythran &
python ../run_benchmark.py -p M -b $BNM -f dace_cpu

wait
