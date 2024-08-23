#!/bin/bash
source ../../conda-npb/bin/activate
#python ../run_benchmark.py -b gemm -f numba &
#python ../run_benchmark.py -b gemm -f numpy &
python ../run_benchmark.py -b gemm -f pythran &
#python ../run_benchmark.py -b gemm -f dace_cpu

wait
