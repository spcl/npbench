OMP_NUM_THREADS=1 likwid-perfctr -C 0 -g FLOPS_SP -m python run_benchmark.py -p paper -f $1 -b $2 -M likwid
OMP_NUM_THREADS=1 likwid-perfctr -C 0 -g L3 -m python run_benchmark.py -p paper -f $1 -b $2 -M likwid
OMP_NUM_THREADS=1 likwid-perfctr -C 0 -g L3CACHE -m python run_benchmark.py -p paper -f $1 -b $2 -M likwid
OMP_NUM_THREADS=1 likwid-perfctr -C 0 -g L2 -m python run_benchmark.py -p paper -f $1 -b $2 -M likwid
OMP_NUM_THREADS=1 likwid-perfctr -C 0 -g L2CACHE -m python run_benchmark.py -p paper -f $1 -b $2 -M likwid
