HL_NUM_THREADS=1 likwid-perfctr -C 0 -g FLOPS_SP -m $1
HL_NUM_THREADS=1 likwid-perfctr -C 0 -g L3 -m $1
HL_NUM_THREADS=1 likwid-perfctr -C 0 -g L3CACHE -m $1
HL_NUM_THREADS=1 likwid-perfctr -C 0 -g L2 -m $1
HL_NUM_THREADS=1 likwid-perfctr -C 0 -g L2CACHE -m $1
