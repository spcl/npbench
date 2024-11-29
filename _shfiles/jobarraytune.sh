#!/bin/bash
#SBATCH -D jtune
#SBATCH -p general
#SBATCH --account=pr28fi --ntasks=1 --ear=off
#SBATCH -J dpnp_cpu
#SBATCH --array=1-26
##SBATCH -d singleton
#SBATCH --export=NONE  ## Added these ones, should be safer (get a new clean environment on the compute nodes)
##SBATCH --get-user-env

# Configuration settings
SZ=paper && VAL=0 && RR=1  && TT=100  # PRODUCTION
#SZ=M     && VAL=1 && RR=1   && TT=5000 # VALIDATION (comment to toggle)

FW=$SLURM_JOB_NAME

# Load required modules
module purge
module swap spack/latest
module load intel
module load slurm_setup
module load git
module load cmake
module load gmake
module load openblas
module load intel-toolkit
module load intel-vtune
#module load intel-oneapi-vtune  # Load VTune module
module list

# Benchmark list
BMS=('adi' 'jacobi_1d' 'jacobi_2d' 'fdtd_2d' 'bicg' 'cavity_flow' \
     'cholesky' 'nbody' 'channel_flow' 'covariance' 'gemm' 'conv2d_bias' \
     'softmax' 'k2mm' 'atax' 'crc16' 'mandelbrot1' 'seidel_2d' 'hdiff' \
     'vadv' 'heat_3d' 'scattering_self_energies' 'contour_integral' 'stockham_fft'\
     'trisolv' 'lu' )

source deactivate
conda activate conda-npb  # Activate the conda environment

BASE=/hppfs/work/pr28fi/di38jil/npb-lrz
IDX=$(( ${SLURM_ARRAY_TASK_ID} - 1 ))
BM=${BMS[${IDX}]}

# VTune analyses list
ANALYSES=("hotspots" "hpc-performance" "anomaly-detection")

# Run VTune analyses
for ANALYSIS in "${ANALYSES[@]}"; do
    vtune -collect "${ANALYSIS}" \
          -result-dir "${BM}_${ANALYSIS}_${SZ}" \
          -- python3 ${BASE}/run_benchmark.py -p $SZ -v $VAL -r $RR -b ${BM} -f $FW -t $TT \
          1> "${BM}_${ANALYSIS}_${FW}.out" 2> "${BM}_${ANALYSIS}_${FW}.err"
done

exit 0

