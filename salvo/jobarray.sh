#!/bin/bash
#SBATCH -D jx
#SBATCH -p general -t 30
#SBATCH -A pr28fi --ntasks-per-node=1 --ear=off
#SBATCH -J dpnp_cpu
#SBATCH --array=1-26
#SBATCH -d singleton
#SBATCH --export=NONE
##SBATCH --get-user-env

SZ=paper && VAL=0 && RR=10  && TT=100  # PRODUCTION
#SZ=M     && VAL=1 && RR=1   && TT=5000 # VALIDATION (comment to toggle)

FW=$SLURM_JOB_NAME

module purge
module swap spack/latest
module load intel
module load slurm_setup
module load anaconda3
module load git
module load cmake
module load gmake
module load openblas
module list

BMS=('adi' 'jacobi_1d' 'jacobi_2d' 'fdtd_2d' 'bicg' 'cavity_flow' \
     'cholesky' 'nbody' 'channel_flow' 'covariance' 'gemm' 'conv2d_bias' \
     'softmax' 'k2mm' 'atax' 'crc16' 'mandelbrot1' 'seidel_2d' 'hdiff' \
     'vadv' 'heat_3d' 'scattering_self_energies' 'contour_integral' 'stockham_fft'\
     'trisolv' 'lu' )

#conda deactivate
source activate npb

BASE=${HOME}/npb-lrz
IDX=$(( ${SLURM_ARRAY_TASK_ID} - 1 ))
BM=${BMS[${IDX}]}

## If python is wrong, check the spack version and the conda commands via salloc
which python3
mkdir -p $SZ ; cd $SZ

python3 ${BASE}/run_benchmark.py -p $SZ -v $VAL -r $RR -b ${BM} -f $FW -t $TT  1> ${BM}_${FW}.out 2> ${BM}_${FW}.err

exit
