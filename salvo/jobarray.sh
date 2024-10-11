#!/bin/bash
#SBATCH -D jd
#SBATCH -p test
#SBATCH --account=pr28fi --ntasks=1 --ear=off
#SBATCH -J dpnp
#SBATCH --array=1-26 -d singleton
##SBATCH --export=NONE  ## Added these ones, should be safer (get a new clean environment on the compute nodes)
##SBATCH --get-user-env

SZ=paper
FW=$SLURM_JOB_NAME

module purge
module swap spack/24.1.1
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
     'vadv' 'heat_3d' 'scattering_self_energies' 'contour_intergral' 'stockham_fft'\
     'trisolv' 'lu' )

source activate /dss/dsshome1/07/di52vum/.conda/envs/npb
BASE=${HOME}/npb-lrz
IDX=$(( ${SLURM_ARRAY_TASK_ID} - 1 ))
BM=${BMS[${IDX}]}

mkdir -p $SZ
cd $SZ

python ${BASE}/run_benchmark.py -p $SZ -v 1 -r 10 -b ${BM} -f $FW -t 100 2> ${BM}_${FW}.err

exit
