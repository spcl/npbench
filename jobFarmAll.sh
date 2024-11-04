#!/bin/bash
#SBATCH -J farm
#SBATCH -D salvo
#SBATCH -N 24 -p general -t 1410
##BATCH -N  1 -p micro   -t 1440
#SBATCH --account=pr28fi
#SBATCH --ear=off
#SBATCH --ntasks-per-node 1
#SBATCH --export=NONE
#SBATCH --get-user-env

module purge
module swap spack/23
module load intel
module load slurm_setup
module load anaconda3
module load git
module load cmake
module load gmake
module load openblas
module list

BASE=/hppfs/work/pr28fi/di38jil/npb-shared/
export CODE=$BASE/npb-lrz

source $BASE/conda-npb/bin/activate

# Fill just the following two
MY_NODES_PER_UNIT=1
#MY_WORK_UNITS=128

# Just algebra, these need no changes
MY_WORKERS=$(( $SLURM_JOB_NUM_NODES / $MY_NODES_PER_UNIT ))
MY_TASKS_PER_WORKER=$(( $SLURM_NTASKS_PER_NODE * $MY_NODES_PER_UNIT ))

MDIR=${SLURM_JOB_NAME}_${SLURM_JOB_ID}
mkdir -p $MDIR
cd $MDIR
for SZ in paper
do
  mkdir -p $SZ
  cd $SZ
  for FW in 'numpy' 'numba' 'pythran' 'dace_cpu'
  do
  for BM in 'adi' 'jacobi_1d' 'jacobi_2d' 'fdtd_2d' 'bicg' 'cavity_flow' 'cholesky' 'nbody' 'channel_flow' 'covariance' \
    'gemm' 'conv2d_bias' 'softmax' 'k2mm' 'atax' 'crc16' 'mandelbrot1' 'seidel_2d' 'hdiff' 'vadv' \
    'heat_3d' 'scattering_self_energies' 'contour_integral' 'stockham_fft' 'trisolv' 'lu'
  do
    while true ; do # Scan for a free worker
     	SUBJOBS=`jobs -r | wc -l` # detect how many subjobs are already running
      if [ $SUBJOBS -lt $MY_WORKERS ] ; then  # submit only if at least one worker is free
        sleep 4 # wait before any submission

        srun -N $MY_NODES_PER_UNIT -n $MY_TASKS_PER_WORKER -J sj.$BM.$FW.$SZ \
          python $CODE/run_benchmark.py -b $BM -f $FW -p $SZ -v 0 -r 25 -t 300 2>> $BM.err 1>> $BM.out  &
  #  	  echo "---jobs" ; jobs -r ; echo "---end"
        break # So we move to next BM
      fi
    done # While true
  done #BM
  done #FM
  cd ..
done #SZ

echo "=> Job submission finished; Waiting for last batch to end."
wait # for the last pool of work units

echo "=> Creating plots"
for iDIR in $(ls -d */)
do
  cd $iDIR
  srun -N $MY_NODES_PER_UNIT -n $MY_TASKS_PER_WORKER -J sj.$iDIR.res \
    python $CODE/plot_results.py &
  srun -N $MY_NODES_PER_UNIT -n $MY_TASKS_PER_WORKER -J sj.$iDIR.lin \
    python $CODE/plot_lines.py &
  wait
done
exit
