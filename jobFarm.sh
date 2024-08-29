#!/bin/bash
#SBATCH -J npb-all
#SBATCH -D salvo
#SBATCH -N 24 -p general --time=24:00:00
##BATCH -N  1 -p micro   -t 360
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

#for SIZE in S M L paper
#do


# Fill just the following two
MY_NODES_PER_UNIT=1
#MY_WORK_UNITS=128

# Just algebra, these need no changes
MY_WORKERS=$(( $SLURM_JOB_NUM_NODES / $MY_NODES_PER_UNIT ))
MY_TASKS_PER_WORKER=$(( $SLURM_NTASKS_PER_NODE * $MY_NODES_PER_UNIT ))

for iBM in 'adi' 'jacobi_1d' 'jacobi_2d' 'fdtd_2d' 'bicg' 'cavity_flow' 'cholesky' 'nbody' 'channel_flow' 'covariance' \
           'gemm' 'conv2d_bias' 'softmax' 'k2mm' 'atax' 'crc16' 'mandelbrot1' 'seidel_2d' 'hdiff' 'vadv' \
           'heat_3d' 'scattering_self_energies' 'contour_integral' 'stockham_fft' 'trisolv' 'lu'
do
  export BM=$iBM
  while true ; do # Scan for a free worker
  	SUBJOBS=`jobs -r | wc -l` # detect how many subjobs are already running
    if [ $SUBJOBS -lt $MY_WORKERS ] ; then  # submit only if at least one worker is free
	  echo "---jobs"
	  jobs -r
	  echo "---end"
      sleep 4 # wait before any submission
      # wrapper could also be an MPI program,`-c $SLURM_CPUS_PER_TASK` is only for OpenMP
      srun -N $MY_NODES_PER_UNIT -n $MY_TASKS_PER_WORKER -J subjob.$BM \
        $CODE/loop.sh 2> $BM.err 1> $BM.out  &
      break # So we move to next BM
    fi
  done
done

wait # for the last pool of work units

python $CODE/plot_results.py 2> plot_results.err
python $CODE/plot_lines.py   2> plot_lines.err
cd ..


exit
