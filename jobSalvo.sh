#!/bin/bash
#SBATCH -J npb-all
#SBATCH -D salvo
#SBATCH --partition=micro
#SBATCH --time=24:00:00
#SBATCH --account=pr28fi
#SBATCH --ntasks=1
#SBATCH --ear=off
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
CODE=$BASE/npb-lrz

source $BASE/conda-npb/bin/activate

for SIZE in S M L paper
do
  mkdir -p $SIZE
  cd $SIZE
#  python $CODE/quickstart.py -p $SIZE -v 0 -d 1 -r 1 -t 1.0 2> quickstart.err
  python $CODE/plot_results.py 2> plot_results.err
  python $CODE/plot_lines.py   2> plot_lines.err
  cd ..
done

exit
