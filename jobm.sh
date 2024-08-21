#!/bin/bash
#SBATCH -D jobm
#SBATCH --partition=micro
#SBATCH --time=24:00:00
#SBATCH --account=pr28fi
#SBATCH --ntasks=1
#SBATCH -J medium
#SBATCH --ear=off
#SBATCH --export=NONE  ## Added these ones, should be safer (get a new clean environment on the compute nodes)
#SBATCH --get-user-env

module purge
module swap spack/23
module load intel
module load slurm_setup
module load anaconda3
module load git
module load cmake
module load gmake
module list

source /hppfs/work/pr28fi/di38jil/npb-shared/conda-npb/bin/activate

python /hppfs/work/pr28fi/di38jil/npb-shared/npb-lrz/main.py -p M -v 1 -d 1 # I made the shortest one I could find
python /hppfs/work/pr28fi/di38jil/npb-shared/npb-lrz/plot_results.py
python /hppfs/work/pr28fi/di38jil/npb-shared/npb-lrz/plot_lines.py

exit
