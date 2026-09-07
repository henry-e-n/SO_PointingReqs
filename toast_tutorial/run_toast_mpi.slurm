#!/bin/bash
# Job name:
# SBATCH --job-name=testtoastmpi
# Job output
#SBATCH --output mpi.out
# Email job changes
#SBATCH --mail-type=ALL
# SBATCH --mail-user=henry.nachman@utexas.edu
# Time limit (1 minute)
#SBATCH --time 00:03:00
# Number of MPI tasks needed for use case (example):
#SBATCH --ntasks=1
# Processors per task:
#SBATCH --cpus-per-task=2
# One task per node
#SBATCH --ntasks-per-node=1
# Set memory requirement
#SBATCH --mem 256GB

# Load openmpi
# module use /shared_home/software/modulefiles/
# module load openmpi # Loads the Default (D) openmpi

# Toy Command to run:
srun python3 /shared_home/henachman/SO_PointingReqs/toast_tutorial/03_TOAST_MPI_Tutorial.py
echo "Done!"
