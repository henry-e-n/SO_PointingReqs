#!/bin/bash
# Job name:
# SBATCH --job-name=toast_sim
# Job output
#SBATCH --output sim_outputs/slurm_outputs/%A_toast_map.out
#SBATCH --error sim_outputs/slurm_outputs/%A_toast_map.err
# Email job changes
#SBATCH --mail-type=FAIL
#SBATCH --mail-user=henry.nachman@utexas.edu
# Time limit (1 minute)
#SBATCH --time 01:00:00
# Number of MPI tasks needed for use case (example):
#SBATCH --ntasks=1
# Processors per task:
#SBATCH --cpus-per-task=8
# One task per node
#SBATCH --ntasks-per-node=1
# Set memory requirement
#SBATCH --mem 256GB

# -----------------------------------------------------------------------
# Environment setup
# -----------------------------------------------------------------------

module use /shared_home/software/modulefiles/
module load soconda/stable


# Fill in the line appropriate for your site:
# source /path/to/your/soconda_or_conda/bin/activate

# -----------------------------------------------------------------------
# Threading
# -----------------------------------------------------------------------
# MLMapmaker parallelizes with OpenMP threads *within* a process group, not
# MPI ranks, so this must match --cpus-per-task above.
export OMP_NUM_THREADS=${SLURM_CPUS_PER_TASK}

# -----------------------------------------------------------------------
# Paths -- fill these in
# If running array
# -----------------------------------------------------------------------
TEST_NAME="simple_lr20arcmin"

# mkdir -p "${OUT_DIR}"

# -----------------------------------------------------------------------
# Run
# -----------------------------------------------------------------------
# NOTE on scaling: because MLMapmaker forces TOAST's process-group size to 1,
# each MPI rank independently loads and processes a subset of the
# observations found in HDF5_VOLUME (TOAST load-balances observations across
# ranks). This means --ntasks-per-node * --nodes should not exceed the
# number of observations in your HDF5 volume -- extra ranks beyond that will
# sit idle. For a single test observation, set --ntasks-per-node=1 and rely
# on --cpus-per-task for speed instead.

# echo "Running with dxi = $SLURM_ARRAY_TASK_ID"

echo "Running run_toast_mapmaker.py"

srun python run_toast_sim/run_toast_mapmaker.py \
--data_vol /shared_home/henachman/SO_PointingReqs/sim_outputs/toast_outputs/$TEST_NAME/toast_obs \
--out_dir /shared_home/henachman/SO_PointingReqs/sim_outputs/toast_outputs/$TEST_NAME/mapmaker \

echo "Done"