#!/bin/bash
# Job name:
# SBATCH --job-name=toast_sim
# Job output
#SBATCH --output sim_outputs/slurm_outputs/%A_%a_toast_sim.out
#SBATCH --error sim_outputs/slurm_outputs/%A_%a_toast_sim.err
# Email job changes
#SBATCH --mail-type=FAIL
#SBATCH --mail-user=henry.nachman@utexas.edu
# Time limit (1 minute)
#SBATCH --time 00:30:00
# Number of MPI tasks needed for use case (example):
#SBATCH --ntasks=1
# Processors per task:
#SBATCH --cpus-per-task=8
# One task per node
#SBATCH --ntasks-per-node=1
# Set memory requirement
#SBATCH --mem 256GB
#SBATCH --array=0-0

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
# OUT_DIR=${SLURM_SUBMIT_DIR}/sim_outputs/toast_outputs/$1_${SLURM_ARRAY_TASK_ID}

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

echo "Running toast_sim.py"

srun python run_toast_sim/toast_sim.py \
--input_map sim_sky_maps/PS_maps/sim_maps/hr_simple_map_10arcmin.fits \
--schedule_file sim_sky_maps/PS_maps/test_schedule.txt \
--save_intermediate \
--out_dir sim_outputs/toast_outputs/simple_hr10arcmin \
--make_map  \
--no_atmosphere  \
--no_scan_sync

echo "Done"