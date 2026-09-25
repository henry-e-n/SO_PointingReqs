#!/bin/bash
# Job name:
# SBATCH --job-name=sim_mlmapmaking
# Job output
#SBATCH --output sim_outputs/slurm_outputs/%A_%a_ml_mapmaker.out
#SBATCH --error sim_outputs/slurm_outputs/%A_%a_ml_mapmaker.err
# Email job changes
#SBATCH --mail-type=FAIL
#SBATCH --mail-user=henry.nachman@utexas.edu
# Time limit (1 minute)
#SBATCH --time 00:10:00
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
# Match whatever you use interactively in the notebooks, e.g. a soconda
# build such as:
#   source /shared_home/software/soconda_builds/soconda_20260813_0.2.5/bin/activate
# or, on a module-based cluster (e.g. NERSC):
#   module load python
#   conda activate <your_toast_env>


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
# -----------------------------------------------------------------------
HDF5_VOLUME=/shared_home/henachman/SO_PointingReqs/sim_outputs/toast_outputs/simple10arcmin/toast_obs   # directory of per-observation .h5 files
AREA_FILE=/shared_home/henachman/SO_PointingReqs/sim_sky_maps/area.fits           # geometry file from the earlier steps
OUT_DIR=${SLURM_SUBMIT_DIR}/sim_outputs/map_outputs/simple10arcmin

mkdir -p "${OUT_DIR}"

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

srun --ntasks=$((SLURM_NTASKS)) toast_so_map \
    --load_hdf5.enable \
    --load_hdf5.volume "${HDF5_VOLUME}" \
    --mlmapmaker.enable \
    --mlmapmaker.area "${AREA_FILE}" \
    --mlmapmaker.write_iter_map 100 \
    --mlmapmaker.maxiter [400] \
    --out_dir "${OUT_DIR}" \
    --log_config "${OUT_DIR}/used_config.yaml"

srun python mapmaking/map_loader.py "${OUT_DIR}"
echo "Done. Output written to ${OUT_DIR}"