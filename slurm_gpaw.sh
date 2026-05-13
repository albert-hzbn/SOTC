#!/bin/bash -l
#SBATCH -p small
#SBATCH -o ./job.out.%j
#SBATCH -e ./job.err.%j
#SBATCH -D ./
#SBATCH -J sqtc_gpaw
#SBATCH --nodes=1
#SBATCH --ntasks=32
#SBATCH --cpus-per-task=4
#SBATCH --mem=64G
#SBATCH --time=12:00:00
#
# SQTC + GPAW job script
# ======================
# GPAW parallelises differently from VASP:
#   - Uses MPI for k-point + band parallelism
#   - Uses OpenMP (--cpus-per-task) for ScaLAPACK / BLAS
#   - Good decomposition for a 32-atom cell with (2,2,2) k-mesh:
#       8 k-points  →  8 MPI ranks  ×  4 OMP threads = 32 cores / node
#
# Resources: 1 node, 32 MPI ranks, 4 OMP threads each = 128 core equivalents.
# GPAW runs one snapshot at a time (serialised by GPAWForceCalculator._GPAW_LOCK)
# so all 32 MPI ranks service each single-point calculation.
#
# Adjust --ntasks and --cpus-per-task if your cluster has different node topology.
# MPCDF Raven: 72 cores/node  →  --ntasks=18 --cpus-per-task=4  (single node)
# MPCDF Viper: 128 cores/node →  --ntasks=32 --cpus-per-task=4  (used here)
#
# Usage:
#   sbatch slurm_gpaw.sh                         # run all GPAW examples
#   sbatch slurm_gpaw.sh --only al               # Al only
#   sbatch slurm_gpaw.sh --only pb_harm          # Pb harmonic only
#   sbatch slurm_gpaw.sh --only pb_anh           # Pb anharmonic only

# ── Environment ───────────────────────────────────────────────────────────────

unset SLURM_EXPORT_ENV

export OMP_NUM_THREADS=$SLURM_CPUS_PER_TASK
export OMP_PLACES=cores
export OMP_PROC_BIND=close

# GPAW requires a PAW dataset directory.  Set GPAW_SETUP_PATH to the directory
# containing the GPAW PAW setup files (e.g. gpaw-setups-0.9.20000).
# Install with:  gpaw install-data /path/to/setups
export GPAW_SETUP_PATH="${GPAW_SETUP_PATH:-/u/alli/Softwares/gpaw-setups}"

# Activate the project virtual environment (contains gpaw, ase, numpy, scipy).
source /u/alli/calculations/sqtc/.venv/bin/activate

# Verify GPAW is available
python3 -c "import gpaw; print(f'GPAW {gpaw.__version__} loaded')" || {
    echo "ERROR: GPAW not found in .venv.  Install with:"
    echo "  .venv/bin/pip install gpaw"
    echo "  .venv/bin/gpaw install-data \$GPAW_SETUP_PATH"
    exit 1
}

# ── Parse optional argument ───────────────────────────────────────────────────
# Pass --only <target> as the first script argument to run a subset.

ONLY=${1:-all}   # default: run everything

# ── Run ───────────────────────────────────────────────────────────────────────

set -e   # abort on first error

if [[ "$ONLY" == "al" || "$ONLY" == "all" ]]; then
    echo "============================================================"
    echo " Step 1: SQTC+GPAW  fcc Al  (harmonic reference)"
    echo "============================================================"
    python3 codes/examples/run_sqtc_al_gpaw.py
fi

if [[ "$ONLY" == "pb_harm" || "$ONLY" == "all" ]]; then
    echo "============================================================"
    echo " Step 2a: SQTC+GPAW  fcc Pb  T=100 K  (harmonic reference)"
    echo "============================================================"
    python3 codes/examples/run_sqtc_pb_gpaw.py --only harmonic
fi

if [[ "$ONLY" == "pb_anh" || "$ONLY" == "all" ]]; then
    echo "============================================================"
    echo " Step 2b: SQTC+GPAW  fcc Pb  T=500 K  (anharmonic)"
    echo "============================================================"
    python3 codes/examples/run_sqtc_pb_gpaw.py --only anharmonic
fi

if [[ "$ONLY" == "compare" || "$ONLY" == "all" ]]; then
    echo "============================================================"
    echo " Step 3: Compare Al vs Pb  (harmonic + anharmonic)"
    echo "============================================================"
    python3 codes/examples/compare_al_pb_gpaw.py --no-plot
fi

echo "Done."
