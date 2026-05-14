#!/bin/bash -l
#SBATCH -p small
#SBATCH -o ./job.out.%j
#SBATCH -e ./job.err.%j
#SBATCH -D ./
#SBATCH -J sqtc_qe
#SBATCH --nodes=1
#SBATCH --ntasks=32
#SBATCH --cpus-per-task=4
#SBATCH --mem=64G
#SBATCH --time=24:00:00
#
# SQTC + Quantum ESPRESSO job script
# ====================================
# QE (pw.x) parallelisation:
#   --ntasks        = MPI ranks for pw.x
#   --cpus-per-task = OpenMP threads per rank (or pure-MPI: set to 1)
#
# For a 32-atom fcc Al supercell with (2,2,2) k-mesh (8 k-points):
#   Good decomposition: 8 MPI ranks (1 per k-point) × 4 OMP threads = 32 cores
#   Or pure MPI: 32 ranks, cpus-per-task=1
#
# Each pw.x call is sequential (controlled by QEForceCalculator semaphore),
# so all MPI ranks are dedicated to the single running pw.x process.
#
# Viper node topology: 128 cores → 32 MPI × 4 OMP = 128 cores (optimal).
#
# Usage:
#   sbatch slurm_qe.sh            # run all QE examples
#   sbatch slurm_qe.sh --only al  # Al only

# ── Environment ───────────────────────────────────────────────────────────────

unset SLURM_EXPORT_ENV

# Intel MKL + MPI: required by pw.x (compiled against Intel oneAPI)
module load mkl/2025.2 intel/2025.3 impi/2021.17

export OMP_NUM_THREADS=$SLURM_CPUS_PER_TASK
export OMP_PLACES=cores
export OMP_PROC_BIND=close

# QE: tell pw.x how many OpenMP threads to use
export ESPRESSO_OMP_NUM_THREADS=$SLURM_CPUS_PER_TASK

# MKL ScaLAPACK libraries (pw.x links against these)
export LD_LIBRARY_PATH=$MKLROOT/lib:$LD_LIBRARY_PATH

# pw.x command: srun launches the MPI job within the SLURM allocation.
# $SLURM_NTASKS is set automatically by SLURM.
export QE_PW_CMD="srun -n $SLURM_NTASKS --cpus-per-task=$SLURM_CPUS_PER_TASK \
    /u/alli/Softwares/qe-7.5/bin/pw.x"

# Activate the conda environment with ASE + SQTC dependencies.
module load python-waterboa/2025.06
eval "$(conda shell.bash hook)"
conda activate sqtc_gpaw

# Quick sanity check
python3 -c "from ase.calculators.espresso import Espresso; print('ASE Espresso OK')"
mpirun -np 1 /u/alli/Softwares/qe-7.5/bin/pw.x --version 2>/dev/null | head -1

# ── Parse optional argument ───────────────────────────────────────────────────
# Accept:  sbatch slurm_qe.sh al
#          sbatch slurm_qe.sh --only al

ONLY="all"
while [[ $# -gt 0 ]]; do
    case "$1" in
        --only) ONLY="$2"; shift 2 ;;
        *)      ONLY="$1"; shift   ;;
    esac
done

# ── Run ───────────────────────────────────────────────────────────────────────

set -e

if [[ "$ONLY" == "al" || "$ONLY" == "all" ]]; then
    echo "============================================================"
    echo " SQTC+QE  fcc Al  T=300 K  (harmonic reference)"
    echo "============================================================"
    python3 codes/examples/run_sqtc_al_qe.py
fi

echo "Done."
