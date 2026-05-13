#!/bin/bash -l
#SBATCH -p small
#SBATCH -o ./job.out.%j
#SBATCH -e ./job.err.%j
#SBATCH -D /u/alli/calculations/sqtc
#SBATCH -J sqtc_al_qha
#SBATCH --nodes=4
#SBATCH --ntasks-per-node=128
#SBATCH --ntasks=512
#SBATCH --cpus-per-task=1
#SBATCH --mem=120G
#SBATCH --time=12:00:00
# ──────────────────────────────────────────────────────────────────────────────
# fcc Al  SQTC + QHA benchmark
#
# Runs SQTC at 3 volumes (V₀×{0.98, 1.00, 1.02}) + 3 static primitive-cell
# DFT calculations, then computes α(T), C_P(T), B(T) via:
#   (a) full Helmholtz QHA volume minimisation
#   (b) Grüneisen parameter shortcut
# Comparison plots saved in sqtc_al_qha_run/.
# ──────────────────────────────────────────────────────────────────────────────

unset SLURM_EXPORT_ENV

export OMP_NUM_THREADS=$SLURM_CPUS_PER_TASK
export OMP_PLACES=cores
export OMP_PROC_BIND=true

module load mkl/2025.2 intel/2025.3 impi/2021.17
export LD_LIBRARY_PATH=$MKLROOT/lib/intel64:$LD_LIBRARY_PATH

export VASP_STD="/u/alli/Softwares/DFT_v/s"
export VASP_PP_BASE="/u/alli/Softwares/VASP/Pseudopotentials"

source /u/alli/calculations/sqtc/.venv/bin/activate

python3 codes/examples/run_sqtc_al_qha_vasp.py
