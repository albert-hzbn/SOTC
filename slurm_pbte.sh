#!/bin/bash -l
#SBATCH -p small
#SBATCH -o ./job.out.%j
#SBATCH -e ./job.err.%j
#SBATCH -D ./
#SBATCH -J sqtc_pbte
#SBATCH --nodes=4
#SBATCH --ntasks-per-node=128
#SBATCH --ntasks=512
#SBATCH --cpus-per-task=1
#SBATCH --mem=120G
#SBATCH --time=16:00:00

# PbTe rocksalt — HIGH anharmonicity benchmark
# Wall time 16h: each SQTC iteration runs 12 VASP calculations across 3 batches
# of 4 nodes; each VASP call takes ~50 min, so one iteration costs ~150 min.
# Convergence expected in 3-5 iterations: 5 × 150 min = 750 min ≈ 12.5 h.
# 16h provides comfortable headroom for the allowed 15 iterations.

unset SLURM_EXPORT_ENV

export OMP_NUM_THREADS=$SLURM_CPUS_PER_TASK
export OMP_PLACES=cores
export OMP_PROC_BIND=true

module load mkl/2025.2 intel/2025.3 impi/2021.17
export LD_LIBRARY_PATH=$MKLROOT/lib/intel64:$LD_LIBRARY_PATH

export VASP_STD="/u/alli/Softwares/DFT_v/s"
export VASP_PP_BASE="/u/alli/Softwares/VASP/Pseudopotentials"

source /u/alli/calculations/sqtc/.venv/bin/activate

python3 codes/examples/run_sqtc_pbte_vasp.py
