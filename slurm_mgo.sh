#!/bin/bash -l
#SBATCH -p small
#SBATCH -o ./job.out.%j
#SBATCH -e ./job.err.%j
#SBATCH -D ./
#SBATCH -J sqtc_mgo
#SBATCH --nodes=4
#SBATCH --ntasks-per-node=128
#SBATCH --ntasks=512
#SBATCH --cpus-per-task=1
#SBATCH --mem=120G
#SBATCH --time=08:00:00

# MgO rocksalt — LOW anharmonicity benchmark
# Wall time 8h: MgO is stiff (T_D=760 K) so SCF cycles are many per snapshot;
# 12 ensemble × 12 iter × ~2 min/SCF ≈ 5-6 h; 8 h is comfortable.

unset SLURM_EXPORT_ENV

export OMP_NUM_THREADS=$SLURM_CPUS_PER_TASK
export OMP_PLACES=cores
export OMP_PROC_BIND=true

module load mkl/2025.2 intel/2025.3 impi/2021.17
export LD_LIBRARY_PATH=$MKLROOT/lib/intel64:$LD_LIBRARY_PATH

export VASP_STD="/u/alli/Softwares/DFT_v/s"
export VASP_PP_BASE="/u/alli/Softwares/VASP/Pseudopotentials"

source /u/alli/calculations/sqtc/.venv/bin/activate

python3 codes/examples/run_sqtc_mgo_vasp.py
