#!/bin/bash -l
#SBATCH -p small
#SBATCH -o ./job.out.%j
#SBATCH -e ./job.err.%j
#SBATCH -D ./
#SBATCH -J sqtc_au
#SBATCH --nodes=4
#SBATCH --ntasks-per-node=128
#SBATCH --ntasks=512
#SBATCH --cpus-per-task=1
#SBATCH --mem=120G
#SBATCH --time=4:00:00

# fcc Au benchmark at 1200 K (extremely anharmonic, T/T_melt = 0.90)

unset SLURM_EXPORT_ENV

export OMP_NUM_THREADS=$SLURM_CPUS_PER_TASK
export OMP_PLACES=cores
export OMP_PROC_BIND=true

module load mkl/2025.2 intel/2025.3 impi/2021.17
export LD_LIBRARY_PATH=$MKLROOT/lib/intel64:$LD_LIBRARY_PATH

export VASP_STD="/u/alli/Softwares/DFT_v/s"
export VASP_PP_BASE="/u/alli/Softwares/VASP/Pseudopotentials"

source /u/alli/calculations/sqtc/.venv/bin/activate

python3 codes/examples/run_sqtc_au_vasp.py
