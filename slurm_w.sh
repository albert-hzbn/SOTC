#!/bin/bash -l
#SBATCH -p small
#SBATCH -o ./job.out.%j
#SBATCH -e ./job.err.%j
#SBATCH -D ./
#SBATCH -J sqtc_w
#SBATCH --nodes=4
#SBATCH --ntasks-per-node=128
#SBATCH --ntasks=512
#SBATCH --cpus-per-task=1
#SBATCH --mem=120G
#SBATCH --time=8:00:00

# bcc W benchmark at 2000 K (ultra-refractory, T/T_melt = 0.54, T/T_D = 5.0)
# v2: r_cutoff 4.0->5.0 A (adds 3rd NN at 4.51 A, 26 neighbors), n_atoms_sc 32->64
# Wall time 8h: 64-atom cells + higher T -> slower SCF; 12 snaps x 3 batches, ~30 min each

unset SLURM_EXPORT_ENV

export OMP_NUM_THREADS=$SLURM_CPUS_PER_TASK
export OMP_PLACES=cores
export OMP_PROC_BIND=true

module load mkl/2025.2 intel/2025.3 impi/2021.17
export LD_LIBRARY_PATH=$MKLROOT/lib/intel64:$LD_LIBRARY_PATH

export VASP_STD="/u/alli/Softwares/DFT_v/s"
export VASP_PP_BASE="/u/alli/Softwares/VASP/Pseudopotentials"

source /u/alli/calculations/sqtc/.venv/bin/activate

python3 codes/examples/run_sqtc_bccw_vasp.py
