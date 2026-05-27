#!/bin/bash -l
#SBATCH -p small
#SBATCH -o sotc_al_qha_qe.out.%j
#SBATCH -e sotc_al_qha_qe.err.%j
#SBATCH -D ./
#SBATCH -J sotc_al_qha_qe
#SBATCH --nodes=1
#SBATCH --ntasks=32
#SBATCH --cpus-per-task=4
#SBATCH --mem=64G
#SBATCH --time=24:00:00
# fcc Al QHA  32-atom  — 3 SOTC runs at V-, V0, V+  (delta=2%)

unset SLURM_EXPORT_ENV
module load mkl/2025.2 intel/2025.3 impi/2021.17

export OMP_NUM_THREADS=$SLURM_CPUS_PER_TASK
export OMP_PLACES=cores
export OMP_PROC_BIND=close
export LD_LIBRARY_PATH=$MKLROOT/lib:$LD_LIBRARY_PATH

export QE_PW_CMD="srun -n $SLURM_NTASKS --cpus-per-task=$SLURM_CPUS_PER_TASK \
    /u/alli/Softwares/qe-7.5/bin/pw.x"
export SSSP_DIR="$PWD/pseudopotentials"

source .venv/bin/activate
export PYTHONPATH="$PWD/codes:$PYTHONPATH"

set -e
python3 codes/benchmarks/run_sotc_al_qha_qe.py
echo "sotc_al_qha_qe done."
