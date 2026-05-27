#!/bin/bash -l
#SBATCH -p small
#SBATCH -o sotc_nacl_qe.out.%j
#SBATCH -e sotc_nacl_qe.err.%j
#SBATCH -D ./
#SBATCH -J sotc_nacl_qe
#SBATCH --nodes=1
#SBATCH --ntasks=64
#SBATCH --cpus-per-task=2
#SBATCH --mem=96G
#SBATCH --time=24:00:00
# rocksalt NaCl  700 K  64-atom  — ionic, 4 SOTC iterations

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
python3 codes/benchmarks/run_sotc_nacl_qe.py
echo "sotc_nacl_qe done."
