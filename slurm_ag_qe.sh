#!/bin/bash -l
#SBATCH -p small
#SBATCH -o sotc_ag_qe.out.%j
#SBATCH -e sotc_ag_qe.err.%j
#SBATCH -D ./
#SBATCH -J sotc_ag_qe
#SBATCH --nodes=1
#SBATCH --ntasks=32
#SBATCH --cpus-per-task=4
#SBATCH --mem=64G
#SBATCH --time=12:00:00
# fcc Ag  1000 K  32-atom  — moderately anharmonic, up to 6 SOTC iterations

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
python3 codes/benchmarks/run_sotc_ag_qe.py
echo "sotc_ag_qe done."
