#!/bin/bash -l
#SBATCH -p small
#SBATCH -o sotc_pbte_qe.out.%j
#SBATCH -e sotc_pbte_qe.err.%j
#SBATCH -D ./
#SBATCH -J sotc_pbte_qe
#SBATCH --nodes=1
#SBATCH --ntasks=64
#SBATCH --cpus-per-task=2
#SBATCH --mem=96G
#SBATCH --time=48:00:00
# rocksalt PbTe  700 K  64-atom  — soft modes, strongly anharmonic, up to 8 iters

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
python3 codes/benchmarks/run_sotc_pbte_qe.py
echo "sotc_pbte_qe done."
