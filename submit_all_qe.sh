#!/bin/bash
# submit_all_qe.sh — submit all SOTC+QE benchmark jobs to SLURM
#
# Usage (from the repo root):
#   bash submit_all_qe.sh          # submit all 10 jobs
#   bash submit_all_qe.sh al cu    # submit only the listed materials
#
# Valid names: al ag au cu mo w nacl mgo pbte al_qha

set -euo pipefail

SCRIPTS=(
    al      slurm_al_qe.sh
    ag      slurm_ag_qe.sh
    au      slurm_au_qe.sh
    cu      slurm_cu_qe.sh
    mo      slurm_mo_qe.sh
    w       slurm_w_qe.sh
    nacl    slurm_nacl_qe.sh
    mgo     slurm_mgo_qe.sh
    pbte    slurm_pbte_qe.sh
    al_qha  slurm_al_qha_qe.sh
)

# Build set of targets
declare -A TARGET
if [[ $# -eq 0 ]]; then
    for ((i=0; i<${#SCRIPTS[@]}; i+=2)); do
        TARGET[${SCRIPTS[$i]}]=1
    done
else
    for name in "$@"; do TARGET[$name]=1; done
fi

# Submit
for ((i=0; i<${#SCRIPTS[@]}; i+=2)); do
    name="${SCRIPTS[$i]}"
    script="${SCRIPTS[$((i+1))]}"
    if [[ -n "${TARGET[$name]+x}" ]]; then
        echo -n "Submitting $name ($script) ... "
        jobid=$(sbatch "$script" | awk '{print $NF}')
        echo "job $jobid"
    fi
done

echo "Done. Check queue with: squeue -u \$USER"
