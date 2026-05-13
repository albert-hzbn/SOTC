#!/usr/bin/env python3
"""
Fast SQTC+VASP example: fcc Al (quick smoke test)
==================================================
Designed to finish much faster than H2/He production examples.
"""

import os
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "../"))

from sqtc import SQTCRunner

# VASP / machine configuration
VASP_STD = os.environ.get("VASP_STD", "/u/alli/Softwares/DFT_v/s")
PP_BASE_DIR = Path(os.environ.get("VASP_PP_BASE", "/u/alli/Softwares/VASP/Pseudopotentials"))

# 4 concurrent SQTC snapshots; each VASP step uses 64 MPI ranks
NCORES_PER_JOB = 64
NCORE_INCAR = 8
# srun --exclusive ensures each of the 4 concurrent VASP calls gets its own node
VASP_CMD = f"srun --nodes=1 --ntasks={NCORES_PER_JOB} --exclusive {VASP_STD}"

# fcc Al primitive cell (1 atom)
a_al = 4.05  # Angstrom
prim_cell_al = 0.5 * a_al * np.array(
    [
        [0.0, 1.0, 1.0],
        [1.0, 0.0, 1.0],
        [1.0, 1.0, 0.0],
    ]
)
prim_pos_al = np.array([[0.0, 0.0, 0.0]])

# Physical inputs
M_Al_amu = 26.9815385
T_D_Al = 390.0
T_design = 300.0

print("\n" + "=" * 68)
print(" Fast SQTC VASP Example: fcc Al")
print("=" * 68)
print(f"  VASP command : {VASP_CMD}")
print(f"  POTCAR path  : {PP_BASE_DIR / 'PAW_PBE' / 'Al' / 'POTCAR'}")

vasp_settings = {
    "encut": 450.0,
    "functional": "PBE",
    "ncore": NCORE_INCAR,
    "kgrid": (4, 4, 4),
    "pp_base_dir": PP_BASE_DIR,
    "pp_set": "PAW_PBE",
    "extra_incar": {
        "KSPACING": "0.25",
        "EDIFF": "1E-7",
        "PREC": "Accurate",
        "ISMEAR": "1",
        "SIGMA": "0.10",
    },
    "timeout": 3600,
}

runner = SQTCRunner(
    element="Al",
    mass_amu=M_Al_amu,
    prim_cell=prim_cell_al,
    prim_positions=prim_pos_al,
    T=T_design,
    T_D=T_D_Al,
    n_atoms_sc=32,
    force_calculator=None,
    vasp_cmd=VASP_CMD,
    vasp_settings=vasp_settings,
    # r_cutoff: 1st + 2nd NN (fcc Al: 2.86 Å and 4.05 Å).  Including the 2nd
    # NN shell captures the dispersion curvature and brings T_D into the
    # experimental range 390–430 K.  A 32-atom supercell is required to be
    # NN-complete for r_cutoff=4.5 Å (18 neighbors per atom: 12 at 2.86 Å +
    # 6 at 4.05 Å).  With 9 unique ±R classes × 9 = 81 parameters and
    # 32×3×6 = 576 observations (obs/params ≈ 7), the fit remains well
    # overdetermined.
    r_cutoff=4.5,
    r_max_corr=8.0,
    n_ensemble=6,
    work_dir="sqtc_al_fast_vasp_run",
    verbosity=1,
    # Ridge regularisation: kept small since the system is overdetermined.
    ridge_alpha=1e-3,
    # Central-force symmetry projection: project each IFC class onto
    # Φ(R)=α·r̂⊗r̂+β·(I−r̂⊗r̂) and average α,β within each shell.
    # Reduces 81 free parameters → 4 (2 per shell), preventing overfitting.
    symmetrize_bonds=True,
)

results = runner.run(
    n_sc_iterations=12,
    epsilon_conv=0.003,
    mixing=0.30,
    T_values=np.array([200.0, 300.0, 400.0]),
    q_mesh_cv=(6, 6, 6),
)

print("\n--- Fast run summary ---")
print(f"Converged: {results['converged']}")
print(f"Iterations: {results['n_iterations']}")
print(f"T_D effective [K]: {results['T_D_effective']:.2f}")
for T_v, cv_v in zip(results['T_values'], results['C_V_scan']):
    print(f"C_V({T_v:.0f} K) [J/mol/K]: {cv_v:.4f}")
print("Results saved to sqtc_al_fast_vasp_run/sqtc_results.json")
