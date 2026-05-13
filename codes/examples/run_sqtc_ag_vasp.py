#!/usr/bin/env python3
"""
SQTC+VASP benchmark: fcc Ag — anharmonic regime test
=====================================================
Ag is a strongly anharmonic FCC metal:
  - Experimental T_D (low-T calorimetric) ≈ 225 K
  - Grüneisen parameter γ ≈ 2.4  (large anharmonicity)
  - At T = 300 K > T_D: thermal occupation is classical-like, making
    anharmonic phonon renormalization significant
  - Harmonic DFT-PBE (small-displacement phonopy) typically gives
    T_D ≈ 200-215 K — underestimated vs experiment due to PBE volume
    overbinding; SQTC at T=300 K samples the thermally renormalized IFCs

Experimental reference values (NIST / Lide CRC):
  C_V(200 K) = 24.0  J/(mol·K)
  C_V(300 K) = 24.9  J/(mol·K)   (≈ Dulong–Petit 3R = 24.94)
  C_V(400 K) = 25.8  J/(mol·K)   (slight super-classical: anharmonic)
  T_D (calorimetric from C_V) ≈ 220–228 K

Harmonic DFT-PBE (literature, e.g. Togo & Tanaka 2015 / phonopy):
  T_D ≈ 200–215 K   (∼5-10% below experiment due to PBE overbinding)
  C_V(300 K) ≈ 24.9 J/(mol·K) (insensitive at T >> T_D)
"""

import os
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "../"))

from sqtc import SQTCRunner

# ── VASP / machine configuration ──────────────────────────────────────────────
VASP_STD = os.environ.get("VASP_STD", "/u/alli/Softwares/DFT_v/s")
PP_BASE_DIR = Path(os.environ.get("VASP_PP_BASE", "/u/alli/Softwares/VASP/Pseudopotentials"))

NCORES_PER_JOB = 64
NCORE_INCAR = 8
VASP_CMD = f"srun --nodes=1 --ntasks={NCORES_PER_JOB} --exclusive {VASP_STD}"

# ── fcc Ag primitive cell ──────────────────────────────────────────────────────
# Use experimental lattice constant to remove PBE volume error.
# PBE overbinds volume by ~5.6% (a_PBE=4.16 vs a_exp=4.085 Å), which
# softens phonons via Grüneisen coupling and lowers T_D by ~10%.
# Using a_exp forces VASP to evaluate forces at the physical volume,
# giving phonon frequencies consistent with experimental T_D ≈ 225 K.
a_ag = 4.085  # experimental lattice constant [Å]
prim_cell_ag = 0.5 * a_ag * np.array(
    [
        [0.0, 1.0, 1.0],
        [1.0, 0.0, 1.0],
        [1.0, 1.0, 0.0],
    ]
)
prim_pos_ag = np.array([[0.0, 0.0, 0.0]])

# ── Physical inputs ────────────────────────────────────────────────────────────
M_Ag_amu  = 107.8682
T_D_Ag    = 225.0   # experimental calorimetric T_D [K] — used as Debye seed
T_design  = 300.0   # target temperature [K]

print("\n" + "=" * 68)
print(" SQTC VASP Benchmark: fcc Ag (anharmonic regime)")
print("=" * 68)
print(f"  VASP command : {VASP_CMD}")
print(f"  POTCAR path  : {PP_BASE_DIR / 'PAW_PBE' / 'Ag' / 'POTCAR'}")
print(f"  a_exp        : {a_ag} Å  (experimental; removes PBE volume error)")
print(f"  T_design     : {T_design} K  (> T_D = {T_D_Ag} K → anharmonic regime)")
print(f"  γ_Grüneisen  : ≈ 2.4  (large — strong thermal phonon renormalization)")
print()
print("  Experimental reference (NIST):")
print("    C_V(200 K) = 24.0  J/(mol·K)")
print("    C_V(300 K) = 24.9  J/(mol·K)")
print("    C_V(400 K) = 25.8  J/(mol·K)")
print("    T_D (calorimetric) ≈ 220–228 K")
print()
print("  Harmonic DFT-PBE reference (phonopy literature):")
print("    T_D ≈ 200–215 K  (PBE overestimates volume → softer phonons)")
print("    C_V(300 K) ≈ 24.9 J/(mol·K)  (classical limit, insensitive)")

vasp_settings = {
    "encut": 400.0,          # Ag: 250 eV hard cutoff, 400 gives well-converged forces
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
    element="Ag",
    mass_amu=M_Ag_amu,
    prim_cell=prim_cell_ag,
    prim_positions=prim_pos_ag,
    T=T_design,
    T_D=T_D_Ag,
    # 32-atom supercell: NN-complete for r_cutoff=4.5 Å
    # (12×1st-NN at 2.94 Å + 6×2nd-NN at 4.16 Å = 18 neighbors per atom)
    n_atoms_sc=32,
    force_calculator=None,
    vasp_cmd=VASP_CMD,
    vasp_settings=vasp_settings,
    # 1st + 2nd NN shell for Ag: 2.94 Å and 4.16 Å, both < 4.5 Å cutoff
    r_cutoff=4.5,
    r_max_corr=8.0,
    n_ensemble=6,
    work_dir="sqtc_ag_exp_vol_run",
    verbosity=1,
    ridge_alpha=1e-3,
    symmetrize_bonds=True,
)

results = runner.run(
    n_sc_iterations=12,
    epsilon_conv=0.003,
    mixing=0.30,
    T_values=np.array([100.0, 200.0, 300.0, 400.0]),
    q_mesh_cv=(6, 6, 6),
)

print("\n--- SQTC Ag benchmark summary ---")
print(f"Converged : {results['converged']}")
print(f"Iterations: {results['n_iterations']}")
print()
print(f"{'Quantity':<30} {'SQTC':>10} {'Harmonic DFT':>14} {'Experiment':>12}")
print("-" * 70)
print(f"{'T_D (calorimetric) [K]':<30} {results['T_D_effective']:>10.1f} {'200–215':>14} {'220–228':>12}")
for T_v, cv_v in zip(results['T_values'], results['C_V_scan']):
    exp = {100.0: "~21.5", 200.0: "24.0", 300.0: "24.9", 400.0: "25.8"}.get(T_v, "—")
    harm = "~24.9" if T_v >= 200.0 else "~21?"
    print(f"  C_V({T_v:.0f} K) [J/mol/K]  {'':<11} {cv_v:>10.4f} {harm:>14} {exp:>12}")
print()
print("Results saved to sqtc_ag_exp_vol_run/sqtc_results.json")
