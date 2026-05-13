#!/usr/bin/env python3
"""
SQTC+VASP benchmark: BCC β-Ti — dynamically unstable in harmonic DFT
======================================================================
β-Ti (BCC, space group Im-3m) is the archetypal test for methods beyond
harmonic DFT:

  HARMONIC DFT FAILURE:
  - At 0 K, harmonic PBE-PAW gives large imaginary phonon branches at the
    N-point [1/2,1/2,0] and along Σ ([q,q,0]) in the Brillouin zone.
  - These imaginary frequencies signal an unstable BCC structure at 0 K
    (physical: β-Ti is metastable below 882°C and transforms to HCP α-Ti).
  - Consequence: harmonic C_V and T_D are UNDEFINED at 0 K; the IFCs give a
    non-positive-definite dynamical matrix with negative eigenvalues.

  WHY β-Ti IS STABLE AT HIGH T:
  - Above 882°C (1155 K), anharmonic phonon–phonon interactions
    (phonon renormalization) shift the imaginary frequencies positive.
  - The thermal population of other phonon modes "stiffens" the soft modes
    via cubic/quartic anharmonic coupling.

  SQTC APPROACH:
  - Run the self-consistency at T=1200 K using thermally displaced snapshots.
  - The IFCs are fitted to forces in the thermally-expanded, thermally-excited
    crystal — capturing the renormalized (positive) force constants directly.
  - No imaginary modes expected after self-consistency.

  Experimental references (β-Ti, NIST/literature):
    T_D (calorimetric, from C_V fitting)  ≈ 350–380 K
    C_V(1200 K)  ≈ 27.5 J/(mol·K)  (super-classical; Dulong-Petit = 24.94)
    C_V(1000 K)  ≈ 26.5 J/(mol·K)

  TDEP/SCAILD literature (thermally renormalized):
    T_D(1200 K) ≈ 330–370 K   (Hellman & Abrikosov 2013; Bouchet & Bottin 2015)
    C_V(1200 K) ≈ 25–26 J/(mol·K)  (harmonic renormalized; anharmonic excess
                                     explains the remaining ~1.5 J/mol/K gap)
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

# ── BCC Ti primitive cell ──────────────────────────────────────────────────────
# Experimental BCC Ti lattice constant at ~1200 K.
# 0 K PBE-PAW value ≈ 3.24 Å; thermal expansion to 1200 K adds ~1%.
# Using the experimental high-T value places the simulation at the physical
# volume where the structure is stable.
a_ti = 3.27  # Å  (experimental BCC Ti at ~1200 K)

# BCC primitive cell (1 atom per cell, body-centred)
prim_cell_ti = 0.5 * a_ti * np.array(
    [
        [-1.0,  1.0,  1.0],
        [ 1.0, -1.0,  1.0],
        [ 1.0,  1.0, -1.0],
    ]
)
prim_pos_ti = np.array([[0.0, 0.0, 0.0]])

# ── Physical inputs ────────────────────────────────────────────────────────────
M_Ti_amu  = 47.867
# Seed T_D from TDEP literature for renormalized β-Ti at 1200 K
T_D_Ti    = 360.0   # K  (TDEP/SCAILD estimate; harmonic DFT has no valid value)
T_design  = 1200.0  # K  (above α→β transition at 1155 K)

# 1st NN distance: a*√3/2 = 2.832 Å; 2nd NN: a = 3.270 Å
# r_cutoff = 4.0 Å includes both shells (14 neighbors total per atom):
#   8 × 1st NN at 2.832 Å  +  6 × 2nd NN at 3.270 Å
# Unique ±R classes for 1st+2nd NN of BCC: 7 → 7×9=63 parameters
# 32 atoms × 3 × 6 snapshots = 576 observations → 9× overdetermined

print("\n" + "=" * 68)
print(" SQTC VASP Benchmark: BCC β-Ti (dynamically unstable in harmonic DFT)")
print("=" * 68)
print(f"  VASP command : {VASP_CMD}")
print(f"  POTCAR path  : {PP_BASE_DIR / 'PAW_PBE' / 'Ti' / 'POTCAR'}")
print(f"  a (exp 1200K): {a_ti} Å")
print(f"  T_design     : {T_design} K  (above α→β transition at 1155 K)")
print()
print("  HARMONIC DFT STATUS: FAILS")
print("    → Imaginary phonon branches at N-point [1/2,1/2,0]")
print("    → Dynamical matrix not positive-definite at 0 K")
print("    → C_V and T_D undefined from harmonic IFCs")
print()
print("  Experimental reference (β-Ti phase):")
print("    T_D (calorimetric)  ≈ 350–380 K")
print("    C_V(1000 K) ≈ 26.5 J/(mol·K)")
print("    C_V(1200 K) ≈ 27.5 J/(mol·K)  (super-Dulong-Petit)")
print()
print("  TDEP/SCAILD at 1200 K (Hellman & Abrikosov 2013):")
print("    T_D ≈ 330–370 K  |  C_V ≈ 25–26 J/(mol·K)")

vasp_settings = {
    "encut": 400.0,
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
        # Larger Fermi smearing appropriate for a refractory metal at 1200 K.
        # σ = k_B T ≈ 0.10 eV at 1200 K.
        "SIGMA": "0.20",
    },
    "timeout": 3600,
}

runner = SQTCRunner(
    element="Ti",
    mass_amu=M_Ti_amu,
    prim_cell=prim_cell_ti,
    prim_positions=prim_pos_ti,
    T=T_design,
    T_D=T_D_Ti,
    # 32-atom supercell: NN-complete for BCC with r_cutoff=4.0 Å
    # (8×1st-NN at 2.83 Å + 6×2nd-NN at 3.27 Å = 14 neighbors per atom)
    n_atoms_sc=32,
    force_calculator=None,
    vasp_cmd=VASP_CMD,
    vasp_settings=vasp_settings,
    # 1st + 2nd NN of BCC Ti (2.832 Å and 3.270 Å), both within 4.0 Å
    r_cutoff=4.0,
    r_max_corr=9.0,
    n_ensemble=20,
    work_dir="sqtc_bccti_vasp_run",
    verbosity=1,
    ridge_alpha=1e-2,
    # symmetrize_bonds=False is CRITICAL for BCC metals.
    # Central-force projection (α·r̂⊗r̂ + β·(I−r̂⊗r̂)) eliminates angular
    # (non-central) IFC components. For BCC, the angular terms between
    # 1st-NN <111> pairs are the primary mechanism that lifts the N-point
    # imaginary branch. Destroying them forces imaginary modes to persist.
    # FCC is nearly close-packed so central forces dominate → True works.
    # BCC is not close-packed → angular forces are essential → must be False.
    symmetrize_bonds=False,
)

results = runner.run(
    n_sc_iterations=12,
    epsilon_conv=0.003,
    mixing=0.30,
    T_values=np.array([300.0, 400.0, 500.0, 800.0, 1000.0, 1200.0]),
    q_mesh_cv=(6, 6, 6),
)

print("\n--- SQTC BCC β-Ti benchmark summary ---")
print(f"Converged : {results['converged']}")
print(f"Iterations: {results['n_iterations']}")
print()
print(f"{'Quantity':<38} {'SQTC':>10} {'TDEP (lit)':>12} {'Experiment':>12}")
print("-" * 76)
td_spectral = results['T_D_effective']        # spectral-moment (robust)
td_caloric  = results.get('T_D_caloric', float('nan'))  # calorimetric
print(f"{'T_D (spectral-moment) [K]':<38} {td_spectral:>10.1f} {'330–370':>12} {'350–380':>12}")
print(f"{'T_D (calorimetric, T=T_D only) [K]':<38} {td_caloric:>10.1f} {'330–370':>12} {'350–380':>12}")
exp = {300.0: "~25.1", 400.0: "~25.4", 500.0: "~25.6", 800.0: "~26.3", 1000.0: "~26.5", 1200.0: "~27.5"}
tdep = {300.0: "~24.5?", 400.0: "~24.9?", 500.0: "~25?", 800.0: "~25?", 1000.0: "~25.5", 1200.0: "~25.8"}
for T_v, cv_v in zip(results['T_values'], results['C_V_scan']):
    print(
        f"  C_V({T_v:.0f} K) [J/mol/K]  {'':>8} {cv_v:>10.4f} "
        f"{tdep.get(T_v,'—'):>12} {exp.get(T_v,'—'):>12}"
    )
print()
print("  Harmonic DFT: UNDEFINED (imaginary modes at N-point)")
print("Results saved to sqtc_bccti_vasp_run/sqtc_results.json")
