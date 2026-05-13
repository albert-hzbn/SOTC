#!/usr/bin/env python3
"""
SQTC+VASP benchmark: fcc Au — extremely anharmonic near melting (T = 1200 K)
=============================================================================
Au is the most anharmonic of the noble-metal FCC triad (Cu/Ag/Au):

  ANHARMONICITY (strongest of Cu/Ag/Au triad):
  - Grüneisen parameter γ ≈ 2.95 — very large (Cu: 1.96, Ag: 2.40)
  - T_D ≈ 165 K — exceptionally low; at T = 1200 K, T/T_D ≈ 7.3
  - At T = 1200 K (T/T_melt = 0.90): second-highest anharmonicity ratio of any
    common FCC metal; explicit phonon–phonon coupling strongly renormalises IFCs
  - C_V exceeds 3R = 24.94 J/(mol·K) substantially: experimentally ~27.4 J/(mol·K)
  - Large Debye-Waller factor: MSD at 1200 K ≈ 0.03–0.04 Å² per atom

  HARMONIC DFT (0 K PBE-PAW, phonopy):
  - T_D ≈ 135–155 K  (PBE overbinds: a_PBE ≈ 4.16 vs a_exp = 4.078 Å)
  - C_V(1200 K) ≈ 24.94 J/(mol·K)  (classical limit — no anharmonic correction)
  - Harmonic DFT misses ~2.5 J/(mol·K) of C_V at 1200 K

  SQTC APPROACH:
  - Thermally renormalised IFCs at 1200 K capture the softened phonon branches
  - Provides a direct test of SQTC in the extreme anharmonic limit (T/T_D ≈ 7)

  Experimental references (NIST / Grimvall 1999 / Barin & Knacke):
    T_D (calorimetric)  ≈ 165 K
    C_V( 300 K) ≈ 24.4 J/(mol·K)
    C_V( 700 K) ≈ 25.7 J/(mol·K)
    C_V(1000 K) ≈ 26.8 J/(mol·K)
    C_V(1200 K) ≈ 27.4 J/(mol·K)   (super-Dulong–Petit by +2.5 J/mol/K)

  TDEP/ab-initio at 1200 K:
    T_D ≈ 120–140 K  |  C_V ≈ 25.4–26.0 J/(mol·K)
    (harmonic-renormalized; anharmonic gap ≈ 1.4 J/mol/K at 1200 K)
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

# ── fcc Au primitive cell ──────────────────────────────────────────────────────
# a_exp(300 K) = 4.078 Å; linear thermal expansion α ≈ 14.2×10⁻⁶ K⁻¹.
# At 1200 K:  a(1200K) = 4.078 × (1 + 14.2e-6 × 900) ≈ 4.130 Å.
# Note: PBE overbinds Au significantly (a_PBE ≈ 4.16 Å, +2% vs experiment);
# using a_exp removes the PBE volume error and gives physical phonon frequencies.
a_au = 4.13  # Å  (experimental fcc Au at ~1200 K)

prim_cell_au = 0.5 * a_au * np.array(
    [
        [0.0, 1.0, 1.0],
        [1.0, 0.0, 1.0],
        [1.0, 1.0, 0.0],
    ]
)
prim_pos_au = np.array([[0.0, 0.0, 0.0]])

# ── Physical inputs ────────────────────────────────────────────────────────────
M_Au_amu  = 196.9665
# Seed with calorimetric T_D; at 1200 K effective value is lower (~120–140 K).
# Use 160 K to start near the right scale; SQTC self-consistency will converge.
T_D_Au    = 160.0   # K
T_design  = 1200.0  # K  (T/T_melt = 0.90 — extreme anharmonic regime)

print("\n" + "=" * 68)
print(" SQTC VASP Benchmark: fcc Au (extremely anharmonic, T/T_melt = 0.90)")
print("=" * 68)
print(f"  VASP command : {VASP_CMD}")
print(f"  POTCAR path  : {PP_BASE_DIR / 'PAW_PBE' / 'Au' / 'POTCAR'}")
print(f"  a (exp 1200K): {a_au} Å")
print(f"  T_design     : {T_design} K  (T/T_melt = {T_design/1337:.2f})")
print()
print("  HARMONIC DFT STATUS: DEFINED (all real modes)")
print("    → Harmonic C_V(1200 K) ≈ 24.94 J/(mol·K)  [classical limit]")
print("    → Anharmonic excess: C_V(exp) − C_V(harm) ≈ +2.5 J/(mol·K)")
print("    → Largest anharmonic gap in the Cu/Ag/Au triad")
print()
print("  Experimental reference (NIST / Grimvall / Barin & Knacke):")
print("    T_D (calorimetric) ≈ 165 K")
print("    C_V( 300 K) ≈ 24.4 J/(mol·K)")
print("    C_V( 700 K) ≈ 25.7 J/(mol·K)")
print("    C_V(1000 K) ≈ 26.8 J/(mol·K)")
print("    C_V(1200 K) ≈ 27.4 J/(mol·K)  (super-Dulong–Petit)")
print()
print("  TDEP at 1200 K:")
print("    T_D ≈ 120–140 K  |  C_V ≈ 25.4–26.0 J/(mol·K)")

vasp_settings = {
    "encut": 400.0,           # Au: hard cutoff 229.9 eV; 400 eV well-converged
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
        # σ = k_B × 1200 K ≈ 0.10 eV — Au is a d-metal; M-P smearing appropriate
        "SIGMA": "0.10",
    },
    "timeout": 3600,
}

runner = SQTCRunner(
    element="Au",
    mass_amu=M_Au_amu,
    prim_cell=prim_cell_au,
    prim_positions=prim_pos_au,
    T=T_design,
    T_D=T_D_Au,
    # 32-atom supercell: NN-complete for r_cutoff = 4.5 Å
    # fcc Au: 1st NN at a/√2 = 2.92 Å (×12), 2nd NN at a = 4.13 Å (×6) → 18 total
    # 9 unique ±R classes → 81 free params (reduced to 18 with symmetrize_bonds=True)
    # 32×3×12 = 1152 observations → 14× overdetermined
    n_atoms_sc=32,
    force_calculator=None,
    vasp_cmd=VASP_CMD,
    vasp_settings=vasp_settings,
    r_cutoff=4.5,
    r_max_corr=9.0,
    n_ensemble=12,
    work_dir="sqtc_au_vasp_run",
    verbosity=1,
    ridge_alpha=1e-3,
    # FCC: central-force projection valid
    symmetrize_bonds=True,
)

results = runner.run(
    n_sc_iterations=12,
    epsilon_conv=0.003,
    mixing=0.30,
    T_values=np.array([300.0, 500.0, 700.0, 1000.0, 1200.0]),
    q_mesh_cv=(6, 6, 6),
)

print("\n--- SQTC fcc Au benchmark summary ---")
print(f"Converged : {results['converged']}")
print(f"Iterations: {results['n_iterations']}")
print()
print(f"{'Quantity':<38} {'SQTC':>10} {'TDEP (lit)':>12} {'Experiment':>12}")
print("-" * 76)
td_spectral = results['T_D_effective']
td_caloric  = results.get('T_D_caloric', float('nan'))
print(f"{'T_D (spectral-moment) [K]':<38} {td_spectral:>10.1f} {'120–140':>12} {'165':>12}")
print(f"{'T_D (calorimetric) [K]':<38} {td_caloric:>10.1f} {'—':>12} {'165':>12}")

exp  = {300.0: "~24.4", 500.0: "~25.0", 700.0: "~25.7", 1000.0: "~26.8", 1200.0: "~27.4"}
tdep = {300.0: "~24.1", 500.0: "~24.5", 700.0: "~24.9", 1000.0: "~25.5", 1200.0: "~25.9"}
for T_v, cv_v in zip(results['T_values'], results['C_V_scan']):
    print(
        f"  C_V({T_v:.0f} K) [J/mol/K]  {'':>8} {cv_v:>10.4f} "
        f"{tdep.get(T_v,'—'):>12} {exp.get(T_v,'—'):>12}"
    )
print()
print("  Harmonic DFT C_V(1200 K) ≈ 24.94 J/(mol·K)  [3R classical limit]")
print("Results saved to sqtc_au_vasp_run/sqtc_results.json")
