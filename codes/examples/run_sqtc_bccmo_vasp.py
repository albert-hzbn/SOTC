#!/usr/bin/env python3
"""
SQTC+VASP benchmark: bcc Mo — moderately anharmonic refractory metal (T = 1500 K)
===================================================================================
Mo (BCC, harmonically stable) provides a controlled contrast to β-Ti:
same BCC structure and temperature regime, but WITHOUT dynamic instability.

  ANHARMONICITY (moderate — refractory BCC):
  - Grüneisen parameter γ ≈ 1.57 — moderate (Cu: 1.96, Au: 2.95)
  - T_D ≈ 450 K (one of the highest in 4d/5d transition metals)
  - At T = 1500 K, T/T_D ≈ 3.3: still in the classical regime where
    anharmonic contributions to C_V are measurable (~1–2 J/mol/K above 3R)
  - At T = 1500 K (T/T_melt = 0.52): significant phonon renormalisation
    but no dynamic instability (Mo is BCC-stable at all temperatures)
  - C_V(1500 K) ≈ 27.0 J/(mol·K): +2.1 J/mol/K above 3R = 24.94 J/(mol·K)

  HARMONIC DFT (0 K PBE-PAW, phonopy):
  - T_D ≈ 430–460 K  (PBE very close to experiment for Mo; γ_PBE ≈ 1.5)
  - C_V(1500 K) ≈ 24.94 J/(mol·K)  [classical limit — no anharmonic correction]
  - Harmonic DFT misses ~2.1 J/(mol·K) at 1500 K

  SQTC APPROACH:
  - Runs at T = 1500 K: halfway between room temperature and melting
  - Provides a benchmark of SQTC on a BCC metal WITHOUT the complication of
    imaginary modes — clean test of the anharmonic IFC fitting at high T

  Experimental references (NIST / Barin & Knacke / Schlosser & Munster):
    T_D (calorimetric) ≈ 450 K
    C_V( 600 K) ≈ 25.3 J/(mol·K)
    C_V(1000 K) ≈ 26.3 J/(mol·K)
    C_V(1200 K) ≈ 26.7 J/(mol·K)
    C_V(1500 K) ≈ 27.2 J/(mol·K)   (super-Dulong–Petit)

  TDEP/ab-initio at 1500 K (Hellman; Bouchet & Bottin):
    T_D ≈ 390–420 K  |  C_V ≈ 25.2–25.8 J/(mol·K)

  BCC β-Ti comparison:
  - β-Ti: imaginary modes at N-point → SQTC must compensate
  - Mo:   all modes real, BCC stable → clean harmonic-to-anharmonic comparison
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

# ── bcc Mo primitive cell ──────────────────────────────────────────────────────
# a_exp(300 K) = 3.147 Å; Mo has very low thermal expansion (α ≈ 4.8×10⁻⁶ K⁻¹).
# At 1500 K:  a(1500K) = 3.147 × (1 + 4.8e-6 × 1200) ≈ 3.165 Å.
# Note: PBE gives a_PBE ≈ 3.17 Å (+0.7% vs experiment) for Mo — very accurate.
# Using experimental high-T value.
a_mo = 3.16  # Å  (experimental bcc Mo at ~1500 K)

# BCC primitive cell (1 atom per cell, body-centred)
prim_cell_mo = 0.5 * a_mo * np.array(
    [
        [-1.0,  1.0,  1.0],
        [ 1.0, -1.0,  1.0],
        [ 1.0,  1.0, -1.0],
    ]
)
prim_pos_mo = np.array([[0.0, 0.0, 0.0]])

# ── Physical inputs ────────────────────────────────────────────────────────────
M_Mo_amu  = 95.96
# Seed with calorimetric T_D (450 K); at 1500 K effective value ~400–420 K
T_D_Mo    = 430.0   # K
T_design  = 1500.0  # K  (T/T_melt = 0.52)

print("\n" + "=" * 68)
print(" SQTC VASP Benchmark: bcc Mo (moderately anharmonic refractory BCC)")
print("=" * 68)
print(f"  VASP command : {VASP_CMD}")
print(f"  POTCAR path  : {PP_BASE_DIR / 'PAW_PBE' / 'Mo' / 'POTCAR'}")
print(f"  a (exp 1500K): {a_mo} Å")
print(f"  T_design     : {T_design} K  (T/T_melt = {T_design/2896:.2f})")
print()
print("  HARMONIC DFT STATUS: DEFINED (all real modes)")
print("    → BCC Mo is harmonically stable at 0 K (no imaginary branches)")
print("    → Harmonic C_V(1500 K) ≈ 24.94 J/(mol·K)  [classical limit]")
print("    → Anharmonic excess: C_V(exp) − C_V(harm) ≈ +2.3 J/(mol·K) at 1500 K")
print()
print("  Comparison with BCC β-Ti:")
print("    β-Ti: imaginary modes at N-point, SQTC must handle dynamic instability")
print("    Mo:   all modes real, clean test of anharmonic IFC renormalisation")
print()
print("  Experimental reference (NIST / Barin & Knacke):")
print("    T_D (calorimetric) ≈ 450 K")
print("    C_V( 600 K) ≈ 25.3 J/(mol·K)")
print("    C_V(1000 K) ≈ 26.3 J/(mol·K)")
print("    C_V(1200 K) ≈ 26.7 J/(mol·K)")
print("    C_V(1500 K) ≈ 27.2 J/(mol·K)  (super-Dulong–Petit)")
print()
print("  TDEP at 1500 K (Hellman; Bouchet & Bottin):")
print("    T_D ≈ 390–420 K  |  C_V ≈ 25.2–25.8 J/(mol·K)")

vasp_settings = {
    "encut": 400.0,           # Mo: hard cutoff 224.9 eV; 400 eV well-converged
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
        # σ = k_B × 1500 K ≈ 0.13 eV; use 0.15 eV for good metallic smearing
        "SIGMA": "0.15",
    },
    "timeout": 3600,
}

runner = SQTCRunner(
    element="Mo",
    mass_amu=M_Mo_amu,
    prim_cell=prim_cell_mo,
    prim_positions=prim_pos_mo,
    T=T_design,
    T_D=T_D_Mo,
    # 64-atom supercell: NN-complete for r_cutoff = 5.0 Å
    # bcc Mo: 1st NN at a√3/2 = 2.74 Å (×8), 2nd NN at a = 3.16 Å (×6),
    #         3rd NN at a√2 = 4.47 Å (×12) → 26 total neighbors
    # CRITICAL: 3rd NN (4.47 Å) carries transverse acoustic stiffness at H-point;
    # missing it with r_cutoff=4.0 Å underestimates T_D by ~15% (338 K vs 390-420 K TDEP).
    # 64-atom supercell: side ≈ 10.0 Å > 2×r_cutoff = 10.0 Å (minimum image ok)
    # 64×3×12 = 2304 force observations; rank ~ 9×26 = 234 → 10× overdetermined
    n_atoms_sc=64,
    force_calculator=None,
    vasp_cmd=VASP_CMD,
    vasp_settings=vasp_settings,
    # Include 1st + 2nd + 3rd NN shells (2.74, 3.16, and 4.47 Å)
    r_cutoff=5.0,
    r_max_corr=10.0,
    n_ensemble=12,
    work_dir="sqtc_bccmo_v2_vasp_run",
    verbosity=1,
    ridge_alpha=1e-2,
    # BCC: angular (non-central) IFC components are essential — do NOT project
    # onto central-force form; same reasoning as β-Ti: BCC is not close-packed
    symmetrize_bonds=False,
)

results = runner.run(
    n_sc_iterations=12,
    epsilon_conv=0.003,
    mixing=0.30,
    T_values=np.array([600.0, 1000.0, 1200.0, 1500.0]),
    q_mesh_cv=(6, 6, 6),
)

print("\n--- SQTC bcc Mo benchmark summary ---")
print(f"Converged : {results['converged']}")
print(f"Iterations: {results['n_iterations']}")
print()
print(f"{'Quantity':<38} {'SQTC':>10} {'TDEP (lit)':>12} {'Experiment':>12}")
print("-" * 76)
td_spectral = results['T_D_effective']
td_caloric  = results.get('T_D_caloric', float('nan'))
print(f"{'T_D (spectral-moment) [K]':<38} {td_spectral:>10.1f} {'390–420':>12} {'450':>12}")
print(f"{'T_D (calorimetric) [K]':<38} {td_caloric:>10.1f} {'—':>12} {'450':>12}")

exp  = {600.0: "~25.3", 1000.0: "~26.3", 1200.0: "~26.7", 1500.0: "~27.2"}
tdep = {600.0: "~24.6", 1000.0: "~25.1", 1200.0: "~25.4", 1500.0: "~25.8"}
for T_v, cv_v in zip(results['T_values'], results['C_V_scan']):
    print(
        f"  C_V({T_v:.0f} K) [J/mol/K]  {'':>8} {cv_v:>10.4f} "
        f"{tdep.get(T_v,'—'):>12} {exp.get(T_v,'—'):>12}"
    )
print()
print("  Harmonic DFT C_V(1500 K) ≈ 24.94 J/(mol·K)  [3R classical limit]")
print("Results saved to sqtc_bccmo_v2_vasp_run/sqtc_results.json")
