#!/usr/bin/env python3
"""
SQTC+VASP benchmark: bcc W — ultra-refractory metal at high temperature (T = 2000 K)
=====================================================================================
W (tungsten) has the highest melting point of all metals (3695 K) and provides
a stringent benchmark for SQTC in the strongly anharmonic high-T BCC regime:

  ANHARMONICITY (moderate-to-strong for a refractory metal):
  - Grüneisen parameter γ ≈ 1.62 — moderate (very similar to Mo: 1.57)
  - T_D ≈ 400 K (calorimetric); exceptionally stiff d-metal bonding
  - At T = 2000 K (T/T_melt = 0.54): T/T_D ≈ 5.0 — deep classical regime
  - Very low thermal expansion (α ≈ 4.5×10⁻⁶ K⁻¹ — lowest of all metals)
  - C_V(2000 K) ≈ 27.5 J/(mol·K): anharmonic excess +2.6 J/(mol·K) above 3R
  - Unlike Mo, W has 5d orbitals (vs 4d for Mo) → larger relativistic effects
    but similar force-constant topology; PAW-PBE well tested for W forces

  HARMONIC DFT (0 K PBE-PAW, phonopy):
  - T_D ≈ 360–400 K  (PBE gives a_PBE ≈ 3.18 Å vs a_exp = 3.165 Å, very accurate)
  - C_V(2000 K) ≈ 24.94 J/(mol·K)  [classical limit]
  - All phonon modes real; BCC W is harmonically stable at 0 K
  - Harmonic DFT misses ~2.6 J/(mol·K) at 2000 K

  BCC STRUCTURE — same topology as bcc Mo, bcc Ti:
  - BCC Bravais lattice, 1-atom primitive cell
  - a_exp(300 K) = 3.165 Å; α ≈ 4.5×10⁻⁶ K⁻¹ (very low)
  - At 2000 K: a(2000K) = 3.165 × (1 + 4.5e-6 × 1700) ≈ 3.189 Å

  COMPARISON WITH Mo:
  - Mo at 1500 K (T/T_D ≈ 3.3):  C_V ≈ 27.2 J/(mol·K) — already well-benchmarked
  - W  at 2000 K (T/T_D ≈ 5.0):  higher T/T_D tests deeper anharmonic regime
  - W is the highest-T benchmark in this set (2000 K vs Au at 1200 K)

  Experimental references (NIST / Chase 1998 / White & Minges 1997):
    T_D (calorimetric) ≈ 400 K
    C_V( 500 K) ≈ 25.1 J/(mol·K)
    C_V(1000 K) ≈ 25.9 J/(mol·K)
    C_V(1500 K) ≈ 26.6 J/(mol·K)
    C_V(2000 K) ≈ 27.5 J/(mol·K)  (super-Dulong–Petit)

  TDEP/ab-initio at 2000 K (Hellman; Bouchet & Bottin for W):
    T_D ≈ 340–380 K  |  C_V ≈ 25.4–26.2 J/(mol·K)
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

# ── bcc W primitive cell ──────────────────────────────────────────────────────
# a_exp(300 K) = 3.165 Å; α ≈ 4.5×10⁻⁶ K⁻¹ (extremely low — lowest of all metals).
# At 2000 K:  a(2000K) = 3.165 × (1 + 4.5e-6 × 1700) ≈ 3.189 Å.
# PBE gives a_PBE ≈ 3.18 Å (+0.5% vs exp) — very accurate for a 5d metal.
a_w = 3.19  # Å  (experimental bcc W at ~2000 K)

# BCC primitive cell (1 atom per cell, body-centred)
prim_cell_w = 0.5 * a_w * np.array(
    [
        [-1.0,  1.0,  1.0],
        [ 1.0, -1.0,  1.0],
        [ 1.0,  1.0, -1.0],
    ]
)
prim_pos_w = np.array([[0.0, 0.0, 0.0]])

# ── Physical inputs ────────────────────────────────────────────────────────────
M_W_amu   = 183.84
# Seed with calorimetric T_D (400 K); at 2000 K effective value ≈ 340–380 K
T_D_W     = 380.0   # K
T_design  = 2000.0  # K  (T/T_melt = 0.54)

print("\n" + "=" * 70)
print(" SQTC VASP Benchmark: bcc W (ultra-refractory, T = 2000 K)")
print("=" * 70)
print(f"  VASP command : {VASP_CMD}")
print(f"  POTCAR path  : {PP_BASE_DIR / 'PAW_PBE' / 'W' / 'POTCAR'}")
print(f"  a (exp 2000K): {a_w} Å")
print(f"  T_design     : {T_design} K  (T/T_melt = {T_design/3695:.2f}, T/T_D = {T_design/T_D_W:.1f})")
print()
print("  HARMONIC DFT STATUS: DEFINED (all real modes)")
print("    → BCC W is harmonically stable at 0 K (unlike β-Ti)")
print("    → Harmonic C_V(2000 K) ≈ 24.94 J/(mol·K)  [classical limit]")
print("    → Anharmonic excess: C_V(exp) − C_V(harm) ≈ +2.6 J/(mol·K) at 2000 K")
print()
print("  Comparison with bcc Mo (1500 K):")
print("    Mo at T/T_D=3.3: tests moderate-anharmonic regime")
print("    W  at T/T_D=5.0: tests deeper anharmonic renormalisation at higher T")
print()
print("  Experimental reference (NIST / White & Minges 1997):")
print("    T_D (calorimetric) ≈ 400 K")
print("    C_V( 500 K) ≈ 25.1 J/(mol·K)")
print("    C_V(1000 K) ≈ 25.9 J/(mol·K)")
print("    C_V(1500 K) ≈ 26.6 J/(mol·K)")
print("    C_V(2000 K) ≈ 27.5 J/(mol·K)  (super-Dulong–Petit)")
print()
print("  TDEP at 2000 K:")
print("    T_D ≈ 340–380 K  |  C_V ≈ 25.4–26.2 J/(mol·K)")

vasp_settings = {
    # W: PAW hard cutoff 223.0 eV; 400 eV gives well-converged forces.
    # The W_sv PAW (5p semi-core) can be used but is unnecessary for force constants;
    # standard W PAW handles 5d4 6s2 valence well.
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
        # σ = k_B × 2000 K ≈ 0.17 eV; use 0.20 eV for robust metallic smearing
        "SIGMA": "0.20",
    },
    # 2000 K snapshots: heavier displaced configurations; allow extra SCF time
    "timeout": 5400,
}

runner = SQTCRunner(
    element="W",
    mass_amu=M_W_amu,
    prim_cell=prim_cell_w,
    prim_positions=prim_pos_w,
    T=T_design,
    T_D=T_D_W,
    # 64-atom supercell: NN-complete for r_cutoff = 5.0 Å
    # bcc W: 1st NN at a√3/2 = 2.76 Å (×8), 2nd NN at a = 3.19 Å (×6),
    #        3rd NN at a√2 = 4.51 Å (×12) → 26 total neighbors
    # CRITICAL: 3rd NN (4.51 Å) carries transverse acoustic stiffness at the
    # H-point and Σ-line; missing it with r_cutoff=4.0 Å underestimates T_D by ~25%.
    # 64-atom supercell: side ≈ 10.1 Å > 2×r_cutoff = 10.0 Å (minimum image ok)
    # 64×3×12 = 2304 force observations; rank ~ 9×26 = 234 → 10× overdetermined
    n_atoms_sc=64,
    force_calculator=None,
    vasp_cmd=VASP_CMD,
    vasp_settings=vasp_settings,
    r_cutoff=5.0,
    r_max_corr=10.0,
    n_ensemble=12,
    work_dir="sqtc_bccw_v2_vasp_run",
    verbosity=1,
    ridge_alpha=1e-2,
    # BCC: angular (non-central) IFC components essential — same reasoning as Mo/Ti
    symmetrize_bonds=False,
)

results = runner.run(
    n_sc_iterations=12,
    epsilon_conv=0.003,
    mixing=0.30,
    T_values=np.array([500.0, 1000.0, 1500.0, 2000.0]),
    q_mesh_cv=(6, 6, 6),
)

print("\n--- SQTC bcc W benchmark summary ---")
print(f"Converged : {results['converged']}")
print(f"Iterations: {results['n_iterations']}")
print()
print(f"{'Quantity':<38} {'SQTC':>10} {'TDEP (lit)':>12} {'Experiment':>12}")
print("-" * 76)
td_spectral = results['T_D_effective']
td_caloric  = results.get('T_D_caloric', float('nan'))
print(f"{'T_D (spectral-moment) [K]':<38} {td_spectral:>10.1f} {'340–380':>12} {'400':>12}")
print(f"{'T_D (calorimetric) [K]':<38} {td_caloric:>10.1f} {'—':>12} {'400':>12}")

exp  = {500.0: "~25.1", 1000.0: "~25.9", 1500.0: "~26.6", 2000.0: "~27.5"}
tdep = {500.0: "~24.6", 1000.0: "~25.1", 1500.0: "~25.6", 2000.0: "~26.0"}
for T_v, cv_v in zip(results['T_values'], results['C_V_scan']):
    print(
        f"  C_V({T_v:.0f} K) [J/mol/K]  {'':>8} {cv_v:>10.4f} "
        f"{tdep.get(T_v,'—'):>12} {exp.get(T_v,'—'):>12}"
    )
print()
print("  Harmonic DFT C_V(2000 K) ≈ 24.94 J/(mol·K)  [3R classical limit]")
print("Results saved to sqtc_bccw_v2_vasp_run/sqtc_results.json")
