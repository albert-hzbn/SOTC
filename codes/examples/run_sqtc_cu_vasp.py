#!/usr/bin/env python3
"""
SQTC+VASP benchmark: fcc Cu — strongly anharmonic near melting (T = 1200 K)
=============================================================================
Cu is a prototypical strongly anharmonic FCC metal at high temperature:

  ANHARMONICITY:
  - Grüneisen parameter γ ≈ 1.96 — large; phonon frequencies shift
    significantly with volume / temperature
  - At T = 1200 K (T/T_melt = 0.88): thermal renormalisation of IFCs is
    significant; harmonic 0 K phonons underestimate C_V
  - C_V slightly exceeds the classical Dulong–Petit limit 3R = 24.94 J/(mol·K)
    from explicit anharmonic contributions (odd-order phonon–phonon scattering)
  - C_V is DEFINED (all real phonon branches): Cu is harmonically stable,
    unlike β-Ti where 0 K IFCs give imaginary modes

  HARMONIC DFT (0 K PBE-PAW, phonopy):
  - T_D ≈ 290–310 K  (PBE slightly overbinds volume: a_PBE ≈ 3.64 vs 3.615 Å)
  - C_V(1200 K) ≈ 24.94 J/(mol·K)  (classical limit — no anharmonic correction)

  SQTC APPROACH:
  - Forces computed at 1200 K thermally displaced snapshots
  - IFCs capture renormalised (temperature-softened) force constants directly
  - Directly samples the anharmonic regime where C_V > 3R

  Experimental references (NIST / Barin & Knacke / CRC Handbook):
    T_D (calorimetric)  ≈ 315 K
    C_V( 300 K) ≈ 24.4 J/(mol·K)
    C_V( 700 K) ≈ 25.3 J/(mol·K)
    C_V(1000 K) ≈ 25.7 J/(mol·K)
    C_V(1200 K) ≈ 26.1 J/(mol·K)   (super-Dulong–Petit)

  TDEP/ab-initio at 1200 K (Hellman & Abrikosov; Born & Hellman 2019):
    T_D ≈ 280–300 K  |  C_V ≈ 25.2–25.5 J/(mol·K)
    (harmonic-renormalized; anharmonic gap ≈ 0.6 J/mol/K at 1200 K)
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

# ── fcc Cu primitive cell ──────────────────────────────────────────────────────
# a_exp(300 K) = 3.615 Å; linear thermal expansion α ≈ 17×10⁻⁶ K⁻¹.
# At 1200 K:  a(1200K) = 3.615 × (1 + 17e-6 × 900) ≈ 3.670 Å.
# Using the high-T experimental volume ensures forces are evaluated at the
# physical density where anharmonic phonon renormalisation occurs.
a_cu = 3.67  # Å  (experimental fcc Cu at ~1200 K)

prim_cell_cu = 0.5 * a_cu * np.array(
    [
        [0.0, 1.0, 1.0],
        [1.0, 0.0, 1.0],
        [1.0, 1.0, 0.0],
    ]
)
prim_pos_cu = np.array([[0.0, 0.0, 0.0]])

# ── Physical inputs ────────────────────────────────────────────────────────────
M_Cu_amu  = 63.546
# Seed from calorimetric T_D; high-T effective value is slightly lower (~290-300 K)
T_D_Cu    = 310.0   # K
T_design  = 1200.0  # K  (T/T_melt = 0.88 — strongly anharmonic regime)

print("\n" + "=" * 68)
print(" SQTC VASP Benchmark: fcc Cu (strongly anharmonic, near melting)")
print("=" * 68)
print(f"  VASP command : {VASP_CMD}")
print(f"  POTCAR path  : {PP_BASE_DIR / 'PAW_PBE' / 'Cu' / 'POTCAR'}")
print(f"  a (exp 1200K): {a_cu} Å")
print(f"  T_design     : {T_design} K  (T/T_melt = {T_design/1358:.2f})")
print()
print("  HARMONIC DFT STATUS: DEFINED")
print("    → All phonon branches real at 0 K")
print("    → Harmonic C_V(1200 K) ≈ 24.94 J/(mol·K)  [classical limit]")
print("    → Anharmonic excess: C_V(exp) − C_V(harm) ≈ +1.2 J/(mol·K)")
print()
print("  Experimental reference (NIST/CRC):")
print("    T_D (calorimetric) ≈ 315 K")
print("    C_V( 300 K) ≈ 24.4 J/(mol·K)")
print("    C_V( 700 K) ≈ 25.3 J/(mol·K)")
print("    C_V(1000 K) ≈ 25.7 J/(mol·K)")
print("    C_V(1200 K) ≈ 26.1 J/(mol·K)  (super-Dulong–Petit)")
print()
print("  TDEP at 1200 K (Hellman & Born):")
print("    T_D ≈ 280–300 K  |  C_V ≈ 25.2–25.5 J/(mol·K)")

vasp_settings = {
    "encut": 400.0,           # Cu: hard cutoff 272.9 eV; 400 eV well-converged
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
        # σ = k_B × 1200 K ≈ 0.10 eV — Methfessel-Paxton smearing
        "SIGMA": "0.10",
    },
    "timeout": 3600,
}

runner = SQTCRunner(
    element="Cu",
    mass_amu=M_Cu_amu,
    prim_cell=prim_cell_cu,
    prim_positions=prim_pos_cu,
    T=T_design,
    T_D=T_D_Cu,
    # 32-atom supercell: NN-complete for r_cutoff = 4.5 Å
    # fcc Cu 1st NN at a/√2 = 2.59 Å (×12), 2nd NN at a = 3.67 Å (×6) → 18 total
    # 9 unique ±R classes → 9×9 = 81 free params; 32×3×12 = 1152 obs → 14× overdetermined
    n_atoms_sc=32,
    force_calculator=None,
    vasp_cmd=VASP_CMD,
    vasp_settings=vasp_settings,
    r_cutoff=4.5,
    r_max_corr=9.0,
    n_ensemble=12,
    work_dir="sqtc_cu_vasp_run",
    verbosity=1,
    ridge_alpha=1e-3,
    # FCC: central-force projection valid; reduces 81 → 18 params (2 per shell × 9 shells)
    symmetrize_bonds=True,
)

results = runner.run(
    n_sc_iterations=12,
    epsilon_conv=0.003,
    mixing=0.30,
    T_values=np.array([300.0, 500.0, 700.0, 1000.0, 1200.0]),
    q_mesh_cv=(6, 6, 6),
)

print("\n--- SQTC fcc Cu benchmark summary ---")
print(f"Converged : {results['converged']}")
print(f"Iterations: {results['n_iterations']}")
print()
print(f"{'Quantity':<38} {'SQTC':>10} {'TDEP (lit)':>12} {'Experiment':>12}")
print("-" * 76)
td_spectral = results['T_D_effective']
td_caloric  = results.get('T_D_caloric', float('nan'))
print(f"{'T_D (spectral-moment) [K]':<38} {td_spectral:>10.1f} {'280–300':>12} {'315':>12}")
print(f"{'T_D (calorimetric) [K]':<38} {td_caloric:>10.1f} {'—':>12} {'315':>12}")

exp  = {300.0: "~24.4", 500.0: "~24.9", 700.0: "~25.3", 1000.0: "~25.7", 1200.0: "~26.1"}
tdep = {300.0: "~24.0", 500.0: "~24.4", 700.0: "~24.8", 1000.0: "~25.2", 1200.0: "~25.5"}
for T_v, cv_v in zip(results['T_values'], results['C_V_scan']):
    print(
        f"  C_V({T_v:.0f} K) [J/mol/K]  {'':>8} {cv_v:>10.4f} "
        f"{tdep.get(T_v,'—'):>12} {exp.get(T_v,'—'):>12}"
    )
print()
print("  Harmonic DFT C_V(1200 K) ≈ 24.94 J/(mol·K)  [3R classical limit]")
print("Results saved to sqtc_cu_vasp_run/sqtc_results.json")
