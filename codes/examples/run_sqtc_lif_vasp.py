#!/usr/bin/env python3
"""
SQTC+VASP benchmark: LiF (rocksalt) — lightest ionic compound (T = 800 K)
==========================================================================
LiF is the lightest member of the alkali fluoride series and provides a
qualitatively different benchmark from NaCl and MgO:

  ANHARMONICITY (moderate-to-high for an ionic crystal):
  - Grüneisen parameter γ ≈ 1.68 — moderate (higher than MgO: 1.5, close to NaCl: 1.6)
  - T_D ≈ 730 K — HIGHEST Debye temperature of any alkali halide (due to light Li)
  - At T = 800 K (T/T_melt = 0.71): T/T_D ≈ 1.10 — barely above Debye T; quantum
    corrections to C_V are still significant even at this temperature
  - C_V(800 K) ≈ 47.5 J/(mol·K): still ~2.4 J/(mol·K) below the classical limit 6R

  CONTRAST WITH NaCl AND MgO:
  - LiF: T_D = 730 K — highest alkali halide Debye T; quantum corrections visible at 800 K
  - NaCl: T_D = 321 K — moderate; fully classical at 700 K
  - MgO:  T_D = 760 K — similar to LiF but much harder oxide; different bonding regime
  - LiF fills the gap between MgO (stiff oxide) and NaCl (soft halide) in the ionic series:
    same rocksalt structure, intermediate stiffness, but unique because T/T_D ≈ 1 at T_design

  LO-TO SPLITTING (most extreme in this benchmark set):
  - Born effective charges: Z*_Li ≈ +1.01 e, Z*_F ≈ −1.01 e (close to formal charges)
  - Infrared-active optical mode at Γ: TO ≈ 9.3 THz, LO ≈ 15.0 THz
  - LO-TO gap ΔΩ ≈ 5.7 THz — larger than NaCl (4.7 THz); see PbTe docstring
  - Short-range IFC framework gives TO frequency only; the absence of LO correction
    will cause T_D(SQTC) to be noticeably underestimated vs experiment.
    LiF is thus a useful calibration point for the LO-TO systematic error.

  STRUCTURE — rocksalt (B1, space group Fm-3m):
  - FCC Bravais lattice, 2-atom primitive cell: Li at (0,0,0), F at (a/2, 0, 0)
  - a_exp(300 K) = 4.027 Å; α ≈ 37×10⁻⁶ K⁻¹ (large for ionic crystal)
  - At 800 K:  a(800K) = 4.027 × (1 + 37e-6 × 500) ≈ 4.101 Å
  - NN distance: Li–F = a/2 = 2.050 Å (×6)
  - 2nd NN: Li–Li = a/√2 = 2.900 Å (×12)

  VASP NOTE:
  - Li PAW: ENMAX = 271 eV; F PAW (hard): ENMAX = 400.0 eV.
    Use ENCUT = 550 eV (>1.3× max) for well-converged forces with F.
  - LiF is a wide-gap insulator (experimental gap ~14 eV; PBE ~9 eV):
    use Gaussian smearing ISMEAR=0, SIGMA=0.05 for safe integration.

  Experimental references (NIST JANAF / Stull & Prophet / Dworkin & Bredig 1968):
    T_D (calorimetric) ≈ 730 K
    C_V( 300 K) ≈ 38.0 J/(mol·K)
    C_V( 500 K) ≈ 44.0 J/(mol·K)
    C_V( 700 K) ≈ 46.5 J/(mol·K)
    C_V( 800 K) ≈ 47.5 J/(mol·K)
    C_V(1000 K) ≈ 48.8 J/(mol·K)

  TDEP/phonopy at 800 K (Hellman; Born & Hellman):
    T_D ≈ 670–710 K  |  C_V ≈ 46.5–47.0 J/(mol·K)
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

# ── LiF rocksalt primitive cell ───────────────────────────────────────────────
# a_exp(300 K) = 4.027 Å; α ≈ 37×10⁻⁶ K⁻¹ (large thermal expansion).
# At 800 K:  a(800K) = 4.027 × (1 + 37e-6 × 500) ≈ 4.101 Å → use 4.10 Å.
a_lif = 4.10  # Å  (experimental LiF at ~800 K)

prim_cell_lif = 0.5 * a_lif * np.array(
    [
        [0.0, 1.0, 1.0],
        [1.0, 0.0, 1.0],
        [1.0, 1.0, 0.0],
    ]
)
# Cartesian positions: Li at origin, F at (a/2, 0, 0)
prim_pos_lif = np.array(
    [
        [0.0,        0.0, 0.0],  # Li
        [a_lif / 2,  0.0, 0.0],  # F
    ]
)

# ── Physical inputs ────────────────────────────────────────────────────────────
M_Li_amu  = 6.941
M_F_amu   = 18.998
# Seed from calorimetric T_D (730 K); at 800 K renormalised T_D ≈ 670–710 K
T_D_LiF   = 710.0   # K
T_design  = 800.0   # K  (T/T_melt = 0.71, T/T_D ≈ 1.10)

print("\n" + "=" * 70)
print(" SQTC VASP Benchmark: LiF rocksalt (lightest ionic halide, T ≈ T_D)")
print("=" * 70)
print(f"  VASP command : {VASP_CMD}")
print(f"  POTCAR       : Li + F  (PAW_PBE)")
print(f"  a (exp 800K) : {a_lif} Å")
print(f"  T_design     : {T_design} K  (T/T_melt = {T_design/1121:.2f}, T/T_D ≈ {T_design/T_D_LiF:.2f})")
print()
print("  KEY FEATURES:")
print("    → T/T_D ≈ 1.10: running near Debye temperature — partial quantum regime")
print("    → C_V still ~2 J/(mol·K) below classical 6R = 49.88 J/(mol·K) at 800 K")
print("    → Highest T_D of alkali halides (due to light Li: mass = 6.941 amu)")
print()
print("  LO-TO SPLITTING (most extreme in benchmark set):")
print("    → TO ≈ 9.3 THz, LO ≈ 15.0 THz (gap ΔΩ ≈ 5.7 THz — larger than NaCl)")
print("    → Short-range IFC framework gives TO only; T_D(SQTC) will be biased low")
print("    → Compare with NaCl: quantifies LO-TO systematic error vs gap magnitude")
print()
print("  HARMONIC DFT STATUS: DEFINED (all real modes)")
print("    → All phonon branches real; LiF is harmonically stable")
print("    → γ ≈ 1.68 — moderate anharmonicity")
print()
print("  Experimental reference (NIST JANAF / Stull & Prophet):")
print("    T_D (calorimetric) ≈ 730 K")
print("    C_V( 300 K) ≈ 38.0 J/(mol·K)")
print("    C_V( 500 K) ≈ 44.0 J/(mol·K)")
print("    C_V( 700 K) ≈ 46.5 J/(mol·K)")
print("    C_V( 800 K) ≈ 47.5 J/(mol·K)")
print("    C_V(1000 K) ≈ 48.8 J/(mol·K)")
print()
print("  TDEP at 800 K:")
print("    T_D ≈ 670–710 K  |  C_V ≈ 46.5–47.0 J/(mol·K)")

vasp_settings = {
    # F PAW: ENMAX_hard = 400.0 eV; Li PAW: ENMAX = 271 eV.
    # Use ENCUT = 550 eV (>1.3× max) for well-converged forces with the hard F PAW.
    "encut": 550.0,
    "functional": "PBE",
    "ncore": NCORE_INCAR,
    "kgrid": (4, 4, 4),
    "pp_base_dir": PP_BASE_DIR,
    "pp_set": "PAW_PBE",
    "extra_incar": {
        "KSPACING": "0.25",
        "EDIFF": "1E-7",
        "PREC": "Accurate",
        # Gaussian smearing for wide-gap ionic insulator (PBE gap ~9 eV).
        "ISMEAR": "0",
        "SIGMA": "0.05",
    },
    "timeout": 3600,
}

runner = SQTCRunner(
    element="Li",
    mass_amu=M_Li_amu,
    elements=["Li", "F"],
    masses_amu=[M_Li_amu, M_F_amu],
    prim_cell=prim_cell_lif,
    prim_positions=prim_pos_lif,
    T=T_design,
    T_D=T_D_LiF,
    # 64-atom supercell (32 f.u.): NN-complete for r_cutoff = 3.5 Å
    # LiF rocksalt: 1st NN Li–F at a/2 = 2.05 Å (×6), 2nd NN Li–Li at a/√2 = 2.90 Å (×12)
    # r_cutoff = 3.5 Å captures both shells with margin
    n_atoms_sc=64,
    force_calculator=None,
    vasp_cmd=VASP_CMD,
    vasp_settings=vasp_settings,
    r_cutoff=3.5,
    r_max_corr=8.0,
    n_ensemble=12,
    work_dir="sqtc_lif_vasp_run",
    verbosity=1,
    ridge_alpha=1e-3,
    # Rocksalt LiF: central-force symmetry valid for each sublattice (FCC-based)
    symmetrize_bonds=True,
)

results = runner.run(
    n_sc_iterations=12,
    epsilon_conv=0.003,
    mixing=0.30,
    T_values=np.array([300.0, 500.0, 700.0, 800.0, 1000.0]),
    q_mesh_cv=(6, 6, 6),
)

print("\n--- SQTC LiF benchmark summary ---")
print(f"Converged : {results['converged']}")
print(f"Iterations: {results['n_iterations']}")
print()
print(f"{'Quantity':<38} {'SQTC':>10} {'TDEP (lit)':>12} {'Experiment':>12}")
print("-" * 76)
td = results['T_D_effective']
td_c = results.get('T_D_caloric', float('nan'))
print(f"{'T_D (spectral-moment) [K]':<38} {td:>10.1f} {'670–710':>12} {'730':>12}")
print(f"{'T_D (calorimetric) [K]':<38} {td_c:>10.1f} {'—':>12} {'730':>12}")

exp  = {300.0: "~38.0", 500.0: "~44.0", 700.0: "~46.5", 800.0: "~47.5", 1000.0: "~48.8"}
tdep = {300.0: "~37.0", 500.0: "~43.5", 700.0: "~46.2", 800.0: "~46.7", 1000.0: "~48.0"}
for T_v, cv_v in zip(results['T_values'], results['C_V_scan']):
    print(f"  C_V({T_v:.0f} K) [J/mol/K]  {'':>5} {cv_v:>10.4f} {tdep.get(T_v,'—'):>12} {exp.get(T_v,'—'):>12}")
print()
print("  Classical limit 6R = 49.883 J/(mol·K)  [per formula unit]")
print("  NOTE: T/T_D ≈ 1.1 — C_V still ~2.4 J/(mol·K) below classical at 800 K")
print("  NOTE: LO-TO splitting (ΔΩ ≈ 5.7 THz) not included; largest gap in set")
print("  NOTE: Compare T_D bias vs NaCl (ΔΩ ≈ 4.7 THz) and MgO (ΔΩ ≈ 6 THz)")
print("Results saved to sqtc_lif_vasp_run/sqtc_results.json")
