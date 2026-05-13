#!/usr/bin/env python3
"""
SQTC+VASP benchmark: NaCl (rocksalt) — MODERATE anharmonicity ionic compound
=============================================================================
NaCl (rock salt) is the prototypical moderately anharmonic ionic compound:

  ANHARMONICITY (moderate):
  - Grüneisen parameter γ ≈ 1.6 — moderate; larger than MgO (1.5), smaller than PbTe
  - T_D ≈ 321 K — moderate; below room temperature for Na but above for Cl
  - Softer than MgO due to larger ion radii and weaker Madelung energy
  - Well-characterized phonon dispersion: significant acoustic-optic coupling
  - C_V(700 K) ≈ 50.5 J/(mol·K): small but measurable excess above 6R = 49.88 J/(mol·K)
  - γ is temperature-dependent: rises from 1.56 at 300 K to ~1.7 at 900 K

  HARMONIC DFT (0 K PBE-PAW, phonopy):
  - T_D ≈ 270–310 K  (PBE overestimates volume by ~2%; softens phonons)
  - C_V(700 K) ≈ 49.9 J/(mol·K)  (near classical; small acoustic anharmonic contribution)
  - All phonon modes real; NO imaginary branches

  STRUCTURE — rocksalt (B1, space group Fm-3m):
  - FCC Bravais lattice, 2-atom primitive cell: Na at (0,0,0), Cl at (a/2, 0, 0)
  - a_exp(300 K) = 5.640 Å  (most precisely measured rocksalt parameter)
  - NN distance: Na–Cl = a/2 = 2.820 Å (×6 per ion)
  - 2nd NN: Na–Na = a/√2 = 3.989 Å (×12 per ion)

  NOTE on LO-TO splitting:
  - NaCl has a large LO-TO gap (ΔTO≈4.7 THz, LO≈7.9 THz at Γ).
  - Short-range IFCs without explicit Born-charge Ewald correction give the
    TO frequency; the LO branch is missing → T_D(SQTC) will be biased low.
  - To isolate this effect, compare T_D(SQTC) with T_D from TO modes only.

  Experimental references (NIST JANAF / Dworkin & Bredig 1968 / Ritter 1985):
    T_D (calorimetric, low-T fit) ≈ 321 K
    C_V( 200 K) ≈ 44.0 J/(mol·K)
    C_V( 300 K) ≈ 49.0 J/(mol·K)
    C_V( 400 K) ≈ 49.9 J/(mol·K)
    C_V( 700 K) ≈ 50.5 J/(mol·K)
    C_V( 900 K) ≈ 51.2 J/(mol·K)

  TDEP/phonopy at 700 K (Hellman et al.; Born & Hellman):
    T_D ≈ 260–290 K  |  C_V ≈ 49.8–50.1 J/(mol·K)
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

# ── NaCl rocksalt primitive cell ──────────────────────────────────────────────
# a_exp(300 K) = 5.640 Å; thermal expansion α ≈ 40×10⁻⁶ K⁻¹ (large for NaCl).
# At 700 K:  a(700K) = 5.640 × (1 + 40e-6 × 400) ≈ 5.730 Å.
a_nacl = 5.73  # Å  (experimental NaCl at ~700 K)

prim_cell_nacl = 0.5 * a_nacl * np.array(
    [
        [0.0, 1.0, 1.0],
        [1.0, 0.0, 1.0],
        [1.0, 1.0, 0.0],
    ]
)
# Cartesian positions: Na at origin, Cl at (a/2, 0, 0)
prim_pos_nacl = np.array([
    [0.0,        0.0, 0.0],  # Na
    [a_nacl/2,   0.0, 0.0],  # Cl
])

# ── Physical inputs ────────────────────────────────────────────────────────────
M_Na_amu  = 22.9898
M_Cl_amu  = 35.453
T_D_NaCl  = 310.0   # K  (seed close to experimental T_D = 321 K)
T_design  = 700.0   # K  (T/T_D ≈ 2.3; above classical transition, moderate anharmonicity)

print("\n" + "=" * 70)
print(" SQTC VASP Benchmark: NaCl rocksalt (moderate anharmonicity)")
print("=" * 70)
print(f"  VASP command : {VASP_CMD}")
print(f"  POTCAR       : Na + Cl  (PAW_PBE)")
print(f"  a (exp 700K) : {a_nacl} Å")
print(f"  T_design     : {T_design} K  (T/T_D = {T_design/T_D_NaCl:.1f})")
print()
print("  HARMONIC DFT STATUS: DEFINED (all real modes)")
print("    → γ ≈ 1.6 — moderate (between MgO: 1.5 and PbTe: ~2.3)")
print("    → LO-TO splitting present but NOT included in IFC framework")
print()
print("  Experimental reference (NIST JANAF / Dworkin & Bredig 1968):")
print("    T_D (calorimetric) ≈ 321 K")
print("    C_V( 300 K) ≈ 49.0 J/(mol·K)")
print("    C_V( 700 K) ≈ 50.5 J/(mol·K)")
print("    C_V( 900 K) ≈ 51.2 J/(mol·K)")

vasp_settings = {
    "encut": 400.0,           # Na/Cl: Cl hard 262.5 eV; 400 eV well-converged
    "functional": "PBE",
    "ncore": NCORE_INCAR,
    "kgrid": (4, 4, 4),
    "pp_base_dir": PP_BASE_DIR,
    "pp_set": "PAW_PBE",
    "extra_incar": {
        "KSPACING": "0.25",
        "EDIFF": "1E-7",
        "PREC": "Accurate",
        # Gaussian smearing for ionic insulator (band gap ~8.9 eV)
        "ISMEAR": "0",
        "SIGMA": "0.05",
    },
    "timeout": 3600,
}

runner = SQTCRunner(
    element="Na",
    mass_amu=M_Na_amu,
    elements=["Na", "Cl"],
    masses_amu=[M_Na_amu, M_Cl_amu],
    prim_cell=prim_cell_nacl,
    prim_positions=prim_pos_nacl,
    T=T_design,
    T_D=T_D_NaCl,
    # 64-atom supercell (32 f.u.): NN-complete for r_cutoff = 4.3 Å
    # NaCl:  1st NN Na–Cl at 2.865 Å (×6), 2nd NN Na–Na at 4.053 Å (×12)
    # r_cutoff = 4.3 Å captures both shells
    n_atoms_sc=64,
    force_calculator=None,
    vasp_cmd=VASP_CMD,
    vasp_settings=vasp_settings,
    r_cutoff=4.3,
    r_max_corr=9.0,
    n_ensemble=12,
    work_dir="sqtc_nacl_vasp_run",
    verbosity=1,
    ridge_alpha=1e-3,
    symmetrize_bonds=True,
)

results = runner.run(
    n_sc_iterations=12,
    epsilon_conv=0.003,
    mixing=0.30,
    T_values=np.array([200.0, 300.0, 400.0, 700.0, 900.0]),
    q_mesh_cv=(6, 6, 6),
)

print("\n--- SQTC NaCl benchmark summary ---")
print(f"Converged : {results['converged']}")
print(f"Iterations: {results['n_iterations']}")
print()
print(f"{'Quantity':<38} {'SQTC':>10} {'TDEP (lit)':>12} {'Experiment':>12}")
print("-" * 76)
td = results['T_D_effective']
td_c = results.get('T_D_caloric', float('nan'))
print(f"{'T_D (spectral-moment) [K]':<38} {td:>10.1f} {'260–290':>12} {'321':>12}")
print(f"{'T_D (calorimetric) [K]':<38} {td_c:>10.1f} {'—':>12} {'321':>12}")

exp  = {200.0: "~44.0", 300.0: "~49.0", 400.0: "~49.9", 700.0: "~50.5", 900.0: "~51.2"}
tdep = {200.0: "~42.5", 300.0: "~48.5", 400.0: "~49.5", 700.0: "~50.0", 900.0: "~50.4"}
for T_v, cv_v in zip(results['T_values'], results['C_V_scan']):
    print(f"  C_V({T_v:.0f} K) [J/mol/K]  {'':>5} {cv_v:>10.4f} {tdep.get(T_v,'—'):>12} {exp.get(T_v,'—'):>12}")
print()
print("  Classical limit 6R = 49.883 J/(mol·K)  [per formula unit]")
print("  NOTE: LO-TO splitting not included; TO-only T_D expected")
print("Results saved to sqtc_nacl_vasp_run/sqtc_results.json")
