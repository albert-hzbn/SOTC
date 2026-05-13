#!/usr/bin/env python3
"""
SQTC+VASP benchmark: MgO (rocksalt) — LOW anharmonicity reference compound
============================================================================
MgO (periclase) is the prototypical stiff ionic oxide with very low anharmonicity:

  ANHARMONICITY (low):
  - Grüneisen parameter γ ≈ 1.5 — modest; one of the least anharmonic oxides
  - T_D ≈ 760 K — very high; quantum effects persist to >500 K
  - Strong ionic bonding: Madelung energy ~25 eV/f.u.; stiff short-range repulsion
  - No dynamical instabilities at any temperature (stable rock-salt structure)
  - C_V barely exceeds 6R = 49.88 J/(mol·K) even at 2000 K: Δ(C_V) < 1 J/(mol·K)
  - Dominant phonon contribution: acoustic modes; optical modes at ≈480 cm⁻¹ (IR)

  HARMONIC DFT (0 K PBE-PAW, phonopy):
  - T_D ≈ 720–760 K  (PBE slightly overestimates volume; a_PBE ≈ 4.25 vs a_exp = 4.211 Å)
  - C_V(1500 K) ≈ 49.6 J/(mol·K)  (near classical limit 6R = 49.88 J/(mol·K))
  - All phonon modes real; stable at 0 K

  STRUCTURE — rocksalt (B1, space group Fm-3m):
  - FCC Bravais lattice, 2-atom primitive cell: Mg at (0,0,0), O at (a/2, 0, 0)
  - Conventional parameter a_exp(300 K) = 4.211 Å
  - NN distance: Mg–O = a/2 = 2.106 Å (×6 per atom)
  - 2nd NN: Mg–Mg = a/√2 = 2.979 Å (×12 per atom)

  NOTE on LO-TO splitting:
  - SQTC uses displacement–force regression (short-range IFCs only).
  - Long-range electrostatic contribution (Born charges + dielectric screening)
    is NOT included via an explicit Ewald sum. VASP forces include LR effects
    implicitly, but the IFC interpolation lacks the LR analytic correction.
  - The optical phonon branch at Γ will show the TO frequency; the LO-TO gap
    (Δω ≈ 6 THz for MgO) is not captured → T_D may be slightly underestimated.
  - This is a known limitation common to all short-range IFC methods without
    explicit Born charge correction.

  Experimental references (NIST JANAF / Stull & Prophet / Isaak et al. 1989):
    T_D (calorimetric) ≈ 760 K  (low-T heat-capacity fit)
    C_V( 300 K) ≈ 37.2 J/(mol·K)
    C_V( 500 K) ≈ 44.6 J/(mol·K)
    C_V(1000 K) ≈ 48.5 J/(mol·K)
    C_V(1500 K) ≈ 49.6 J/(mol·K)

  TDEP/phonopy at 1500 K (Hellman et al.; Karki et al. 2000):
    T_D ≈ 700–750 K  |  C_V ≈ 49.2–49.5 J/(mol·K)
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

# ── MgO rocksalt primitive cell ───────────────────────────────────────────────
# a_exp(300 K) = 4.211 Å; thermal expansion α ≈ 10.5×10⁻⁶ K⁻¹.
# At 1500 K:  a(1500K) = 4.211 × (1 + 10.5e-6 × 1200) ≈ 4.264 Å.
a_mgo = 4.26  # Å  (experimental MgO at ~1500 K)

# FCC primitive cell — rocksalt is FCC Bravais + 2-atom basis
prim_cell_mgo = 0.5 * a_mgo * np.array(
    [
        [0.0, 1.0, 1.0],
        [1.0, 0.0, 1.0],
        [1.0, 1.0, 0.0],
    ]
)
# Cartesian positions in the primitive cell:
#   Mg at origin, O displaced by a/2 along [100]
prim_pos_mgo = np.array([
    [0.0,       0.0, 0.0],   # Mg
    [a_mgo/2,   0.0, 0.0],   # O
])

# ── Physical inputs ────────────────────────────────────────────────────────────
M_Mg_amu  = 24.305
M_O_amu   = 15.999
# Seed T_D close to experimental; PBE harmonic gives 720–760 K
T_D_MgO   = 740.0   # K
T_design  = 1500.0  # K  (T/T_D = 2.0; well into classical regime)

print("\n" + "=" * 70)
print(" SQTC VASP Benchmark: MgO rocksalt (low anharmonicity)")
print("=" * 70)
print(f"  VASP command : {VASP_CMD}")
print(f"  POTCAR       : Mg + O  (PAW_PBE)")
print(f"  a (exp 1500K): {a_mgo} Å")
print(f"  T_design     : {T_design} K  (T/T_D = {T_design/T_D_MgO:.1f})")
print()
print("  HARMONIC DFT STATUS: DEFINED (all real modes)")
print("    → All phonon branches real; no dynamic instabilities")
print("    → γ ≈ 1.5 — lowest Grüneisen of Cu/Ag/Au/NaCl/PbTe comparison set")
print("    → LO-TO splitting NOT included (short-range IFCs only)")
print()
print("  Experimental reference (NIST JANAF / Isaak 1989):")
print("    T_D (calorimetric) ≈ 760 K")
print("    C_V( 300 K) ≈ 37.2 J/(mol·K)")
print("    C_V( 500 K) ≈ 44.6 J/(mol·K)")
print("    C_V(1000 K) ≈ 48.5 J/(mol·K)")
print("    C_V(1500 K) ≈ 49.6 J/(mol·K)")

vasp_settings = {
    "encut": 500.0,           # Mg/O: high ENCUT needed for O (hard PAW); 500 eV converged
    "functional": "PBE",
    "ncore": NCORE_INCAR,
    "kgrid": (4, 4, 4),
    "pp_base_dir": PP_BASE_DIR,
    "pp_set": "PAW_PBE",
    "extra_incar": {
        "KSPACING": "0.25",
        "EDIFF": "1E-7",
        "PREC": "Accurate",
        # Gaussian smearing for insulator (band gap ~7.8 eV)
        "ISMEAR": "0",
        "SIGMA": "0.05",
        # LDA+U not needed for MgO in ground state; use PBE directly
    },
    "timeout": 3600,
}

runner = SQTCRunner(
    element="Mg",             # primary element label (used for display)
    mass_amu=M_Mg_amu,        # fallback single-species mass
    elements=["Mg", "O"],     # per-basis-atom species
    masses_amu=[M_Mg_amu, M_O_amu],  # per-basis-atom masses [amu]
    prim_cell=prim_cell_mgo,
    prim_positions=prim_pos_mgo,
    T=T_design,
    T_D=T_D_MgO,
    # 64-atom supercell (32 f.u.): NN-complete for r_cutoff = 3.2 Å
    # Rocksalt MgO:  1st NN Mg–O at 2.13 Å (×6), 2nd NN Mg–Mg at 3.01 Å (×12)
    # r_cutoff = 3.2 Å captures both shells; 32 f.u. × 2 atoms = 64 atoms total
    n_atoms_sc=64,
    force_calculator=None,
    vasp_cmd=VASP_CMD,
    vasp_settings=vasp_settings,
    r_cutoff=3.2,
    r_max_corr=8.0,
    n_ensemble=12,
    work_dir="sqtc_mgo_vasp_run",
    verbosity=1,
    ridge_alpha=1e-3,
    # FCC-based rocksalt: central-force symmetry valid for each sublattice
    symmetrize_bonds=True,
)

results = runner.run(
    n_sc_iterations=12,
    epsilon_conv=0.003,
    mixing=0.30,
    T_values=np.array([300.0, 500.0, 800.0, 1000.0, 1200.0, 1500.0]),
    q_mesh_cv=(6, 6, 6),
)

print("\n--- SQTC MgO benchmark summary ---")
print(f"Converged : {results['converged']}")
print(f"Iterations: {results['n_iterations']}")
print()
print(f"{'Quantity':<38} {'SQTC':>10} {'TDEP (lit)':>12} {'Experiment':>12}")
print("-" * 76)
td = results['T_D_effective']
td_c = results.get('T_D_caloric', float('nan'))
print(f"{'T_D (spectral-moment) [K]':<38} {td:>10.1f} {'700–750':>12} {'760':>12}")
print(f"{'T_D (calorimetric) [K]':<38} {td_c:>10.1f} {'—':>12} {'760':>12}")

exp  = {300.0: "~37.2", 500.0: "~44.6", 800.0: "~47.6", 1000.0: "~48.5", 1200.0: "~49.2", 1500.0: "~49.6"}
tdep = {300.0: "~36.5", 500.0: "~44.0", 800.0: "~47.3", 1000.0: "~48.2", 1200.0: "~48.9", 1500.0: "~49.3"}
for T_v, cv_v in zip(results['T_values'], results['C_V_scan']):
    print(f"  C_V({T_v:.0f} K) [J/mol/K]  {'':>5} {cv_v:>10.4f} {tdep.get(T_v,'—'):>12} {exp.get(T_v,'—'):>12}")
print()
print("  Classical limit 6R = 49.883 J/(mol·K)  [per formula unit]")
print("  NOTE: LO-TO splitting not included in short-range IFC framework")
print("Results saved to sqtc_mgo_vasp_run/sqtc_results.json")
