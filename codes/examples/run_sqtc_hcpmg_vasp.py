#!/usr/bin/env python3
"""
SQTC+VASP benchmark: hcp Mg — hexagonal close-packed metal at 700 K
=====================================================================
Mg is an ideal HCP benchmark because it provides the first non-cubic structure
in this test suite. HCP requires a non-trivial primitive cell with a 2-atom basis
and a c/a ratio that differs from FCC/BCC:

  ANHARMONICITY (moderate):
  - Grüneisen parameter γ ≈ 1.46 — moderate (between Mo: 1.57 and Si: 1.06)
  - c/a ≈ 1.624 (close to ideal 1.633); near-ideal hexagonal geometry
  - T_D ≈ 400 K (calorimetric Debye fit)
  - At T = 700 K (T/T_melt = 0.76): T/T_D ≈ 1.75 — moderately anharmonic regime
  - C_V(700 K) ≈ 26.5 J/(mol·K): moderate anharmonic excess above 3R = 24.94

  HCP STRUCTURE — first hexagonal benchmark in this set:
  - Hexagonal Bravais lattice, 2-atom primitive cell
  - a₁ = [a, 0, 0],  a₂ = [-a/2, a√3/2, 0],  a₃ = [0, 0, c]
  - Mg atom 1 at (0, 0, 0); Mg atom 2 at fractional (1/3, 2/3, 1/2)
    = Cartesian [0, a/√3, c/2]
  - a_exp(300 K) = 3.209 Å, c_exp(300 K) = 5.211 Å  (c/a = 1.624)
  - α_a ≈ 25×10⁻⁶ K⁻¹, α_c ≈ 26×10⁻⁶ K⁻¹ (mildly anisotropic)

  KEY PHYSICS:
  - 1st NN shell: 12 atoms at distance a ≈ 3.22 Å (6 in basal plane + 6 across layer)
    (ideal c/a: all 12 exactly equidistant; Mg is very close to ideal)
  - 2nd NN shell: 6 atoms at distance c ≈ 5.24 Å (across 2 layers along c-axis)
  - HCP lacks inversion symmetry in the same sense as FCC/BCC; the 2-atom basis
    means IFCs have more independent components than in monatomic FCC/BCC
  - Light mass (24.3 amu) → quantum zero-point energy is non-negligible even at 700 K:
    ½ × ℏω_D / k_B ≈ 200 K; at T = 700 K, T >> T_D/2 so classical approx is adequate

  HARMONIC DFT (0 K PBE-PAW, phonopy):
  - T_D ≈ 375–405 K  (PBE slightly underestimates a by ~1%, stiffens phonons slightly)
  - C_V(700 K) ≈ 24.94 J/(mol·K)  [classical limit]
  - All phonon modes real; Mg is harmonically stable

  Experimental references (NIST JANAF / Barin & Knacke / Touloukian 1970):
    T_D (calorimetric) ≈ 400 K
    C_V( 300 K) ≈ 24.9 J/(mol·K)
    C_V( 400 K) ≈ 25.6 J/(mol·K)
    C_V( 500 K) ≈ 26.0 J/(mol·K)
    C_V( 700 K) ≈ 26.5 J/(mol·K)

  TDEP/phonopy at 700 K (Hellman; Mozafari & Hellman 2018):
    T_D ≈ 360–390 K  |  C_V ≈ 25.0–25.3 J/(mol·K)
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

# ── hcp Mg primitive cell ──────────────────────────────────────────────────────
# a_exp(300 K) = 3.209 Å, c_exp = 5.211 Å (c/a = 1.624, near ideal 1.633)
# α_a ≈ 25×10⁻⁶ K⁻¹, α_c ≈ 26×10⁻⁶ K⁻¹ (nearly isotropic expansion).
# At 700 K:  a(700K) = 3.209 × (1 + 25e-6 × 400) ≈ 3.241 Å → use 3.24 Å
#            c(700K) = 5.211 × (1 + 26e-6 × 400) ≈ 5.265 Å → use 5.26 Å
#            c/a = 5.26 / 3.24 = 1.623 (maintained near ideal)
a_mg = 3.24  # Å  (experimental hcp Mg at ~700 K)
c_mg = 5.26  # Å  (experimental hcp Mg at ~700 K, c/a = 1.623)

# HCP primitive cell — hexagonal lattice vectors (right-hand convention):
#   a₁ along [100], a₂ in the basal plane at 120°, a₃ along [001]
prim_cell_mg = np.array(
    [
        [a_mg,          0.0,                   0.0],   # a₁
        [-a_mg / 2,     a_mg * np.sqrt(3) / 2, 0.0],   # a₂
        [0.0,           0.0,                   c_mg],   # a₃
    ]
)
# Cartesian atom positions:
#   Mg₁ at (0, 0, 0)
#   Mg₂ at fractional (1/3, 2/3, 1/2) = Cartesian (0, a/√3, c/2)
#   Verify: (1/3)a₁ + (2/3)a₂ + (1/2)a₃
#         = [a/3, 0, 0] + [-a/3, a/√3, 0] + [0, 0, c/2]
#         = [0,   a/√3, c/2]  ✓
prim_pos_mg = np.array(
    [
        [0.0,                  0.0,                   0.0],   # Mg₁
        [0.0,  a_mg / np.sqrt(3),                c_mg / 2],   # Mg₂
    ]
)

# ── Physical inputs ────────────────────────────────────────────────────────────
M_Mg_amu  = 24.305
# Seed from calorimetric T_D; at 700 K effective renormalised value ≈ 360–390 K
T_D_Mg    = 390.0   # K
T_design  = 700.0   # K  (T/T_melt = 0.76, T/T_D ≈ 1.75)

print("\n" + "=" * 70)
print(" SQTC VASP Benchmark: hcp Mg (first HCP structure in benchmark set)")
print("=" * 70)
print(f"  VASP command : {VASP_CMD}")
print(f"  POTCAR path  : {PP_BASE_DIR / 'PAW_PBE' / 'Mg' / 'POTCAR'}")
print(f"  a (exp 700K) : {a_mg} Å")
print(f"  c (exp 700K) : {c_mg} Å  (c/a = {c_mg/a_mg:.3f}, ideal = 1.633)")
print(f"  T_design     : {T_design} K  (T/T_melt = {T_design/923:.2f}, T/T_D ≈ {T_design/T_D_Mg:.2f})")
print()
print("  STRUCTURE TYPE: HCP — first hexagonal benchmark in this set")
print("    → 2-atom primitive cell; c/a ≈ 1.623 (near ideal 1.633)")
print("    → IFCs have more independent components than FCC/BCC due to 2-atom basis")
print()
print("  HARMONIC DFT STATUS: DEFINED (all real modes)")
print("    → Mg is harmonically stable in hcp phase")
print("    → Harmonic C_V(700 K) ≈ 24.94 J/(mol·K)  [classical limit]")
print("    → Anharmonic excess: C_V(exp) − C_V(harm) ≈ +1.5 J/(mol·K) at 700 K")
print()
print("  Experimental reference (NIST JANAF / Touloukian 1970):")
print("    T_D (calorimetric) ≈ 400 K")
print("    C_V( 300 K) ≈ 24.9 J/(mol·K)")
print("    C_V( 400 K) ≈ 25.6 J/(mol·K)")
print("    C_V( 500 K) ≈ 26.0 J/(mol·K)")
print("    C_V( 700 K) ≈ 26.5 J/(mol·K)")
print()
print("  TDEP at 700 K (Hellman; Mozafari & Hellman):")
print("    T_D ≈ 360–390 K  |  C_V ≈ 25.0–25.3 J/(mol·K)")

vasp_settings = {
    # Mg: PAW soft (default) cutoff 200 eV; 400 eV gives well-converged forces.
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
        # σ = k_B × 700 K ≈ 0.06 eV; use 0.08 eV for metallic Mg
        "SIGMA": "0.08",
    },
    "timeout": 3600,
}

runner = SQTCRunner(
    element="Mg",
    mass_amu=M_Mg_amu,
    prim_cell=prim_cell_mg,
    prim_positions=prim_pos_mg,
    T=T_design,
    T_D=T_D_Mg,
    # 72-atom supercell (36 f.u.): NN-complete for r_cutoff = 5.5 Å
    # hcp Mg NN shells:
    #   1st NN: 12 atoms at ~3.23 Å (6 in-plane Mg1-Mg1 + 6 inter-basis Mg1-Mg2)
    #   2nd NN: 6 Mg1-Mg2 inter-basis pairs at ~4.57 Å  ← critical, just outside old 4.5 Å cutoff
    #   3rd NN: 2 Mg1-Mg1 at c = 5.26 Å (along c-axis)
    # r_cutoff = 5.5 Å includes all three shells; supercell > 2×5.5 = 11 Å in all directions
    n_atoms_sc=72,
    force_calculator=None,
    vasp_cmd=VASP_CMD,
    vasp_settings=vasp_settings,
    r_cutoff=5.5,
    r_max_corr=11.0,
    n_ensemble=12,
    work_dir="sqtc_hcpmg_v2_vasp_run",
    verbosity=1,
    ridge_alpha=1e-3,
    # HCP: non-central-force components exist for the out-of-plane bonds;
    # central-force projection is a less accurate approximation for hcp than FCC.
    # Use symmetrize_bonds=False to retain full tensorial IFCs.
    symmetrize_bonds=False,
)

results = runner.run(
    n_sc_iterations=12,
    epsilon_conv=0.003,
    mixing=0.30,
    T_values=np.array([200.0, 300.0, 400.0, 500.0, 700.0]),
    q_mesh_cv=(6, 6, 6),
)

print("\n--- SQTC hcp Mg benchmark summary ---")
print(f"Converged : {results['converged']}")
print(f"Iterations: {results['n_iterations']}")
print()
print(f"{'Quantity':<38} {'SQTC':>10} {'TDEP (lit)':>12} {'Experiment':>12}")
print("-" * 76)
td_spectral = results['T_D_effective']
td_caloric  = results.get('T_D_caloric', float('nan'))
print(f"{'T_D (spectral-moment) [K]':<38} {td_spectral:>10.1f} {'360–390':>12} {'400':>12}")
print(f"{'T_D (calorimetric) [K]':<38} {td_caloric:>10.1f} {'—':>12} {'400':>12}")

exp  = {200.0: "~23.5", 300.0: "~24.9", 400.0: "~25.6", 500.0: "~26.0", 700.0: "~26.5"}
tdep = {200.0: "~23.0", 300.0: "~24.8", 400.0: "~25.0", 500.0: "~25.1", 700.0: "~25.2"}
n_b = len(prim_pos_mg)  # 2 atoms per primitive cell; SQTC outputs per f.u.
for T_v, cv_v in zip(results['T_values'], results['C_V_scan']):
    cv_per_atom = cv_v / n_b  # convert to per-atom (per mole of Mg)
    print(
        f"  C_V({T_v:.0f} K) [J/mol/K]  {'':>8} {cv_per_atom:>10.4f} "
        f"{tdep.get(T_v,'—'):>12} {exp.get(T_v,'—'):>12}"
    )
print()
print(f"  (SQTC C_V divided by n_b={n_b} to give per-atom [J/(mol-atom·K)] for comparison)")
print("  Harmonic DFT C_V(700 K) ≈ 24.94 J/(mol·K)  [3R classical limit]")
print("  NOTE: first HCP result — validates non-cubic primitive cell handling in SQTC")
print("Results saved to sqtc_hcpmg_v2_vasp_run/sqtc_results.json")
