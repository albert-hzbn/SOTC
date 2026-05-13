#!/usr/bin/env python3
"""
SQTC+VASP benchmark: Si (diamond cubic) — covalent semiconductor at 1000 K
============================================================================
Silicon provides a fundamentally different benchmark from all metals and ionic
compounds in this suite: covalent directional bonding, diamond structure, and
very low anharmonicity:

  ANHARMONICITY (low — weakly anharmonic covalent solid):
  - Grüneisen parameter γ ≈ 1.06 — one of the lowest among non-molecular solids
  - Negative thermal expansion below ~120 K (transverse acoustic mode softening)
  - T_D ≈ 640 K (calorimetric Debye fit) — very high for a semiconductor
  - At T = 1000 K (T/T_melt = 0.59): T/T_D ≈ 1.56 — barely classical regime
  - C_V(1000 K) ≈ 24.0 J/(mol·K): close to but below classical 3R (quantum effects)
  - C_V BELOW 3R at 1000 K because T/T_D = 1.56 is not fully classical

  DIAMOND STRUCTURE — cubic with 2-atom basis:
  - FCC Bravais lattice, 2-atom primitive cell (space group Fd-3m)
  - Primitive lattice vectors: a₁ = a/2 × [0,1,1], a₂ = a/2 × [1,0,1],
    a₃ = a/2 × [1,1,0]
  - Si₁ at (0, 0, 0);  Si₂ at (a/4, a/4, a/4)  — tetrahedral bond geometry
  - a_exp(300 K) = 5.431 Å; very low thermal expansion: α ≈ 2.6×10⁻⁶ K⁻¹
  - NN Si–Si distance = a√3/4 = 2.35 Å (×4 tetrahedral bonds per atom)
  - 2nd NN: a/√2 = 3.84 Å (×12)

  KEY PHYSICS:
  - Purely covalent sp³ bonding: very stiff, low anharmonicity (γ ≈ 1.06)
  - No d-electrons, no magnetic effects, no LO-TO ionic splitting → cleanest
    comparison between SQTC and harmonic DFT (anharmonic correction is small)
  - Semiconductor (indirect gap ~1.1 eV PBE underestimates to ~0.6 eV):
    use Gaussian smearing with small σ for displaced-cell calculations
  - SQTC at 1000 K: captures mild phonon renormalisation of acoustic modes;
    the TO optical mode at Γ (≈15.6 THz) softens slightly with temperature
  - Useful cross-check: if SQTC converges close to harmonic DFT for Si,
    it confirms the method does not over-anharmonicise weakly coupled systems

  HARMONIC DFT (0 K PBE-PAW, phonopy):
  - T_D ≈ 600–640 K  (PBE gives a_PBE ≈ 5.468 Å, +0.7% vs exp; slightly softens)
  - C_V(1000 K) ≈ 23.9 J/(mol·K)  (below 3R: quantum corrections still visible)
  - All phonon modes real; Si is harmonically stable

  Experimental references (NIST / Flubacher et al. 1959 / Shanks et al. 1963):
    T_D (calorimetric) ≈ 640 K
    C_V( 300 K) ≈ 20.0 J/(mol·K)
    C_V( 500 K) ≈ 22.8 J/(mol·K)
    C_V( 700 K) ≈ 23.5 J/(mol·K)
    C_V(1000 K) ≈ 24.0 J/(mol·K)

  TDEP/phonopy at 1000 K (Hellman; Knoop et al. 2020):
    T_D ≈ 590–630 K  |  C_V ≈ 23.7–23.9 J/(mol·K)
    (very close to harmonic: anharmonic correction ≈ 0.1 J/(mol·K))
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

# ── Si diamond primitive cell ──────────────────────────────────────────────────
# a_exp(300 K) = 5.431 Å; α ≈ 2.6×10⁻⁶ K⁻¹ (among the lowest of all solids).
# At 1000 K:  a(1000K) = 5.431 × (1 + 2.6e-6 × 700) ≈ 5.441 Å.
# Note: PBE overestimates a by ~0.7% (a_PBE ≈ 5.468 Å); using experimental volume
# ensures force constants are evaluated at the correct physical lattice spacing.
a_si = 5.44  # Å  (experimental Si at ~1000 K)

# Diamond = FCC Bravais lattice with 2-atom basis
prim_cell_si = 0.5 * a_si * np.array(
    [
        [0.0, 1.0, 1.0],   # a₁
        [1.0, 0.0, 1.0],   # a₂
        [1.0, 1.0, 0.0],   # a₃
    ]
)
# Tetrahedral Si–Si bond: Si₂ at (1/4, 1/4, 1/4) in fractional = (a/4, a/4, a/4) Cartesian
prim_pos_si = np.array(
    [
        [0.0,      0.0,      0.0],   # Si₁
        [a_si/4,   a_si/4,   a_si/4], # Si₂  (tetrahedral site)
    ]
)

# ── Physical inputs ────────────────────────────────────────────────────────────
M_Si_amu  = 28.0855
# Seed from calorimetric T_D; at 1000 K effective renormalised value ≈ 590–630 K
T_D_Si    = 620.0   # K
T_design  = 1000.0  # K  (T/T_melt = 0.59, T/T_D ≈ 1.56)

print("\n" + "=" * 70)
print(" SQTC VASP Benchmark: Si diamond (covalent semiconductor, low anharmonicity)")
print("=" * 70)
print(f"  VASP command : {VASP_CMD}")
print(f"  POTCAR path  : {PP_BASE_DIR / 'PAW_PBE' / 'Si' / 'POTCAR'}")
print(f"  a (exp 1000K): {a_si} Å")
print(f"  T_design     : {T_design} K  (T/T_melt = {T_design/1687:.2f}, T/T_D ≈ {T_design/T_D_Si:.2f})")
print()
print("  STRUCTURE TYPE: Diamond cubic (FCC Bravais + 2-atom basis)")
print("    → First purely covalent semiconductor in this benchmark set")
print("    → sp³ tetrahedral bonding: Si–Si NN at 2.35 Å (×4), highly directional")
print()
print("  HARMONIC DFT STATUS: DEFINED (all real modes)")
print("    → Si is harmonically stable (no dynamic instabilities)")
print(f"    → T/T_D = {T_design/T_D_Si:.2f}: partially quantum regime, C_V < 3R at 1000 K")
print("    → Anharmonic correction expected to be small (γ ≈ 1.06)")
print("    → SQTC should converge close to harmonic DFT — validates low-γ limit")
print()
print("  Experimental reference (NIST / Flubacher et al. 1959):")
print("    T_D (calorimetric) ≈ 640 K")
print("    C_V( 300 K) ≈ 20.0 J/(mol·K)  (well below classical limit)")
print("    C_V( 500 K) ≈ 22.8 J/(mol·K)")
print("    C_V( 700 K) ≈ 23.5 J/(mol·K)")
print("    C_V(1000 K) ≈ 24.0 J/(mol·K)  (approaching but not at 3R)")
print()
print("  TDEP at 1000 K (Hellman; Knoop et al. 2020):")
print("    T_D ≈ 590–630 K  |  C_V ≈ 23.7–23.9 J/(mol·K)")

vasp_settings = {
    # Si: PAW hard cutoff 245.3 eV; 400 eV gives well-converged forces.
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
        # Gaussian smearing for semiconductor (PBE gap ~0.6 eV underestimated but non-zero).
        # Displaced snapshots at 1000 K may have slightly different electronic structure;
        # Gaussian smearing with small σ is safer than Methfessel-Paxton for gapped systems.
        "ISMEAR": "0",
        "SIGMA": "0.05",
    },
    "timeout": 3600,
}

runner = SQTCRunner(
    element="Si",
    mass_amu=M_Si_amu,
    prim_cell=prim_cell_si,
    prim_positions=prim_pos_si,
    T=T_design,
    T_D=T_D_Si,
    # 64-atom supercell (32 f.u.): NN-complete for r_cutoff = 4.5 Å
    # Si diamond: 1st NN at a√3/4 = 2.35 Å (×4 tetrahedral), 2nd NN at a/√2 = 3.84 Å (×12)
    # r_cutoff = 4.5 Å includes both 1st and 2nd NN shells
    # 64 atoms × 3 × 12 snaps = 2304 force observations
    n_atoms_sc=64,
    force_calculator=None,
    vasp_cmd=VASP_CMD,
    vasp_settings=vasp_settings,
    r_cutoff=4.5,
    r_max_corr=10.0,
    n_ensemble=12,
    work_dir="sqtc_si_vasp_run",
    verbosity=1,
    ridge_alpha=1e-3,
    # Diamond: central-force projection NOT valid for tetrahedral sp³ bonds.
    # The restoring force has strong angular (bending) components perpendicular
    # to the Si–Si bond axis; symmetrize_bonds=False retains full 3×3 IFC tensors.
    symmetrize_bonds=False,
)

results = runner.run(
    n_sc_iterations=12,
    epsilon_conv=0.003,
    mixing=0.30,
    T_values=np.array([300.0, 500.0, 700.0, 1000.0]),
    q_mesh_cv=(6, 6, 6),
)

print("\n--- SQTC Si diamond benchmark summary ---")
print(f"Converged : {results['converged']}")
print(f"Iterations: {results['n_iterations']}")
print()
print(f"{'Quantity':<38} {'SQTC':>10} {'TDEP (lit)':>12} {'Experiment':>12}")
print("-" * 76)
td_spectral = results['T_D_effective']
td_caloric  = results.get('T_D_caloric', float('nan'))
print(f"{'T_D (spectral-moment) [K]':<38} {td_spectral:>10.1f} {'590–630':>12} {'640':>12}")
print(f"{'T_D (calorimetric) [K]':<38} {td_caloric:>10.1f} {'—':>12} {'640':>12}")

exp  = {300.0: "~20.0", 500.0: "~22.8", 700.0: "~23.5", 1000.0: "~24.0"}
tdep = {300.0: "~19.8", 500.0: "~22.6", 700.0: "~23.3", 1000.0: "~23.8"}
for T_v, cv_v in zip(results['T_values'], results['C_V_scan']):
    print(
        f"  C_V({T_v:.0f} K) [J/mol/K]  {'':>8} {cv_v:>10.4f} "
        f"{tdep.get(T_v,'—'):>12} {exp.get(T_v,'—'):>12}"
    )
print()
print("  Classical limit 3R = 24.94 J/(mol·K)  [per mol Si]")
print("  NOTE: C_V < 3R at 1000 K (T/T_D = 1.56); quantum ZPE still relevant")
print("  NOTE: Low γ ≈ 1.06 → anharmonic correction to SQTC vs harmonic DFT is small")
print("Results saved to sqtc_si_vasp_run/sqtc_results.json")
