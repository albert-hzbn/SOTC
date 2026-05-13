#!/usr/bin/env python3
"""
SQTC+VASP benchmark: PbTe (rocksalt) — HIGH anharmonicity thermoelectric
=========================================================================
PbTe is among the most anharmonic group-IV tellurides; its giant anharmonicity
drives the remarkably low thermal conductivity (κ ≈ 1.7 W/m/K at 300 K) that
makes it the prototypical thermoelectric.

  ANHARMONICITY (high):
  - Grüneisen parameter γ ≈ 2.3–2.5  (extremely large for a stable compound)
  - Resonant bonding: partially occupied p-orbital topology → soft TA modes
  - Near-ferroelectric instability: in harmonic DFT, the transverse acoustic
    (TA) branch along Σ [110] and at X/L zone boundary is anomalously soft.
    Some PBE calculations show imaginary TA frequencies at the X-point.
    SQTC self-consistency stabilises these modes via thermally renormalised IFCs.
  - T_D ≈ 130–150 K — very low; almost purely classical at 700 K  (T/T_D ≈ 5)
  - C_V(700 K) ≈ 53–56 J/(mol·K): noticeable super-classical excess above 6R

  HARMONIC DFT (0 K PBE-PAW):
  - TA modes near X [0.5, 0.5, 0] can be imaginary or very small (<0.5 THz)
    depending on exact PBE volume (slight underbinding of Pb softens TA further)
  - T_D(harmonic) ≈ 100–120 K (dominated by soft acoustic)
  - If PbTe shows imaginary modes → SQTC imaginary-mode compensation activates
    (analogous to β-Ti behaviour)

  STRUCTURE — rocksalt (B1, space group Fm-3m):
  - FCC Bravais, 2-atom primitive cell: Pb at (0,0,0), Te at (a/2, 0, 0)
  - a_exp(300 K) = 6.462 Å; thermal expansion α ≈ 20×10⁻⁶ K⁻¹
  - At 700 K:  a(700K) = 6.462 × (1 + 20e-6 × 400) ≈ 6.514 Å
  - NN distance: Pb–Te = a/2 = 3.257 Å (×6 per atom)
  - 2nd NN: Pb–Pb = a/√2 = 4.607 Å (×12 per atom)

  NOTE on LO-TO splitting:
  - PbTe has a HUGE LO-TO gap near Γ: the off-centring tendency of Pb gives
    enormous Born effective charges (Z*_Pb ≈ 5.9, Z*_Te ≈ −5.9) → large
    LO-TO splitting (ΔΩ ≈ 1.5 THz at Γ).
  - Short-range IFC framework without Ewald correction gives TO frequency;
    the LO branch is absent → T_D(SQTC) is significantly underestimated.
  - This is the MOST visible LO-TO limitation in this compound set.

  Experimental references (NIST JANAF / Delaire et al. 2011 Nature Materials):
    T_D (Debye, low-T Cp) ≈ 130 K
    C_V( 300 K) ≈ 50.7 J/(mol·K)   (already above 6R — anharmonic excess)
    C_V( 500 K) ≈ 52.0 J/(mol·K)
    C_V( 700 K) ≈ 53.5 J/(mol·K)
    C_V( 900 K) ≈ 55.0 J/(mol·K)

  Delaire et al. (Nature Mater. 10, 614, 2011) — inelastic neutron scattering:
    Giant anharmonic phonon scattering; TA modes at zone boundary anomalously soft
    and heavily scattered → large 3-phonon scattering; κ = 1.7 W/m/K at 300 K

  TDEP at 700 K (Hellman / Knoop et al.):
    T_D ≈ 120–135 K  (temperature-dependent Debye fit)  |  C_V ≈ 52–54 J/(mol·K)
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

# ── PbTe rocksalt primitive cell ──────────────────────────────────────────────
# a_exp(300 K) = 6.462 Å; α ≈ 20×10⁻⁶ K⁻¹.
# At 700 K:  a(700K) = 6.462 × (1 + 20e-6 × 400) ≈ 6.514 Å.
a_pbte = 6.51  # Å  (experimental PbTe at ~700 K)

prim_cell_pbte = 0.5 * a_pbte * np.array(
    [
        [0.0, 1.0, 1.0],
        [1.0, 0.0, 1.0],
        [1.0, 1.0, 0.0],
    ]
)
# Cartesian positions: Pb at origin, Te at (a/2, 0, 0)
prim_pos_pbte = np.array([
    [0.0,        0.0, 0.0],  # Pb
    [a_pbte/2,   0.0, 0.0],  # Te
])

# ── Physical inputs ────────────────────────────────────────────────────────────
M_Pb_amu  = 207.2
M_Te_amu  = 127.6
T_D_PbTe  = 130.0   # K  (seed from low-T Cp experiment; T/T_D = 5.4 at 700 K)
T_design  = 700.0   # K  (well into classical regime; anharmonic excess visible)

print("\n" + "=" * 70)
print(" SQTC VASP Benchmark: PbTe rocksalt (high anharmonicity)")
print("=" * 70)
print(f"  VASP command : {VASP_CMD}")
print(f"  POTCAR       : Pb + Te  (PAW_PBE)")
print(f"  a (exp 700K) : {a_pbte} Å")
print(f"  T_design     : {T_design} K  (T/T_D = {T_design/T_D_PbTe:.1f})")
print()
print("  HARMONIC DFT STATUS: POSSIBLY IMAGINARY (TA near zone boundary)")
print("    → TA modes at X/L zone boundary can be imaginary in PBE harmonic calc")
print("    → SQTC imaginary-mode compensation will activate if needed")
print("    → γ ≈ 2.3–2.5 — highest Grüneisen of compound benchmark set")
print("    → Resonant bonding → anomalously large Born charges (Z* ≈ ±5.9)")
print("    → LO-TO gap ~1.5 THz NOT included in short-range IFC framework")
print()
print("  Experimental reference (NIST JANAF / Delaire et al. 2011):")
print("    T_D ≈ 130 K")
print("    C_V( 300 K) ≈ 50.7 J/(mol·K)  [already above 6R = 49.88]")
print("    C_V( 500 K) ≈ 52.0 J/(mol·K)")
print("    C_V( 700 K) ≈ 53.5 J/(mol·K)")

vasp_settings = {
    # Pb: ENMAX_PAW ≈ 131 eV; Te: ENMAX_PAW ≈ 174 eV.
    # Use 400 eV for safety (>1.4× max). Pb_d (4f14 5d10 6s2 6p2) vs Pb (6s2 6p2):
    # the standard Pb PAW is sufficient; Pb_d adds 5d semi-core → increases ENCUT.
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
        # Metallic-like screening; use Methfessel-Paxton smearing
        # (PbTe is a narrow-gap semiconductor, ~0.19 eV PBE gap; ISMEAR=1 avoids
        # integration artefacts at displaced configurations where gap fluctuates)
        "ISMEAR": "1",
        "SIGMA": "0.10",
    },
    # timeout per VASP subprocess call [s]. The clock starts when srun is launched,
    # not when VASP begins (srun blocks until nodes are free).
    # With 12 snaps on 4 exclusive nodes there are 3 sequential batches of 4.
    # The last batch waits: 2 × max_snap_time ≈ 2 × 50 min = 100 min = 6000 s
    # before VASP starts, then needs ~50 min = 3000 s.  Total ≈ 9000 s.
    # Use 14400 s (4 h) so there is ample headroom even if individual SCF cycles
    # are slow due to heavy Pb/Te PAW and displaced-cell convergence difficulties.
    "timeout": 14400,
}

runner = SQTCRunner(
    element="Pb",
    mass_amu=M_Pb_amu,
    elements=["Pb", "Te"],
    masses_amu=[M_Pb_amu, M_Te_amu],
    prim_cell=prim_cell_pbte,
    prim_positions=prim_pos_pbte,
    T=T_design,
    T_D=T_D_PbTe,
    # 64-atom supercell (32 f.u.): NN-complete for r_cutoff = 5.0 Å
    # PbTe:  1st NN Pb–Te at 3.255 Å (×6), 2nd NN Pb–Pb at 4.607 Å (×12)
    # r_cutoff = 5.0 Å captures both shells
    n_atoms_sc=64,
    force_calculator=None,
    vasp_cmd=VASP_CMD,
    vasp_settings=vasp_settings,
    r_cutoff=5.0,
    r_max_corr=10.0,
    n_ensemble=12,
    work_dir="sqtc_pbte_vasp_run",
    verbosity=1,
    ridge_alpha=1e-2,         # slightly higher ridge for soft-mode IFCs
    # symmetrize_bonds=False: PbTe has significant higher-order (odd-rank)
    # off-diagonal IFCs due to resonant bonding → DO NOT force central-force
    # symmetry which sets antisymmetric bonds to zero.
    symmetrize_bonds=False,
)

results = runner.run(
    n_sc_iterations=15,      # allow extra iterations for soft-mode self-consistency
    epsilon_conv=0.003,
    mixing=0.25,             # slightly conservative mixing for anharmonic system
    T_values=np.array([300.0, 500.0, 700.0, 900.0]),
    q_mesh_cv=(6, 6, 6),
)

print("\n--- SQTC PbTe benchmark summary ---")
print(f"Converged : {results['converged']}")
print(f"Iterations: {results['n_iterations']}")
print()
print(f"{'Quantity':<38} {'SQTC':>10} {'TDEP (lit)':>12} {'Experiment':>12}")
print("-" * 76)
td = results['T_D_effective']
td_c = results.get('T_D_caloric', float('nan'))
unstab = results.get('unstable_fraction', float('nan'))
print(f"{'T_D (spectral-moment) [K]':<38} {td:>10.1f} {'120–135':>12} {'130':>12}")
print(f"{'T_D (calorimetric) [K]':<38} {td_c:>10.1f} {'—':>12} {'130':>12}")
print(f"{'Imaginary mode fraction [-]':<38} {unstab:>10.3f} {'—':>12} {'—':>12}")

exp  = {300.0: "~50.7", 500.0: "~52.0", 700.0: "~53.5", 900.0: "~55.0"}
tdep = {300.0: "~48.5", 500.0: "~51.0", 700.0: "~52.5", 900.0: "~54.0"}
for T_v, cv_v in zip(results['T_values'], results['C_V_scan']):
    print(f"  C_V({T_v:.0f} K) [J/mol/K]  {'':>5} {cv_v:>10.4f} {tdep.get(T_v,'—'):>12} {exp.get(T_v,'—'):>12}")
print()
print("  Classical limit 6R = 49.883 J/(mol·K)  [per formula unit]")
print("  NOTE: LO-TO splitting (Δω ≈ 1.5 THz) not included; Z* ≈ ±5.9 (large)")
print("  NOTE: Super-classical C_V(700 K) − 6R ≈ 3.6 J/(mol·K) is anharmonic excess")
print("Results saved to sqtc_pbte_vasp_run/sqtc_results.json")
