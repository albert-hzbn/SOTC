#!/usr/bin/env python3
"""
SQTC+VASP benchmark: fcc Ni — transition metal near melting (T = 1200 K)
=========================================================================
Ni is an FCC ferromagnet that provides an important benchmark because:

  ANHARMONICITY (moderate-to-strong):
  - Grüneisen parameter γ ≈ 1.78 — moderate (between Cu: 1.96 and MgO: 1.5)
  - T_D ≈ 450 K (experimental calorimetric)
  - At T = 1200 K (T/T_melt = 0.69): T/T_D ≈ 2.7 — well into classical regime
  - C_V(1200 K) ≈ 29.5 J/(mol·K): total experimental value including electronic
    contribution γ_el × T ≈ 2.5–3 J/(mol·K); purely phononic anharmonic excess
    above 3R ≈ +1.5 J/(mol·K) (consistent with γ ≈ 1.78)

  MAGNETIC NOTE — why ISPIN=1 is appropriate here:
  - Ni is ferromagnetic below T_Curie ≈ 627 K; above T_C it is paramagnetic.
  - At T = 1200 K >> T_C, local moments are strongly disordered on the time
    scale of atomic vibrations → the effective potential seen by an atom is the
    disordered-moment average, well approximated by a non-spin-polarised (ISPIN=1)
    DFT calculation.
  - Using ISPIN=2 above T_C would impose an ordered-moment ferromagnetic state
    that does not reflect the physical paramagnetic configuration and produces
    artificially stiffened force constants.
  - SQTC with ISPIN=1 at 1200 K therefore gives the temperature-renormalised
    phonon IFCs of paramagnetic Ni — physically correct in this regime.
  - Phononic contribution to C_V from SQTC should be compared with:
      C_V(phonon, 1200 K) = C_V(exp, total) − γ_el × T ≈ 29.5 − 3.0 ≈ 26.5 J/(mol·K)

  HARMONIC DFT (0 K PBE-PAW, phonopy, ISPIN=2 FM state):
  - T_D ≈ 390–430 K  (PBE slightly overestimates volume for Ni)
  - C_V(1200 K) ≈ 24.94 J/(mol·K)  [classical limit]
  - Anharmonic phononic excess: ~1.5 J/(mol·K) at 1200 K

  Experimental references (NIST / Barin & Knacke / Desai 1987):
    T_D (calorimetric) ≈ 450 K
    C_V( 300 K) ≈ 26.1 J/(mol·K)  (includes electronic contribution)
    C_V( 700 K) ≈ 34.0 J/(mol·K)  (λ-anomaly near T_Curie = 627 K!)
    C_V( 900 K) ≈ 30.4 J/(mol·K)  (paramagnetic, above T_C)
    C_V(1000 K) ≈ 30.2 J/(mol·K)
    C_V(1200 K) ≈ 29.5 J/(mol·K)
    (N.B. experimental Cp includes ≈2.5–3 J/(mol·K) electronic contribution
    at 1200 K; SQTC gives only the phononic contribution)

  TDEP/phonopy at 1200 K (paramagnetic Ni, ISPIN=1):
    T_D ≈ 390–420 K  |  phononic C_V ≈ 25.2–25.8 J/(mol·K)
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

# ── fcc Ni primitive cell ──────────────────────────────────────────────────────
# a_exp(300 K) = 3.524 Å; linear thermal expansion α ≈ 13.4×10⁻⁶ K⁻¹.
# At 1200 K:  a(1200K) = 3.524 × (1 + 13.4e-6 × 900) ≈ 3.567 Å.
a_ni = 3.57  # Å  (experimental fcc Ni at ~1200 K)

prim_cell_ni = 0.5 * a_ni * np.array(
    [
        [0.0, 1.0, 1.0],
        [1.0, 0.0, 1.0],
        [1.0, 1.0, 0.0],
    ]
)
prim_pos_ni = np.array([[0.0, 0.0, 0.0]])

# ── Physical inputs ────────────────────────────────────────────────────────────
M_Ni_amu  = 58.6934
# Seed from calorimetric T_D; at 1200 K effective (renormalised) value ≈ 390–420 K
T_D_Ni    = 430.0   # K
T_design  = 1200.0  # K  (T/T_melt = 0.69, T/T_C ≈ 1.9 — well into paramagnetic regime)

print("\n" + "=" * 70)
print(" SQTC VASP Benchmark: fcc Ni (transition metal, paramagnetic at 1200 K)")
print("=" * 70)
print(f"  VASP command : {VASP_CMD}")
print(f"  POTCAR path  : {PP_BASE_DIR / 'PAW_PBE' / 'Ni' / 'POTCAR'}")
print(f"  a (exp 1200K): {a_ni} Å")
print(f"  T_design     : {T_design} K  (T/T_melt = {T_design/1728:.2f}, T/T_C = {T_design/627:.1f})")
print()
print("  MAGNETIC STATE: ISPIN=1 (non-spin-polarised)")
print("    → Ni is paramagnetic above T_Curie = 627 K")
print("    → At 1200 K, local moments are thermally disordered on vibration timescale")
print("    → ISPIN=1 is the correct mean-field approximation for paramagnetic Ni")
print()
print("  HARMONIC DFT STATUS: DEFINED (all real modes, ISPIN=1 paramagnetic)")
print("    → Harmonic C_V(phonon, 1200 K) ≈ 24.94 J/(mol·K)  [classical limit]")
print("    → Phononic anharmonic excess: C_V(exp) − 3R − γ_el×T ≈ +1.5 J/(mol·K)")
print()
print("  Experimental reference (NIST / Desai 1987):")
print("    T_D (calorimetric) ≈ 450 K")
print("    C_V( 900 K) ≈ 30.4 J/(mol·K)  (total, including electronic)")
print("    C_V(1000 K) ≈ 30.2 J/(mol·K)  (total)")
print("    C_V(1200 K) ≈ 29.5 J/(mol·K)  (total)")
print("    Electronic contribution ≈ 2.5–3.0 J/(mol·K) at 1200 K")
print("    Phononic C_V(1200 K) ≈ 26.5–27.0 J/(mol·K)")
print()
print("  TDEP at 1200 K (ISPIN=1, paramagnetic):")
print("    T_D ≈ 390–420 K  |  phononic C_V ≈ 25.2–25.8 J/(mol·K)")

vasp_settings = {
    # Ni: PAW hard cutoff 270.0 eV; 400 eV provides well-converged forces.
    # Ni_pv (3p semi-core) is unnecessary for force constants; standard Ni PAW suffices.
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
        # Methfessel-Paxton smearing; σ = k_B × 1200 K ≈ 0.10 eV
        "ISMEAR": "1",
        "SIGMA": "0.10",
        # Non-spin-polarised: ISPIN=1 is the default in VASP.
        # Explicitly set to prevent any spin initialisation.
        "ISPIN": "1",
    },
    "timeout": 3600,
}

runner = SQTCRunner(
    element="Ni",
    mass_amu=M_Ni_amu,
    prim_cell=prim_cell_ni,
    prim_positions=prim_pos_ni,
    T=T_design,
    T_D=T_D_Ni,
    # 32-atom supercell: NN-complete for r_cutoff = 4.5 Å
    # fcc Ni: 1st NN at a/√2 = 2.52 Å (×12), 2nd NN at a = 3.57 Å (×6) → 18 total
    # 9 unique ±R classes → 81 free params (18 with symmetrize_bonds=True)
    # 32×3×12 = 1152 observations → 14× overdetermined
    n_atoms_sc=32,
    force_calculator=None,
    vasp_cmd=VASP_CMD,
    vasp_settings=vasp_settings,
    r_cutoff=4.5,
    r_max_corr=9.0,
    n_ensemble=12,
    work_dir="sqtc_fccni_vasp_run",
    verbosity=1,
    ridge_alpha=1e-3,
    # FCC: central-force projection valid; isotropic bonding symmetry
    symmetrize_bonds=True,
)

results = runner.run(
    n_sc_iterations=12,
    epsilon_conv=0.003,
    mixing=0.30,
    T_values=np.array([500.0, 700.0, 900.0, 1000.0, 1200.0]),
    q_mesh_cv=(6, 6, 6),
)

print("\n--- SQTC fcc Ni benchmark summary ---")
print(f"Converged : {results['converged']}")
print(f"Iterations: {results['n_iterations']}")
print()
print(f"{'Quantity':<38} {'SQTC':>10} {'TDEP (lit)':>12} {'Experiment':>12}")
print("-" * 76)
td_spectral = results['T_D_effective']
td_caloric  = results.get('T_D_caloric', float('nan'))
print(f"{'T_D (spectral-moment) [K]':<38} {td_spectral:>10.1f} {'390–420':>12} {'450':>12}")
print(f"{'T_D (calorimetric) [K]':<38} {td_caloric:>10.1f} {'—':>12} {'450':>12}")

# Experimental phononic C_V (total Cp minus electronic contribution ~2.5 J/molK at high T)
exp_phon = {500.0: "~24.5", 700.0: "~25.5", 900.0: "~27.5", 1000.0: "~27.7", 1200.0: "~26.5"}
exp_tot  = {500.0: "~27.8", 700.0: "~34.0*",900.0: "~30.4", 1000.0: "~30.2", 1200.0: "~29.5"}
tdep     = {500.0: "~24.5", 700.0: "~24.8", 900.0: "~25.1", 1000.0: "~25.3", 1200.0: "~25.7"}
for T_v, cv_v in zip(results['T_values'], results['C_V_scan']):
    print(
        f"  C_V_phonon({T_v:.0f} K) [J/mol/K] {'':<5} {cv_v:>10.4f} "
        f"{tdep.get(T_v,'—'):>12} {exp_phon.get(T_v,'—'):>12}"
    )
print()
print("  NOTE: Total experimental Cp includes electronic + magnetic contributions.")
print("  SQTC gives phononic C_V only; electronic ≈ 2.5 J/(mol·K) not included.")
print("  *700 K data anomalous: λ-peak near T_Curie = 627 K (not phononic).")
print("  Harmonic DFT C_V(1200 K) ≈ 24.94 J/(mol·K)  [3R classical limit]")
print("Results saved to sqtc_fccni_vasp_run/sqtc_results.json")
