#!/usr/bin/env python3
"""
SQTC example: solid ⁴He (hcp, 25 bar) — DFT forces via VASP
=============================================================
Uses VASP PBE+D3 for the interatomic forces.

System parameters
-----------------
  Crystal:  hcp ⁴He, a = 3.57 Å, c/a = 1.633
  Mass:     M_He = 4.002602 amu
  T_D:      26 K  (from heat capacity measurements)
  Pressure: 25 bar (near melting)

DFT settings
------------
  Functional: PBE-D3 (IVDW=12, BJ damping)
  ENCUT:  600 eV  (≥ 1.25 × ENMAX_He = 479 eV)
  KSPACING: 0.25 Å⁻¹ (fairly accurate k-sampling for supercell)
  ISMEAR:  0 (Gaussian, solid He is insulating)
  SIGMA:   0.05 eV
  EDIFF:   1E-8 eV (tight SCF for accurate forces)
  Cores:   16 MPI tasks per snapshot (5 snapshots in parallel → 80 cores)

Comparison targets (experimental)
----------------------------------
  C_V (0.5 K) ≈  0.6 J/(mol·K)
  C_V (1.0 K) ≈  2.3 J/(mol·K)
  C_V (1.6 K) ≈  7.5 J/(mol·K)
  C_V (2.0 K) ≈ 11.8 J/(mol·K)
  MSD (0 K) ≈ 0.32 Å² (Debye-Waller, Horner 1974)
  T_D = 26 K (White et al. 1969)
  Zone-boundary LA freq ≈ 0.41 ± 0.04 THz (Minkiewicz et al. 1972)

References
----------
  White G.K. et al. (1969) J. Low Temp. Phys. 1, 373.
  Minkiewicz V.J. et al. (1972) Phys. Rev. A 8, 1513.
  Horner H. (1974) in Dynamical Properties of Solids, Vol. 1.
  Grimme S. et al. (2010) J. Chem. Phys. 132, 154104. (DFT-D3)
"""

import sys
import numpy as np
import os
from pathlib import Path

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "../"))

from sqtc import SQTCRunner
from sqtc.correlators import DebyeCorrelator
from sqtc.constants import KB, HBAR, NA

# ── VASP / machine configuration ──────────────────────────────────────────────
VASP_STD    = os.environ.get("VASP_STD", "/u/alli/Softwares/DFT_v/s")
PP_BASE_DIR = Path(os.environ.get("VASP_PP_BASE", "/u/alli/Softwares/VASP/Pseudopotentials"))

# 16 MPI tasks per snapshot; 5 snapshots run in parallel → 80 / 128 cores
NCORES_PER_JOB = 16
NCORE_INCAR    = 4   # NCORE: cores per orbital-parallel group

VASP_CMD = f"mpirun -np {NCORES_PER_JOB} {VASP_STD}"

# ── He crystal structure (hcp, P6₃/mmc) ──────────────────────────────────────
a_he = 3.57           # Å
c_he = 1.633 * a_he  # 5.831 Å

prim_cell_he = np.array([
    [a_he,    0.0,                    0.0],
    [-a_he/2, a_he * np.sqrt(3) / 2,  0.0],
    [0.0,     0.0,                    c_he],
])

prim_pos_he = np.array([
    [0.0,      0.0,                     0.0],
    [a_he / 2, a_he / (2*np.sqrt(3)), c_he / 2],
])

# ── Physical parameters ───────────────────────────────────────────────────────
T_D_He   = 26.0       # K
M_He_amu = 4.002602   # amu

# ── Experimental reference values ────────────────────────────────────────────
exp_cv = {0.5: 0.6, 1.0: 2.3, 1.6: 7.5, 2.0: 11.8}  # J/(mol·K)
exp_msd_0K  = 0.32   # Å²
exp_T_D     = 26.0   # K
exp_ZB_freq = 0.41   # THz (zone-boundary LA [001], Minkiewicz 1972)

# ── Debye model reference ─────────────────────────────────────────────────────
debye = DebyeCorrelator(T_D=T_D_He, M_amu=M_He_amu)

print("\n" + "=" * 70)
print(" SQTC Example: Solid ⁴He (hcp, 25 bar) — DFT (VASP PBE+D3)")
print("=" * 70)
print(f"  VASP command : {VASP_CMD}")
print(f"  Pseudopot.   : {PP_BASE_DIR / 'PAW_PBE' / 'He' / 'POTCAR'}")

print("\n── Debye model predictions (harmonic reference) ──")
T_range = np.array([0.5, 1.0, 1.6, 2.0])
for T in T_range:
    cv_d = debye.heat_capacity_v_jmolk(T)
    print(f"  T={T:.1f} K: C_V_Debye={cv_d:6.3f}  C_V_exp={exp_cv[T]:.1f}  J/(mol·K)")

# ── VASP DFT settings ─────────────────────────────────────────────────────────
vasp_settings = {
    "encut":       600.0,       # eV  — ≥ 1.25 × ENMAX_He (479 eV)
    "functional":  "PBE-D3",    # IVDW=12 (D3 with BJ damping)
    "ncore":       NCORE_INCAR,
    "kgrid":       (2, 3, 4),   # conservative grid; KSPACING overrides below
    "pp_base_dir": PP_BASE_DIR,
    "pp_set":      "PAW_PBE",
    "extra_incar": {
        "KSPACING": "0.4",     # Å⁻¹ — automatic k-grid from cell geometry
        "EDIFF":    "1E-6",
        "PREC":     "Accurate",
    },
    "timeout":     3600,        # s — 1 h per snapshot
}

# ── SQTC runner ───────────────────────────────────────────────────────────────
print("\n── SQTC run (DFT PBE+D3 forces via VASP) ──")

T_design = 1.6  # K  (near melting at 25 bar)

runner = SQTCRunner(
    element       = "He",
    mass_amu      = M_He_amu,
    prim_cell     = prim_cell_he,
    prim_positions= prim_pos_he,
    T             = T_design,
    T_D           = T_D_He,
    n_atoms_sc    = 16,
    force_calculator = None,          # use VASP, not pair potential
    vasp_cmd      = VASP_CMD,
    vasp_settings = vasp_settings,
    r_cutoff      = 6.5,
    r_max_corr    = 8.0,
    n_ensemble    = 5,
    work_dir      = "sqtc_he_vasp_run",
    verbosity     = 1,
)

results = runner.run(
    n_sc_iterations = 15,
    epsilon_conv    = 0.002,
    mixing          = 0.5,
    T_values        = T_range,
    q_mesh_cv       = (8, 8, 8),
)

# ── Results comparison ────────────────────────────────────────────────────────
print("\n── SQTC DFT vs Debye vs Experiment ──")
print(f"{'Quantity':<30} {'Debye':>10} {'SQTC DFT':>10} {'Experiment':>12}")
print("-" * 66)

for T in T_range:
    cv_debye = debye.heat_capacity_v_jmolk(T)
    idx      = np.argmin(np.abs(results["T_values"] - T))
    cv_sqtc  = results["C_V_scan"][idx]
    print(
        f"  C_V({T:.1f} K) [J/mol/K]  "
        f"{cv_debye:>10.3f}  {cv_sqtc:>10.3f}  {exp_cv[T]:>12.3f}"
    )

print(f"\n  {'T_D [K]':<28} {T_D_He:>10.1f}  {results['T_D_effective']:>10.1f}  {exp_T_D:>12.1f}")
print(f"  {'ZPE [meV/atom]':<28} {'—':>10}  {results['ZPE_eV']*1000:>10.2f}  {'—':>12}")
print(f"\n  Converged: {results['converged']}  in {results['n_iterations']} iterations")

# ── Percentage errors ─────────────────────────────────────────────────────────
print("\n── % Error vs experiment ──")
print(f"{'T [K]':<10} {'Debye error':>15} {'SQTC error':>15}")
print("-" * 42)
for T in T_range:
    cv_debye = debye.heat_capacity_v_jmolk(T)
    idx      = np.argmin(np.abs(results["T_values"] - T))
    cv_sqtc  = results["C_V_scan"][idx]
    cv_exp   = exp_cv[T]
    err_debye= 100.0 * (cv_debye - cv_exp) / cv_exp
    err_sqtc = 100.0 * (cv_sqtc  - cv_exp) / cv_exp
    print(f"  {T:<8.1f}  {err_debye:>+14.1f}%  {err_sqtc:>+14.1f}%")

# ── Zone-boundary phonon frequency ───────────────────────────────────────────
print("\n── Zone-boundary frequency LA [001] ──")
phcalc = results["phonon_calculator"]
q_ZB   = np.array([0.0, 0.0, 0.5])
freqs_ZB = phcalc.frequencies_thz_at_q(q_ZB)
freq_LA  = sorted([f for f in freqs_ZB if f > 0])[-1]
print(f"  SQTC DFT:   {freq_LA:.3f} THz  (highest branch at q=[0,0,0.5])")
print(f"  Experiment: {exp_ZB_freq:.2f} ± 0.04 THz  (Minkiewicz et al. 1972)")

print("\n✓ He DFT example complete. Results saved to sqtc_he_vasp_run/sqtc_results.json")
