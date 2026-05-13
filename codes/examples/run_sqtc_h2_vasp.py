#!/usr/bin/env python3
"""
SQTC example: solid para-H₂ Phase I — DFT forces via VASP
===========================================================
Uses VASP PBE+D3 for the interatomic forces.
H₂ is treated as an effective monatomic solid (each site = H₂ COM, M = 2·M_H).

System parameters
-----------------
  Crystal:  hcp para-H₂, a = 3.78 Å, c/a = 1.631
  Mass:     M_H2 = 2.01588 amu (effective COM mass)
  T_D:      110 K (translational, Silvera 1980)
  Pressure: 1 atm

DFT settings
------------
  Functional: PBE-D3 (IVDW=12, BJ damping)
  ENCUT:  500 eV  (≥ 2 × ENMAX_H = 250 eV)
  KSPACING: 0.25 Å⁻¹ (fairly accurate k-sampling for supercell)
  ISMEAR:  0 (Gaussian, H₂ is insulating)
  SIGMA:   0.05 eV
  EDIFF:   1E-8 eV (tight SCF for accurate forces)
  Cores:   16 MPI tasks per snapshot (5 snapshots in parallel → 80 cores)

Comparison targets (experimental)
----------------------------------
  C_V (4 K)  ≈ 1.0  J/(mol·K)  (Silvera 1980)
  C_V (8 K)  ≈ 3.5  J/(mol·K)
  C_V (13 K) ≈ 5.7  J/(mol·K)

References
----------
  Silvera I.F. (1980) Rev. Mod. Phys. 52, 393.
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
NCORE_INCAR    = 4   # NCORE: cores per orbital-parallel group (4 is robust)

VASP_CMD = f"mpirun -np {NCORES_PER_JOB} {VASP_STD}"

# ── H₂ crystal structure (hcp, P6₃/mmc) ──────────────────────────────────────
a_h2 = 3.78           # Å
c_h2 = 1.631 * a_h2  # 6.165 Å

prim_cell_h2 = np.array([
    [a_h2,    0.0,                   0.0],
    [-a_h2/2, a_h2 * np.sqrt(3) / 2, 0.0],
    [0.0,     0.0,                   c_h2],
])

prim_pos_h2 = np.array([
    [0.0,      0.0,                    0.0],
    [a_h2 / 2, a_h2 / (2*np.sqrt(3)), c_h2 / 2],
])

# ── Physical parameters ───────────────────────────────────────────────────────
T_D_H2   = 110.0      # K  (translational Debye temperature, Silvera 1980)
M_H2_amu = 2.01588    # amu (effective H₂ COM mass)

# ── Experimental reference (Silvera 1980) ────────────────────────────────────
exp_cv = {4.0: 1.0, 8.0: 3.5, 13.0: 5.7}   # J/(mol·K)

# ── Debye model reference ─────────────────────────────────────────────────────
debye = DebyeCorrelator(T_D=T_D_H2, M_amu=M_H2_amu)

print("\n" + "=" * 70)
print(" SQTC Example: Solid para-H₂ Phase I — DFT (VASP PBE+D3)")
print("=" * 70)
print(f"  VASP command : {VASP_CMD}")
print(f"  Pseudopot.   : {PP_BASE_DIR / 'PAW_PBE' / 'H' / 'POTCAR'}")

print("\n── Debye model predictions (harmonic reference) ──")
for T in [4.0, 8.0, 13.0]:
    cv_d = debye.heat_capacity_v_jmolk(T)
    cv_exp = exp_cv[T]
    print(f"  T={T:4.0f} K: C_V_Debye={cv_d:5.2f}  C_V_exp={cv_exp:.1f}  J/(mol·K)")

# ── VASP DFT settings ─────────────────────────────────────────────────────────
vasp_settings = {
    "encut":       500.0,       # eV  — well above ENMAX_H = 250 eV
    "functional":  "PBE-D3",    # IVDW=12 (D3 with BJ damping)
    "ncore":       NCORE_INCAR,
    "kgrid":       (2, 3, 4),   # conservative grid; KSPACING overrides if set
    "pp_base_dir": PP_BASE_DIR,
    "pp_set":      "PAW_PBE",
    "extra_incar": {
        "KSPACING": "0.4",     # Å⁻¹ — applies to all supercell shapes
        "EDIFF":    "1E-6",
        "PREC":     "Accurate",
    },
    "timeout":     3600,        # s — 1 h per snapshot (generous for 16-atom cell)
}

# ── SQTC runner ───────────────────────────────────────────────────────────────
print("\n── SQTC run (DFT PBE+D3 forces via VASP) ──")

T_design = 8.0   # K  (mid-range design temperature)
T_values = np.array([4.0, 6.0, 8.0, 10.0, 13.0])

runner = SQTCRunner(
    element       = "H",
    mass_amu      = M_H2_amu,
    prim_cell     = prim_cell_h2,
    prim_positions= prim_pos_h2,
    T             = T_design,
    T_D           = T_D_H2,
    n_atoms_sc    = 16,
    force_calculator = None,          # use VASP, not pair potential
    vasp_cmd      = VASP_CMD,
    vasp_settings = vasp_settings,
    r_cutoff      = 6.0,
    r_max_corr    = 8.0,
    n_ensemble    = 5,
    work_dir      = "sqtc_h2_vasp_run",
    verbosity     = 1,
)

results = runner.run(
    n_sc_iterations = 15,
    epsilon_conv    = 0.002,
    mixing          = 0.5,
    T_values        = T_values,
    q_mesh_cv       = (8, 8, 8),
)

# ── Results comparison ────────────────────────────────────────────────────────
print("\n── SQTC DFT vs Debye vs Experiment ──")
print(f"{'T [K]':<8} {'C_V Debye':>12} {'C_V SQTC':>12} {'C_V exp':>12}")
print("-" * 48)
for T in T_values:
    cv_d = debye.heat_capacity_v_jmolk(T)
    idx  = np.argmin(np.abs(results["T_values"] - T))
    cv_sqtc = results["C_V_scan"][idx]
    cv_exp  = exp_cv.get(T)
    exp_str = f"{cv_exp:12.2f}" if cv_exp else f"{'—':>12}"
    print(f"  {T:<6.0f} {cv_d:>12.3f} {cv_sqtc:>12.3f} {exp_str}")

print(f"\n  T_D input = {T_D_H2:.0f} K,  T_D effective = {results['T_D_effective']:.1f} K")
print(f"  ZPE = {results['ZPE_eV']*1000:.2f} meV/H₂")
print(f"  MSD ({T_design:.0f} K) = {results['MSD_ang2']:.4f} Å²")
print(f"\n  Converged: {results['converged']}  in {results['n_iterations']} iterations")

# ── Percentage errors ─────────────────────────────────────────────────────────
print("\n── % Error vs experiment ──")
print(f"{'T [K]':<10} {'Debye error':>15} {'SQTC error':>15}")
print("-" * 42)
for T in [4.0, 8.0, 13.0]:
    cv_d    = debye.heat_capacity_v_jmolk(T)
    idx     = np.argmin(np.abs(results["T_values"] - T))
    cv_sqtc = results["C_V_scan"][idx]
    cv_exp  = exp_cv[T]
    err_d   = 100.0 * (cv_d    - cv_exp) / cv_exp
    err_sqtc= 100.0 * (cv_sqtc - cv_exp) / cv_exp
    print(f"  {T:<8.0f}  {err_d:>+14.1f}%  {err_sqtc:>+14.1f}%")

print("\n✓ H₂ DFT example complete. Results saved to sqtc_h2_vasp_run/sqtc_results.json")
