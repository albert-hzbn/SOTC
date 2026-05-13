#!/usr/bin/env python3
"""
SQTC example: solid para-H₂ Phase I (1 atm, 4–13 K)
======================================================
Uses the Silvera–Goldman isotropic potential (no VASP licence required).
H₂ is treated as an effective monatomic solid with M = 2·M_H.

System parameters
-----------------
  Crystal:  hcp para-H₂, a = 3.78 Å, c/a = 1.631
  Mass:     M_H2 = 2.01588 amu (effective COM mass)
  T_D:      110 K (translational, Silvera 1980)
  Pressure: 1 atm

Comparison targets (experimental)
----------------------------------
  C_V (4 K)  ≈ 1.0  J/(mol·K)  (Silvera 1980)
  C_V (8 K)  ≈ 3.5  J/(mol·K)
  C_V (13 K) ≈ 5.7  J/(mol·K)
  MSD (4 K) ≈ 0.20 Å²  (Silvera 1980, Table 3)
  MSD (13 K) ≈ 0.27 Å²

References
----------
  Silvera I.F. (1980) Rev. Mod. Phys. 52, 393.
  Silvera I.F. & Goldman V.V. (1978) J. Chem. Phys. 69, 4209.
"""

import sys
import numpy as np
import os

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "../"))

from sqtc import SQTCRunner
from sqtc.mock_forces import SilveraGoldmanPotential
from sqtc.correlators import DebyeCorrelator
from sqtc.constants import KB, HBAR, NA

# ── H₂ crystal structure (hcp, P6_3/mmc) ─────────────────────────────────────
a_h2 = 3.78   # Å
c_h2 = 1.631 * a_h2  # 6.165 Å

prim_cell_h2 = np.array([
    [a_h2,    0.0,             0.0],
    [-a_h2/2, a_h2*np.sqrt(3)/2, 0.0],
    [0.0,     0.0,             c_h2],
])

prim_pos_h2 = np.array([
    [0.0,         0.0,              0.0],
    [a_h2/2,      a_h2/(2*np.sqrt(3)), c_h2/2],
])

# ── Physical parameters ───────────────────────────────────────────────────────
T_D_H2 = 110.0       # K  (translational Debye temperature)
M_H2_amu = 2.01588   # amu (effective H₂ COM mass = 2 × M_H)

# ── Experimental reference (Silvera 1980) ────────────────────────────────────
exp_cv = {   # J/(mol·K)
    4.0:  1.0,
    8.0:  3.5,
    13.0: 5.7,
}
exp_msd = {  # Å²
    4.0:  0.20,
    13.0: 0.27,
}

# ── Debye model reference ─────────────────────────────────────────────────────
debye = DebyeCorrelator(T_D=T_D_H2, M_amu=M_H2_amu)

print("\n" + "="*70)
print(" SQTC Example: Solid para-H₂ Phase I (1 atm, 4–13 K)")
print("="*70)

print("\n── Debye model predictions (harmonic reference) ──")
for T, cv_exp in exp_cv.items():
    cv_d = debye.heat_capacity_v_jmolk(T)
    msd_d = debye.msd_ang2(T)
    msd_exp = exp_msd.get(T)
    msd_str = f"{msd_exp:.3f}" if msd_exp else "  —  "
    print(
        f"  T={T:4.0f} K: C_V={cv_d:5.2f} (exp {cv_exp:.1f})  "
        f"MSD={msd_d:.4f} Å² (exp {msd_str} Å²)"
    )

# ── SQTC run ──────────────────────────────────────────────────────────────────
print("\n── SQTC run (Silvera–Goldman pair potential) ──")

T_design = 8.0   # K  (mid-range design temperature)
T_values = np.array([4.0, 6.0, 8.0, 10.0, 13.0])

runner = SQTCRunner(
    element="H",
    mass_amu=M_H2_amu,
    prim_cell=prim_cell_h2,
    prim_positions=prim_pos_h2,
    T=T_design,
    T_D=T_D_H2,
    n_atoms_sc=16,
    force_calculator=SilveraGoldmanPotential(r_cut=8.0),
    r_cutoff=6.0,
    r_max_corr=8.0,
    n_ensemble=5,
    work_dir="sqtc_h2_run",
    verbosity=1,
)

results = runner.run(
    n_sc_iterations=15,
    epsilon_conv=0.002,
    mixing=0.5,
    T_values=T_values,
    q_mesh_cv=(8, 8, 8),
)

# ── Results ───────────────────────────────────────────────────────────────────
print("\n── SQTC vs Debye vs Experiment ──")
print(f"{'T [K]':<8} {'C_V Debye':>12} {'C_V SQTC':>12} {'C_V exp':>12}")
print("-" * 48)
for T in T_values:
    cv_d = debye.heat_capacity_v_jmolk(T)
    idx = np.argmin(np.abs(results["T_values"] - T))
    cv_sqtc = results["C_V_scan"][idx]
    cv_exp = exp_cv.get(T)
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
    cv_d = debye.heat_capacity_v_jmolk(T)
    idx = np.argmin(np.abs(results["T_values"] - T))
    cv_sqtc = results["C_V_scan"][idx]
    cv_exp = exp_cv[T]
    err_d = 100.0 * (cv_d - cv_exp) / cv_exp
    err_sqtc = 100.0 * (cv_sqtc - cv_exp) / cv_exp
    print(f"  {T:<8.0f}  {err_d:>+14.1f}%  {err_sqtc:>+14.1f}%")

print("\n✓ H₂ example complete. Results saved to sqtc_h2_run/sqtc_results.json")
