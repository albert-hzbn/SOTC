#!/usr/bin/env python3
"""
SQTC example: solid ⁴He (hcp, 25 bar, T = 0.5–2.0 K)
========================================================
Demonstrates the full SQTC workflow using the Aziz HFD-B pair potential
(no VASP licence required).

System parameters
-----------------
  Crystal:  hcp ⁴He, a = 3.57 Å, c/a = 1.633
  Mass:     M_He = 4.0026 amu
  T_D:      26 K  (from heat capacity measurements)
  Pressure: 25 bar (near melting)

Comparison targets (experimental)
----------------------------------
  C_V (0.5 K) ≈ 0.6  J/(mol·K)
  C_V (1.0 K) ≈ 2.3  J/(mol·K)
  C_V (1.6 K) ≈ 7.5  J/(mol·K)
  C_V (2.0 K) ≈ 11.8 J/(mol·K)
  MSD (0 K) ≈ 0.32 Å² (Debye-Waller, Horner 1974)
  T_D = 26 K (White et al. 1969)
"""

import sys
import numpy as np
import os

# Add parent directory to path so 'sqtc' package is importable
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "../"))

from sqtc import SQTCRunner
from sqtc.mock_forces import AzizHFDPotential, EinsteinBackground
from sqtc.correlators import DebyeCorrelator
from sqtc.constants import KB, HBAR, NA

# ── He crystal structure (hcp, P6_3/mmc) ─────────────────────────────────────
# Primitive cell of hcp: 2 atoms, a = 3.57 Å, c = 1.633·a = 5.831 Å
a_he = 3.57   # Å
c_he = 1.633 * a_he

prim_cell_he = np.array([
    [a_he,   0.0,           0.0],
    [-a_he/2, a_he*np.sqrt(3)/2, 0.0],
    [0.0,    0.0,           c_he],
])

prim_pos_he = np.array([
    [0.0,        0.0,           0.0],
    [a_he/2,     a_he/(2*np.sqrt(3)), c_he/2],
])

# ── Physical parameters ───────────────────────────────────────────────────────
T_D_He = 26.0   # K
M_He_amu = 4.002602

# ── Experimental reference values (for comparison) ───────────────────────────
# White et al. (1969); Minkiewicz et al. (1972); Horner (1974)
exp_cv = {   # C_V [J/(mol·K)] at given T [K]
    0.5: 0.6,
    1.0: 2.3,
    1.6: 7.5,
    2.0: 11.8,
}
exp_msd_0K_ang2 = 0.32  # Å² (zero-point MSD)
exp_T_D = 26.0           # K

# ── Debye model reference (harmonic, wrong physics) ──────────────────────────
debye = DebyeCorrelator(T_D=T_D_He, M_amu=M_He_amu)

print("\n" + "="*70)
print(" SQTC Example: Solid ⁴He (hcp, 25 bar, T = 0.5–2.0 K)")
print("="*70)

print("\n── Debye model predictions (harmonic reference) ──")
print(f"  ⟨u²⟩_0K = {debye.msd_ang2(0.001):.4f} Å²   (exp: {exp_msd_0K_ang2:.2f} Å²)")
print(f"  Note: Debye MSD underestimates He due to strong anharmonicity.")
print(f"  C_V values from Debye model (J/mol/K):")
T_range = np.array([0.5, 1.0, 1.6, 2.0])
for T in T_range:
    cv_debye = debye.heat_capacity_v_jmolk(T)
    print(f"    T = {T:.1f} K: C_V = {cv_debye:6.3f}  (exp: {exp_cv[T]:.1f})")

# ── SQTC run ──────────────────────────────────────────────────────────────────
print("\n── SQTC run (Aziz HFD-B pair potential) ──")

# Use T = 1.6 K as the design temperature (near melting at 25 bar)
T_design = 1.6  # K

runner = SQTCRunner(
    element="He",
    mass_amu=M_He_amu,
    prim_cell=prim_cell_he,
    prim_positions=prim_pos_he,
    T=T_design,
    T_D=T_D_He,
    n_atoms_sc=16,           # 16-atom supercell
    force_calculator=EinsteinBackground(
        # Aziz HFD-B bare potential.
        AzizHFDPotential(r_cut=10.0),
        # k_spring [eV/Å²]: quantum zero-point pressure that stabilizes He
        # at the experimental lattice (a=3.57 Å). At this lattice the bare
        # Aziz pair spring constant is -0.00091 eV/Å² (mechanically unstable
        # classically). A spring of 0.003 eV/Å² gives an effective spring
        # constant consistent with T_D ≈ 26 K (experimental).
        k_spring=0.003,
    ),
    r_cutoff=6.5,            # IFC cutoff
    r_max_corr=8.0,          # correlator matching range
    n_ensemble=5,            # 5 SQTC configurations
    work_dir="sqtc_he_run",
    verbosity=1,
)

results = runner.run(
    n_sc_iterations=15,
    epsilon_conv=0.002,
    mixing=0.5,
    T_values=T_range,
    q_mesh_cv=(8, 8, 8),
)

# ── Results comparison ────────────────────────────────────────────────────────
print("\n── SQTC vs Debye vs Experiment ──")
print(f"{'Quantity':<30} {'Debye':>12} {'SQTC':>12} {'Experiment':>12}")
print("-" * 70)

for T in T_range:
    cv_debye = debye.heat_capacity_v_jmolk(T)
    # Find SQTC C_V at this T from the scan
    idx = np.argmin(np.abs(results["T_values"] - T))
    cv_sqtc = results["C_V_scan"][idx]
    print(
        f"  C_V({T:.1f} K) [J/mol/K]  "
        f"{cv_debye:>12.3f}  {cv_sqtc:>12.3f}  {exp_cv[T]:>12.3f}"
    )

print(f"\n  {'T_D [K]':<28} {T_D_He:>12.1f}  {results['T_D_effective']:>12.1f}  {exp_T_D:>12.1f}")
print(f"  {'ZPE [meV/atom]':<28} {'—':>12}  {results['ZPE_eV']*1000:>12.2f}  {'—':>12}")
print(f"\n  Converged: {results['converged']}  in {results['n_iterations']} iterations")

# ── Percentage errors ─────────────────────────────────────────────────────────
print("\n── % Error vs experiment ──")
print(f"{'T [K]':<10} {'Debye error':>15} {'SQTC error':>15}")
print("-" * 42)
for T in T_range:
    cv_debye = debye.heat_capacity_v_jmolk(T)
    idx = np.argmin(np.abs(results["T_values"] - T))
    cv_sqtc = results["C_V_scan"][idx]
    cv_exp = exp_cv[T]
    err_debye = 100.0 * (cv_debye - cv_exp) / cv_exp
    err_sqtc  = 100.0 * (cv_sqtc  - cv_exp) / cv_exp
    print(f"  {T:<8.1f}  {err_debye:>+14.1f}%  {err_sqtc:>+14.1f}%")

# ── Phonon frequencies ────────────────────────────────────────────────────────
print("\n── Zone-boundary frequency (LA [001]) ──")
print(f"  Debye:      {T_D_He * KB / HBAR / (2*np.pi*1e12):.3f} THz")
print(f"  Experiment: 0.41 ± 0.04 THz  (Minkiewicz et al. 1972)")
phcalc = results["phonon_calculator"]
q_ZB = np.array([0.0, 0.0, 0.5])
freqs_ZB = phcalc.frequencies_thz_at_q(q_ZB)
print(f"  SQTC:       {sorted(freqs_ZB)[-1]:.3f} THz  (highest branch at q_ZB)")

print("\n✓ He example complete. Results saved to sqtc_he_run/sqtc_results.json")
