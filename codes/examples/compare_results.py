#!/usr/bin/env python3
"""
SQTC comparison: C_V(T) for He and H₂
=======================================
Plots SQTC predictions alongside the Debye model and experimental data.
Produces two panels:
  Left:  ⁴He at 25 bar (T_D = 26 K)
  Right: para-H₂ Phase I (T_D = 110 K)

Usage
-----
    python examples/compare_results.py [--no-plot]

The results JSON files (sqtc_he_run/sqtc_results.json and
sqtc_h2_run/sqtc_results.json) must exist before running this script.
Run run_sqtc_he.py and run_sqtc_h2.py first.

If --no-plot is given, results are printed as a table without plotting.
"""

import sys
import json
import argparse
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).parents[1]))

from sqtc.correlators import DebyeCorrelator
from sqtc.constants import R_GAS

# ── Experimental data ─────────────────────────────────────────────────────────
# He: White et al. (1969) & Wanner & Meyer (1973)
exp_he_T   = np.array([0.5, 0.8, 1.0, 1.3, 1.6, 1.8, 2.0])
exp_he_cv  = np.array([0.6, 1.4, 2.3, 4.7, 7.5, 9.9, 11.8])  # J/(mol·K)

# H₂: Silvera (1980) Table 6
exp_h2_T   = np.array([4.0, 5.0, 6.0, 8.0, 10.0, 13.0])
exp_h2_cv  = np.array([1.0, 1.6, 2.1, 3.5,  4.6,   5.7])   # J/(mol·K)


def load_sqtc_results(path: Path):
    with open(path) as f:
        return json.load(f)


def debye_cv_scan(T_D, M_amu, T_values):
    corr = DebyeCorrelator(T_D=T_D, M_amu=M_amu)
    return np.array([corr.heat_capacity_v_jmolk(T) for T in T_values])


def print_table(label, T_vals, cv_debye, cv_sqtc, exp_T, exp_cv):
    print(f"\n── {label} ──")
    print(f"{'T [K]':>8}  {'C_V Debye':>12}  {'C_V SQTC':>12}  {'C_V exp':>12}")
    print("-" * 50)
    for T in T_vals:
        cv_d = np.interp(T, T_vals, cv_debye)
        cv_s = np.interp(T, T_vals, cv_sqtc)
        cv_e = np.interp(T, exp_T, exp_cv) if exp_T.min() <= T <= exp_T.max() else float("nan")
        exp_str = f"{cv_e:12.3f}" if not np.isnan(cv_e) else f"{'—':>12}"
        print(f"  {T:>6.1f}  {cv_d:>12.3f}  {cv_s:>12.3f}  {exp_str}")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--no-plot", action="store_true")
    args = parser.parse_args()

    he_path = Path("sqtc_he_run/sqtc_results.json")
    h2_path = Path("sqtc_h2_run/sqtc_results.json")

    he_ok = he_path.exists()
    h2_ok = h2_path.exists()

    if not he_ok and not h2_ok:
        print("ERROR: No SQTC results found.")
        print("Run 'python examples/run_sqtc_he.py' and 'python examples/run_sqtc_h2.py' first.")
        sys.exit(1)

    # ── He ─────────────────────────────────────────────────────────────────────
    if he_ok:
        he = load_sqtc_results(he_path)
        he_T = np.array(he["T_values"])
        he_sqtc = np.array(he["C_V_scan"])
        he_debye = debye_cv_scan(T_D=26.0, M_amu=4.0026, T_values=he_T)
        print_table("⁴He (25 bar)", he_T, he_debye, he_sqtc, exp_he_T, exp_he_cv)
        print(f"\n  T_D effective = {he['T_D_effective']:.1f} K  (input: 26 K)")
        print(f"  ZPE = {he['ZPE_eV']*1000:.2f} meV/atom")
        print(f"  MSD_0K (Debye eff) = {he['MSD_ang2']:.4f} Å²  (exp: 0.32 Å²)")
    else:
        print("He results not found. Run run_sqtc_he.py first.")

    # ── H₂ ─────────────────────────────────────────────────────────────────────
    if h2_ok:
        h2 = load_sqtc_results(h2_path)
        h2_T = np.array(h2["T_values"])
        h2_sqtc = np.array(h2["C_V_scan"])
        h2_debye = debye_cv_scan(T_D=110.0, M_amu=2.01588, T_values=h2_T)
        print_table("para-H₂ Phase I (1 atm)", h2_T, h2_debye, h2_sqtc, exp_h2_T, exp_h2_cv)
        print(f"\n  T_D effective = {h2['T_D_effective']:.1f} K  (input: 110 K)")
    else:
        print("H₂ results not found. Run run_sqtc_h2.py first.")

    # ── Plot ───────────────────────────────────────────────────────────────────
    if args.no_plot:
        return

    try:
        import matplotlib.pyplot as plt
    except ImportError:
        print("\nmatplotlib not available — skipping plot.")
        return

    fig, axes = plt.subplots(1, 2, figsize=(11, 5))
    fig.suptitle("SQTC vs Debye vs Experiment", fontsize=13)

    # ── He panel ───────────────────────────────────────────────────────────────
    ax = axes[0]
    if he_ok:
        ax.plot(he_T, he_debye, "b--", lw=1.5, label="Debye")
        ax.plot(he_T, he_sqtc,  "r-",  lw=2.0, label="SQTC")
    ax.errorbar(exp_he_T, exp_he_cv, fmt="ko", ms=5, label="Experiment")
    ax.axhline(3 * R_GAS, color="gray", lw=0.8, ls=":")
    ax.text(exp_he_T[-1] * 0.5, 3 * R_GAS + 0.3, "3R", color="gray", fontsize=9)
    ax.set_xlabel("Temperature (K)")
    ax.set_ylabel("C_V  (J mol⁻¹ K⁻¹)")
    ax.set_title("⁴He  (hcp, 25 bar)")
    ax.legend(loc="upper left")
    ax.set_xlim(0, 2.2)
    ax.set_ylim(0, 14)

    # ── H₂ panel ───────────────────────────────────────────────────────────────
    ax = axes[1]
    if h2_ok:
        ax.plot(h2_T, h2_debye, "b--", lw=1.5, label="Debye")
        ax.plot(h2_T, h2_sqtc,  "r-",  lw=2.0, label="SQTC")
    ax.errorbar(exp_h2_T, exp_h2_cv, fmt="ko", ms=5, label="Experiment")
    ax.axhline(3 * R_GAS, color="gray", lw=0.8, ls=":")
    ax.text(exp_h2_T[0], 3 * R_GAS + 0.2, "3R", color="gray", fontsize=9)
    ax.set_xlabel("Temperature (K)")
    ax.set_title("para-H₂  (hcp Phase I, 1 atm)")
    ax.legend(loc="upper left")
    ax.set_ylim(0, 8)

    plt.tight_layout()
    out = "sqtc_comparison.pdf"
    plt.savefig(out, dpi=150)
    print(f"\nPlot saved to {out}")
    plt.show()


if __name__ == "__main__":
    main()
