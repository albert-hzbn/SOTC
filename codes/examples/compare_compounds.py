#!/usr/bin/env python3
"""
Compare SQTC compound benchmarks: MgO / NaCl / PbTe
====================================================
Three rocksalt compounds spanning low to high anharmonicity:

  MgO  — LOW    anharmonicity  (γ ≈ 1.5, T_D ≈ 760 K)
  NaCl — MEDIUM anharmonicity  (γ ≈ 1.6, T_D ≈ 321 K)
  PbTe — HIGH   anharmonicity  (γ ≈ 2.3, T_D ≈ 130 K; giant soft TA modes)

Key metrics compared:
  1. T_D (effective, SQTC) vs T_D (harmonic DFT ref) vs T_D (experiment)
  2. C_V per formula unit vs experiment over the T_values range
  3. Anharmonic excess ΔC_V = C_V − 6R  (6R = 49.88 J/mol/K is classical limit)
  4. Imaginary mode fraction (PbTe expected > 0)
  5. Convergence quality

Usage:
    python compare_compounds.py            # text tables only
    python compare_compounds.py --plot     # also save PNG figure
"""

import argparse
import json
from pathlib import Path

import numpy as np

WORK_ROOT = Path(__file__).resolve().parent.parent.parent

R_GAS = 8.31446  # J/(mol·K)
SIX_R = 6.0 * R_GAS  # 49.883 J/(mol·K) — classical limit per f.u. (2 atoms)

# ── Experimental / literature references ─────────────────────────────────────
# Sources: NIST JANAF, Isaak et al. (MgO), Dworkin & Bredig (NaCl),
#          Delaire et al. 2011 Nature Materials (PbTe)
EXPERIMENTAL = {
    "MgO": {
        "formula": "MgO",
        "label": "MgO  (low  γ≈1.5)",
        "T_D_exp": 760.0,
        "T_D_harmonic_dft": 740.0,   # PBE phonopy/phonon3py
        "gamma_exp": 1.50,
        "kappa_300K": 60.0,          # W/(m·K)  — ref. value
        "harmonic_status": "stable",
        "imaginary_expected": False,
        "C_V": {                      # J/(mol·K) per formula unit
            300.0:  37.2,
            500.0:  44.6,
            800.0:  47.6,
            1000.0: 48.5,
            1200.0: 49.2,
            1500.0: 49.6,
        },
    },
    "NaCl": {
        "formula": "NaCl",
        "label": "NaCl (med  γ≈1.6)",
        "T_D_exp": 321.0,
        "T_D_harmonic_dft": 300.0,   # PBE phonopy
        "gamma_exp": 1.60,
        "kappa_300K": 6.7,
        "harmonic_status": "stable",
        "imaginary_expected": False,
        "C_V": {
            200.0: 44.0,
            300.0: 49.0,
            400.0: 49.9,
            700.0: 50.5,
            900.0: 51.2,
        },
    },
    "PbTe": {
        "formula": "PbTe",
        "label": "PbTe (high γ≈2.3)",
        "T_D_exp": 130.0,
        "T_D_harmonic_dft": 115.0,   # PBE harmonic (imaginary TA at X)
        "gamma_exp": 2.30,
        "kappa_300K": 1.7,
        "harmonic_status": "imaginary_TA_near_X",
        "imaginary_expected": True,
        "C_V": {
            300.0: 50.7,
            500.0: 52.0,
            700.0: 53.5,
            900.0: 55.0,
        },
    },
}

# ── SQTC result directories ────────────────────────────────────────────────────
RESULT_FILES = {
    "MgO":  WORK_ROOT / "sqtc_mgo_vasp_run"  / "sqtc_results.json",
    "NaCl": WORK_ROOT / "sqtc_nacl_vasp_run" / "sqtc_results.json",
    "PbTe": WORK_ROOT / "sqtc_pbte_vasp_run" / "sqtc_results.json",
}


def load_results():
    """Load SQTC JSON results; mark missing entries clearly."""
    data = {}
    for key, path in RESULT_FILES.items():
        if path.exists():
            with open(path) as f:
                data[key] = json.load(f)
        else:
            data[key] = None
    return data


def print_summary(sqtc_data):
    """Print the top-level convergence and T_D comparison table."""
    print("\n" + "=" * 78)
    print("  SQTC COMPOUND BENCHMARKS: MgO / NaCl / PbTe")
    print("  Rocksalt B1 structure — low / moderate / high anharmonicity")
    print("=" * 78)

    headers = f"  {'System':<24}  {'Converged':>9}  {'Iter':>4}  "
    headers += f"{'T_D(SQTC)':>10}  {'T_D(harm)':>10}  {'T_D(exp)':>10}  {'γ(exp)':>8}"
    print(headers)
    print("  " + "-" * 74)

    for key in ["MgO", "NaCl", "PbTe"]:
        res  = sqtc_data[key]
        ref  = EXPERIMENTAL[key]
        label = ref["label"]

        if res is None:
            print(f"  {label:<24}  {'NOT RUN':>9}")
            continue

        conv = res.get("converged", False)
        nit  = res.get("n_iterations", -1)
        td   = res.get("T_D_effective", float("nan"))
        td_harm = ref["T_D_harmonic_dft"]
        td_exp  = ref["T_D_exp"]
        gamma   = ref["gamma_exp"]

        print(
            f"  {label:<24}  {str(conv):>9}  {nit:>4}  "
            f"{td:>10.1f}  {td_harm:>10.1f}  {td_exp:>10.1f}  {gamma:>8.2f}"
        )

    print()
    print("  T_D(harm) = harmonic DFT (0 K PBE-PAW phonopy reference)")
    print("  T_D(exp)  = experimental low-T Cp calorimetric Debye temperature")
    print("  γ(exp)    = mode-average Grüneisen parameter near room temperature")
    print()
    print("  NOTE: LO-TO splitting NOT included in SQTC short-range IFC framework.")
    print("  Expected bias: T_D(SQTC) < T_D(exp) due to missing LO branch.")
    print("  Effect largest for PbTe (Z* ≈ ±5.9; Δω_LO-TO ≈ 1.5 THz)")


def print_cv_table(sqtc_data):
    """Print C_V(T) comparison table for each system."""
    print("\n" + "=" * 78)
    print("  HEAT CAPACITY C_V  [J/(mol·K) per formula unit]")
    print(f"  Classical limit  6R = {SIX_R:.3f} J/(mol·K)")
    print("=" * 78)

    for key in ["MgO", "NaCl", "PbTe"]:
        res  = sqtc_data[key]
        ref  = EXPERIMENTAL[key]
        T_design = {"MgO": 1500.0, "NaCl": 700.0, "PbTe": 700.0}[key]

        print(f"\n  --- {ref['label']} ---   (T_design = {T_design:.0f} K)")
        print(f"  {'T [K]':>7}  {'C_V(SQTC)':>10}  {'ΔC_V(SQTC)':>12}  {'C_V(exp)':>10}  {'ΔC_V(exp)':>10}")
        print(f"  {'-'*57}")

        if res is None:
            print("    ** Result not available **")
            continue

        T_arr  = res.get("T_values", [])
        cv_arr = res.get("C_V_scan", [])
        exp_cv = ref["C_V"]

        for T_v, cv_v in zip(T_arr, cv_arr):
            T_v = float(T_v)
            dcv_sqtc = cv_v - SIX_R
            cv_exp  = exp_cv.get(T_v, None)
            exp_str  = f"{cv_exp:>10.2f}" if cv_exp is not None else f"{'—':>10}"
            dcv_exp  = f"{cv_exp - SIX_R:>10.3f}" if cv_exp is not None else f"{'—':>10}"
            print(
                f"  {T_v:>7.0f}  {cv_v:>10.4f}  {dcv_sqtc:>12.4f}  "
                f"{exp_str}  {dcv_exp}"
            )

    print()
    print("  ΔC_V = C_V − 6R  (positive = super-classical anharmonic excess)")


def print_anharmonic_table(sqtc_data):
    """Ranked anharmonicity summary across all three compounds."""
    print("\n" + "=" * 78)
    print("  ANHARMONICITY RANKING — compound benchmarks")
    print("=" * 78)
    print()

    headers = (
        f"  {'System':<8}  {'γ(exp)':>8}  {'κ(300K)':>10}  "
        f"{'T_D(SQTC)':>10}  {'T_D(exp)':>10}  "
        f"{'f_imag':>8}  {'Harm. status':>22}"
    )
    print(headers)
    print("  " + "-" * 74)

    for key in ["MgO", "NaCl", "PbTe"]:
        res  = sqtc_data[key]
        ref  = EXPERIMENTAL[key]

        td   = res.get("T_D_effective", float("nan")) if res else float("nan")
        f_im = res.get("unstable_fraction", float("nan")) if res else float("nan")

        print(
            f"  {key:<8}  {ref['gamma_exp']:>8.2f}  {ref['kappa_300K']:>10.1f}  "
            f"{td:>10.1f}  {ref['T_D_exp']:>10.1f}  "
            f"{f_im:>8.3f}  {ref['harmonic_status']:>22}"
        )

    print()
    print("  κ(300K): thermal conductivity [W/(m·K)] — independent measure of anharmonicity")
    print("  f_imag:  imaginary mode fraction in self-consistent SQTC phonon calculation")
    print("  Harm. status: stability of harmonic DFT phonon spectrum at 0 K")


def plot_cv_comparison(sqtc_data, outfile="compound_cv_comparison.png"):
    """Save a matplotlib figure comparing C_V(T) for all three compounds."""
    import matplotlib.pyplot as plt

    fig, axes = plt.subplots(1, 3, figsize=(14, 5), sharey=False)
    colors = {"MgO": "steelblue", "NaCl": "darkorange", "PbTe": "crimson"}

    for ax, key in zip(axes, ["MgO", "NaCl", "PbTe"]):
        res  = sqtc_data[key]
        ref  = EXPERIMENTAL[key]
        color = colors[key]

        # Plot SQTC result
        if res is not None:
            T_arr  = np.asarray(res.get("T_values", []))
            cv_arr = np.asarray(res.get("C_V_scan", []))
            ax.plot(T_arr, cv_arr, "o-", color=color, lw=2, ms=6,
                    label=f"SQTC (T_D={res.get('T_D_effective', 0):.0f} K)")

        # Plot experimental data
        T_exp = sorted(ref["C_V"].keys())
        cv_exp = [ref["C_V"][T] for T in T_exp]
        ax.plot(T_exp, cv_exp, "s--", color="black", lw=1.5, ms=5,
                label=f"Experiment (T_D={ref['T_D_exp']:.0f} K)")

        # Classical limit
        T_range = np.linspace(50, max(max(T_exp), 200), 200)
        ax.axhline(SIX_R, ls=":", color="gray", lw=1, label="6R = 49.88 J/mol/K")

        ax.set_title(ref["label"], fontsize=11, fontweight="bold")
        ax.set_xlabel("Temperature [K]", fontsize=10)
        ax.set_ylabel("$C_V$ [J mol$^{-1}$ K$^{-1}$]", fontsize=10)
        ax.legend(fontsize=8)
        ax.grid(True, alpha=0.3)
        ax.set_ylim(bottom=0)

    fig.suptitle(
        "SQTC Compound Benchmarks: MgO / NaCl / PbTe\n"
        "Low / Moderate / High Anharmonicity (Rocksalt B1)",
        fontsize=12, fontweight="bold",
    )
    fig.tight_layout()
    fig.savefig(outfile, dpi=150, bbox_inches="tight")
    print(f"\n  Figure saved to: {outfile}")


def main():
    parser = argparse.ArgumentParser(description="Compare SQTC compound benchmark results")
    parser.add_argument("--plot", action="store_true", help="Save C_V comparison figure")
    args = parser.parse_args()

    sqtc_data = load_results()
    n_available = sum(1 for v in sqtc_data.values() if v is not None)
    print(f"\n  Loaded {n_available}/3 compound result files.")
    if n_available < 3:
        missing = [k for k, v in sqtc_data.items() if v is None]
        print(f"  MISSING: {', '.join(missing)}  —  run corresponding SQTC jobs first.")

    print_summary(sqtc_data)
    print_cv_table(sqtc_data)
    print_anharmonic_table(sqtc_data)

    if args.plot:
        try:
            plot_cv_comparison(sqtc_data)
        except ImportError:
            print("\n  matplotlib not available; skipping figure.")


if __name__ == "__main__":
    main()
