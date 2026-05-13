#!/usr/bin/env python3
"""
SQTC anharmonic metals comparison: Cu, Au, Mo vs β-Ti and Ag
=============================================================
Compares SQTC predictions for five systems spanning a range of anharmonicity:

  System       Structure  T_design  T/T_melt  γ_Grüneisen  Instability
  ────────────────────────────────────────────────────────────────────────
  β-Ti (BCC)   BCC        1200 K    1.04*     —            YES (imaginary modes)
  Mo   (BCC)   BCC        1500 K    0.52      ~1.57        no
  Cu   (FCC)   FCC        1200 K    0.88      ~1.96        no
  Ag   (FCC)   FCC         300 K    0.25      ~2.40        no
  Au   (FCC)   FCC        1200 K    0.90      ~2.95        no
  (*β-Ti exists only above T_trans = 1155 K)

Key comparison axes:
  1. T_D (spectral-moment): SQTC vs TDEP literature vs experiment
  2. C_V at the design temperature: anharmonic excess above 3R = 24.94 J/(mol·K)
  3. Unstable fraction: how many imaginary modes SQTC had to compensate

Usage
-----
    python codes/examples/compare_anharmonic.py [--no-plot]

All result JSON files must exist before running this script.
Results directories:
  sqtc_bccti_vasp_run/sqtc_results.json   (β-Ti, 1200 K)
  sqtc_bccmo_vasp_run/sqtc_results.json   (Mo, 1500 K)
  sqtc_cu_vasp_run/sqtc_results.json      (Cu, 1200 K)
  sqtc_ag_vasp_run/sqtc_results.json      (Ag, 300 K)  -- or sqtc_ag_exp_vol_run
  sqtc_au_vasp_run/sqtc_results.json      (Au, 1200 K)
"""

import sys
import json
import argparse
from pathlib import Path

import numpy as np

# ── System metadata ────────────────────────────────────────────────────────────
# Experimental and literature references for each system.
SYSTEMS = {
    "β-Ti (BCC)": {
        "path": "sqtc_bccti_vasp_run/sqtc_results.json",
        "structure": "BCC",
        "T_design": 1200.0,
        "T_melt": 1941.0,   # K (Ti melting; β phase above 1155 K)
        "gruneisen": "—",
        "harmonically_stable": False,
        "T_D_exp": "350–380",
        "T_D_tdep": "330–370",
        # Experimental C_V at available T points [K]: value [J/(mol·K)]
        "exp_cv": {300: "~25.1", 400: "~25.4", 500: "~25.6",
                   800: "~26.3", 1000: "~26.5", 1200: "~27.5"},
        "tdep_cv": {1200: "~25.8"},
        "cv_3R_excess_exp": "+2.6",   # C_V(exp) - 3R at T_design
    },
    "Mo (BCC)": {
        "path": "sqtc_bccmo_vasp_run/sqtc_results.json",
        "structure": "BCC",
        "T_design": 1500.0,
        "T_melt": 2896.0,
        "gruneisen": "1.57",
        "harmonically_stable": True,
        "T_D_exp": "450",
        "T_D_tdep": "390–420",
        "exp_cv": {600: "~25.3", 1000: "~26.3", 1200: "~26.7", 1500: "~27.2"},
        "tdep_cv": {1500: "~25.8"},
        "cv_3R_excess_exp": "+2.3",
    },
    "Cu (FCC)": {
        "path": "sqtc_cu_vasp_run/sqtc_results.json",
        "structure": "FCC",
        "T_design": 1200.0,
        "T_melt": 1358.0,
        "gruneisen": "1.96",
        "harmonically_stable": True,
        "T_D_exp": "315",
        "T_D_tdep": "280–300",
        "exp_cv": {300: "~24.4", 500: "~24.9", 700: "~25.3", 1000: "~25.7", 1200: "~26.1"},
        "tdep_cv": {1200: "~25.5"},
        "cv_3R_excess_exp": "+1.2",
    },
    "Ag (FCC)": {
        "path": "sqtc_ag_vasp_run/sqtc_results.json",
        "structure": "FCC",
        "T_design": 300.0,
        "T_melt": 1235.0,
        "gruneisen": "2.40",
        "harmonically_stable": True,
        "T_D_exp": "220–228",
        "T_D_tdep": "200–215",
        "exp_cv": {100: "~22.7", 200: "~24.0", 300: "~24.9", 400: "~25.8"},
        "tdep_cv": {300: "~24.9"},
        "cv_3R_excess_exp": "−0.0",   # at 300 K, Ag ≈ classical limit
    },
    "Au (FCC)": {
        "path": "sqtc_au_vasp_run/sqtc_results.json",
        "structure": "FCC",
        "T_design": 1200.0,
        "T_melt": 1337.0,
        "gruneisen": "2.95",
        "harmonically_stable": True,
        "T_D_exp": "165",
        "T_D_tdep": "120–140",
        "exp_cv": {300: "~24.4", 500: "~25.0", 700: "~25.7", 1000: "~26.8", 1200: "~27.4"},
        "tdep_cv": {1200: "~25.9"},
        "cv_3R_excess_exp": "+2.5",
    },
}

# Fallback: prefer exp-vol run for Ag if vasp run not found
_ag_fallback = "sqtc_ag_exp_vol_run/sqtc_results.json"


def load_results(path: str):
    """Load SQTC JSON result; return None if file not found."""
    p = Path(path)
    if not p.exists():
        return None
    with open(p) as f:
        return json.load(f)


def format_float(v, fmt=".1f"):
    try:
        return format(float(v), fmt)
    except (TypeError, ValueError):
        return "—"


def print_summary_table(loaded: dict):
    """Print the main comparison table."""
    DP = 24.943  # 3R in J/(mol·K)

    print("\n" + "=" * 90)
    print("  SQTC anharmonic metals benchmark — summary")
    print("=" * 90)
    print(f"  {'System':<12} {'Struct':>6} {'T[K]':>6} {'T/Tm':>5} {'γ':>5}  "
          f"{'T_D SQTC':>9} {'T_D exp':>9} {'C_V(T) SQTC':>12} {'C_V exp':>10} {'ΔC_V':>7}")
    print("-" * 90)

    for name, meta in SYSTEMS.items():
        r = loaded.get(name)
        if r is None:
            print(f"  {name:<12} {'—':>6} {'—':>6} {'—':>5} {meta['gruneisen']:>5}  "
                  f"  [NOT RUN]")
            continue

        T_des = meta["T_design"]
        T_m   = meta["T_melt"]
        t_tm  = f"{T_des / T_m:.2f}"

        # T_D
        td_sqtc = format_float(r.get("T_D_effective", float("nan")))
        td_exp  = meta["T_D_exp"]

        # C_V at design T
        T_vals = np.array(r.get("T_values", []))
        cv_scan = np.array(r.get("C_V_scan", []))
        if len(T_vals) > 0 and T_des in T_vals:
            idx = list(T_vals).index(T_des)
            cv_sqtc = cv_scan[idx]
        elif len(T_vals) > 0:
            cv_sqtc = float(np.interp(T_des, T_vals, cv_scan))
        else:
            cv_sqtc = float("nan")

        cv_exp_str = meta["exp_cv"].get(int(T_des), "—")
        delta_cv   = f"{cv_sqtc - DP:+.2f}" if not np.isnan(cv_sqtc) else "—"

        unstable = r.get("unstable_fraction", 0.0)
        stab_str = f"{'UNSTABLE':>8}" if not meta["harmonically_stable"] else f"{'stable':>8}"

        print(f"  {name:<12} {meta['structure']:>6} {T_des:>6.0f} {t_tm:>5}  "
              f"{meta['gruneisen']:>5}  "
              f"{td_sqtc:>9} {td_exp:>9} "
              f"{cv_sqtc:>12.3f} {cv_exp_str:>10} {delta_cv:>7}   {stab_str}")

    print("-" * 90)
    print(f"  3R (Dulong–Petit) = {DP:.3f} J/(mol·K);  ΔC_V = C_V(SQTC) − 3R")


def print_cv_scan_table(loaded: dict):
    """Print per-system C_V vs T scan."""
    print("\n" + "─" * 70)
    print("  C_V(T) scan [J/(mol·K)] — SQTC vs experiment")
    print("─" * 70)

    for name, meta in SYSTEMS.items():
        r = loaded.get(name)
        if r is None:
            continue
        T_vals = np.array(r.get("T_values", []))
        cv_scan = np.array(r.get("C_V_scan", []))
        exp_cv  = meta["exp_cv"]

        print(f"\n  ── {name} (T_design = {meta['T_design']:.0f} K,"
              f" T_D_eff = {r.get('T_D_effective', float('nan')):.1f} K,"
              f" unstable_frac = {r.get('unstable_fraction', 0.0):.3f})")
        print(f"    {'T [K]':>8}  {'C_V SQTC':>12}  {'C_V exp':>10}  {'error':>8}")
        print("    " + "-" * 46)

        for T_v, cv_v in zip(T_vals, cv_scan):
            exp_str = exp_cv.get(int(T_v), "—")
            try:
                exp_val = float(exp_str.replace("~", "").replace(">", "").strip())
                err_str = f"{cv_v - exp_val:+.3f}"
            except (ValueError, AttributeError):
                err_str = "—"
            print(f"    {T_v:>8.1f}  {cv_v:>12.4f}  {exp_str:>10}  {err_str:>8}")


def print_anharmonic_ranking(loaded: dict):
    """Rank systems by SQTC anharmonic C_V excess at design T."""
    DP = 24.943
    print("\n" + "─" * 60)
    print("  Anharmonic C_V excess at design T: SQTC vs experiment")
    print("─" * 60)
    print(f"  {'System':<12}  {'C_V SQTC':>10}  {'C_V exp':>10}  "
          f"{'SQTC-3R':>9}  {'exp-3R':>9}")
    print("  " + "-" * 56)

    rows = []
    for name, meta in SYSTEMS.items():
        r = loaded.get(name)
        if r is None:
            continue
        T_des  = meta["T_design"]
        T_vals = np.array(r.get("T_values", []))
        cv_scan = np.array(r.get("C_V_scan", []))

        if len(T_vals) == 0:
            continue
        if T_des in T_vals:
            idx = list(T_vals).index(T_des)
            cv_sqtc = cv_scan[idx]
        else:
            cv_sqtc = float(np.interp(T_des, T_vals, cv_scan))

        exp_str = meta["exp_cv"].get(int(T_des), "—")
        try:
            exp_val = float(exp_str.replace("~", "").replace(">", "").strip())
        except (ValueError, AttributeError):
            exp_val = float("nan")

        rows.append((name, cv_sqtc, exp_val, cv_sqtc - DP, exp_val - DP))

    # sort by experimental anharmonic excess descending
    rows.sort(key=lambda x: x[4] if not np.isnan(x[4]) else -99, reverse=True)

    for name, cv_s, cv_e, ds, de in rows:
        e_str = f"{cv_e:>10.3f}" if not np.isnan(cv_e) else f"{'—':>10}"
        de_str = f"{de:>+9.3f}" if not np.isnan(de) else f"{'—':>9}"
        print(f"  {name:<12}  {cv_s:>10.4f}  {e_str}  {ds:>+9.4f}  {de_str}")


def main():
    parser = argparse.ArgumentParser(
        description="Compare SQTC anharmonic metal benchmarks"
    )
    parser.add_argument("--no-plot", action="store_true",
                        help="Print tables only, skip matplotlib plot")
    args = parser.parse_args()

    # Load all available results
    loaded = {}
    for name, meta in SYSTEMS.items():
        r = load_results(meta["path"])
        # Try Ag fallback
        if r is None and name == "Ag (FCC)":
            r = load_results(_ag_fallback)
            if r is not None:
                print(f"  [INFO] Using {_ag_fallback} for Ag results")
        if r is not None:
            loaded[name] = r
        else:
            print(f"  [MISSING] {meta['path']} — run {name} first")

    if not loaded:
        print("\nERROR: No results found. Run the SQTC benchmarks first.")
        sys.exit(1)

    print_summary_table(loaded)
    print_cv_scan_table(loaded)
    print_anharmonic_ranking(loaded)

    if args.no_plot or len(loaded) < 2:
        return

    # ── Optional matplotlib plot ───────────────────────────────────────────────
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
    except ImportError:
        print("\n  [INFO] matplotlib not available — skipping plot")
        return

    DP = 24.943
    fig, axes = plt.subplots(1, 2, figsize=(13, 5))
    fig.suptitle("SQTC anharmonic metals — C_V comparison", fontsize=13)

    colors = {
        "β-Ti (BCC)": "#e63946",
        "Mo (BCC)":   "#457b9d",
        "Cu (FCC)":   "#f4a261",
        "Ag (FCC)":   "#6a994e",
        "Au (FCC)":   "#e9c46a",
    }
    markers = {
        "β-Ti (BCC)": "^",
        "Mo (BCC)":   "s",
        "Cu (FCC)":   "o",
        "Ag (FCC)":   "D",
        "Au (FCC)":   "*",
    }

    ax_cv, ax_excess = axes

    for name, r in loaded.items():
        T_vals  = np.array(r.get("T_values", []))
        cv_scan = np.array(r.get("C_V_scan", []))
        if len(T_vals) == 0:
            continue
        c  = colors.get(name, "grey")
        mk = markers.get(name, "o")
        label = f"{name}  [T_D={r.get('T_D_effective', 0):.0f} K]"
        ax_cv.plot(T_vals, cv_scan, marker=mk, color=c, label=label, lw=1.5, ms=6)
        ax_excess.plot(T_vals, cv_scan - DP, marker=mk, color=c, label=name, lw=1.5, ms=6)

    # Reference lines
    ax_cv.axhline(DP, color="k", lw=0.8, ls="--", label="3R (Dulong–Petit)")
    ax_excess.axhline(0, color="k", lw=0.8, ls="--")

    ax_cv.set_xlabel("T [K]")
    ax_cv.set_ylabel("C_V  [J / (mol·K)]")
    ax_cv.set_title("C_V(T)  — SQTC vs 3R")
    ax_cv.legend(fontsize=7)
    ax_cv.grid(True, alpha=0.3)

    ax_excess.set_xlabel("T [K]")
    ax_excess.set_ylabel(r"$C_V - 3R$  [J / (mol·K)]")
    ax_excess.set_title("Anharmonic C_V excess above 3R")
    ax_excess.legend(fontsize=8)
    ax_excess.grid(True, alpha=0.3)

    plt.tight_layout()
    out = Path("sqtc_anharmonic_comparison.png")
    fig.savefig(out, dpi=150, bbox_inches="tight")
    print(f"\n  Plot saved to {out}")


if __name__ == "__main__":
    main()
