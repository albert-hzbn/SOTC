#!/usr/bin/env python3
"""
SQTC+GPAW comparison: Al (harmonic) vs Pb (harmonic + anharmonic)
==================================================================
Produces a four-panel figure and a terminal table:

  Panel 1  C_V(T) for Al  — GPAW vs VASP vs Debye vs experiment
  Panel 2  C_V(T) for Pb  — harmonic SQTC vs anharmonic SQTC vs experiment
  Panel 3  Anharmonic correction ΔC_V^anh(T) for Pb
           = C_V^anh(T) − C_V^harm(T)  [also as % of 3R]
  Panel 4  T-dependent Debye temperature T_D^SQTC(T):
           shows phonon softening in Pb (T_D drops) while Al stays flat

Physical picture
----------------
  Al: weakly anharmonic (γ ≈ 2.17, T_D = 390 K)
    At 300 K (T/T_D ≈ 0.77) the harmonic approximation is excellent.
    GPAW and VASP results should agree; both agree with experiment.

  Pb: strongly anharmonic (γ ≈ 2.73, T_D = 88 K)
    At 500 K (T/T_D ≈ 5.7, T/T_melt ≈ 0.83) anharmonic phonon softening
    reduces phonon frequencies → shifts C_V above the harmonic prediction.
    ΔC_V^anh grows from ~0 at 100 K to ~+2 J/(mol·K) near melting.

Requirements
------------
  sqtc_al_gpaw_run/sqtc_results.json      (from run_sqtc_al_gpaw.py)
  sqtc_pb_gpaw_run/pb_harm_vs_anh.json    (from run_sqtc_pb_gpaw.py, both T)

Optional (shown if present):
  sqtc_al_fast_vasp_run/sqtc_results.json  (VASP reference for Al)

Usage
-----
    python codes/examples/compare_al_pb_gpaw.py [--no-plot]

With --no-plot: print table only (no matplotlib required).
"""

from __future__ import annotations

import sys
import json
import argparse
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).parents[1]))

from sqtc.correlators import DebyeCorrelator
from sqtc.constants import R_GAS


# ── Paths ─────────────────────────────────────────────────────────────────────

ROOT = Path(".")
AL_GPAW_PATH  = ROOT / "sqtc_al_gpaw_run"  / "sqtc_results.json"
AL_VASP_PATH  = ROOT / "sqtc_al_fast_vasp_run" / "sqtc_results.json"
PB_MERGE_PATH = ROOT / "sqtc_pb_gpaw_run"  / "pb_harm_vs_anh.json"
PB_HARM_PATH  = ROOT / "sqtc_pb_gpaw_run"  / "harmonic"  / "sqtc_results.json"
PB_ANH_PATH   = ROOT / "sqtc_pb_gpaw_run"  / "anharmonic" / "sqtc_results.json"


# ── Experimental data ─────────────────────────────────────────────────────────

# Al: Desnoyers & Morrison (1958) + NIST Webbook
EXP_AL_T  = np.array([100., 200., 300., 400., 500., 600., 700., 800., 900.])
EXP_AL_CV = np.array([18.8,  23.6,  24.4,  25.2,  26.1,  27.1,  28.1,  29.2,  30.3])

# Pb: Hultgren et al. (1973) selected values [J/(mol·K)]
EXP_PB_T  = np.array([50.,  100., 150., 200., 250., 300., 400., 500., 600.])
EXP_PB_CV = np.array([18.2,  24.7,  25.7,  26.4,  26.9,  27.4,  28.7,  30.3,  32.1])


# ── Helpers ───────────────────────────────────────────────────────────────────

def load_json(path: Path):
    if not path.exists():
        return None
    with open(path) as fh:
        return json.load(fh)

def debye_cv_scan(T_D: float, M_amu: float, T_values: np.ndarray) -> np.ndarray:
    """Debye model C_V [J/(mol·K)] on a temperature array."""
    corr = DebyeCorrelator(T_D=T_D, M_amu=M_amu)
    return np.array([corr.heat_capacity_v_jmolk(T) for T in T_values])

THREE_R = 3.0 * R_GAS   # Dulong-Petit limit  [J/(mol·K)]


# ── Load data ─────────────────────────────────────────────────────────────────

def load_all():
    data = {}

    # Al — GPAW
    d = load_json(AL_GPAW_PATH)
    if d is not None:
        data["al_gpaw"] = {
            "T_design": d["T"],
            "T_D": d["T_D_effective"],
            "T_scan": np.array(d["T_values"]),
            "cv": np.array(d["C_V_scan"]),
            "cv_at_design": d["C_V_jmolk"],
            "converged": d["converged"],
        }
    else:
        print(f"WARNING: {AL_GPAW_PATH} not found — run run_sqtc_al_gpaw.py first.")

    # Al — VASP (optional)
    d = load_json(AL_VASP_PATH)
    if d is not None:
        data["al_vasp"] = {
            "T_D": d["T_D_effective"],
            "T_scan": np.array(d["T_values"]),
            "cv": np.array(d["C_V_scan"]),
        }

    # Pb — merged comparison JSON (preferred)
    d = load_json(PB_MERGE_PATH)
    if d is not None:
        data["pb"] = {
            "T_scan": np.array(d["T_scan"]),
            "cv_harm": np.array(d["cv_harm_jmolk"]),
            "cv_anh":  np.array(d["cv_anh_jmolk"]),
            "delta_cv": np.array(d["delta_cv_jmolk"]),
            "T_D_harm": d["T_D_harm"],
            "T_D_anh":  d["T_D_anh"],
            "converged_harm": d["converged_harm"],
            "converged_anh":  d["converged_anh"],
        }
    else:
        # Fallback: load individual JSONs
        dh = load_json(PB_HARM_PATH)
        da = load_json(PB_ANH_PATH)
        if dh is not None and da is not None:
            T_scan = np.array(dh["T_values"])
            cv_harm = np.array(dh["C_V_scan"])
            cv_anh  = np.array(da["C_V_scan"])
            data["pb"] = {
                "T_scan": T_scan,
                "cv_harm": cv_harm,
                "cv_anh":  cv_anh,
                "delta_cv": cv_anh - cv_harm,
                "T_D_harm": dh["T_D_effective"],
                "T_D_anh":  da["T_D_effective"],
                "converged_harm": dh["converged"],
                "converged_anh":  da["converged"],
            }
        elif dh is not None:
            print("WARNING: Pb anharmonic run missing — only harmonic will be shown.")
            T_scan = np.array(dh["T_values"])
            data["pb"] = {
                "T_scan": T_scan,
                "cv_harm": np.array(dh["C_V_scan"]),
                "cv_anh":  None,
                "delta_cv": None,
                "T_D_harm": dh["T_D_effective"],
                "T_D_anh":  None,
                "converged_harm": dh["converged"],
                "converged_anh":  False,
            }
        else:
            print(f"WARNING: Pb results not found — run run_sqtc_pb_gpaw.py first.")

    return data


# ── Terminal table ─────────────────────────────────────────────────────────────

def print_table(data: dict) -> None:
    print()
    print("=" * 80)
    print("  SQTC+GPAW: Al vs Pb  —  harmonic vs anharmonic comparison")
    print("=" * 80)

    if "al_gpaw" in data:
        al = data["al_gpaw"]
        print(f"\n── Al (fcc, γ ≈ 2.17)  [T_design = {al['T_design']:.0f} K] ──")
        print(f"  T_D (SQTC+GPAW) = {al['T_D']:.1f} K  "
              f"(exp ≈ 390 K, expected slight underestimate with PBE)")
        print(f"  Converged       : {al['converged']}")
        if "al_vasp" in data:
            vasp = data["al_vasp"]
            print(f"  T_D (SQTC+VASP) = {vasp['T_D']:.1f} K  (VASP reference)")
        print()
        print(f"  {'T [K]':>8}  {'C_V GPAW':>12}  {'C_V VASP':>12}  "
              f"{'C_V Debye':>12}  {'C_V exp':>12}")
        print("  " + "-" * 62)
        cv_debye_al = debye_cv_scan(al["T_D"], 26.98, al["T_scan"])
        cv_vasp_al  = (np.interp(al["T_scan"], data["al_vasp"]["T_scan"],
                                  data["al_vasp"]["cv"])
                       if "al_vasp" in data else None)
        for i, T in enumerate(al["T_scan"]):
            cv_e = np.interp(T, EXP_AL_T, EXP_AL_CV) if (
                EXP_AL_T.min() <= T <= EXP_AL_T.max()) else float("nan")
            cv_v = f"{cv_vasp_al[i]:12.4f}" if cv_vasp_al is not None else f"{'—':>12}"
            cv_e_s = f"{cv_e:12.4f}" if not np.isnan(cv_e) else f"{'—':>12}"
            print(f"  {T:8.1f}  {al['cv'][i]:12.4f}  {cv_v}  "
                  f"{cv_debye_al[i]:12.4f}  {cv_e_s}")

    if "pb" in data:
        pb = data["pb"]
        print(f"\n── Pb (fcc, γ ≈ 2.73)  [T_melt = 600.6 K] ──")
        print(f"  T_D^harm  = {pb['T_D_harm']:.1f} K  (100 K run, ≈harmonic IFCs)")
        if pb["T_D_anh"] is not None:
            print(f"  T_D^anh   = {pb['T_D_anh']:.1f} K  (500 K run, renormalized IFCs)")
            print(f"  Δ T_D     = {pb['T_D_harm'] - pb['T_D_anh']:+.1f} K  "
                  f"(negative → phonon softening at high T)")
        print()
        print(f"  {'T [K]':>8}  {'C_V^harm':>12}  {'C_V^anh':>12}  "
              f"{'ΔC_V^anh':>12}  {'ΔC_V/3R':>10}  {'C_V exp':>12}")
        print("  " + "-" * 80)
        for i, T in enumerate(pb["T_scan"]):
            cv_h = pb["cv_harm"][i]
            cv_a_s = f"{pb['cv_anh'][i]:12.4f}" if pb["cv_anh"] is not None else f"{'—':>12}"
            dcv_s  = f"  {pb['delta_cv'][i]:+11.4f}  {pb['delta_cv'][i]/THREE_R:+9.4f}" \
                     if pb["delta_cv"] is not None else f"  {'—':>11}  {'—':>9}"
            cv_e = np.interp(T, EXP_PB_T, EXP_PB_CV) if (
                EXP_PB_T.min() <= T <= EXP_PB_T.max()) else float("nan")
            cv_e_s = f"{cv_e:12.4f}" if not np.isnan(cv_e) else f"{'—':>12}"
            print(f"  {T:8.1f}  {cv_h:12.4f}  {cv_a_s}  {dcv_s}  {cv_e_s}")

    print()
    print("  3R (Dulong-Petit) = {:.4f} J/(mol·K)".format(THREE_R))


# ── Plotting ──────────────────────────────────────────────────────────────────

def plot_comparison(data: dict) -> None:
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    import matplotlib.gridspec as gridspec

    fig = plt.figure(figsize=(14, 11))
    gs  = gridspec.GridSpec(2, 2, figure=fig, hspace=0.38, wspace=0.32)
    ax1 = fig.add_subplot(gs[0, 0])  # Al C_V(T)
    ax2 = fig.add_subplot(gs[0, 1])  # Pb C_V(T) harm vs anh
    ax3 = fig.add_subplot(gs[1, 0])  # Pb ΔC_V^anh(T)
    ax4 = fig.add_subplot(gs[1, 1])  # T_D vs T for both

    dulong_petit = THREE_R

    # ── Panel 1: Al ──────────────────────────────────────────────────────────
    ax = ax1
    ax.axhline(dulong_petit, color="gray", lw=0.8, ls="--", label="3R (Dulong–Petit)")

    if "al_gpaw" in data:
        al = data["al_gpaw"]
        ax.plot(al["T_scan"], al["cv"], "b-o", ms=5, lw=1.8,
                label=f"SQTC+GPAW  ($T_D$={al['T_D']:.0f} K)")
        T_plot = np.linspace(al["T_scan"].min(), al["T_scan"].max(), 200)
        ax.plot(T_plot, debye_cv_scan(al["T_D"], 26.98, T_plot),
                "b--", lw=1.2, alpha=0.55, label="Debye (GPAW $T_D$)")

    if "al_vasp" in data:
        vasp = data["al_vasp"]
        ax.plot(vasp["T_scan"], vasp["cv"], "g-s", ms=4, lw=1.4,
                label=f"SQTC+VASP  ($T_D$={vasp['T_D']:.0f} K)")

    # Experimental Al
    ax.scatter(EXP_AL_T, EXP_AL_CV, marker="^", c="k", s=50, zorder=5,
               label="Experiment")
    # Debye reference at exp T_D
    T_deb = np.linspace(10, 950, 300)
    ax.plot(T_deb, debye_cv_scan(390.0, 26.98, T_deb),
            "k:", lw=1.0, alpha=0.45, label="Debye ($T_D$=390 K)")

    ax.set_xlabel("$T$ [K]")
    ax.set_ylabel("$C_V$ [J mol$^{-1}$ K$^{-1}$]")
    ax.set_title("Al (fcc)  —  harmonic regime\n$\\gamma_{Gr}\\approx2.17$,  "
                 "$T_D=390$ K")
    ax.legend(fontsize=7.5)
    ax.set_xlim(0, 950)
    ax.set_ylim(0, dulong_petit * 1.30)
    ax.grid(True, alpha=0.25)

    # ── Panel 2: Pb harm vs anh ───────────────────────────────────────────────
    ax = ax2
    ax.axhline(dulong_petit, color="gray", lw=0.8, ls="--", label="3R")
    ax.axvline(600.6, color="tomato", lw=0.8, ls=":", alpha=0.7, label="$T_{melt}$")

    if "pb" in data:
        pb = data["pb"]
        ax.plot(pb["T_scan"], pb["cv_harm"], "b-o", ms=5, lw=1.8,
                label=f"SQTC+GPAW harmonic  ($T_D$={pb['T_D_harm']:.0f} K)")
        T_deb_pb = np.linspace(pb["T_scan"].min(), pb["T_scan"].max(), 200)
        ax.plot(T_deb_pb, debye_cv_scan(pb["T_D_harm"], 207.2, T_deb_pb),
                "b--", lw=1.2, alpha=0.55, label="Debye (harm $T_D$)")

        if pb["cv_anh"] is not None:
            ax.plot(pb["T_scan"], pb["cv_anh"], "r-o", ms=5, lw=1.8,
                    label=f"SQTC+GPAW anharmonic  ($T_D$={pb['T_D_anh']:.0f} K)")
            ax.fill_between(pb["T_scan"], pb["cv_harm"], pb["cv_anh"],
                            alpha=0.12, color="red", label="$\\Delta C_V^{anh}$")

    ax.scatter(EXP_PB_T, EXP_PB_CV, marker="^", c="k", s=50, zorder=5,
               label="Experiment (Hultgren)")
    ax.set_xlabel("$T$ [K]")
    ax.set_ylabel("$C_V$ [J mol$^{-1}$ K$^{-1}$]")
    ax.set_title("Pb (fcc)  —  strongly anharmonic\n$\\gamma_{Gr}\\approx2.73$,  "
                 "$T_D=88$ K,  $T_{melt}=600$ K")
    ax.legend(fontsize=7.5)
    ax.set_xlim(0, 650)
    ax.set_ylim(0, dulong_petit * 1.40)
    ax.grid(True, alpha=0.25)

    # ── Panel 3: ΔC_V^anh(T) for Pb ──────────────────────────────────────────
    ax = ax3
    ax.axhline(0, color="gray", lw=0.8, ls="-")
    ax.axvline(600.6, color="tomato", lw=0.8, ls=":", alpha=0.7, label="$T_{melt}$")

    if "pb" in data and data["pb"]["delta_cv"] is not None:
        pb = data["pb"]
        ax.bar(pb["T_scan"], pb["delta_cv"],
               width=(pb["T_scan"][1] - pb["T_scan"][0]) * 0.75,
               color="salmon", edgecolor="firebrick", alpha=0.8,
               label="$\\Delta C_V^{anh} = C_V^{anh} - C_V^{harm}$")
        ax2_r = ax.twinx()
        ax2_r.plot(pb["T_scan"], pb["delta_cv"] / THREE_R * 100,
                   "r--", lw=1.2, alpha=0.8, label="$\\Delta C_V / 3R$ [%]")
        ax2_r.set_ylabel("$\\Delta C_V / 3R$ [%]", color="firebrick", fontsize=9)
        ax2_r.tick_params(axis="y", colors="firebrick")
        ax2_r.axhline(0, color="firebrick", lw=0.5, ls="--", alpha=0.3)

    if "al_gpaw" in data and "al_vasp" in data:
        # Show Al harmonic correction as near-zero reference
        al = data["al_gpaw"]
        vasp = data["al_vasp"]
        cv_al_interp = np.interp(al["T_scan"], vasp["T_scan"], vasp["cv"])
        delta_al = al["cv"] - cv_al_interp
        ax.plot(al["T_scan"], delta_al, "b-o", ms=4, lw=1.2, alpha=0.7,
                label="Al: GPAW − VASP (≈ 0 expected)")

    ax.set_xlabel("$T$ [K]")
    ax.set_ylabel("$\\Delta C_V^{anh}$ [J mol$^{-1}$ K$^{-1}$]")
    ax.set_title("Anharmonic correction $\\Delta C_V^{anh}$ for Pb\n"
                 "(= 0 for harmonic; grows near melting)")
    ax.legend(fontsize=8, loc="upper left")
    ax.set_xlim(0, 650)
    ax.grid(True, alpha=0.25)

    # ── Panel 4: T_D vs T ─────────────────────────────────────────────────────
    ax = ax4
    # Al: show a flat line (harmonic → T_D doesn't depend on T)
    if "al_gpaw" in data:
        al = data["al_gpaw"]
        # Single T_D from one run; plot as horizontal band
        ax.axhline(al["T_D"], color="blue", lw=1.8, ls="-",
                   label=f"Al SQTC+GPAW ($T_D$={al['T_D']:.0f} K)")
        ax.fill_between([0, 950], al["T_D"] * 0.97, al["T_D"] * 1.03,
                        alpha=0.12, color="blue", label="Al ±3% band")

    if "pb" in data:
        pb = data["pb"]
        T_pb = [100.0, 500.0]
        T_D_pb = [pb["T_D_harm"]]
        labels_pb = [100.0]
        if pb["T_D_anh"] is not None:
            T_pb_pts = np.array([100.0, 500.0])
            T_D_pts  = np.array([pb["T_D_harm"], pb["T_D_anh"]])
            ax.plot(T_pb_pts, T_D_pts, "r-o", ms=8, lw=2.0,
                    label="Pb SQTC+GPAW: $T_D$ vs $T$")
            ax.annotate(f"Harm ({pb['T_D_harm']:.0f} K)",
                        (100.0, pb["T_D_harm"]), textcoords="offset points",
                        xytext=(8, 5), fontsize=8, color="firebrick")
            ax.annotate(f"Anh ({pb['T_D_anh']:.0f} K)",
                        (500.0, pb["T_D_anh"]), textcoords="offset points",
                        xytext=(8, -12), fontsize=8, color="firebrick")
            delta_td = pb["T_D_harm"] - pb["T_D_anh"]
            ax.annotate("", xy=(500.0, pb["T_D_anh"]),
                        xytext=(500.0, pb["T_D_harm"]),
                        arrowprops=dict(arrowstyle="<->", color="red", lw=1.5))
            mid_y = 0.5 * (pb["T_D_harm"] + pb["T_D_anh"])
            ax.text(510, mid_y, f"ΔT_D = {delta_td:+.1f} K", fontsize=8.5,
                    color="firebrick", va="center")
        else:
            ax.scatter([100.0], [pb["T_D_harm"]], s=80, color="red", zorder=5,
                       label=f"Pb harm ($T_D$={pb['T_D_harm']:.0f} K)")

    ax.axhline(390.0, color="steelblue", lw=0.8, ls="--", alpha=0.5,
               label="Al exp $T_D$ = 390 K")
    ax.axhline(88.0, color="tomato", lw=0.8, ls="--", alpha=0.5,
               label="Pb exp $T_D$ = 88 K")

    ax.set_xlabel("$T$ [K]")
    ax.set_ylabel("$T_D^{SQTC}$ [K]")
    ax.set_title("Debye temperature vs run temperature\n"
                 "(flat = harmonic; dropping = phonon softening)")
    ax.legend(fontsize=8)
    ax.set_xlim(0, 650)
    ax.set_ylim(0, max(420.0, ax.get_ylim()[1]))
    ax.grid(True, alpha=0.25)

    fig.suptitle("SQTC+GPAW: Al (harmonic) vs Pb (anharmonic)\n"
                 "Harmonic + anharmonic decomposition of $C_V(T)$",
                 fontsize=13, fontweight="bold")

    out = Path("sqtc_al_pb_gpaw_comparison.pdf")
    fig.savefig(out, bbox_inches="tight", dpi=150)
    print(f"\n  Figure saved → {out}")
    out_png = out.with_suffix(".png")
    fig.savefig(out_png, bbox_inches="tight", dpi=150)
    print(f"  Figure saved → {out_png}")


# ── Main ──────────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--no-plot", action="store_true",
                        help="Print table only; skip matplotlib.")
    args = parser.parse_args()

    data = load_all()

    if not data:
        print("No result files found.  Run the production scripts first:\n"
              "  python codes/examples/run_sqtc_al_gpaw.py\n"
              "  python codes/examples/run_sqtc_pb_gpaw.py")
        sys.exit(1)

    print_table(data)

    if not args.no_plot:
        try:
            plot_comparison(data)
        except ImportError:
            print("\n  matplotlib not available — skipping plot "
                  "(use --no-plot to suppress this message).")


if __name__ == "__main__":
    main()
