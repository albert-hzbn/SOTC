#!/usr/bin/env python3
"""Figures of the design comparison and the trial update (JAP manuscript).

* ``fig_selfconsistent.pdf``: mesh RMSE against the iteration-1 effective
  FC2 versus force calls for all eleven crystals: matched trial (SOTC-D and
  canonical sampling from the iteration-1 trial, against the disjoint
  effective FC2) and equal cost (bootstrapped SOTC-D and one-shot canonical
  pairs from the harmonic trial, and two-iteration sTDEP, against the full
  effective FC2).
* ``fig_dispersions.pdf``: dispersions of the converged large-cell harmonic
  FD reference, the 40-snapshot iteration-1 effective FC2, and SOTC-D with
  eight force calls (iteration-1 trial and bootstrapped).

Reads ``robustness/design_comparison.json`` (``analyze_designs.py``) and the
primary-cell force archives.
"""

from __future__ import annotations

import json
from pathlib import Path
import sys

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "codes"))
sys.path.insert(0, str(Path(__file__).resolve().parent))

from plot_robustness import AQUA, BLUE, GRAY, INK, MUTED, ORANGE, YELLOW, style  # noqa: E402

OUT = ROOT / "manuscript" / "jap" / "figures"
MATERIALS = ["al", "ag", "au", "cu", "mo", "w", "nacl", "mgo", "pbte", "si", "mg"]
LABELS = {"al": "Al", "ag": "Ag", "au": "Au", "cu": "Cu", "mo": "Mo", "w": "W",
          "nacl": "NaCl", "mgo": "MgO", "pbte": "PbTe", "si": "Si", "mg": "Mg"}
DISPERSION_MATERIALS = ["al", "w", "mgo", "pbte", "si", "mg"]
PATHS = {  # reduced coordinates of the primitive cell
    "fcc": [("$\\Gamma$", (0, 0, 0)), ("X", (0.5, 0, 0.5)), ("W", (0.5, 0.25, 0.75)),
            ("K", (0.375, 0.375, 0.75)), ("$\\Gamma$", (0, 0, 0)), ("L", (0.5, 0.5, 0.5))],
    "bcc": [("$\\Gamma$", (0, 0, 0)), ("H", (0.5, -0.5, 0.5)), ("N", (0, 0, 0.5)),
            ("$\\Gamma$", (0, 0, 0)), ("P", (0.25, 0.25, 0.25)), ("H", (0.5, -0.5, 0.5))],
    "hcp": [("$\\Gamma$", (0, 0, 0)), ("M", (0.5, 0, 0)), ("K", (1 / 3, 1 / 3, 0)),
            ("$\\Gamma$", (0, 0, 0)), ("A", (0, 0, 0.5))],
}
COUNTS = (2, 4, 8, 16)


def report(material):
    path = ROOT / "validation_results" / material / "robustness" / "design_comparison.json"
    return json.loads(path.read_text()) if path.exists() else None


def selfconsistent_figure():
    mats = [m for m in MATERIALS if report(m) and "iter1" in report(m)["targets"]]
    ncol = 6
    nrow = int(np.ceil(len(mats) / ncol))
    fig, axes = plt.subplots(nrow, ncol, figsize=(7.0, 1.55 * nrow + 0.75), sharex=True)
    axes = axes.ravel()
    x = np.asarray(COUNTS)
    for ax, m in zip(axes, mats):
        e = report(m)["targets"]["iter1"]["supercell"]
        for kind, color, marker in (("canonical_unpaired", GRAY, "s"),
                                    ("canonical_paired", AQUA, "^")):
            vals = [[t["mesh_vs_disjoint"] for t in e[kind][str(c)]] for c in COUNTS]
            ax.fill_between(x, [np.percentile(v, 10) for v in vals],
                            [np.percentile(v, 90) for v in vals], color=color,
                            alpha=0.15, linewidth=0)
            ax.plot(x, [np.median(v) for v in vals], color=color, marker=marker,
                    markersize=3, linewidth=1.2)
        other = [np.median([t["mesh_vs_full_reference"]
                            for t in e["other_iteration_canonical"][str(c)]]) for c in COUNTS]
        ax.plot(x, other, color=GRAY, linestyle="--", marker="v", markersize=3,
                linewidth=1.0)
        ax.plot(x, [e["designs"]["boot"][str(c)]["mesh_vs_full_reference"] for c in COUNTS],
                color=ORANGE, linestyle="--", marker="D", markersize=3, linewidth=1.2)
        if "stdep" in e:
            budgets = sorted(int(c) for c in e["stdep"]["stdep_pool"])
            ax.plot(budgets, [np.median([t["mesh_vs_full_reference"]
                                         for t in e["stdep"]["stdep_pool"][str(c)]])
                              for c in budgets],
                    color=YELLOW, linestyle="--", marker="P", markersize=4,
                    markeredgecolor=INK, markeredgewidth=0.4, linewidth=1.2)
        ax.plot(x, [e["designs"]["sotcd1"][str(c)]["mesh_vs_disjoint"] for c in COUNTS],
                color=BLUE, marker="o", markersize=3.5, linewidth=1.6)
        ax.set_xscale("log", base=2)
        ax.set_yscale("log")
        ax.set_xticks(COUNTS)
        ax.set_xticklabels([str(c) for c in COUNTS])
        ax.minorticks_off()
        ax.set_title(LABELS[m], color=INK, pad=2)
        style(ax)
    for ax in axes[len(mats):]:
        ax.axis("off")
    handles = [
        plt.Line2D([], [], color=BLUE, marker="o", markersize=3.5, label="SOTC-D (iter. 1)"),
        plt.Line2D([], [], color=AQUA, marker="^", markersize=3, label=r"canonical $\pm$ (iter. 1)"),
        plt.Line2D([], [], color=GRAY, marker="s", markersize=3, label="canonical (iter. 1)"),
        plt.Line2D([], [], color=ORANGE, linestyle="--", marker="D", markersize=3,
                   label="bootstrapped SOTC-D"),
        plt.Line2D([], [], color=GRAY, linestyle="--", marker="v", markersize=3,
                   label=r"canonical $\pm$ (harmonic)"),
        plt.Line2D([], [], color=YELLOW, linestyle="--", marker="P", markersize=4,
                   markeredgecolor=INK, markeredgewidth=0.4, label="sTDEP, pooled"),
    ]
    fig.supxlabel("force calls", fontsize=8, color=INK, y=0.09)
    fig.supylabel("mesh RMSE vs iteration-1 effective FC2 (THz)", fontsize=8, color=INK)
    fig.legend(handles=handles, loc="lower center", ncol=3, frameon=False,
               fontsize=6.5, handlelength=2.0, columnspacing=1.2)
    fig.tight_layout(pad=0.4, w_pad=0.3, h_pad=0.5, rect=(0.0, 0.08, 1.0, 1.0))
    fig.savefig(OUT / "fig_selfconsistent.pdf")
    fig.savefig(OUT / "fig_selfconsistent.png", dpi=200)
    plt.close(fig)


def band_path(bravais, n_per=40):
    nodes = PATHS["fcc" if bravais == "diamond" else bravais]
    points, x, ticks = [], [], [0.0]
    total = 0.0
    for (_, a), (_, b) in zip(nodes[:-1], nodes[1:]):
        a, b = np.asarray(a, float), np.asarray(b, float)
        seg = a + np.linspace(0.0, 1.0, n_per)[:, None] * (b - a)
        points.append(seg)
        x.append(total + np.linspace(0.0, 1.0, n_per) * np.linalg.norm(b - a))
        total += np.linalg.norm(b - a)
        ticks.append(total)
    return np.concatenate(points), np.concatenate(x), ticks, [n for n, _ in nodes]


def dispersion_figure():
    import phonopy

    from analyze_designs import N_TRAIN, Fitter, datasets
    from analyze_primary_archives import PRIMARY, phonopy_frequencies, symfc_fc2_basis

    mats = [m for m in DISPERSION_MATERIALS if report(m)]
    fig, axes = plt.subplots(2, 3, figsize=(7.0, 4.4))
    for ax, m in zip(axes.ravel(), mats):
        spec = PRIMARY[m]
        vr = ROOT / "validation_results" / m
        large = phonopy.load(vr / "fd_references" / "production_large" / "phonopy_harmonic.yaml")
        phonon = phonopy.load(vr / "primary_cell" / "fd" / "phonopy_harmonic.yaml")
        q, x, ticks, names = band_path(spec["bravais"])
        # Path coordinates refer to the conventional primitive cell of
        # ``primitive_structure``; rotate onto each model's primitive cell.
        def freqs(ph):
            prim = np.asarray(ph.primitive.cell)
            ref = np.asarray(large.primitive.cell)
            return phonopy_frequencies(ph, q @ ref @ np.linalg.inv(prim))
        sup = phonon.supercell
        fit = Fitter(symfc_fc2_basis(np.asarray(sup.positions), np.asarray(sup.cell),
                                     list(sup.symbols), spec["cutoff"]))
        data = datasets(m)
        lines = [(freqs(large), GRAY, "-", 1.6, "harmonic FD, large cell")]
        for name, n, color, ls, lw, label in (
                ("canonical1", N_TRAIN, INK, "-", 0.9, "effective FC2, 40 snapshots"),
                ("sotcd1", 8, BLUE, "--", 1.1, "SOTC-D, 8 calls"),
                ("boot", 8, ORANGE, ":", 1.3, "bootstrapped SOTC-D, 8 calls")):
            phonon.force_constants = fit(*[v[:n] for v in data[name]])
            lines.append((freqs(phonon), color, ls, lw, label))
        for f, color, ls, lw, label in lines:
            for k in range(f.shape[1]):
                ax.plot(x, f[:, k], color=color, linestyle=ls, linewidth=lw,
                        label=label if k == 0 else None)
        for t in ticks[1:-1]:
            ax.axvline(t, color="#d6d5d0", linewidth=0.5)
        ax.set_xticks(ticks)
        ax.set_xticklabels(names)
        ax.set_xlim(0, ticks[-1])
        ax.set_ylim(bottom=0)
        ax.set_title(f"{LABELS[m]}, {spec['T']:.0f} K", color=INK, pad=2)
        ax.spines[["top", "right"]].set_visible(False)
    axes[0, 0].set_ylabel("frequency (THz)")
    axes[1, 0].set_ylabel("frequency (THz)")
    handles, labels = axes[0, 0].get_legend_handles_labels()
    fig.legend(handles, labels, loc="lower center", ncol=4, frameon=False, fontsize=7)
    fig.tight_layout(pad=0.4, rect=(0, 0.05, 1, 1))
    fig.savefig(OUT / "fig_dispersions.pdf")
    fig.savefig(OUT / "fig_dispersions.png", dpi=200)
    plt.close(fig)


def main():
    OUT.mkdir(parents=True, exist_ok=True)
    selfconsistent_figure()
    dispersion_figure()


if __name__ == "__main__":
    main()
