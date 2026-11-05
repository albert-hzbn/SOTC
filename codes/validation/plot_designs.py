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

import matplotlib.pyplot as plt
import matplotlib.ticker
import numpy as np

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "codes"))
sys.path.insert(0, str(Path(__file__).resolve().parent))

from plot_style import AQUA, BLUE, GRAY, INK, ORANGE, PAGE, RULE, YELLOW, plain_log, save  # noqa: E402

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
    fig, axes = plt.subplots(nrow, ncol, figsize=(PAGE, 1.5 * nrow + 0.75), sharex=True)
    axes = axes.ravel()
    x = np.asarray(COUNTS)
    for k, (ax, m) in enumerate(zip(axes, mats)):
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
        ax.xaxis.set_minor_locator(matplotlib.ticker.NullLocator())
        plain_log(ax)
        ax.margins(y=0.2)
        ax.tick_params(axis="y", labelsize=7, pad=1.5)
        ax.text(0.95, 0.94, f"({chr(97 + k)}) {LABELS[m]}", transform=ax.transAxes,
                ha="right", va="top", fontsize=7.5)
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
    fig.supxlabel("force calls", fontsize=8.5, y=0.115)
    fig.supylabel("mesh RMSE vs iteration-1 effective FC2 (THz)", fontsize=8.5, x=0.006)
    fig.legend(handles=handles, loc="lower center", ncol=3, handlelength=2.4)
    fig.tight_layout(pad=0.3, w_pad=0.15, h_pad=0.3, rect=(0.022, 0.135, 1.0, 1.0))
    save(fig, OUT, "fig_selfconsistent")


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
    fig, axes = plt.subplots(2, 3, figsize=(PAGE, 4.2))
    for k_panel, (ax, m) in enumerate(zip(axes.ravel(), mats)):
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
        lines = [(freqs(large), "#b5b4af", "-", 1.5, "harmonic FD, large cell")]
        for name, n, color, ls, lw, label in (
                ("canonical1", N_TRAIN, INK, "-", 0.7, "effective FC2, 40 snapshots"),
                ("sotcd1", 8, BLUE, "--", 0.9, "SOTC-D, 8 calls"),
                ("boot", 8, ORANGE, ":", 1.1, "bootstrapped SOTC-D, 8 calls")):
            phonon.force_constants = fit(*[v[:n] for v in data[name]])
            lines.append((freqs(phonon), color, ls, lw, label))
        for f, color, ls, lw, label in lines:
            for k in range(f.shape[1]):
                ax.plot(x, f[:, k], color=color, linestyle=ls, linewidth=lw,
                        label=label if k == 0 else None)
        for t in ticks[1:-1]:
            ax.axvline(t, color=RULE, linewidth=0.4, zorder=0)
        ax.set_xticks(ticks)
        ax.set_xticklabels(names)
        ax.xaxis.set_tick_params(top=False)
        ax.set_xlim(0, ticks[-1])
        ax.set_ylim(bottom=0)
        ax.yaxis.set_major_locator(matplotlib.ticker.MaxNLocator(nbins=5, integer=True))
        ax.set_title(f"({chr(97 + k_panel)}) {LABELS[m]}, {spec['T']:.0f} K", loc="left")
    axes[0, 0].set_ylabel("frequency (THz)")
    axes[1, 0].set_ylabel("frequency (THz)")
    handles, labels = axes[0, 0].get_legend_handles_labels()
    fig.legend(handles, labels, loc="lower center", ncol=4, handlelength=2.6)
    fig.tight_layout(pad=0.3, rect=(0, 0.05, 1, 1))
    save(fig, OUT, "fig_dispersions")


def convergence_dispersion_figure(materials=("al", "au"), counts=COUNTS):
    """Branch-resolved convergence of SOTC-D (updated trial) with the number
    of force calls, with the neutron-derived Born--von Karman model of Al."""
    import phonopy

    from al_neutron_bvk import bvk_frequencies
    from analyze_designs import N_TRAIN, Fitter, datasets
    from analyze_primary_archives import PRIMARY, phonopy_frequencies, symfc_fc2_basis

    fig, axes = plt.subplots(len(materials), len(counts), figsize=(PAGE, 2.05 * len(materials) + 0.25),
                             sharey="row")
    for row, m in zip(axes, materials):
        spec = PRIMARY[m]
        vr = ROOT / "validation_results" / m
        large = phonopy.load(vr / "fd_references" / "production_large" / "phonopy_harmonic.yaml")
        phonon = phonopy.load(vr / "primary_cell" / "fd" / "phonopy_harmonic.yaml")
        q, x, ticks, names = band_path(spec["bravais"])
        ref = np.asarray(large.primitive.cell)

        def freqs(ph):
            return phonopy_frequencies(ph, q @ ref @ np.linalg.inv(np.asarray(ph.primitive.cell)))

        sup = phonon.supercell
        fit = Fitter(symfc_fc2_basis(np.asarray(sup.positions), np.asarray(sup.cell),
                                     list(sup.symbols), spec["cutoff"]))
        data = datasets(m)
        phonon.force_constants = fit(*[v[:N_TRAIN] for v in data["canonical1"]])
        fixed = [(freqs(large), "#b5b4af", "-", 1.5, "harmonic FD, large cell"),
                 (freqs(phonon), INK, "-", 0.7, "effective FC2, 40 snapshots")]
        if m == "al":
            q_cart = q @ np.linalg.inv(ref).T
            fixed += [(bvk_frequencies(q_cart, 10), YELLOW, "-", 0.8, "neutron BvK model, 10 K"),
                      (bvk_frequencies(q_cart, 300), AQUA, "-", 0.8, "neutron BvK model, 300 K")]
        for ax, n in zip(row, counts):
            phonon.force_constants = fit(*[v[:n] for v in data["sotcd1"]])
            lines = fixed + [(freqs(phonon), BLUE, "--", 0.9, "SOTC-D, updated trial")]
            for f, color, ls, lw, label in lines:
                for k in range(f.shape[1]):
                    ax.plot(x, f[:, k], color=color, linestyle=ls, linewidth=lw,
                            label=label if k == 0 else None)
            for t in ticks[1:-1]:
                ax.axvline(t, color=RULE, linewidth=0.4, zorder=0)
            ax.set_xticks(ticks)
            # W and K are too close to label separately in these narrow panels
            ax.set_xticklabels([("W,K" if name == "W" else "" if name == "K" else name)
                                for name in names])
            ax.xaxis.set_tick_params(top=False)
            ax.set_xlim(0, ticks[-1])
            ax.set_ylim(bottom=0)
            ax.set_title(f"{LABELS[m]}, {spec['T']:.0f} K, {n} calls")
        row[0].set_ylabel("frequency (THz)")
    handles, labels = axes[0, 0].get_legend_handles_labels()
    fig.legend(handles, labels, loc="lower center", ncol=3, handlelength=2.6)
    fig.tight_layout(pad=0.3, w_pad=0.3, rect=(0, 0.09, 1, 1))
    save(fig, OUT, "fig_dispersion_convergence")


def main():
    OUT.mkdir(parents=True, exist_ok=True)
    selfconsistent_figure()
    dispersion_figure()
    convergence_dispersion_figure()


if __name__ == "__main__":
    main()
