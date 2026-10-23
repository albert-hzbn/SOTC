#!/usr/bin/env python3
"""Figures for the matched robustness analysis (JAP manuscript).

Reads ``validation_results/<material>/robustness/robustness_summary.json``
and writes

* ``fig_reference_accuracy.pdf``: phonon-mesh RMSE against the converged
  harmonic FD reference and against the same-cell effective reference;
* ``fig_matched_forces.pdf``: held-out canonical force RMSE of SOTC versus
  random canonical subsets of identical size in the identical cell.
"""

from __future__ import annotations

import json
from pathlib import Path

import matplotlib
import matplotlib.ticker

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402

ROOT = Path(__file__).resolve().parents[2]
OUT = ROOT / "manuscript" / "jap" / "figures"
MATERIALS = ["al", "ag", "au", "cu", "mo", "w", "nacl", "mgo", "pbte"]
LABELS = {"al": "Al", "ag": "Ag", "au": "Au", "cu": "Cu", "mo": "Mo",
          "w": "W", "nacl": "NaCl", "mgo": "MgO", "pbte": "PbTe"}
# Validated categorical slots (blue, orange, aqua, yellow) plus neutral gray.
BLUE, ORANGE, AQUA, YELLOW = "#2a78d6", "#eb6834", "#1baf7a", "#eda100"
GRAY, INK, MUTED = "#8a8985", "#0b0b0b", "#52514e"

plt.rcParams.update({
    "font.size": 8, "axes.labelsize": 8, "axes.titlesize": 8,
    "xtick.labelsize": 7.5, "ytick.labelsize": 7.5, "legend.fontsize": 7,
    "axes.edgecolor": MUTED, "axes.labelcolor": INK, "xtick.color": MUTED,
    "ytick.color": MUTED, "axes.linewidth": 0.6, "pdf.fonttype": 42,
})


def load(material):
    path = ROOT / "validation_results" / material / "robustness" / "robustness_summary.json"
    return json.loads(path.read_text()) if path.exists() else None


def model(report, name):
    for row in report["models"]:
        if row["model"] == name:
            return row
    return None


def metric(row, key, sub="rmse_thz"):
    if row is None or key not in row:
        return np.nan
    return row[key][sub]


def canonical_stats(report, label, key, sub="rmse_thz"):
    block = report.get("matched_canonical", {}).get(label)
    if not block:
        return np.nan, np.nan, np.nan
    if key == "held_out_force_rmse_ev_ang":
        values = np.asarray([row[key] for row in block["canonical"]])
    else:
        values = np.asarray([row[key][sub] for row in block["canonical"]])
    return np.median(values), np.percentile(values, 10), np.percentile(values, 90)


def style(ax):
    ax.spines[["top", "right"]].set_visible(False)
    ax.yaxis.grid(True, color="#e4e3df", linewidth=0.5)
    ax.set_axisbelow(True)


def accuracy_figure(reports):
    mats = [m for m in MATERIALS if reports.get(m)]
    x = np.arange(len(mats))
    fig, axes = plt.subplots(2, 1, figsize=(7.0, 4.6), sharex=True)
    width = 0.2

    ax = axes[0]
    series = [("fd_3x3x3", "FD 3$\\times$3$\\times$3 (harmonic)", GRAY, "//"),
              ("fd_same_cell", "FD, SOTC cell (harmonic)", AQUA, ""),
              ("sotc_selected", "SOTC-C (selected)", BLUE, ""),
              ("sotc_pooled", "SOTC-C (all iterations)", ORANGE, "")]
    for k, (name, label, color, hatch) in enumerate(series):
        values = [metric(model(reports[m], name), "mesh_vs_fd_large") for m in mats]
        ax.bar(x + (k - 1.5) * width, values, width * 0.9, color=color,
               hatch=hatch, edgecolor="white", linewidth=0.4, label=label)
    ax.set_ylabel("RMSE vs converged FD (THz)")
    ax.set_title("(a) Harmonic reference: 5$^3$/4$^3$-primitive-cell finite displacement",
                 loc="left")
    style(ax)
    ax.legend(ncol=2, frameon=False, loc="upper left")

    ax = axes[1]
    for k, (label, name, color) in enumerate((
            ("sotc_selected", "SOTC-C (selected)", BLUE),
            ("sotc_pooled", "SOTC-C (all iterations)", ORANGE))):
        sotc = [metric(model(reports[m], label), "mesh_vs_effective") for m in mats]
        stats = np.asarray([canonical_stats(reports[m], label, "mesh_vs_effective")
                            for m in mats])
        offset = (2 * k - 1.5) * width
        ax.bar(x + offset, sotc, width * 0.9, color=color, edgecolor="white",
               linewidth=0.4, label=name)
        ax.bar(x + offset + width, stats[:, 0], width * 0.9, color="white",
               edgecolor=color, hatch="////", linewidth=0.8,
               label=f"Canonical, same count as {name.split(' (')[1][:-1]}")
        ax.errorbar(x + offset + width, stats[:, 0],
                    yerr=[stats[:, 0] - stats[:, 1], stats[:, 2] - stats[:, 0]],
                    fmt="none", ecolor=INK, elinewidth=0.6, capsize=1.5)
    ax.set_ylabel("RMSE vs effective FC2 (THz)")
    ax.set_title("(b) Effective reference: 40 canonical snapshots at $T_{\\rm des}$, "
                 "same cell", loc="left")
    style(ax)
    ax.legend(ncol=2, frameon=False, loc="upper left")
    ax.set_xticks(x, [LABELS[m] for m in mats])
    fig.tight_layout()
    fig.savefig(OUT / "fig_reference_accuracy.pdf")
    fig.savefig(OUT / "fig_reference_accuracy.png", dpi=200)
    plt.close(fig)


def forces_figure(reports):
    mats = [m for m in MATERIALS if reports.get(m) and reports[m]["canonical_available"]]
    x = np.arange(len(mats))
    fig, ax = plt.subplots(figsize=(7.0, 2.6))
    width = 0.2
    eff = np.asarray([model(reports[m], "effective_reference_symfc_40")
                      ["held_out_force_rmse_ev_ang"] for m in mats])
    for k, (label, name, color) in enumerate((
            ("sotc_selected", "SOTC-C (selected)", BLUE),
            ("sotc_pooled", "SOTC-C (all iterations)", ORANGE))):
        sotc = np.asarray([model(reports[m], label)["held_out_force_rmse_ev_ang"]
                           for m in mats])
        stats = np.asarray([canonical_stats(reports[m], label,
                                            "held_out_force_rmse_ev_ang") for m in mats])
        offset = (2 * k - 1.5) * width
        ax.bar(x + offset, sotc / eff, width * 0.9, color=color, edgecolor="white",
               linewidth=0.4, label=name)
        ax.bar(x + offset + width, stats[:, 0] / eff, width * 0.9, color="white",
               edgecolor=color, hatch="////", linewidth=0.8,
               label=f"Canonical, same count")
        ax.errorbar(x + offset + width, stats[:, 0] / eff,
                    yerr=[(stats[:, 0] - stats[:, 1]) / eff,
                          (stats[:, 2] - stats[:, 0]) / eff],
                    fmt="none", ecolor=INK, elinewidth=0.6, capsize=1.5)
    ax.axhline(1.0, color=GRAY, linewidth=0.8, linestyle="--")
    ax.text(len(mats) - 0.5, 1.0, "40-snapshot effective FC2", color=MUTED,
            ha="right", va="bottom", fontsize=7)
    ax.set_ylabel("Held-out force RMSE\n(relative to effective FC2)")
    ax.set_xticks(x, [LABELS[m] for m in mats])
    style(ax)
    ax.legend(ncol=4, frameon=False, loc="upper left", bbox_to_anchor=(0, 1.18))
    fig.tight_layout()
    fig.savefig(OUT / "fig_matched_forces.pdf")
    fig.savefig(OUT / "fig_matched_forces.png", dpi=200)
    plt.close(fig)


def count_figure(reports, fitter="symfc"):
    """Error versus force calls: SOTC-D against canonical baselines."""
    mats = [m for m in MATERIALS if reports.get(m) and "count_series" in reports[m]
            and reports[m]["count_series"][fitter].get("sotcd")]
    ncol = 3
    nrow = int(np.ceil(len(mats) / ncol))
    fig, axes = plt.subplots(nrow, ncol, figsize=(7.0, 1.9 * nrow + 0.5),
                             squeeze=False)
    for ax, m in zip(axes.ravel(), mats):
        cs = reports[m]["count_series"]
        counts = np.asarray(cs["counts"])
        block = cs[fitter]
        for kind, color, marker, label in (
                ("canonical_unpaired", GRAY, "s", "Canonical"),
                ("canonical_paired", AQUA, "^", "Canonical $\\pm$ pairs")):
            values = np.asarray([[t["mesh_vs_disjoint_effective"] for t in trials]
                                 for trials in block[kind]])
            med = np.median(values, axis=1)
            ax.fill_between(counts, np.percentile(values, 10, axis=1),
                            np.percentile(values, 90, axis=1), color=color,
                            alpha=0.18, linewidth=0)
            ax.plot(counts, med, color=color, marker=marker, markersize=4,
                    linewidth=1.4, label=label)
        sotcd = [row["mesh_vs_disjoint_effective"] for row in block["sotcd"]]
        ax.plot(counts, sotcd, color=BLUE, marker="o", markersize=4.5,
                linewidth=2.0, label="SOTC-D")
        ax.set_xscale("log", base=2)
        ax.set_yscale("log")
        ax.yaxis.set_major_formatter(matplotlib.ticker.FormatStrFormatter("%g"))
        ax.yaxis.set_minor_locator(matplotlib.ticker.LogLocator(base=10, subs=(2, 3, 5)))
        ax.yaxis.set_minor_formatter(matplotlib.ticker.FormatStrFormatter("%g"))
        ax.tick_params(axis="y", which="minor", labelsize=6.5)
        ax.set_xticks(counts, [str(c) for c in counts])
        ax.set_title(LABELS[m], loc="left")
        style(ax)
    for ax in axes.ravel()[len(mats):]:
        ax.axis("off")
    for ax in axes[:, 0]:
        ax.set_ylabel("Mesh RMSE (THz)")
    for ax in axes[-1, :]:
        ax.set_xlabel("Displaced force calls")
    handles, labels = axes.ravel()[0].get_legend_handles_labels()
    fig.legend(handles, labels, ncol=4, frameon=False, loc="upper center")
    fig.tight_layout(rect=(0, 0, 1, 0.95))
    fig.savefig(OUT / "fig_count_series.pdf")
    fig.savefig(OUT / "fig_count_series.png", dpi=200)
    plt.close(fig)


def main():
    OUT.mkdir(parents=True, exist_ok=True)
    reports = {m: load(m) for m in MATERIALS}
    accuracy_figure(reports)
    forces_figure(reports)
    count_figure(reports)


if __name__ == "__main__":
    main()
