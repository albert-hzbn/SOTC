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

import matplotlib.pyplot as plt
import matplotlib.ticker
import numpy as np

from plot_style import AQUA, BLUE, GRAY, INK, MUTED, ORANGE, PAGE, RULE, YELLOW, panel, plain_log, save  # noqa: F401

ROOT = Path(__file__).resolve().parents[2]
OUT = ROOT / "manuscript" / "jap" / "figures"
MATERIALS = ["al", "ag", "au", "cu", "mo", "w", "nacl", "mgo", "pbte"]
LABELS = {"al": "Al", "ag": "Ag", "au": "Au", "cu": "Cu", "mo": "Mo",
          "w": "W", "nacl": "NaCl", "mgo": "MgO", "pbte": "PbTe"}

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
    ax.yaxis.grid(True, color=RULE, linewidth=0.4)
    ax.xaxis.set_tick_params(top=False)
    ax.set_axisbelow(True)


def accuracy_figure(reports):
    mats = [m for m in MATERIALS if reports.get(m)]
    x = np.arange(len(mats))
    fig, axes = plt.subplots(2, 1, figsize=(PAGE, 4.4), sharex=True)
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
    ax.set_ylabel("RMSE vs FD (THz)")
    ax.set_title("(a) Harmonic reference: 5$^3$/4$^3$-primitive-cell finite displacement",
                 loc="left")
    style(ax)
    ax.set_ylim(0, 1.75)
    ax.legend(ncol=2, loc="upper left")

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
    ax.set_ylabel("RMSE vs effective (THz)")
    ax.set_title("(b) Effective reference: 40 canonical snapshots at $T_{\\rm des}$, "
                 "same cell", loc="left")
    style(ax)
    ax.legend(ncol=2, loc="upper right")
    ax.set_xticks(x, [LABELS[m] for m in mats])
    fig.tight_layout(pad=0.3)
    save(fig, OUT, "fig_reference_accuracy")


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
    fig.tight_layout(pad=0.3)
    save(fig, OUT, "fig_matched_forces")


def count_figure(reports, fitter="symfc"):
    """Error versus force calls: SOTC-D against canonical baselines."""
    mats = [m for m in MATERIALS if reports.get(m) and "count_series" in reports[m]
            and reports[m]["count_series"][fitter].get("sotcd")]
    ncol = 3
    nrow = int(np.ceil(len(mats) / ncol))
    fig, axes = plt.subplots(nrow, ncol, figsize=(PAGE, 1.65 * nrow + 0.35),
                             sharex=True, squeeze=False)
    for k, (ax, m) in enumerate(zip(axes.ravel(), mats)):
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
                            alpha=0.16, linewidth=0)
            ax.plot(counts, med, color=color, marker=marker, markersize=3,
                    linewidth=1.0, label=label)
        sotcd = [row["mesh_vs_disjoint_effective"] for row in block["sotcd"]]
        ax.plot(counts, sotcd, color=BLUE, marker="o", markersize=3.5,
                linewidth=1.4, label="SOTC-D")
        ax.set_xscale("log", base=2)
        ax.set_yscale("log")
        plain_log(ax)
        ax.set_xticks(counts, [str(c) for c in counts])
        ax.xaxis.set_minor_locator(matplotlib.ticker.NullLocator())
        ax.margins(y=0.18)
        ax.text(0.97, 0.94, f"({chr(97 + k)}) {LABELS[m]}", transform=ax.transAxes,
                ha="right", va="top")
    for ax in axes.ravel()[len(mats):]:
        ax.axis("off")
    for ax in axes[:, 0]:
        ax.set_ylabel("mesh RMSE (THz)")
    for ax in axes[-1, :]:
        ax.set_xlabel("displaced force calls")
    handles, labels = axes.ravel()[0].get_legend_handles_labels()
    fig.legend(handles, labels, ncol=4, loc="upper center")
    fig.tight_layout(pad=0.3, rect=(0, 0, 1, 0.965))
    save(fig, OUT, "fig_count_series")


def main():
    OUT.mkdir(parents=True, exist_ok=True)
    reports = {m: load(m) for m in MATERIALS}
    accuracy_figure(reports)
    forces_figure(reports)
    count_figure(reports)


if __name__ == "__main__":
    main()
