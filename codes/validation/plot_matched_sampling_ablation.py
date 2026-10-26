#!/usr/bin/env python3
"""Summarize and plot the matched SOTC-versus-canonical sampling ablation."""

from __future__ import annotations

import json
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np


ROOT = Path(__file__).resolve().parents[2]
MATERIALS = {"al": "Al", "w": "W", "nacl": "NaCl"}
METRICS = (
    ("dispersion", "rmse_thz", "Dispersion RMSE (THz)"),
    ("held_out_force", "rmse_ev_ang", r"Held-out force RMSE (eV $\AA^{-1}$)"),
)


def percentile_summary(values):
    values = np.asarray(values, dtype=float)
    return {
        "median": np.median(values, axis=0).tolist(),
        "p10": np.percentile(values, 10, axis=0).tolist(),
        "p90": np.percentile(values, 90, axis=0).tolist(),
        "minimum": np.min(values, axis=0).tolist(),
        "maximum": np.max(values, axis=0).tolist(),
    }


def main():
    reports = {}
    for material in MATERIALS:
        path = (ROOT / "validation_results" / material / "matched_sampling"
                / "matched_sampling_summary.json")
        reports[material] = json.loads(path.read_text())

    figure, axes = plt.subplots(2, 3, figsize=(7.2, 4.7), sharex=True,
                                constrained_layout=True)
    summary = {
        "scope": ("Same cells, electronic settings, force processing, IFC "
                  "priors, held-out set, and q paths; only displacement "
                  "selection and Debye initialization vary."),
        "materials": {},
    }
    for column, (material, label) in enumerate(MATERIALS.items()):
        report = reports[material]
        counts = np.asarray(report["counts"], dtype=int)
        material_summary = {
            "counts": counts.tolist(),
            "canonical_seed_count": report["canonical_seed_count"],
            "sotc_initialization_count": len(report["sotc_initialization_series"]),
            "metrics": {},
        }
        for row, (group, key, ylabel) in enumerate(METRICS):
            axis = axes[row, column]
            canonical = np.asarray([
                [entry[group][key] for entry in seed["counts"]]
                for seed in report["canonical_series"]
            ])
            sotc = np.asarray([
                [entry[group][key] for entry in series]
                for series in report["sotc_initialization_series"].values()
            ])
            cstats = percentile_summary(canonical)
            sstats = percentile_summary(sotc)
            axis.fill_between(counts, cstats["p10"], cstats["p90"],
                              color="0.75", alpha=0.65,
                              label="canonical 10--90%" if row == 0 else None)
            axis.plot(counts, cstats["median"], color="0.25", marker="o",
                      linewidth=1.15, markersize=3.2,
                      label="canonical median" if row == 0 else None)
            axis.fill_between(counts, sstats["minimum"], sstats["maximum"],
                              color="#79aee3", alpha=0.35,
                              label="SOTC init. range" if row == 0 else None)
            for index, (init_label, series) in enumerate(
                    report["sotc_initialization_series"].items()):
                values = [entry[group][key] for entry in series]
                axis.plot(counts, values, color="#0057b8", marker="s",
                          linewidth=0.8, markersize=2.7, alpha=0.45,
                          label="SOTC initializations"
                          if row == 0 and index == 0 else None)
            axis.set_xscale("log", base=2)
            axis.set_yscale("log")
            axis.set_xticks(counts, [str(value) for value in counts])
            axis.grid(True, which="both", color="0.9", linewidth=0.45)
            axis.tick_params(labelsize=7)
            if row == 0:
                axis.set_title(label, fontsize=9)
            if column == 0:
                axis.set_ylabel(ylabel, fontsize=8)
            if row == 1:
                axis.set_xlabel("Displaced configurations", fontsize=8)
            material_summary["metrics"][f"{group}_{key}"] = {
                "canonical": cstats,
                "sotc_initialization_range": sstats,
                "sotc_by_initialization": {
                    init_label: [entry[group][key] for entry in series]
                    for init_label, series in
                    report["sotc_initialization_series"].items()
                },
            }
        summary["materials"][material] = material_summary

    handles, labels = axes[0, 0].get_legend_handles_labels()
    figure.legend(handles, labels, loc="upper center", ncol=4,
                  bbox_to_anchor=(0.5, 1.045), frameon=False, fontsize=7)
    destination = ROOT / "manuscript" / "prb" / "figures" / "fig_matched_sampling"
    figure.savefig(destination.with_suffix(".pdf"), bbox_inches="tight")
    figure.savefig(destination.with_suffix(".png"), dpi=240,
                   bbox_inches="tight")
    plt.close(figure)
    summary["figure"] = str(destination.with_suffix(".pdf"))
    summary_path = ROOT / "validation_results" / "matched_sampling_summary.json"
    summary_path.write_text(json.dumps(summary, indent=2) + "\n")
    print(summary_path)
    print(destination.with_suffix(".pdf"))


if __name__ == "__main__":
    main()
