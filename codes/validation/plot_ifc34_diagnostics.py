#!/usr/bin/env python3
"""Plot post-fit validation diagnostics for the real QE IFC3/IFC4 campaign."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np


ROOT = Path(__file__).resolve().parents[2]
DEFAULT_SUMMARY = ROOT / "validation_results" / "ifc34_campaign_summary.json"
DEFAULT_FIGURE = ROOT / "manuscript" / "figures" / "fig_ifc34_diagnostics"
DEFAULT_PLOT_SUMMARY = (
    ROOT / "validation_results" / "ifc34_diagnostics_plot_summary.json"
)
MATERIAL_LABELS = {
    "al": "Al (300 K)", "nacl": "NaCl (700 K)", "w": "W (1500 K)"
}
COLORS = {"al": "#0072b2", "nacl": "#d55e00", "w": "#009e73"}
MARKERS = {"al": "o", "nacl": "s", "w": "^"}


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--summary", type=Path, default=DEFAULT_SUMMARY)
    parser.add_argument("--figure", type=Path, default=DEFAULT_FIGURE)
    parser.add_argument(
        "--plot-summary", type=Path, default=DEFAULT_PLOT_SUMMARY
    )
    args = parser.parse_args()

    campaign = json.loads(args.summary.read_text())
    materials = campaign["materials"]
    models = [
        "finite_displacement_fc2", "fc2", "fc2_fc3", "fc2_fc3_fc4"
    ]
    model_labels = ["FD FC2", "T-fit FC2", "+FC3", "+FC4"]
    snapshots = [20, 40, 60, 80]

    figure, axes = plt.subplots(2, 2, figsize=(10.0, 7.2))
    axes = axes.ravel()
    for material in materials:
        key = material["material_key"]
        style = {
            "color": COLORS[key], "marker": MARKERS[key],
            "linewidth": 1.7, "markersize": 5,
        }
        axes[0].plot(
            np.arange(len(models)),
            [
                material["held_out_rmse_force_component_ev_ang"][name]
                for name in models
            ],
            label=MATERIAL_LABELS[key], **style,
        )
        axes[1].plot(
            snapshots,
            [
                material["snapshot_count_convergence"][str(count)]
                for count in snapshots
            ],
            **style,
        )
        axes[2].plot(
            [-0.5, 0.0, 0.5],
            [
                item["held_out_rmse_force_component_ev_ang"]
                for item in material["cutoff_sensitivity"]
            ],
            **style,
        )

    axes[0].set_xticks(
        np.arange(len(models)), model_labels, rotation=18, ha="right"
    )
    axes[0].set_yscale("log")
    axes[0].set_ylabel(r"Held-out force RMSE (eV $\AA^{-1}$)")
    axes[0].set_title("(a) Incremental model comparison")

    axes[1].set_xticks(snapshots)
    axes[1].set_yscale("log")
    axes[1].set_xlabel("Training snapshots")
    axes[1].set_ylabel(r"Held-out force RMSE (eV $\AA^{-1}$)")
    axes[1].set_title("(b) Snapshot convergence, FC2+FC3+FC4")

    axes[2].set_xticks(
        [-0.5, 0.0, 0.5], ["shorter", "selected", "longer"]
    )
    axes[2].set_yscale("log")
    axes[2].set_xlabel(
        r"Interaction-cutoff bracket ($\pm0.5$ $\AA$)"
    )
    axes[2].set_ylabel(r"Held-out force RMSE (eV $\AA^{-1}$)")
    axes[2].set_title("(c) Cutoff sensitivity")

    x = np.arange(len(materials))
    width = 0.23
    diagnostics = [
        "final fit", r"higher $E_\mathrm{cut}$", r"denser $k$ mesh"
    ]
    hatches = [None, "//", "xx"]
    for index, (label, pattern) in enumerate(zip(diagnostics, hatches)):
        values = []
        for material in materials:
            if index == 0:
                value = material[
                    "held_out_rmse_force_component_ev_ang"
                ]["fc2_fc3_fc4"]
            elif index == 1:
                value = material[
                    "electronic_convergence_force_rmse_ev_ang"
                ]["higher_cutoff"]
            else:
                value = material[
                    "electronic_convergence_force_rmse_ev_ang"
                ]["denser_kmesh"]
            values.append(value)
        axes[3].bar(
            x + (index - 1) * width, values, width, label=label,
            color="0.75" if index else "#56b4e9",
            edgecolor="0.2", hatch=pattern,
        )
    axes[3].set_xticks(
        x,
        [
            MATERIAL_LABELS[item["material_key"]].split()[0]
            for item in materials
        ],
    )
    axes[3].set_yscale("log")
    axes[3].set_ylabel(r"Force RMSE/difference (eV $\AA^{-1}$)")
    axes[3].set_title("(d) Electronic-setting sensitivity")
    diagnostic_handles, diagnostic_labels = axes[3].get_legend_handles_labels()

    for axis in axes:
        axis.grid(axis="y", alpha=0.22)
    handles, labels = axes[0].get_legend_handles_labels()
    figure.legend(
        handles, labels, loc="upper center", bbox_to_anchor=(0.5, 1.01),
        ncol=3, frameon=False,
    )
    figure.legend(
        diagnostic_handles, diagnostic_labels,
        loc="upper center", bbox_to_anchor=(0.5, 0.965),
        ncol=3, frameon=False, fontsize=8,
    )
    figure.tight_layout(rect=[0.0, 0.0, 1.0, 0.88])
    args.figure.parent.mkdir(parents=True, exist_ok=True)
    args.plot_summary.parent.mkdir(parents=True, exist_ok=True)
    figure.savefig(args.figure.with_suffix(".pdf"), bbox_inches="tight")
    figure.savefig(
        args.figure.with_suffix(".png"), dpi=300, bbox_inches="tight"
    )
    plt.close(figure)

    plot_summary = {
        "source": str(args.summary),
        "status": campaign["status"],
        "materials": [
            material["material_key"] for material in materials
        ],
        "panels": [
            "held-out incremental model comparison",
            "snapshot-count convergence",
            "interaction-cutoff sensitivity",
            "electronic-setting force sensitivity",
        ],
        "figure_pdf": str(args.figure.with_suffix(".pdf")),
        "figure_png": str(args.figure.with_suffix(".png")),
    }
    args.plot_summary.write_text(
        json.dumps(plot_summary, indent=2) + "\n"
    )
    print(json.dumps(plot_summary, indent=2))


if __name__ == "__main__":
    main()
