#!/usr/bin/env python3
"""Scaling figure (JAP manuscript): SOTC-D-SC and pooled sTDEP in cells of
increasing size.

Reads ``validation_results/summaries/scaling/scaling_compare.json``
(``scaling_compare.py``) and writes ``fig_scaling.pdf``:

* (a)-(c) mesh RMSE versus force calls per material, seed means, one line per
  cell size (solid: SOTC-D-SC, dashed: pooled sTDEP);
* (d) calls after which the error stays below 0.1 THz (median over seeds)
  versus the number of FC2 parameters P (marker size: cell size);
* (e) distance between the SOTC-D-SC and pooled-sTDEP fixed points versus the
  number of modes M = 3N, with a 1/M line through the smallest cell of each
  material [the O(1/M) bias of Eq. (8)].
"""

from __future__ import annotations

import json

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402

from plot_robustness import AQUA, BLUE, MUTED, ORANGE, OUT, ROOT  # noqa: E402

CELLS = {"Al, 300 K": ["al", "al_108", "al_256"], "Si, 600 K": ["si", "si_128", "si_216"],
         "NaCl, 700 K": ["nacl_128", "nacl_216"]}
COLORS = {"Al, 300 K": BLUE, "Si, 600 K": ORANGE, "NaCl, 700 K": AQUA}
SHADES = (0.35, 0.65, 1.0)


def seed_mean(runs, proto):
    curves = [(e["calls"], e["rmse_thz"]) for k, e in runs.items() if k.split("/")[0] == proto]
    n = min(len(c) for c, _ in curves)
    return np.asarray(curves[0][0][:n]), np.mean([e[:n] for _, e in curves], axis=0)


def median_calls(runs, proto, key):
    vals = [e[key] for k, e in runs.items() if k.split("/")[0] == proto]
    vals = [np.inf if v is None else v for v in vals]
    return float(np.median(vals))


def main():
    data = json.loads((ROOT / "validation_results" / "summaries" / "scaling"
                       / "scaling_compare.json").read_text())
    fig = plt.figure(figsize=(7.0, 4.6))
    grid = fig.add_gridspec(2, 6)
    top = [fig.add_subplot(grid[0, 2 * i:2 * i + 2]) for i in range(3)]
    ax_p = fig.add_subplot(grid[1, 0:3])
    ax_m = fig.add_subplot(grid[1, 3:6])
    for ax, (material, cells) in zip(top, CELLS.items()):
        cells = [c for c in cells if c in data]
        for shade, cell in zip(SHADES[-len(cells):], cells):
            r = data[cell]
            color = matplotlib.colors.to_rgba(COLORS[material], shade)
            for proto, ls in (("sotcd_sc", "-"), ("stdep_pooled", "--")):
                calls, err = seed_mean(r["runs"], proto)
                ax.plot(calls, err, ls, color=color, lw=1.1,
                        label=f"{r['n_atoms']} atoms" if proto == "sotcd_sc" else None)
        ax.set_xscale("log", base=2)
        ax.set_yscale("log")
        ax.set_xticks([2, 4, 8, 16, 32])
        ax.set_xticklabels(["2", "4", "8", "16", "32"])
        ax.axhline(0.05, color=MUTED, lw=0.5, ls=":")
        ax.set_title(material)
        ax.set_xlabel("DFT force calls")
        ax.legend(frameon=False, fontsize=6.5, loc="upper right")
    top[0].set_ylabel("mesh RMSE (THz)")
    for material, cells in CELLS.items():
        color = COLORS[material]
        for k, cell in enumerate(cells):
            if cell not in data:
                continue
            r = data[cell]
            # cells with equal P are offset slightly, larger cells to the right
            x = r["n_parameters"] * (1 + 0.06 * k)
            for proto, marker, fill in (("sotcd_sc", "o", color), ("stdep_pooled", "s", "none")):
                y = median_calls(r["runs"], proto, "calls_to_0.1")
                ax_p.plot(x, min(y, 48), marker, mfc=fill, mec=color, ms=3 + 1.5 * k)
        present = [c for c in cells if c in data]
        m = np.array([data[c]["n_modes"] for c in present])
        d = np.array([data[c]["fixed_point"]["sotcd_sc_vs_stdep_pooled"] for c in present])
        ax_m.plot(m, d, "o-", color=color, ms=4, lw=1, label=material)
        mm = np.geomspace(m[0], 800, 20)
        ax_m.plot(mm, d[0] * m[0] / mm, ":", color=color, lw=0.8)
    ax_p.plot([], [], "o", color=MUTED, label="SOTC-D-SC")
    ax_p.plot([], [], "s", mfc="none", mec=MUTED, label="pooled sTDEP")
    ax_p.set_xscale("log")
    ax_p.set_xticks([5, 8, 10, 30])
    ax_p.set_xticklabels(["5", "8", "10", "30"])
    ax_p.minorticks_off()
    ax_p.set_xlabel("FC2 parameters $P$")
    ax_p.set_ylabel("calls to 0.1 THz (median)")
    ax_p.legend(frameon=False)
    ax_m.set_xscale("log")
    ax_m.set_yscale("log")
    ax_m.set_xlabel("modes $M=3N$")
    ax_m.set_ylabel("fixed-point offset (THz)")
    ax_m.legend(frameon=False)
    for ax, letter in zip(top + [ax_p, ax_m], "abcde"):
        ax.text(0.02, 0.97, f"({letter})", transform=ax.transAxes, va="top", fontsize=8)
    fig.tight_layout()
    fig.savefig(OUT / "fig_scaling.pdf")
    fig.savefig(OUT / "fig_scaling.png", dpi=200)


if __name__ == "__main__":
    main()
