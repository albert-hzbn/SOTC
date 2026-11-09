#!/usr/bin/env python3
"""Scaling figure (JAP manuscript): SOTC-D-SC and pooled sTDEP in cells of
increasing size.

Reads ``validation_results/summaries/scaling/scaling_compare.json``
(``scaling_compare.py``) and writes ``fig_scaling.pdf``:

* (a)-(c) mesh RMSE versus force calls per material, seed means, one line per
  cell size (solid: SOTC-D-SC, dashed: pooled sTDEP);
* (d) calls after which the error stays below 0.1 THz for every cell, median
  (marker) and range (bar) over three seeds, with the number of FC2
  parameters P under each cell;
* (e) distance between the SOTC-D-SC and pooled-sTDEP fixed points versus the
  number of modes M = 3N, with a 1/M line through the smallest cell of each
  material [the O(1/M) bias of Eq. (8)]; points labeled with the atom count.
"""

from __future__ import annotations

import json

import matplotlib
import matplotlib.pyplot as plt
import numpy as np

from plot_robustness import OUT, ROOT
from plot_style import AQUA, BLUE, GRAY, MUTED, ORANGE, PAGE, RULE, panel, save

CELLS = {"Al, 300 K": ["al", "al_108", "al_256"], "Si, 600 K": ["si", "si_128", "si_216"],
         "NaCl, 700 K": ["nacl_128", "nacl_216"]}
COLORS = {"Al, 300 K": BLUE, "Si, 600 K": ORANGE, "NaCl, 700 K": AQUA}
# tints of the material color, mixed with white (no transparency)
SHADES = (0.45, 0.72, 1.0)
WIDTHS = (0.9, 1.0, 1.25)
# atom-count labels in (e): below the point where another line is above it
LABEL_OFFSETS = {"nacl_128": (0, -4.5, "top"), "nacl_216": (0, -4.5, "top"),
                 "si_216": (0, -4.5, "top")}


def tint(color, shade):
    rgb = np.asarray(matplotlib.colors.to_rgb(color))
    return tuple(1 - shade * (1 - rgb))


def seed_mean(runs, proto):
    curves = [(e["calls"], e["rmse_thz"]) for k, e in runs.items() if k.split("/")[0] == proto]
    n = min(len(c) for c, _ in curves)
    return np.asarray(curves[0][0][:n]), np.mean([e[:n] for _, e in curves], axis=0)


def seed_calls(runs, proto, key):
    return [np.inf if e[key] is None else e[key]
            for k, e in runs.items() if k.split("/")[0] == proto]


def main():
    data = json.loads((ROOT / "validation_results" / "summaries" / "scaling"
                       / "scaling_compare.json").read_text())
    fig = plt.figure(figsize=(PAGE, 4.6))
    grid = fig.add_gridspec(2, 6, height_ratios=(1, 0.92))
    top = [fig.add_subplot(grid[0, 2 * i:2 * i + 2]) for i in range(3)]
    ax_p = fig.add_subplot(grid[1, 0:3])
    ax_m = fig.add_subplot(grid[1, 3:6])
    for ax, (material, cells) in zip(top, CELLS.items()):
        cells = [c for c in cells if c in data]
        n = len(cells)
        for shade, lw, cell in zip(SHADES[-n:], WIDTHS[-n:], cells):
            r = data[cell]
            color = tint(COLORS[material], shade)
            for proto, ls in (("sotcd_sc", "-"), ("stdep_pooled", (0, (3.5, 1.8)))):
                calls, err = seed_mean(r["runs"], proto)
                ax.plot(calls, err, ls=ls, color=color, lw=lw, solid_capstyle="round",
                        label=f"{r['n_atoms']} atoms" if proto == "sotcd_sc" else None)
        ax.set_xscale("log", base=2)
        ax.set_yscale("log")
        ax.set_xlim(1.8, 36)
        ax.set_xticks([2, 4, 8, 16, 32])
        ax.set_xticklabels(["2", "4", "8", "16", "32"])
        ax.set_ylim(0.008, 0.7)
        ax.set_yticks([0.01, 0.03, 0.1, 0.3])
        ax.set_yticklabels(["0.01", "0.03", "0.1", "0.3"])
        ax.yaxis.set_minor_formatter(matplotlib.ticker.NullFormatter())
        ax.minorticks_off()
        ax.axhline(0.05, color=GRAY, lw=0.6, ls=":", zorder=0)
        ax.set_title(material)
        ax.set_xlabel("DFT force calls")
        ax.legend(loc="upper right")
    top[0].set_ylabel("mesh RMSE (THz)")
    top[1].plot([], [], "-", color=MUTED, label="SOTC-D-SC")
    top[1].plot([], [], ls=(0, (3.5, 1.8)), color=MUTED, label="pooled sTDEP")
    top[1].plot([], [], ":", color=GRAY, lw=0.6, label="0.05 THz")
    h, l = top[1].get_legend_handles_labels()
    top[1].add_artist(top[1].legend(h[:-3], l[:-3], loc="upper right"))
    top[1].legend(h[-3:], l[-3:], loc="lower left")

    # (d) calls to 0.1 THz, one slot per cell, grouped by material
    x, ticks, labels = 0.0, [], []
    for material, cells in CELLS.items():
        color = COLORS[material]
        cells = [c for c in cells if c in data]
        first = x
        for cell in cells:
            r = data[cell]
            for proto, dx, marker, fill in (("sotcd_sc", -0.17, "o", color),
                                            ("stdep_pooled", 0.17, "s", "white")):
                v = seed_calls(r["runs"], proto, "calls_to_0.1")
                ax_p.plot([x + dx] * 2, [min(v), max(v)], color=color, lw=1.0,
                          solid_capstyle="butt", zorder=1)
                ax_p.plot(x + dx, np.median(v), marker, mfc=fill, mec=color, mew=1.0,
                          ms=4.5, zorder=2)
            ticks.append(x)
            labels.append(f"{r['n_atoms']}\n$P$={r['n_parameters']}")
            x += 1
        ax_p.text((first + x - 1) / 2, 31, material.split(",")[0], ha="center",
                  va="top", color=color)
        if x < 8:
            ax_p.axvline(x - 0.5, color=RULE, lw=0.5, zorder=0)
        x += 0
    ax_p.set_xticks(ticks)
    ax_p.set_xticklabels(labels, fontsize=7, linespacing=1.1)
    ax_p.tick_params(axis="x", which="both", top=False, length=0, pad=2.5)
    ax_p.set_xlim(-0.6, x - 0.4)
    ax_p.set_yscale("log", base=2)
    ax_p.set_ylim(1.6, 34)
    ax_p.set_yticks([2, 4, 8, 16])
    ax_p.set_yticklabels(["2", "4", "8", "16"])
    ax_p.minorticks_off()
    ax_p.set_xlabel("cell (atoms)", labelpad=1)
    ax_p.set_ylabel("calls to 0.1 THz")
    ax_p.plot([], [], "o", color=MUTED, ms=4.5, label="SOTC-D-SC")
    ax_p.plot([], [], "s", mfc="white", mec=MUTED, ms=4.5, label="pooled sTDEP")
    ax_p.legend(loc="upper center", bbox_to_anchor=(0.56, 0.9), ncol=1)

    # (e) fixed-point offset against the number of modes
    for material, cells in CELLS.items():
        color = COLORS[material]
        present = [c for c in cells if c in data]
        m = np.array([data[c]["n_modes"] for c in present])
        d = np.array([data[c]["fixed_point"]["sotcd_sc_vs_stdep_pooled"] for c in present])
        mm = np.geomspace(m[0], 800, 20)
        ax_m.plot(mm, d[0] * m[0] / mm, ":", color=color, lw=0.8, zorder=1)
        ax_m.plot(m, d, "-", color=color, lw=1.0, zorder=2, label=material)
        ax_m.plot(m, d, "o", mfc=color, mec="white", mew=0.8, ms=5, zorder=3)
        for mi, di, c in zip(m, d, present):
            dx, dy, va = LABEL_OFFSETS.get(c, (0, 4.5, "bottom"))
            ax_m.annotate(str(data[c]["n_atoms"]), (mi, di), xytext=(dx, dy),
                          textcoords="offset points", ha="center", va=va,
                          fontsize=6.5, color=MUTED)
    ax_m.plot([], [], ":", color=MUTED, lw=0.8, label="$1/M$")
    ax_m.set_xscale("log")
    ax_m.set_yscale("log")
    ax_m.set_xlim(80, 900)
    ax_m.set_ylim(0.0045, 0.11)
    ax_m.set_xticks([100, 200, 400, 800])
    ax_m.set_xticklabels(["100", "200", "400", "800"])
    ax_m.set_yticks([0.005, 0.01, 0.02, 0.05, 0.1])
    ax_m.set_yticklabels(["0.005", "0.01", "0.02", "0.05", "0.1"])
    ax_m.minorticks_off()
    ax_m.set_xlabel("modes $M=3N$")
    ax_m.set_ylabel("fixed-point offset (THz)")
    ax_m.legend(loc="lower left", ncol=2)
    for ax, letter in zip(top + [ax_p, ax_m], "abcde"):
        panel(ax, letter)
    fig.tight_layout(pad=0.3, h_pad=0.6)
    save(fig, OUT, "fig_scaling")


if __name__ == "__main__":
    main()
