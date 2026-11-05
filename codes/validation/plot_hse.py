#!/usr/bin/env python3
"""HSE06 figure (JAP manuscript): phonon-mesh RMSE of SOTC-D-SC and sTDEP
against the composite HSE06 reference after every DFT call.

Reads ``validation_results/summaries/<material>/robustness/hse_compare.json``
(``hse_compare.py``) and writes ``fig_hse.pdf``. Lines are seed means, bands
the seed range (SOTC-D-SC and pooled sTDEP); dashed lines are the same protocols in PBE against the PBE
self-consistent reference.
"""

from __future__ import annotations

import json

import matplotlib.pyplot as plt
import numpy as np

from plot_robustness import OUT, ROOT
from plot_style import BLUE, COLUMN, GRAY, MUTED, ORANGE, save

MATERIALS = {"si_hse": "Si, 600 K", "nacl_hse": "NaCl, 700 K"}
PROTOCOLS = [("sotcd_sc", "SOTC-D-SC", BLUE), ("stdep_pooled", "sTDEP (pooled)", ORANGE),
             ("stdep_last", "sTDEP (last)", GRAY)]


def seed_curves(runs, proto):
    curves = [(np.asarray(e["calls"]), np.asarray(e["rmse_thz"]))
              for name, e in runs.items() if name.split("/")[0] == proto]
    calls = curves[0][0]
    errors = np.array([e for c, e in curves if np.array_equal(c, calls)])
    return calls, errors


def main():
    reports = {}
    for m in MATERIALS:
        path = ROOT / "validation_results" / "summaries" / m / "robustness" / "hse_compare.json"
        if path.exists():
            reports[m] = json.loads(path.read_text())
    fig, axes = plt.subplots(len(reports), 1, figsize=(COLUMN, 2.0 * len(reports) + 0.1),
                             sharex=True, squeeze=False)
    for k, (ax, (m, r)) in enumerate(zip(axes[:, 0], reports.items())):
        for proto, label, color in PROTOCOLS:
            calls, errors = seed_curves(r["hse_runs"], proto)
            ax.plot(calls, errors.mean(axis=0), "o-", color=color, ms=2.5, lw=1.1, label=label)
            if proto != "stdep_last":
                ax.fill_between(calls, errors.min(axis=0), errors.max(axis=0), color=color,
                                alpha=0.14, lw=0)
            pbe = r.get("pbe_summary", {}).get(proto)
            if pbe:
                b = sorted(int(x) for x in pbe)
                ax.plot(b, [pbe[str(x)][0] for x in b], "--", color=color, lw=0.7)
        ax.axhline(r["fixed_point"]["stdep_pooled_vs_reference_thz"], color=MUTED, lw=0.6,
                   ls=":")
        ax.set_ylabel("mesh RMSE (THz)")
        ax.set_ylim(0, None)
        ax.set_xlim(1.5, 16.5)
        ax.set_xticks([2, 4, 8, 12, 16])
        ax.text(0.97, 0.94, f"({chr(97 + k)}) {MATERIALS[m]}", transform=ax.transAxes,
                ha="right", va="top")
    axes[-1, 0].set_xlabel("DFT force calls")
    axes[0, 0].plot([], [], "--", color=MUTED, lw=0.7, label="same, PBE")
    axes[0, 0].plot([], [], ":", color=MUTED, lw=0.6, label="sTDEP fixed point")
    axes[0, 0].legend(loc="upper center", ncol=2, bbox_to_anchor=(0.5, 1.28))
    fig.tight_layout(pad=0.3, h_pad=0.4)
    save(fig, OUT, "fig_hse")


if __name__ == "__main__":
    main()
