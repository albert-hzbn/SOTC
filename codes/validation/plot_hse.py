#!/usr/bin/env python3
"""HSE06 figure (JAP manuscript): phonon-mesh RMSE of SOTC-D-SC and sTDEP
against the composite HSE06 reference after every DFT call.

Reads ``validation_results/summaries/<material>/robustness/hse_compare.json``
(``hse_compare.py``) and writes ``fig_hse.pdf``. Lines are seed means, bands
the seed range; dashed lines are the same protocols in PBE against the PBE
self-consistent reference.
"""

from __future__ import annotations

import json
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402

from plot_robustness import BLUE, GRAY, MUTED, ORANGE, OUT, ROOT  # noqa: E402

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
    fig, axes = plt.subplots(1, len(reports), figsize=(3.4 * len(reports), 2.6), squeeze=False)
    for ax, (m, r) in zip(axes[0], reports.items()):
        for proto, label, color in PROTOCOLS:
            calls, errors = seed_curves(r["hse_runs"], proto)
            ax.plot(calls, errors.mean(axis=0), "o-", color=color, ms=3, lw=1.2, label=label)
            ax.fill_between(calls, errors.min(axis=0), errors.max(axis=0), color=color,
                            alpha=0.15, lw=0)
            pbe = r.get("pbe_summary", {}).get(proto)
            if pbe:
                b = sorted(int(k) for k in pbe)
                ax.plot(b, [pbe[str(k)][0] for k in b], "--", color=color, lw=0.8, alpha=0.7)
        ax.axhline(r["fixed_point"]["stdep_pooled_vs_reference_thz"], color=MUTED, lw=0.6,
                   ls=":")
        ax.set_xlabel("DFT force calls")
        ax.set_ylabel("mesh RMSE vs HSE06 reference (THz)")
        ax.set_title(MATERIALS[m])
        ax.set_ylim(0, None)
        ax.set_xticks([2, 4, 8, 12, 16])
    axes[0][0].legend(frameon=False, loc="upper right")
    fig.tight_layout()
    fig.savefig(OUT / "fig_hse.pdf")
    fig.savefig(OUT / "fig_hse.png", dpi=200)


if __name__ == "__main__":
    main()
