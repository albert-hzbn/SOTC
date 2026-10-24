#!/usr/bin/env python3
"""U_iso(300 K) of Cu and Au from SOTC models at the room-temperature lattice.

The production cells of Cu and Au are at their 1200 K lattice parameters, so
the SOTC-D runs for the Debye--Waller comparison use the same 32-atom cell
scaled to the room-temperature lattice parameter (``dft_qha.py <m> <a>
--temperatures 300 --budget 8``). Reported: the harmonic FD FC2 of that cell,
and the self-consistent SOTC-D FC2 after 2, 4, and 8 force calls, Fourier
interpolated on a 20^3 mesh; U_iso = <|u|^2>/3.

Writes validation_results/summaries/sotc_uiso_room.json.
"""

from __future__ import annotations

import json
from pathlib import Path
import sys

import numpy as np

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "codes"))
sys.path.insert(0, str(Path(__file__).resolve().parent))

from al_uiso_models import u_iso  # noqa: E402
from dft_qha import scaled_phonon  # noqa: E402
from run_fd_references import ROOM_LATTICE  # noqa: E402


def main():
    report = {}
    for m in ("cu", "au"):
        a = ROOM_LATTICE[m]
        d = ROOT / "validation_results" / m / "primary_cell" / "qha" / f"a{a:.4f}"
        phonon = scaled_phonon(m, a)
        r = np.load(d / "T300.npz")
        models = {"harmonic_fd_cell": np.load(d / "fd.npz")["force_constants"]}
        for k, fc in enumerate(r["force_constants"]):
            models[f"sotcd_sc_{2 * (k + 1)}"] = fc
        report[m] = {"a_ang": a, "u_iso_ang2": {k: u_iso(phonon, fc) for k, fc in models.items()}}
        print(m, {k: round(v, 5) for k, v in report[m]["u_iso_ang2"].items()}, flush=True)
    out = ROOT / "validation_results" / "summaries" / "sotc_uiso_room.json"
    out.write_text(json.dumps(report, indent=1) + "\n")


if __name__ == "__main__":
    main()
