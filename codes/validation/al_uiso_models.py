#!/usr/bin/env python3
"""U_iso(300 K) of Al from the same-cell FC2 models (SOTC cell, 4.05 A).

Every model is Fourier-interpolated from the 32-atom SOTC cell with phonopy
(20^3 mesh); U_iso = <|u|^2>/3 with quantum (Bose--Einstein) occupations.
Models: harmonic FD trial; effective FC2 of the 40-snapshot canonical
ensembles from the harmonic (iteration 0) and updated (iteration 1) trials;
SOTC-D, bootstrapped SOTC-D, and SOTC-D from the updated trial with 2 and 8
force calls (supercell space-group basis).

Writes validation_results/al/robustness/al_uiso_models.json.
"""

from __future__ import annotations

import json
from pathlib import Path
import sys

import numpy as np

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "codes"))
sys.path.insert(0, str(Path(__file__).resolve().parent))

from analyze_designs import N_TRAIN, Fitter, datasets  # noqa: E402
from analyze_primary_archives import PRIMARY, symfc_fc2_basis  # noqa: E402

T = 300.0


def u_iso(phonon, fc):
    phonon.force_constants = fc
    phonon.run_mesh([20, 20, 20], with_eigenvectors=True, is_mesh_symmetry=False)
    phonon.run_thermal_displacements(temperatures=[T], freq_min=1e-3)
    disp = phonon.get_thermal_displacements_dict()["thermal_displacements"][0]
    return float(np.mean(disp))


def main(material="al"):
    import phonopy

    base = ROOT / "validation_results" / material / "primary_cell"
    phonon = phonopy.load(base / "fd" / "phonopy_harmonic.yaml")
    harmonic = np.array(phonon.force_constants)
    sup = phonon.supercell
    fit = Fitter(symfc_fc2_basis(np.asarray(sup.positions), np.asarray(sup.cell),
                                 list(sup.symbols), PRIMARY[material]["cutoff"]))
    data = datasets(material)
    models = {"harmonic_fd": harmonic,
              "effective_iter0": fit(*[x[:N_TRAIN] for x in data["canonical"]]),
              "effective_iter1": fit(*[x[:N_TRAIN] for x in data["canonical1"]])}
    for name in ("sotcd", "boot", "sotcd1"):
        u, f = data[name]
        for c in (2, 8):
            models[f"{name}_{c}"] = fit(u[:c], f[:c])
    report = {"material": material, "temperature_k": T,
              "u_iso_ang2": {k: u_iso(phonon, fc) for k, fc in models.items()}}
    out = ROOT / "validation_results" / material / "robustness" / "al_uiso_models.json"
    out.write_text(json.dumps(report, indent=1) + "\n")
    for k, v in report["u_iso_ang2"].items():
        print(f"{k:16s} {v:.5f}")


if __name__ == "__main__":
    main()
