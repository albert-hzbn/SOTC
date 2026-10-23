#!/usr/bin/env python3
"""Reference accuracy of the new material classes (diamond Si, hcp Mg).

Si and Mg have no archived SOTC runs, so the reference and effective models
are compared directly: same-cell harmonic FD versus the converged large-cell
FD reference (supercell truncation), the 40-snapshot effective FC2 of
iterations 0 and 1 versus the harmonic models (thermal renormalization and
trial update), and SOTC-D with eight force calls versus the effective FC2 of
its trial level.  All fits use the supercell space-group basis of
``analyze_designs.py``.
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
from analyze_primary_archives import (  # noqa: E402
    MESH, PRIMARY, frequency_metrics, mesh_points, phonopy_frequencies, symfc_fc2_basis,
)
from analyze_robustness import thermo  # noqa: E402

N_DESIGN = 8


def analyze(material):
    import phonopy

    spec = PRIMARY[material]
    vr = ROOT / "validation_results" / material
    qmesh = mesh_points(MESH)
    fd_large = phonopy.load(vr / "fd_references" / "production_large" / "phonopy_harmonic.yaml")
    phonon = phonopy.load(vr / "primary_cell" / "fd" / "phonopy_harmonic.yaml")
    n_prim = len(phonon.primitive.masses)
    mesh = {"fd_large": phonopy_frequencies(fd_large, qmesh),
            "fd_same_cell": phonopy_frequencies(phonon, qmesh)}
    sup = phonon.supercell
    fit = Fitter(symfc_fc2_basis(np.asarray(sup.positions), np.asarray(sup.cell),
                                 list(sup.symbols), spec["cutoff"]))
    data = datasets(material)

    def model_mesh(u, df):
        phonon.force_constants = fit(u, df)
        return phonopy_frequencies(phonon, qmesh)

    for name, pool in (("effective_iter0", "canonical"), ("effective_iter1", "canonical1")):
        if data[pool] is not None:
            mesh[name] = model_mesh(*[x[:N_TRAIN] for x in data[pool]])
    for name in ("sotcd", "sotcd1", "boot"):
        if data[name] is not None:
            mesh[f"{name}_{N_DESIGN}"] = model_mesh(*[x[:N_DESIGN] for x in data[name]])

    pairs = [("fd_same_cell", "fd_large"), ("effective_iter0", "fd_same_cell"),
             ("effective_iter0", "fd_large"), ("effective_iter1", "effective_iter0"),
             ("effective_iter1", "fd_large"), (f"sotcd_{N_DESIGN}", "effective_iter0"),
             (f"sotcd1_{N_DESIGN}", "effective_iter1"), (f"boot_{N_DESIGN}", "effective_iter1")]
    report = {
        "material": material, "temperature_k": spec["T"],
        "supercell_atoms": len(sup.positions),
        "large_supercell_atoms": len(fd_large.supercell.positions),
        "min_frequency_thz": {k: float(v.min()) for k, v in mesh.items()},
        "mesh_rmse_thz": {f"{a}_vs_{b}": frequency_metrics(mesh[b], mesh[a])["rmse_thz"]
                          for a, b in pairs if a in mesh and b in mesh},
        "thermo": {k: thermo(v, spec["T"], n_prim) for k, v in mesh.items()},
    }
    out = vr / "robustness"
    out.mkdir(parents=True, exist_ok=True)
    (out / "new_class_summary.json").write_text(json.dumps(report, indent=1, default=float) + "\n")
    print(json.dumps({k: report[k] for k in ("min_frequency_thz", "mesh_rmse_thz")}, indent=1))


def main():
    for material in sys.argv[1:] or ["si", "mg"]:
        analyze(material)


if __name__ == "__main__":
    main()
