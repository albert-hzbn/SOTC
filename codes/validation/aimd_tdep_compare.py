#!/usr/bin/env python3
"""AIMD-TDEP baseline in the SOTC cell (``aimd_tdep.py`` trajectories).

TDEP-type least-squares FC2 fits (supercell space-group basis, as for every
other protocol) to the first n production steps of each trajectory after
discarding N_EQ equilibration steps; every MD step is one force call, and
the equilibration is counted in the cost. The converged AIMD-TDEP FC2 is
the fit to all production steps of all trajectories (classical anharmonic
ensemble at T). Errors are reported against this limit and against the
quantum self-consistent references (iteration-1 effective FC2 and the
N=256 DFT SSCHA result), so that the classical/quantum difference is
separated from the sampling error.

Writes validation_results/<m>/robustness/aimd_tdep.json.
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
    PRIMARY, frequency_metrics, mesh_points, symfc_fc2_basis,
)
from surrogate_design_study import mesh_frequencies  # noqa: E402

N_EQ = 250
LENGTHS = (16, 32, 64, 128, 256, 512, 1024)


def main(material):
    import phonopy

    base = ROOT / "validation_results" / material / "primary_cell"
    phonon = phonopy.load(base / "fd" / "phonopy_harmonic.yaml")
    sup = phonon.supercell
    basis = symfc_fc2_basis(np.asarray(sup.positions), np.asarray(sup.cell),
                            list(sup.symbols), PRIMARY[material]["cutoff"])
    flat = basis.reshape(len(basis), -1)
    fit = Fitter(basis)
    qmesh = mesh_points(10)
    mesh = lambda fc: mesh_frequencies(phonon, fc, qmesh)  # noqa: E731
    rms = lambda a, b: frequency_metrics(a, b)["rmse_thz"]  # noqa: E731

    trajs = []
    for path in sorted((base / "aimd").glob("traj*/trajectory.npz")):
        t = np.load(path)
        trajs.append((t["displacements_ang"][N_EQ:], t["delta_forces_ev_ang"][N_EQ:]))
    u_all = np.concatenate([u for u, _ in trajs])
    f_all = np.concatenate([f for _, f in trajs])
    limit = mesh(fit(u_all, f_all))
    iter1 = mesh(fit(*[x[:N_TRAIN] for x in datasets(material)["canonical1"]]))
    sscha_path = base / "sscha_dft" / "N256_s1" / "history.npz"
    refs = {"aimd_limit": limit, "iter1_quantum": iter1}
    if sscha_path.exists():
        fc = np.load(sscha_path)["force_constants"][-1]
        coef, *_ = np.linalg.lstsq(flat.T, fc.ravel(), rcond=None)
        refs["sscha256_quantum"] = mesh(np.einsum("k,kijab->ijab", coef, basis))
    report = {"material": material, "n_equilibration": N_EQ,
              "n_production_total": int(len(u_all)),
              "limit_vs_iter1_thz": rms(iter1, limit),
              "limit_vs_harmonic_thz": rms(mesh(np.array(np.load(
                  base / "fd" / "harmonic_reference.npz")["force_constants_ev_ang2"])), limit),
              "u_rms_ang": float(np.sqrt((u_all ** 2).sum(-1).mean())),
              "convergence": {}}
    for n in LENGTHS:
        if n > min(len(u) for u, _ in trajs):
            break
        rows = {k: [] for k in refs}
        for u, f in trajs:
            freq = mesh(fit(u[:n], f[:n]))
            for k, r in refs.items():
                rows[k].append(rms(r, freq))
        report["convergence"][str(n + N_EQ)] = {k: float(np.median(v)) for k, v in rows.items()}
        print(n + N_EQ, {k: round(float(np.median(v)), 3) for k, v in rows.items()}, flush=True)
    out = ROOT / "validation_results" / material / "robustness" / "aimd_tdep.json"
    out.write_text(json.dumps(report, indent=1) + "\n")
    print({k: v for k, v in report.items() if k != "convergence"})


if __name__ == "__main__":
    main(sys.argv[1] if len(sys.argv) > 1 else "al")
