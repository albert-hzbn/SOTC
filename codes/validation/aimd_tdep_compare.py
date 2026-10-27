#!/usr/bin/env python3
"""AIMD-TDEP baseline in the SOTC cell (``aimd_tdep.py`` trajectories).

TDEP-type least-squares FC2 fits (supercell space-group basis, as for every
other protocol) to the first n production steps of each trajectory after
discarding N_EQ equilibration steps; every MD step is one force call, and
the equilibration is counted in the cost. The converged AIMD-TDEP FC2 is
the fit to all production steps of all trajectories (classical anharmonic
ensemble at T). Errors are reported against this limit and against the
quantum self-consistent references (iteration-1 effective FC2, the
N=256 DFT SSCHA result, and the converged sTDEP reference of
``dft_selfconsistent_compare.py``), so that the classical/quantum difference
is separated from the sampling error. Each fit is done twice: FC2 alone, and
FC2 jointly with FC3 (``fit_symfc_model``, FC3 cutoff FC3_CUTOFF). MD samples
have nonzero odd displacement moments, so an FC2-only fit absorbs part of
the cubic forces; the joint fit removes that.

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
from sotc.symfc_fit import fit_symfc_model, symfc_atoms  # noqa: E402

N_EQ = 250
LENGTHS = (16, 32, 64, 128, 256, 512, 1024)
FC3_CUTOFF = 4.0  # Angstrom, within half of the shortest cell length for Al and Si


def main(material):
    import phonopy

    base = ROOT / "validation_results" / material / "primary_cell"
    phonon = phonopy.load(base / "fd" / "phonopy_harmonic.yaml")
    sup = phonon.supercell
    basis = symfc_fc2_basis(np.asarray(sup.positions), np.asarray(sup.cell),
                            list(sup.symbols), PRIMARY[material]["cutoff"])
    flat = basis.reshape(len(basis), -1)
    fit2 = Fitter(basis)
    atoms = symfc_atoms(sup.numbers, sup.scaled_positions, sup.cell)
    cutoffs = {2: PRIMARY[material]["cutoff"], 3: FC3_CUTOFF}

    def project(fc):
        coef, *_ = np.linalg.lstsq(flat.T, np.asarray(fc).ravel(), rcond=None)
        return np.einsum("k,kijab->ijab", coef, basis)

    def fit23(u, f):
        model, _ = fit_symfc_model(atoms, u, f, [2, 3], cutoffs, batch_size=20)
        return project(model.force_constants[2])

    qmesh = mesh_points(10)
    mesh = lambda fc: np.array(mesh_frequencies(phonon, fc, qmesh))  # noqa: E731
    rms = lambda a, b: frequency_metrics(a, b)["rmse_thz"]  # noqa: E731

    trajs = []
    for path in sorted((base / "aimd").glob("traj*/trajectory.npz")):
        t = np.load(path)
        trajs.append((t["displacements_ang"][N_EQ:], t["delta_forces_ev_ang"][N_EQ:]))
    u_all = np.concatenate([u for u, _ in trajs])
    f_all = np.concatenate([f for _, f in trajs])
    limit = mesh(fit2(u_all, f_all))
    limit23 = mesh(fit23(u_all, f_all))
    canonical1 = datasets(material)["canonical1"]
    iter1 = mesh(fit2(*[x[:N_TRAIN] for x in canonical1])) if canonical1 else None
    sscha_path = base / "sscha_dft" / "N256_s1" / "history.npz"
    refs = {"aimd_limit": limit, "aimd_limit_fc23": limit23}
    if iter1 is not None:
        refs["iter1_quantum"] = iter1
    if sscha_path.exists():
        refs["sscha256_quantum"] = mesh(project(np.load(sscha_path)["force_constants"][-1]))
    sc_runs = sorted((base / "stdep_reference").glob("s*/history.npz"))
    if sc_runs:
        refs["sc_reference"] = mesh(np.mean([np.load(p)["force_constants_last"][-1]
                                             for p in sc_runs], axis=0))
    harmonic = mesh(project(np.load(base / "fd" / "harmonic_reference.npz")
                            ["force_constants_ev_ang2"]))
    report = {"material": material, "n_equilibration": N_EQ,
              "n_production_total": int(len(u_all)), "fc3_cutoff_ang": FC3_CUTOFF,
              "u_rms_ang": float(np.sqrt((u_all ** 2).sum(-1).mean())),
              "mean_frequency_thz": {"harmonic": float(harmonic.mean()),
                                     "aimd_limit": float(limit.mean()),
                                     "aimd_limit_fc23": float(limit23.mean())},
              "limits": {f"{name}_vs_{k}": rms(r, lim) for name, lim in
                         (("fc2", limit), ("fc23", limit23))
                         for k, r in [("harmonic", harmonic)] + list(refs.items())
                         if not k.startswith("aimd")},
              "fc2_vs_fc23_limit_thz": rms(limit, limit23),
              "convergence": {}, "convergence_fc23": {}}
    if "sc_reference" in refs:
        report["mean_frequency_thz"]["sc_reference"] = float(refs["sc_reference"].mean())
    for n in LENGTHS:
        if n > min(len(u) for u, _ in trajs):
            break
        for key, fitter in (("convergence", fit2), ("convergence_fc23", fit23)):
            rows = {k: [] for k in refs}
            for u, f in trajs:
                freq = mesh(fitter(u[:n], f[:n]))
                for k, r in refs.items():
                    rows[k].append(rms(r, freq))
            report[key][str(n + N_EQ)] = {k: float(np.median(v)) for k, v in rows.items()}
            print(key, n + N_EQ, {k: round(float(np.median(v)), 3) for k, v in rows.items()},
                  flush=True)
    out = ROOT / "validation_results" / material / "robustness" / "aimd_tdep.json"
    out.write_text(json.dumps(report, indent=1) + "\n")
    print({k: v for k, v in report.items() if k != "convergence"})


if __name__ == "__main__":
    main(sys.argv[1] if len(sys.argv) > 1 else "al")
