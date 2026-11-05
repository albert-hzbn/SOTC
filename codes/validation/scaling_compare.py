#!/usr/bin/env python3
"""Scaling study: SOTC-D-SC, sTDEP, and one-shot designs against the
self-consistent reference in cells of increasing size.

For each cell (Al 300 K: 32/108/256 atoms; Si 600 K: 64/128/216; NaCl
700 K: 128/216), all scored by the 10^3-mesh frequency RMSE against the
chunked sTDEP reference (``stdep_reference``, ``dft_sc_reference_chunked.py``;
for the base cells the 128-call ``dft_sc_protocols.py`` references), taken as
the mean of the last-iteration fits of iterations >= 1 (iteration 0 is drawn
from the harmonic trial):

* SOTC-D-SC (3 seeds) and sTDEP (3 seeds, last and pooled fits) after every
  call, and the calls after which the error stays below 0.1 / 0.05 THz;
* one-shot designs from the harmonic trial: the first b SOTC-D
  configurations, and the median over 20 random subsets of b/2 canonical
  pairs;
* the O(1/M) bias of Eq. (8): the self-consistent SOTC-D fixed point, the
  mean over seeds of the fits after the second half of the budget, against
  the same mean for pooled sTDEP. The difference should fall with the number
  of modes M = 3N, while both carry the residual sampling noise.

Writes validation_results/summaries/scaling/scaling_compare.json.
"""

from __future__ import annotations

import json
from pathlib import Path
import sys

import numpy as np

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "codes"))
sys.path.insert(0, str(Path(__file__).resolve().parent))

from analyze_designs import Fitter, load_stage  # noqa: E402
from analyze_primary_archives import (  # noqa: E402
    PRIMARY, frequency_metrics, mesh_points, symfc_fc2_basis,
)
from surrogate_design_study import mesh_frequencies  # noqa: E402

CELLS = {"al": ["al", "al_108", "al_256"], "si": ["si", "si_128", "si_216"],
         "nacl": ["nacl_128", "nacl_216"]}
BUDGETS = (2, 4, 8, 16, 32)
THRESHOLDS = (0.1, 0.05)


def calls_to(calls, errors, threshold):
    good = [e < threshold for e in errors]
    for k in range(len(good)):
        if all(good[k:]):
            return int(calls[k])
    return None


def at_budget(calls, errors, b):
    calls = np.asarray(calls)
    k = np.searchsorted(calls, b, side="right") - 1
    return float(errors[k]) if k >= 0 else None


def analyze(cell):
    import phonopy

    base = ROOT / "validation_results" / cell / "primary_cell"
    refs = sorted((base / "stdep_reference").glob("s*/history.npz"))
    if not refs:
        return None
    phonon = phonopy.load(base / "fd" / "phonopy_harmonic.yaml")
    sup = phonon.supercell
    basis = symfc_fc2_basis(np.asarray(sup.positions), np.asarray(sup.cell),
                            list(sup.symbols), PRIMARY[cell]["cutoff"])
    qmesh = mesh_points(10)
    mesh = lambda fc: mesh_frequencies(phonon, fc, qmesh)  # noqa: E731
    lasts = [np.load(p)["force_constants_last"] for p in refs]
    ref_fc = np.mean([f for h in lasts for f in h[1:]], axis=0)
    ref = mesh(ref_fc)
    err = lambda fc: frequency_metrics(ref, mesh(fc))["rmse_thz"]  # noqa: E731
    out = {"cell": cell, "n_atoms": len(sup.masses), "n_modes": 3 * len(sup.masses),
           "n_parameters": len(basis),
           "reference": {"runs": [p.parent.name for p in refs],
                         "calls_per_iteration": int(np.load(refs[0])["force_calls"][0]),
                         "iterations": [len(h) for h in lasts],
                         # noise floor: change between consecutive iterations
                         "iteration_changes_thz": [
                             [frequency_metrics(mesh(h[k]), mesh(h[k + 1]))["rmse_thz"]
                              for k in range(len(h) - 1)] for h in lasts]},
           "runs": {}}
    finals = {"sotcd_sc": [], "stdep_pooled": []}
    for proto, kinds in (("sotcd_sc", [None]), ("stdep", ["last", "pooled"])):
        for path in sorted((base / f"{proto}_dft").glob("s*/history.npz")):
            h = np.load(path)
            calls = h["force_calls"].tolist()
            for kind in kinds:
                fcs = h["force_constants"] if kind is None else h[f"force_constants_{kind}"]
                errors = [err(f) for f in fcs]
                name = proto if kind is None else f"{proto}_{kind}"
                out["runs"][f"{name}/{path.parent.name}"] = {
                    "calls": calls, "rmse_thz": errors,
                    **{f"calls_to_{t}": calls_to(calls, errors, t) for t in THRESHOLDS},
                    **{f"at_{b}": at_budget(calls, errors, b) for b in BUDGETS}}
                if name in finals:
                    half = [f for f, c in zip(fcs, calls) if c > calls[-1] / 2]
                    finals[name].append(np.mean(half, axis=0))
    out["fixed_point"] = {k: err(np.mean(v, axis=0)) for k, v in finals.items() if v}
    if all(finals.values()):
        out["fixed_point"]["sotcd_sc_vs_stdep_pooled"] = frequency_metrics(
            mesh(np.mean(finals["stdep_pooled"], axis=0)),
            mesh(np.mean(finals["sotcd_sc"], axis=0)))["rmse_thz"]
    fit = Fitter(basis)
    oneshot = {}
    sotcd = load_stage(cell, "sotcd_displacements.npz", ["sotcd_qe"])
    if sotcd is not None:
        oneshot["sotcd"] = {str(b): err(fit(sotcd[0][:b], sotcd[1][:b]))
                            for b in BUDGETS if b <= len(sotcd[0])}
    canonical = load_stage(cell, "canonical_displacements.npz", ["canonical_qe"])
    if canonical is not None:
        oneshot["canonical_median"] = {}
        n_pairs = len(canonical[0]) // 2
        for b in BUDGETS:
            vals = []
            for seed in range(20):
                rng = np.random.default_rng(5000 + 31 * b + seed)
                pairs = rng.choice(n_pairs, size=b // 2, replace=False)
                idx = np.ravel(np.stack((2 * pairs, 2 * pairs + 1), axis=1))
                vals.append(err(fit(canonical[0][idx], canonical[1][idx])))
            oneshot["canonical_median"][str(b)] = float(np.median(vals))
    out["oneshot"] = oneshot
    return out


def main():
    cells = sys.argv[1:] or [c for group in CELLS.values() for c in group]
    path = ROOT / "validation_results" / "summaries" / "scaling" / "scaling_compare.json"
    path.parent.mkdir(parents=True, exist_ok=True)
    report = json.loads(path.read_text()) if path.exists() else {}
    for cell in cells:
        r = analyze(cell)
        if r is None:
            print(cell, "no reference yet", flush=True)
            continue
        report[cell] = r
        print(cell, r["n_atoms"], "P", r["n_parameters"], "fixed point", r["fixed_point"],
              flush=True)
        for k, v in r["runs"].items():
            print(f"  {k:22s}", [v[f"at_{b}"] and round(v[f"at_{b}"], 3) for b in BUDGETS],
                  "to0.1", v["calls_to_0.1"], "to0.05", v["calls_to_0.05"], flush=True)
        print("  oneshot", r["oneshot"], flush=True)
        path.write_text(json.dumps(report, indent=1) + "\n")


if __name__ == "__main__":
    main()
