#!/usr/bin/env python3
"""DFT-level comparison of self-consistent protocols in the SOTC cell.

Reference: the auxiliary FC2 of the largest converged DFT SSCHA run
(``dft_sscha.py``), the self-consistent effective FC2 of the cell. All FC2
models, including the SSCHA ones, are projected onto the supercell
space-group basis within r_cut, so that every protocol is compared in the
same model space; the effect of the projection on the reference is
reported separately.

Scored (mesh RMSE, 10^3 mesh, THz):
* SSCHA with smaller populations, after every population (force calls);
* SOTC-D-SC (``dft_sotcd_sc.py``), after every pair, per seed;
* bootstrapped SOTC-D, two-iteration sTDEP (last, pooled), one-shot
  canonical pairs (20 subsets), all from the existing archives;
* the 40-snapshot iteration-1 effective FC2.

Writes validation_results/<m>/robustness/dft_selfconsistent.json.
"""

from __future__ import annotations

import json
from pathlib import Path
import sys

import numpy as np

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "codes"))
sys.path.insert(0, str(Path(__file__).resolve().parent))

from analyze_designs import N_TRAIN, Fitter, datasets, stdep_blocks  # noqa: E402
from analyze_primary_archives import (  # noqa: E402
    PRIMARY, frequency_metrics, mesh_points, symfc_fc2_basis,
)
from surrogate_design_study import mesh_frequencies  # noqa: E402

BUDGETS = (4, 8, 16, 32)


def main(material):
    import phonopy

    base = ROOT / "validation_results" / material / "primary_cell"
    phonon = phonopy.load(base / "fd" / "phonopy_harmonic.yaml")
    harmonic_fc = np.array(phonon.force_constants)
    sup = phonon.supercell
    basis = symfc_fc2_basis(np.asarray(sup.positions), np.asarray(sup.cell),
                            list(sup.symbols), PRIMARY[material]["cutoff"])
    flat = basis.reshape(len(basis), -1)

    def project(fc):
        coef, *_ = np.linalg.lstsq(flat.T, np.asarray(fc).ravel(), rcond=None)
        return np.einsum("k,kijab->ijab", coef, basis)

    qmesh = mesh_points(10)
    mesh = lambda fc: mesh_frequencies(phonon, fc, qmesh)  # noqa: E731
    runs = {}
    for path in sorted((base / "sscha_dft").glob("N*_s*/history.npz")):
        h = np.load(path)
        runs[path.parent.name] = {"fcs": h["force_constants"],
                                  "calls": h["force_calls"].tolist(),
                                  "converged": h["converged"].tolist()}
    ref_name = max((k for k in runs if runs[k]["converged"][-1]),
                   key=lambda k: int(k[1:].split("_")[0]))
    ref_full = runs[ref_name]["fcs"][-1]
    ref = project(ref_full)
    ref_freq = mesh(ref)
    err = lambda fc: frequency_metrics(ref_freq, mesh(fc))["rmse_thz"]  # noqa: E731
    report = {"material": material, "reference": ref_name,
              "reference_force_calls": runs[ref_name]["calls"][-1],
              "projection_effect_thz": frequency_metrics(ref_freq, mesh(ref_full))["rmse_thz"],
              "harmonic_vs_reference_thz": err(project(harmonic_fc)),
              "sscha": {}, "sotcd_sc": {}, "designs": {}}
    for name, r in runs.items():
        report["sscha"][name] = [{"force_calls": c, "converged": v, "mesh_rmse_thz": err(project(f))}
                                 for f, c, v in zip(r["fcs"], r["calls"], r["converged"])]
    for path in sorted((base / "sotcd_sc_dft").glob("s*/history.npz")):
        h = np.load(path)
        report["sotcd_sc"][path.parent.name] = [
            {"force_calls": int(c), "mesh_rmse_thz": err(f)}
            for f, c in zip(h["force_constants"], h["force_calls"])]

    fit = Fitter(basis)
    data = datasets(material)
    d = report["designs"]
    iter1 = fit(*[x[:N_TRAIN] for x in data["canonical1"]])
    d["iter1_effective_40"] = err(iter1)
    # Second reference: the 40-snapshot iteration-1 effective FC2 used in the
    # DFT design tables, so SSCHA and SOTC-D-SC are on the same footing.
    iter1_freq = mesh(iter1)
    err1 = lambda fc: frequency_metrics(iter1_freq, mesh(fc))["rmse_thz"]  # noqa: E731
    report["vs_iter1"] = {
        "harmonic": err1(project(harmonic_fc)),
        "sscha": {n: [{"force_calls": c, "mesh_rmse_thz": err1(project(f))}
                      for f, c in zip(r["fcs"], r["calls"])] for n, r in runs.items()},
        "sotcd_sc": {path.parent.name: [{"force_calls": int(c), "mesh_rmse_thz": err1(f)}
                                        for f, c in zip(np.load(path)["force_constants"],
                                                        np.load(path)["force_calls"])]
                     for path in sorted((base / "sotcd_sc_dft").glob("s*/history.npz"))},
    }
    u_b, f_b = data["boot"]
    d["boot"] = {str(b): err(fit(u_b[:b], f_b[:b])) for b in (2, 4, 8, 16) if b <= len(u_b)}
    u0, f0 = data["canonical"]
    oneshot = {}
    for b in BUDGETS[:3]:
        vals = []
        for seed in range(20):
            rng = np.random.default_rng(5000 + 31 * b + seed)
            pairs = rng.choice(N_TRAIN // 2, size=b // 2, replace=False)
            idx = np.ravel(np.stack((2 * pairs, 2 * pairs + 1), axis=1))
            vals.append(err(fit(u0[idx], f0[idx])))
        oneshot[str(b)] = float(np.median(vals))
    d["canonical_oneshot_median"] = oneshot
    u_st, f_st = data["stdep"]
    start, last, pool = 0, {}, {}
    for seed, budget, first in stdep_blocks(material):
        n2 = budget // 2
        u2, f2 = u_st[start:start + n2], f_st[start:start + n2]
        start += n2
        last.setdefault(str(budget), []).append(err(fit(u2, f2)))
        pool.setdefault(str(budget), []).append(err(fit(np.concatenate([u0[first], u2]),
                                                        np.concatenate([f0[first], f2]))))
    d["stdep_last_median"] = {b: float(np.median(v)) for b, v in last.items()}
    d["stdep_pool_median"] = {b: float(np.median(v)) for b, v in pool.items()}
    out = ROOT / "validation_results" / material / "robustness" / "dft_selfconsistent.json"
    out.write_text(json.dumps(report, indent=1) + "\n")
    print(json.dumps({k: v for k, v in report.items() if k not in ("sscha", "sotcd_sc")}, indent=1))
    for name, rows in report["sscha"].items():
        print("SSCHA", name, [(r["force_calls"], round(r["mesh_rmse_thz"], 3)) for r in rows])
    for name, rows in report["sotcd_sc"].items():
        print("SOTC-D-SC", name, [(r["force_calls"], round(r["mesh_rmse_thz"], 3))
                                  for r in rows if r["force_calls"] in BUDGETS])
    v = report["vs_iter1"]
    print("vs iter1: harmonic", round(v["harmonic"], 3))
    for name, rows in v["sscha"].items():
        print("  SSCHA", name, [(r["force_calls"], round(r["mesh_rmse_thz"], 3)) for r in rows])
    for name, rows in v["sotcd_sc"].items():
        print("  SOTC-D-SC", name, [(r["force_calls"], round(r["mesh_rmse_thz"], 3))
                                    for r in rows if r["force_calls"] in BUDGETS])


if __name__ == "__main__":
    main(sys.argv[1] if len(sys.argv) > 1 else "al")
