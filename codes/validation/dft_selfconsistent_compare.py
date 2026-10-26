#!/usr/bin/env python3
"""DFT-level comparison of self-consistent protocols in the SOTC cell.

Reference: the auxiliary FC2 of the largest converged DFT SSCHA run
(``dft_sscha.py``), the self-consistent effective FC2 of the cell. All FC2
models, including the SSCHA ones, are projected onto the supercell
space-group basis within r_cut (in the metric of its thermal displacement
covariance, i.e. as a force regression would see it), so that every protocol is compared in the
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

import functools
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
from sotc.optimal_design import thermal_modes  # noqa: E402
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

    def euclidean_project(fc):
        coef, *_ = np.linalg.lstsq(flat.T, np.asarray(fc).ravel(), rcond=None)
        return np.einsum("k,kijab->ijab", coef, basis)

    n3 = 3 * len(sup.masses)
    as_matrix = lambda fc: np.asarray(fc).transpose(0, 2, 1, 3).reshape(n3, n3)  # noqa: E731
    mats = [as_matrix(b) for b in basis]

    def thermal_metric(fc):
        _, patterns, sigma = thermal_modes(fc, np.asarray(sup.masses), PRIMARY[material]["T"])
        p = patterns.reshape(len(sigma), -1)
        return p.T @ (sigma[:, None] ** 2 * p)

    def project(fc, cov):
        # Thermal-metric projection: the basis FC2 closest to fc in the force
        # it predicts for displacements of covariance cov, which is the
        # quantity a force regression on such displacements estimates.
        # A Euclidean projection weights force constants beyond r_cut
        # differently and shifts the spectrum by ~0.1 THz.
        bs = [b @ cov for b in mats]
        gram = np.array([[np.sum(bk * bl) for bl in mats] for bk in bs])
        rhs = np.array([np.sum(bk * as_matrix(fc)) for bk in bs])
        return np.einsum("k,kijab->ijab", np.linalg.solve(gram, rhs), basis)

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
    cov = thermal_metric(ref_full)
    project = functools.partial(project, cov=cov)
    ref = project(ref_full)
    ref_freq = mesh(ref)
    err = lambda fc: frequency_metrics(ref_freq, mesh(fc))["rmse_thz"]  # noqa: E731
    report = {"material": material, "reference": ref_name,
              "reference_force_calls": runs[ref_name]["calls"][-1],
              "projection": "thermal metric of the reference at T",
              "projection_effect_thz": frequency_metrics(ref_freq, mesh(ref_full))["rmse_thz"],
              "euclidean_vs_thermal_projection_thz": err(euclidean_project(ref_full)),
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
    # Third reference, free of any projection: the DFT forces of the
    # reference SSCHA population refitted in the basis with the same
    # regression (the 256-call effective FC2 of the harmonic-trial ensemble).
    eq = np.asarray(sup.positions)
    with np.load(base / "fd" / "harmonic_reference.npz") as h:
        f_ref0 = np.asarray(h["reference_forces_ev_ang"])
    inv = np.linalg.inv(np.asarray(sup.cell))

    def population(name):
        us, fs = [], []
        for state in sorted((base / "sscha_dft" / name).glob("pop*/state_*/result.npz")):
            with np.load(state) as s:
                d_frac = (np.asarray(s["positions_ang"]) - eq) @ inv
                u = (d_frac - np.rint(d_frac)) @ np.asarray(sup.cell)
                f = np.asarray(s["forces_ev_ang"]) - f_ref0
            us.append(u - u.mean(axis=0))
            fs.append(f - f.mean(axis=0))
        return np.asarray(us), np.asarray(fs)

    pop_fc = fit(*population(ref_name))
    pop_freq = mesh(pop_fc)
    errp = lambda fc: frequency_metrics(pop_freq, mesh(fc))["rmse_thz"]  # noqa: E731
    report["vs_population"] = {
        "reference": f"{ref_name} population refitted",
        "harmonic": errp(project(harmonic_fc)),
        "sscha_reference_projected": errp(ref),
        "iter1_effective_40": errp(iter1),
        "sscha": {n: [{"force_calls": c, "mesh_rmse_thz": errp(project(f))}
                      for f, c in zip(r["fcs"], r["calls"])] for n, r in runs.items()},
        "sscha_refit": {n: errp(fit(*population(n))) for n in runs if n != ref_name},
        "sotcd_sc": {path.parent.name: [{"force_calls": int(c), "mesh_rmse_thz": errp(f)}
                                        for f, c in zip(np.load(path)["force_constants"],
                                                        np.load(path)["force_calls"])]
                     for path in sorted((base / "sotcd_sc_dft").glob("s*/history.npz"))},
    }
    # Iterated sTDEP (last and pooled fits) and self-consistent ZG
    # (``dft_sc_protocols.py``), against all three references.
    report["iterated"] = {}
    for proto in ("stdep", "zgsc"):
        for path in sorted((base / f"{proto}_dft").glob("s*/history.npz")):
            h = np.load(path)
            for fit_kind in ("last", "pooled"):
                key = f"{proto}_{fit_kind}"
                report["iterated"].setdefault(key, {})[path.parent.name] = [
                    {"force_calls": int(c), "vs_reference": err(f), "vs_iter1": err1(f),
                     "vs_population": errp(f)}
                    for f, c in zip(h[f"force_constants_{fit_kind}"], h["force_calls"])]
    # Fourth reference: the converged self-consistent FC2 from large-population
    # sTDEP (``dft_sc_protocols.py <m> stdep <seed> --per-iter 128 --name
    # stdep_reference``), the final last-iteration fits averaged over seeds.
    sc_runs = sorted((base / "stdep_reference").glob("s*/history.npz"))
    if sc_runs:
        finals = [np.load(p)["force_constants_last"] for p in sc_runs]
        sc_fc = np.mean([f[-1] for f in finals], axis=0)
        sc_freq = mesh(sc_fc)
        errs = lambda fc: frequency_metrics(sc_freq, mesh(fc))["rmse_thz"]  # noqa: E731
        between = lambda a, b: frequency_metrics(mesh(a), mesh(b))["rmse_thz"]  # noqa: E731
        vs = {
            "reference": f"stdep_reference, {len(sc_runs)} seeds x "
                         f"{int(np.load(sc_runs[0])['force_calls'][0])} calls per iteration",
            "iteration_changes": [[between(f[k], f[k + 1]) for k in range(len(f) - 1)]
                                  for f in finals],
            "seed_to_seed": between(finals[0][-1], finals[-1][-1]) if len(finals) > 1 else None,
            "harmonic": errs(project(harmonic_fc)),
            "iteration0_population": errs(pop_fc),
            "sscha_reference_projected": errs(ref),
            "iter1_effective_40": errs(iter1),
            "sscha": {n: [{"force_calls": c, "mesh_rmse_thz": errs(project(f))}
                          for f, c in zip(r["fcs"], r["calls"])] for n, r in runs.items()},
            "sscha_refit": {n: errs(fit(*population(n))) for n in runs},
            "sotcd_sc": {p.parent.name: [{"force_calls": int(c), "mesh_rmse_thz": errs(f)}
                                         for f, c in zip(np.load(p)["force_constants"],
                                                         np.load(p)["force_calls"])]
                         for p in sorted((base / "sotcd_sc_dft").glob("s*/history.npz"))},
        }
        # One-shot designs from the harmonic trial: the first b configurations
        # of the deterministic or sequential designs, and the median over 20
        # random subsets of b/2 pairs for canonical sampling.
        oneshot_sc = {}
        for kind in ("sotcd", "zg", "rattle"):
            u_k, f_k = data[kind]
            oneshot_sc[kind] = {str(b): errs(fit(u_k[:b], f_k[:b]))
                                for b in (2, 4, 8, 16) if b <= len(u_k)}
        oneshot_sc["canonical"] = {}
        for b in (2, 4, 8, 16):
            vals = []
            for seed in range(20):
                rng = np.random.default_rng(5000 + 31 * b + seed)
                pairs = rng.choice(N_TRAIN // 2, size=b // 2, replace=False)
                idx = np.ravel(np.stack((2 * pairs, 2 * pairs + 1), axis=1))
                vals.append(errs(fit(data["canonical"][0][idx], data["canonical"][1][idx])))
            oneshot_sc["canonical"][str(b)] = float(np.median(vals))
        vs["oneshot_first_n"] = oneshot_sc
        for key, seeds in report["iterated"].items():
            proto, fit_kind = key.split("_")
            for seed in seeds:
                h = np.load(base / f"{proto}_dft" / seed / "history.npz")
                for row, f in zip(seeds[seed], h[f"force_constants_{fit_kind}"]):
                    row["vs_sc"] = errs(f)
        report["vs_sc"] = vs
    u_b, f_b = data["boot"]
    d["boot"] ={str(b): err(fit(u_b[:b], f_b[:b])) for b in (2, 4, 8, 16) if b <= len(u_b)}
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
