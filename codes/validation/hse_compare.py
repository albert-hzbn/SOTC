#!/usr/bin/env python3
"""HSE06 flagship: SOTC-D-SC against sTDEP with a hybrid functional.

A self-consistent HSE06 reference is out of reach (hundreds of hybrid calls),
so the reference is composed as

    FC2_ref(HSE) = FC2_SC(PBE) + [FC2_FD(HSE) - FC2_FD(PBE)],

the self-consistent PBE reference (``<pbe>/primary_cell/stdep_reference``,
mean of the last-iteration fits of iterations >= 1, as in scaling_compare.py)
plus the harmonic HSE - PBE correction from finite displacements in the same
cell with the same pseudopotentials. It assumes that the anharmonic
renormalization is the same in both functionals. As a check that does not
rely on that assumption, every run is also scored against the HSE pooled
sTDEP fixed point (seed mean of the pooled fits over the second half of the
budget).

For SOTC-D-SC (3 seeds) and sTDEP (3 seeds, last and pooled fits) in HSE the
10^3-mesh RMSE is reported after every call, at 4/8/16 calls, and the calls
after which it stays below 0.1 / 0.05 THz. The same quantities for the PBE
runs against the PBE reference are listed for comparison.

Usage: hse_compare.py [si_hse|nacl_hse ...]
Writes validation_results/summaries/<material>/robustness/hse_compare.json.
"""

from __future__ import annotations

import json
from pathlib import Path
import sys

import numpy as np

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "codes"))
sys.path.insert(0, str(Path(__file__).resolve().parent))

from analyze_primary_archives import frequency_metrics, mesh_points  # noqa: E402
from surrogate_design_study import mesh_frequencies  # noqa: E402

# HSE material -> (PBE material of the self-consistent reference and runs,
# PBE material of the harmonic FD with the HSE pseudopotentials)
PBE = {"si_hse": ("si", "si"), "nacl_hse": ("nacl_nc", "nacl_nc")}
BUDGETS = (4, 8, 16)
THRESHOLDS = (0.1, 0.05)


def calls_to(calls, errors, threshold):
    good = [e < threshold for e in errors]
    for k in range(len(good)):
        if all(good[k:]):
            return int(calls[k])
    return None


def load_runs(base):
    runs = {}
    for path in sorted((base / "sotcd_sc_dft").glob("s*/history.npz")):
        h = np.load(path)
        runs[f"sotcd_sc/{path.parent.name}"] = (h["force_calls"], h["force_constants"])
    for path in sorted((base / "stdep_dft").glob("s*/history.npz")):
        h = np.load(path)
        for kind in ("last", "pooled"):
            runs[f"stdep_{kind}/{path.parent.name}"] = (h["force_calls"],
                                                        h[f"force_constants_{kind}"])
    return runs


def fixed_point(runs, prefix):
    finals = []
    for name, (calls, fcs) in runs.items():
        if name.startswith(prefix + "/"):
            finals.append(np.mean([f for f, c in zip(fcs, calls) if c > calls[-1] / 2], axis=0))
    return np.mean(finals, axis=0) if finals else None


def score(runs, err, budget_max):
    out = {}
    for name, (calls, fcs) in runs.items():
        keep = calls <= budget_max
        calls, fcs = calls[keep], fcs[keep]
        errors = [err(f) for f in fcs]
        entry = {"calls": [int(c) for c in calls], "rmse_thz": errors}
        for b in BUDGETS:
            k = np.searchsorted(calls, b, side="right") - 1
            entry[f"at_{b}"] = errors[k] if k >= 0 else None
        for t in THRESHOLDS:
            entry[f"calls_to_{t}"] = calls_to(calls, errors, t)
        out[name] = entry
    return out


def summary(scored):
    """Seed mean and spread of the error at each budget, per protocol."""
    out = {}
    for name, entry in scored.items():
        proto = name.split("/")[0]
        for b in BUDGETS:
            if entry[f"at_{b}"] is not None:
                out.setdefault(proto, {}).setdefault(str(b), []).append(entry[f"at_{b}"])
    return {p: {b: [float(np.mean(v)), float(np.std(v))] for b, v in d.items()}
            for p, d in out.items()}


def analyze(material):
    import phonopy

    pbe, pbe_fd = PBE[material]
    res = ROOT / "validation_results"
    base = res / material / "primary_cell"
    pbe_base = res / pbe / "primary_cell"
    phonon = phonopy.load(base / "fd" / "phonopy_harmonic.yaml")
    phonon_pbe = phonopy.load(res / pbe_fd / "primary_cell" / "fd" / "phonopy_harmonic.yaml")
    qmesh = mesh_points(10)
    mesh = lambda fc: mesh_frequencies(phonon, fc, qmesh)  # noqa: E731

    refs = sorted((pbe_base / "stdep_reference").glob("s*/history.npz"))
    lasts = [np.load(p)["force_constants_last"] for p in refs]
    sc_pbe = np.mean([f for h in lasts for f in h[1:]], axis=0)
    fd_hse = np.asarray(phonon.force_constants)
    fd_pbe = np.asarray(phonon_pbe.force_constants)
    ref = sc_pbe + fd_hse - fd_pbe
    ref_mesh = mesh(ref)

    runs = load_runs(base)
    budget = int(max(c[-1] for c, _ in runs.values()))
    fp_stdep = fixed_point(runs, "stdep_pooled")
    fp_sotcd = fixed_point(runs, "sotcd_sc")
    fp_mesh = mesh(fp_stdep)
    rmse = lambda a, b: frequency_metrics(a, b)["rmse_thz"]  # noqa: E731

    report = {
        "material": material, "budget": budget,
        "reference": {"pbe_sc_runs": [p.parent.name for p in refs],
                      "pbe_fd": pbe_fd,
                      "hse_minus_pbe_harmonic_thz": rmse(mesh(fd_pbe), mesh(fd_hse)),
                      "pbe_anharmonic_shift_thz": rmse(mesh(fd_pbe), mesh(sc_pbe)),
                      "stdep_pooled_fixed_point_vs_reference_thz": rmse(ref_mesh, fp_mesh)},
        "fixed_point": {"sotcd_sc_vs_reference_thz": rmse(ref_mesh, mesh(fp_sotcd)),
                        "stdep_pooled_vs_reference_thz": rmse(ref_mesh, fp_mesh),
                        "sotcd_sc_vs_stdep_pooled_thz": rmse(fp_mesh, mesh(fp_sotcd))},
        "hse_runs": score(runs, lambda fc: rmse(ref_mesh, mesh(fc)), budget),
        "hse_runs_vs_stdep_fixed_point": summary(
            score(runs, lambda fc: rmse(fp_mesh, mesh(fc)), budget)),
    }
    report["hse_summary"] = summary(report["hse_runs"])
    pbe_mesh = mesh(sc_pbe)
    pbe_runs = load_runs(pbe_base)
    if pbe_runs:
        report["pbe_summary"] = summary(score(pbe_runs, lambda fc: rmse(pbe_mesh, mesh(fc)),
                                              budget))
    out = res / "summaries" / material / "robustness"
    out.mkdir(parents=True, exist_ok=True)
    (out / "hse_compare.json").write_text(json.dumps(report, indent=1) + "\n")
    return report


def main():
    for material in sys.argv[1:] or ["si_hse"]:
        r = analyze(material)
        print(material, json.dumps({k: r[k] for k in ("reference", "fixed_point")}, indent=1))
        for key in ("hse_summary", "hse_runs_vs_stdep_fixed_point", "pbe_summary"):
            print(key, json.dumps(r.get(key), indent=1))
        for name, e in r["hse_runs"].items():
            print(f"  {name:20s}", [round(x, 3) for x in e["rmse_thz"]],
                  "to0.1", e["calls_to_0.1"], "to0.05", e["calls_to_0.05"])


if __name__ == "__main__":
    main()
