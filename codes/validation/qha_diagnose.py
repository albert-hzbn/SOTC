#!/usr/bin/env python3
"""Diagnosis of the self-consistent thermal expansion (Sec. V K).

For each volume-grid point the designed configurations, forces, and
energies are stored (``dft_qha.py``). This script separates the error
sources of alpha(T) without new DFT calculations:

* statistical noise: the 16 +- pairs of every volume are resampled with
  replacement (independently per volume), the FC2 is refitted (ordinary
  least squares in the supercell space-group basis, as in the protocol),
  and F_h, S, the energy term, and alpha are recomputed;
* the free-energy estimator: besides the first-order SCHA correction
  <dU>, dU = E - E0 - u.Phi.u/2, the second-order cumulant
  -Var(dU)/(2 k_B T) of free-energy perturbation is added;
* the fit form: cubic and quartic F(V).

Writes validation_results/<m>/robustness/qha_diagnosis_<grid>.json.

Usage: qha_diagnose.py <material> <grid_dir> [--boot 40] [--calls 32]
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys

import numpy as np

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "codes"))
sys.path.insert(0, str(Path(__file__).resolve().parent))

from analyze_designs import Fitter  # noqa: E402
from analyze_primary_archives import PRIMARY, symfc_fc2_basis  # noqa: E402
from dft_qha import scaled_phonon  # noqa: E402
from dft_qha_analyze import EV_PER_KJMOL, KB_EV, birch_murnaghan  # noqa: E402


def expansion(v, f, s, degree=3):
    c = np.polyfit(v, f, degree)
    roots = np.roots(np.polyder(c))
    roots = roots[np.isreal(roots)].real
    roots = roots[np.polyval(np.polyder(c, 2), roots) > 0]
    if not len(roots):
        return np.nan, np.nan
    v0 = float(roots[np.argmin(abs(roots - v.mean()))])
    bulk = v0 * np.polyval(np.polyder(c, 2), v0)
    ds = np.polyval(np.polyder(np.polyfit(v, s, 2)), v0)
    return 1e6 * ds / (3.0 * bulk), v0


def main():
    from scipy.optimize import curve_fit

    parser = argparse.ArgumentParser()
    parser.add_argument("material")
    parser.add_argument("grid")
    parser.add_argument("--boot", type=int, default=40)
    parser.add_argument("--calls", type=int, default=32)
    parser.add_argument("--temps", type=float, nargs="*", default=None)
    args = parser.parse_args()
    m = args.material
    base = ROOT / "validation_results" / m
    eos = json.loads((base / "fd_references" / "eos_summary.json").read_text())
    vols, ens = np.asarray(eos["volumes_ang3"]), np.asarray(eos["energies_ev"])
    popt, _ = curve_fit(birch_murnaghan, vols, ens,
                        p0=(ens.min(), vols[np.argmin(ens)], 0.5, 4.0), maxfev=20000)

    points = []
    for d in sorted((base / "primary_cell" / args.grid).glob("a*")):
        temps = sorted(float(p.stem[1:]) for p in d.glob("T*.npz"))
        if temps:
            points.append((float(d.name[1:]), d, temps))
    temps = [t for t in points[0][2] if args.temps is None or t in args.temps]
    a = np.asarray([p[0] for p in points])
    v = a ** 3 / 4.0
    e_st = birch_murnaghan(v, *popt)
    phonons = [scaled_phonon(m, x) for x in a]
    n_prim = len(phonons[0].primitive)
    fitters = []
    for ph in phonons:
        sup = ph.supercell
        fitters.append(Fitter(symfc_fc2_basis(np.asarray(sup.positions), np.asarray(sup.cell),
                                              list(sup.symbols), PRIMARY[m]["cutoff"])))

    def thermal(ph, fc, t):
        ph.force_constants = fc
        ph.run_mesh([20, 20, 20])
        ph.run_thermal_properties(temperatures=[t])
        p = ph.get_thermal_properties_dict()
        return (float(p["free_energy"][0]) * EV_PER_KJMOL,
                float(p["entropy"][0]) * EV_PER_KJMOL / 1000.0)

    rng = np.random.default_rng(0)
    report = {"material": m, "grid": args.grid, "calls": args.calls,
              "lattice_ang": a.tolist(), "temperatures": {}}
    for t in temps:
        data = []
        for (_, d, _), ph, fit in zip(points, phonons, fitters):
            r = np.load(d / f"T{t:.0f}.npz")
            u = r["displacements_ang"][:args.calls]
            f = r["delta_forces_ev_ang"][:args.calls]
            e = r["delta_energies_ev"][:args.calls]
            data.append((u, f, e, len(u[0]), ph, fit))

        def evaluate(pair_index):
            fh, s, du1, du2, check = [], [], [], [], []
            for (u, f, e, n, ph, fit), idx in zip(data, pair_index):
                rows = np.concatenate([[2 * k, 2 * k + 1] for k in idx])
                phi = fit(u[rows], f[rows])
                fq, sq = thermal(ph, phi, t)
                half = rows[len(rows) // 2:]
                du = e[half] - 0.5 * np.einsum("sia,ijab,sjb->s", u[half], phi, u[half])
                fh.append(fq)
                s.append(sq)
                du1.append(n_prim * du.mean() / n)
                du2.append(-n_prim * du.var() / (2 * KB_EV * t) / n)
                check.append(phi)
            fh, s, du1, du2 = map(np.asarray, (fh, s, du1, du2))
            fit1 = lambda y: np.polyval(np.polyfit(v, y, 1), v)  # noqa: E731
            f_sc = e_st + fh + fit1(du1)
            return {"cubic": expansion(v, f_sc, s)[0],
                    "quartic": expansion(v, f_sc, s, 4)[0],
                    "fep2": expansion(v, f_sc + fit1(du2), s)[0],
                    "no_du": expansion(v, e_st + fh, s)[0],
                    "f_sc": f_sc, "du1": du1, "du2": du2}, check

        n_pairs = args.calls // 2
        full, phis = evaluate([np.arange(n_pairs)] * len(data))
        stored = [np.load(d / f"T{t:.0f}.npz")["force_constants"][n_pairs - 1] for _, d, _ in points]
        refit_dev = max(float(np.abs(p - q).max()) for p, q in zip(phis, stored))
        boots = [evaluate([rng.integers(0, n_pairs, n_pairs) for _ in data])[0]
                 for _ in range(args.boot)]
        stat = {k: [float(np.nanmean([b[k] for b in boots])), float(np.nanstd([b[k] for b in boots]))]
                for k in ("cubic", "quartic", "fep2", "no_du")}
        f_noise = np.std([b["f_sc"] - b["f_sc"].mean() for b in boots], axis=0)
        curvature = np.abs(np.diff(full["f_sc"], 2))
        report["temperatures"][f"{t:.0f}"] = {
            "alpha_1e6": {k: float(full[k]) for k in ("cubic", "quartic", "fep2", "no_du")},
            "bootstrap_mean_std_1e6": stat,
            "du1_mev_per_prim": (1000 * full["du1"]).tolist(),
            "du2_mev_per_prim": (1000 * full["du2"]).tolist(),
            "f_noise_mev_per_prim": (1000 * f_noise).tolist(),
            "f_second_difference_mev_per_prim": (1000 * curvature).tolist(),
            "refit_vs_stored_max_ev_ang2": refit_dev}
        print(f"T={t:.0f} alpha cubic {full['cubic']:.1f} (boot {stat['cubic'][0]:.1f}"
              f"+-{stat['cubic'][1]:.1f}) quartic {full['quartic']:.1f}"
              f"+-{stat['quartic'][1]:.1f} fep2 {full['fep2']:.1f}; F noise "
              f"{1000 * f_noise.mean():.2f} meV, 2nd diff {1000 * curvature.mean():.2f} meV;"
              f" refit dev {refit_dev:.1e}", flush=True)
    out = base / "robustness" / f"qha_diagnosis_{args.grid}_{args.calls}.json"
    out.write_text(json.dumps(report, indent=1) + "\n")


if __name__ == "__main__":
    main()
