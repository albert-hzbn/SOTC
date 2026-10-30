#!/usr/bin/env python3
"""Numerical checks of the estimator theory (Sec. II E, Supplementary S4).

A. Single coordinate: exact bias and variance of the FC2 slope for
   canonical, paired canonical, and fixed-amplitude (SOTC-D) designs
   (``sotc.design_theory``) against Monte Carlo, and the crossover budget.
B. Multi-parameter variance on the real FC2 design matrices (DFT SOTC cells
   of Al, Si, NaCl, harmonic FD trial): A-criterion tr[(X^T X)^-1] of each
   design relative to SOTC-D, as a function of the number of calls.
C. Fixed-amplitude bias versus the number of modes M: exact enumeration of
   sign patterns for random M-mode quartic models; the bias relative to the
   renormalization falls as 1/M.
D. FC2/FC3 separation: for +-u pairs the FC2--FC3 cross-information
   vanishes exactly, but each pair contributes a single quadratic feature
   vector, so the FC3 block reaches full rank only with twice the calls of
   an unpaired design.

Writes validation_results/summaries/theory_checks.json.
"""

from __future__ import annotations

import itertools
import json
from pathlib import Path
import sys

import numpy as np

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "codes"))
sys.path.insert(0, str(Path(__file__).resolve().parent))

from analyze_primary_archives import PRIMARY, symfc_fc2_basis  # noqa: E402
from sotc import design_theory as th  # noqa: E402
from sotc.optimal_design import (  # noqa: E402
    configurations, information_matrix, sequential_d_optimal, thermal_modes,
)
from surrogate_design_study import gaussian_configs, paired  # noqa: E402

COUNTS = (2, 4, 8, 16, 32, 64)


def part_a(rng):
    sig2, w2, c, d, s2 = 1.0, 4.0, 0.6, 0.2, 0.5
    target = w2 + 3 * d * sig2
    rows = []
    for k in (4, 6, 8, 16, 32, 64):
        for des in ("canonical", "paired", "fixed"):
            if des == "paired" and k < 6:
                continue
            x = th.simulate(k, des, sig2, w2, c, d, s2, 400000, rng)
            rows.append({"calls": k, "design": des,
                         "var_mc": float(x.var()),
                         "var_theory": th.noise_variance(k, des, sig2, s2)
                         + th.cubic_variance(k, des, sig2, c)
                         + th.quartic_variance(k, des, sig2, d),
                         "bias_mc": float(x.mean() - target),
                         "bias_theory": th.quartic_bias(k, des, sig2, d)})
    return {"parameters": {"sigma2": sig2, "w2": w2, "c": c, "d": d, "s2": s2},
            "rows": rows,
            "crossover_vs_paired": th.crossover_calls(sig2, c, d, s2, "paired"),
            "crossover_vs_canonical": th.crossover_calls(sig2, c, d, s2, "canonical")}


def a_criterion(u, basis):
    info = information_matrix(u, basis)
    try:
        return float(np.trace(np.linalg.inv(info)))
    except np.linalg.LinAlgError:
        return float("inf")


def part_b(material, rng, reps=40):
    import phonopy

    base = ROOT / "validation_results" / material / "primary_cell"
    phonon = phonopy.load(base / "fd" / "phonopy_harmonic.yaml")
    sup = phonon.supercell
    basis = symfc_fc2_basis(np.asarray(sup.positions), np.asarray(sup.cell),
                            list(sup.symbols), PRIMARY[material]["cutoff"])
    # Column normalization so that the criterion weights parameters equally.
    norm = np.sqrt((basis.reshape(len(basis), -1) ** 2).sum(axis=1))
    basis = basis / norm[:, None, None, None, None]
    _, patterns, sigma = thermal_modes(np.asarray(phonon.force_constants),
                                       np.asarray(sup.masses), PRIMARY[material]["T"])
    out = {"n_parameters": int(len(basis)), "n_modes": int(len(sigma)), "calls": {}}
    for k in COUNTS:
        vals = {"canonical": [], "paired": [], "random_signs": [], "sotcd": []}
        for r in range(reps):
            vals["canonical"].append(a_criterion(gaussian_configs(rng, k, patterns, sigma), basis))
            vals["paired"].append(a_criterion(paired(gaussian_configs(rng, k // 2, patterns, sigma)),
                                              basis))
            signs = rng.choice([-1.0, 1.0], (k // 2, len(sigma)))
            vals["random_signs"].append(a_criterion(paired(configurations(signs, patterns, sigma)),
                                                    basis))
            if r < 4:  # the D-optimal search is deterministic up to its seed
                s, _ = sequential_d_optimal(patterns, sigma, basis, k // 2, seed=r)
                vals["sotcd"].append(a_criterion(paired(configurations(s, patterns, sigma)), basis))
        med = {key: float(np.median(v)) for key, v in vals.items()}
        out["calls"][str(k)] = {"median_a_criterion": med,
                                "ratio_to_sotcd": {key: med[key] / med["sotcd"] for key in med}}
        print(material, k, {key: round(x, 3) for key, x in out["calls"][str(k)]["ratio_to_sotcd"]
                            .items()}, flush=True)
    return out


def part_c(rng, reps=20):
    """Fixed-amplitude bias of the diagonal FC2 against the number of modes M.

    Quartic model V4 = (1/8) sum_{a!=b} W_ab Q_a^2 Q_b^2 + (1/24) sum_a W_aa Q_a^4
    with random positive W and sigma = 1. The Gaussian-averaged curvature of
    mode a is (1/2) sum_b W_ab. The regression of f_a on Q_a is averaged
    exactly over all 2^M sign patterns of +-1 amplitudes."""
    out = {}
    for m in (2, 4, 6, 8, 10, 12):
        rel, preds = [], []
        for _ in range(reps):
            w = np.abs(rng.normal(1.0, 0.3, (m, m)))
            w = 0.5 * (w + w.T)
            off = w - np.diag(np.diag(w))
            target = 0.5 * w.sum(axis=1)
            est = np.zeros(m)
            for s in itertools.product([-1.0, 1.0], repeat=m):
                q = np.asarray(s)
                f = -0.5 * (off @ (q * q)) * q - np.diag(w) * q ** 3 / 6.0
                est += -f * q / (q * q)
            est /= 2 ** m
            rel.append(float(np.mean((target - est) / target)))
            # exact prediction of Eq. (8): (W_aa / 3) / ((1/2) sum_b W_ab)
            preds.append(float(np.mean((np.diag(w) / 3.0) / target)))
        out[str(m)] = {"relative_bias": float(np.mean(rel)), "eq8_prediction": float(np.mean(preds)),
                       "times_m": float(np.mean(rel) * m)}
        print("M", m, out[str(m)], flush=True)
    return out


def part_d(rng):
    """FC2/FC3/FC4 design statistics in the Al surrogate cell (3x3x3, hiPhive
    cluster space, cutoffs 4.5/3.5/3.0 A), with thermal modes of the
    surrogate's harmonic FC2 at 300 K.

    Reported per design and budget: the relative FC2--FC3 cross-information
    ||X2^T X3|| / sqrt(||X2^T X2|| ||X3^T X3||); the FC3 A-criterion of the
    Schur complement times the number of calls (variance per call); and the
    numerical rank and smallest singular value of the joint FC2+FC4 design."""
    from ase import Atoms
    from hiphive import ClusterSpace, StructureContainer
    from sotc.optimal_design import moment_matched_factors
    from surrogate_design_study import load_surrogate

    phonon, _, _, _ = load_surrogate("al")
    sup, prim = phonon.supercell, phonon.primitive
    atoms_p = Atoms(list(prim.symbols), positions=np.asarray(prim.positions),
                    cell=np.asarray(prim.cell), pbc=True)
    atoms_s = Atoms(list(sup.symbols), positions=np.asarray(sup.positions),
                    cell=np.asarray(sup.cell), pbc=True)
    _, patterns, sigma = thermal_modes(np.asarray(phonon.force_constants),
                                       np.asarray(sup.masses), 300.0)
    spaces = {"2": ClusterSpace(atoms_p, [4.5]), "23": ClusterSpace(atoms_p, [4.5, 3.5]),
              "234": ClusterSpace(atoms_p, [4.5, 3.5, 3.0])}
    n2 = spaces["2"].n_dofs
    n3 = spaces["23"].n_dofs - n2

    def matrix(cs, us):
        sc = StructureContainer(cs)
        for u in us:
            a = atoms_s.copy()
            a.new_array("displacements", u)
            a.new_array("forces", np.zeros_like(u))
            sc.add_structure(a)
        return sc.get_fit_data()[0]

    def designs(k):
        n_pairs = k // 2
        signs = rng.choice([-1.0, 1.0], (n_pairs, len(sigma)))
        mm = moment_matched_factors(n_pairs, len(sigma), rng)
        return {
            "canonical": gaussian_configs(rng, k, patterns, sigma),
            "canonical_pairs": paired(gaussian_configs(rng, n_pairs, patterns, sigma)),
            "fixed_pairs": paired(configurations(signs, patterns, sigma)),
            "fixed_unpaired": configurations(rng.choice([-1.0, 1.0], (k, len(sigma))),
                                             patterns, sigma),
            "moment_matched_pairs": paired(configurations(signs * mm, patterns, sigma))}

    out = {"n_fc2": int(n2), "n_fc3": int(n3),
           "n_fc4": int(spaces["234"].n_dofs - n2 - n3), "calls": {}}
    for k in (4, 8, 16):
        row = {}
        for name, us in designs(k).items():
            a = matrix(spaces["23"], us)
            x2, x3 = a[:, :n2], a[:, n2:]
            cross = np.linalg.norm(x2.T @ x3) / np.sqrt(np.linalg.norm(x2.T @ x2)
                                                        * np.linalg.norm(x3.T @ x3))
            info = a.T @ a
            schur = info[n2:, n2:] - info[n2:, :n2] @ np.linalg.solve(info[:n2, :n2],
                                                                      info[:n2, n2:])
            b = matrix(spaces["234"], us)
            b24 = np.concatenate([b[:, :n2], b[:, n2 + n3:]], axis=1)
            sv = np.linalg.svd(b24 / np.linalg.norm(b24, axis=0), compute_uv=False)
            row[name] = {"relative_cross_fc2_fc3": float(cross),
                         "fc3_variance_per_call": float(np.trace(np.linalg.pinv(schur)) * k),
                         "fc24_rank": int(np.sum(sv > 1e-8 * sv[0])),
                         "fc24_min_singular_value": float(sv[-1])}
        out["calls"][str(k)] = row
        print("FC234", k, {n: {a: round(b, 4) for a, b in v.items()} for n, v in row.items()},
              flush=True)
    return out


def main():
    rng = np.random.default_rng(2026)
    report = {"A_single_mode": part_a(rng)}
    print("A crossover", report["A_single_mode"]["crossover_vs_paired"],
          report["A_single_mode"]["crossover_vs_canonical"], flush=True)
    report["B_multi_parameter"] = {m: part_b(m, rng) for m in ("al", "si", "nacl")}
    report["C_bias_vs_modes"] = part_c(rng)
    report["D_fc3"] = part_d(rng)
    out = ROOT / "validation_results" / "summaries" / "theory_checks.json"
    out.write_text(json.dumps(report, indent=1) + "\n")


if __name__ == "__main__":
    main()
