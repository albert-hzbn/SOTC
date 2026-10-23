#!/usr/bin/env python3
"""Self-consistent effective FC2 on the anharmonic surrogates.

Target: the self-consistent canonical effective FC2 Phi_SC, i.e. the fixed
point Phi = argmin_Phi <|F + Phi u|^2> over Gaussian u ~ N(0, C(Phi, T)).
It equals the SCHA auxiliary FC2 for crystals without free internal
coordinates, and is computed here by iterating the canonical fit with 6000
samples per iteration until the phonon mesh changes by < 1 mTHz.

Protocols, all started from the harmonic FD trial of the 3x3x3 cell and
counted in surrogate force calls B:

* canonical_oneshot / sotcd_oneshot: no trial update (iteration-0 target);
* stdep_last: sTDEP, three iterations of B/3 paired canonical samples, the
  model fitted to the last iteration only;
* stdep_pool: the same, fitted to all iterations;
* sotcd_sc: SOTC-D whose trial is refitted after every +/- pair (minimum-
  norm correction of the harmonic FC2 while unidentifiable) and whose next
  pair is D-optimal given all previous designs;
* sscha: python-sscha (surrogate_sscha.py), scored from its archive.
"""

from __future__ import annotations

import json
from pathlib import Path
import sys

import numpy as np

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "codes"))
sys.path.insert(0, str(Path(__file__).resolve().parent))

from analyze_primary_archives import frequency_metrics, mesh_points, symfc_fc2_basis  # noqa: E402
from sotc.optimal_design import (  # noqa: E402
    _design_blocks, configurations, information_matrix, sequential_d_optimal,
    thermal_modes,
)
from surrogate_design_study import (  # noqa: E402
    SURROGATES, gaussian_configs, load_surrogate, mesh_frequencies, paired,
    surrogate_forces,
)

BUDGETS = (4, 8, 16, 32, 64)
REPEATS = 12
N_SC = 6000
N_TEST = 200


def design_matrix(basis, u):
    return np.concatenate([_design_blocks(ui, basis) for ui in u])


def fit_coefficients(basis, u, f, anchor=None):
    """Least squares in the FC2 basis; minimum-norm correction of ``anchor``
    when the data do not identify every coefficient."""
    x = design_matrix(basis, u)
    y = (f - f.mean(axis=1, keepdims=True)).ravel()
    if anchor is None:
        anchor = np.zeros(len(basis))
    delta, *_ = np.linalg.lstsq(x, y - x @ anchor, rcond=None)
    return anchor + delta


def to_fc(basis, coef):
    return np.einsum("k,kijab->ijab", coef, basis)


def modes(fc, masses, temperature):
    _, patterns, sigma = thermal_modes(fc, masses, temperature)
    return patterns, sigma


def self_consistent_reference(ctx, rng):
    fc = ctx["harmonic"]
    history = []
    previous = mesh_frequencies(ctx["phonon"], fc, ctx["qmesh"])
    for _ in range(12):
        patterns, sigma = modes(fc, ctx["masses"], ctx["T"])
        u = paired(gaussian_configs(rng, N_SC // 2, patterns, sigma))
        u -= u.mean(axis=1, keepdims=True)
        f = ctx["forces"](u)
        fc = to_fc(ctx["basis"], fit_coefficients(ctx["basis"], u, f))
        freq = mesh_frequencies(ctx["phonon"], fc, ctx["qmesh"])
        change = frequency_metrics(previous, freq)["rmse_thz"]
        history.append(change)
        previous = freq
        if change < 1.0e-3:
            break
    return fc, previous, history


def oneshot(ctx, kind, budget, rng):
    patterns, sigma = modes(ctx["harmonic"], ctx["masses"], ctx["T"])
    if kind == "canonical":
        u = paired(gaussian_configs(rng, budget // 2, patterns, sigma))
    else:
        signs, _ = sequential_d_optimal(patterns, sigma, ctx["basis"], budget // 2,
                                        seed=int(rng.integers(1 << 30)))
        u = paired(configurations(signs, patterns, sigma))
    u -= u.mean(axis=1, keepdims=True)
    f = ctx["forces"](u)
    return to_fc(ctx["basis"], fit_coefficients(ctx["basis"], u, f, ctx["c_harm"]))


def stdep(ctx, budget, rng, pool):
    n_iter = 3 if budget >= 12 else 2
    sizes = [2 * (budget // (2 * n_iter))] * n_iter
    sizes[-1] += budget - sum(sizes)
    fc = ctx["harmonic"]
    coef = ctx["c_harm"]
    us, fs = [], []
    for size in sizes:
        patterns, sigma = modes(fc, ctx["masses"], ctx["T"])
        u = paired(gaussian_configs(rng, size // 2, patterns, sigma))
        u -= u.mean(axis=1, keepdims=True)
        f = ctx["forces"](u)
        us.append(u)
        fs.append(f)
        if pool:
            coef = fit_coefficients(ctx["basis"], np.concatenate(us), np.concatenate(fs), coef)
        else:
            coef = fit_coefficients(ctx["basis"], u, f, coef)
        fc = to_fc(ctx["basis"], coef)
    return fc


def sotcd_sc(ctx, budget, rng):
    basis = ctx["basis"]
    coef = ctx["c_harm"]
    fc = ctx["harmonic"]
    us, fs = [], []
    prior = np.zeros((len(basis), len(basis)))
    for _ in range(budget // 2):
        patterns, sigma = modes(fc, ctx["masses"], ctx["T"])
        signs, _ = sequential_d_optimal(patterns, sigma, basis, 1,
                                        seed=int(rng.integers(1 << 30)),
                                        prior_information=prior)
        u = paired(configurations(signs, patterns, sigma))
        u -= u.mean(axis=1, keepdims=True)
        us.append(u)
        fs.append(ctx["forces"](u))
        prior = prior + information_matrix(u, basis)
        coef = fit_coefficients(basis, np.concatenate(us), np.concatenate(fs), ctx["c_harm"])
        fc = to_fc(basis, coef)
    return fc


def score(ctx, fc):
    freq = mesh_frequencies(ctx["phonon"], fc, ctx["qmesh"])
    pred = -np.einsum("ijab,sjb->sia", fc, ctx["u_test"])
    return {
        "mesh_rmse_thz": frequency_metrics(ctx["freq_sc"], freq)["rmse_thz"],
        "held_out_force_rmse": float(np.sqrt(np.mean((pred - ctx["f_test"]) ** 2))),
        "unstable_fraction": float(np.mean(freq < -0.05)),
    }


def study(material):
    from phonopy.harmonic.force_constants import compact_fc_to_full_fc

    temperature, cutoff = SURROGATES[material]
    phonon, fcs, perms, p2s = load_surrogate(material)
    harmonic = np.asarray(phonon.force_constants)
    if harmonic.shape[0] != harmonic.shape[1]:
        harmonic = compact_fc_to_full_fc(phonon.primitive, harmonic)
    pos = np.asarray(phonon.supercell.positions)
    basis = symfc_fc2_basis(pos, np.asarray(phonon.supercell.cell),
                            list(phonon.supercell.symbols), cutoff)
    flat = basis.reshape(len(basis), -1)
    c_harm, *_ = np.linalg.lstsq(flat.T, harmonic.ravel(), rcond=None)
    ctx = {
        "phonon": phonon, "T": temperature, "basis": basis, "c_harm": c_harm,
        "harmonic": to_fc(basis, c_harm),
        "masses": np.asarray(phonon.supercell.masses), "qmesh": mesh_points(10),
        "forces": lambda u: surrogate_forces(u, fcs, perms, p2s),
    }
    rng = np.random.default_rng(97)
    fc_sc, freq_sc, history = self_consistent_reference(ctx, rng)
    ctx["freq_sc"] = freq_sc
    patterns, sigma = modes(fc_sc, ctx["masses"], temperature)
    u_test = gaussian_configs(rng, N_TEST, patterns, sigma)
    ctx["u_test"] = u_test - u_test.mean(axis=1, keepdims=True)
    ctx["f_test"] = ctx["forces"](ctx["u_test"])
    freq_h = mesh_frequencies(phonon, ctx["harmonic"], ctx["qmesh"])
    report = {
        "material": material, "temperature_k": temperature,
        "sc_iteration_mesh_changes_thz": history,
        "harmonic_vs_sc_mesh_thz": frequency_metrics(freq_sc, freq_h)["rmse_thz"],
        "sc_held_out_force_rmse": score(ctx, fc_sc)["held_out_force_rmse"],
        "budgets": list(BUDGETS), "results": {},
    }
    protocols = {
        "canonical_oneshot": lambda b, r: oneshot(ctx, "canonical", b, r),
        "sotcd_oneshot": lambda b, r: oneshot(ctx, "sotcd", b, r),
        "stdep_last": lambda b, r: stdep(ctx, b, r, pool=False),
        "stdep_pool": lambda b, r: stdep(ctx, b, r, pool=True),
        "sotcd_sc": lambda b, r: sotcd_sc(ctx, b, r),
    }
    for p_index, (name, protocol) in enumerate(protocols.items()):
        block = {}
        for budget in BUDGETS:
            rows = []
            for rep in range(REPEATS):
                r = np.random.default_rng(100_000 * p_index + 1000 * budget + rep)
                rows.append(score(ctx, protocol(budget, r)))
            block[str(budget)] = rows
            print(f"{material} {name:18s} B={budget:3d} median mesh "
                  f"{np.median([x['mesh_rmse_thz'] for x in rows]):.4f}", flush=True)
        report["results"][name] = block
    sscha = ROOT / "validation_results" / material / "surrogate_design" / "sscha_runs.npz"
    if sscha.exists():
        data = np.load(sscha)
        meta = json.loads(str(data["meta"]))
        report["results"]["sscha"] = [
            dict(score(ctx, fc), **m) for fc, m in zip(data["force_constants"], meta)
        ]
    out = ROOT / "validation_results" / material / "surrogate_design"
    out.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(out / "self_consistent_reference.npz", force_constants=fc_sc)
    (out / "surrogate_self_consistent.json").write_text(json.dumps(report, indent=1) + "\n")


def main():
    for material in sys.argv[1:] or list(SURROGATES):
        study(material)


if __name__ == "__main__":
    main()
