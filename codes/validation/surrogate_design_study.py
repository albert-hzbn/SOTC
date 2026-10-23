#!/usr/bin/env python3
"""Displacement-design study on DFT-fitted anharmonic surrogates.

The FC2+FC3+FC4 models fitted to 80 QE snapshots for Al (300 K), W (1500 K)
and NaCl (700 K) in 3x3x3 primitive supercells provide exact, noise-free
forces at negligible cost.  The converged effective FC2 of each surrogate is
fitted to many canonical samples; every finite design is then scored by its
phonon-mesh RMSE against that reference and by held-out force RMSE.

Designs (all from the same harmonic FD trial model and temperature):

* canonical / canonical_paired: TDEP-style Gaussian samples (+/- pairs);
* rademacher_random: every mode at its exact r.m.s. amplitude, random signs;
* sotcd: exact amplitudes, D-optimal signs, +/- pairs (production SOTC-D);
* sotcdq: stratified Gaussian-quantile amplitudes (exact second moment,
  approximately Gaussian fourth moment), D-optimal signs, +/- pairs;
* canonical_dopt: Gaussian amplitudes with D-optimal signs, +/- pairs;
* rattle: hiPhive-style independent Gaussian Cartesian displacements with the
  per-atom thermal r.m.s. of the trial model, +/- pairs;
* zg: Zacharias--Giustino type real commensurate modes at their thermal
  amplitudes with orthogonal (Walsh) alternating sign patterns, +/- pairs.
"""

from __future__ import annotations

import json
from pathlib import Path
import sys

import numpy as np
from scipy.special import erfinv

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "codes"))
sys.path.insert(0, str(Path(__file__).resolve().parent))

from analyze_primary_archives import frequency_metrics, mesh_points, symfc_fc2_basis  # noqa: E402
from sotc.optimal_design import (  # noqa: E402
    _design_blocks, configurations, thermal_modes, thermal_rattle, walsh_signs, zg_modes,
)
from sotc.symfc_fit import forces_from_compact_fcs  # noqa: E402

SURROGATES = {"al": (300.0, 5.0), "w": (1500.0, 5.0), "nacl": (700.0, 4.3),
              # fitted to the primary-cell DFT data (fit_primary_surrogates.py)
              "si": (600.0, 5.0), "mg": (600.0, 4.6)}
COUNTS = (2, 4, 8, 16)
REPEATS = 30
N_REFERENCE = 6000
N_TEST = 200


def load_surrogate(material):
    import h5py
    import phonopy

    base = ROOT / "validation_results" / material / "real_ifc34"
    phonon = phonopy.load(base / "phonopy_harmonic.yaml")
    ff = base / "final_force_constants"
    with h5py.File(ff / "fc234_compact.hdf5") as handle:
        fcs = {k: np.asarray(handle[f"fc{k}"]) for k in (2, 3, 4)}
    perms = np.load(ff / "translation_permutations.npy")
    p2s = np.load(ff / "p2s_map.npy")
    return phonon, fcs, perms, p2s


def surrogate_forces(u, fcs, perms, p2s, batch=200):
    out = []
    for start in range(0, len(u), batch):
        out.append(forces_from_compact_fcs(u[start:start + batch], fcs, perms, p2s))
    return np.concatenate(out)


def gaussian_configs(rng, n, patterns, sigma):
    return np.einsum("km,m,mia->kia", rng.standard_normal((n, len(sigma))),
                     sigma, patterns)


def paired(u):
    return np.stack((u, -u), axis=1).reshape(-1, *u.shape[1:])


def dopt_signs(amplitudes, mode_rows, rng, sweeps=6):
    """D-optimal signs for given per-config, per-mode amplitudes (K, M)."""
    k_conf, n_modes = amplitudes.shape
    n_basis = mode_rows.shape[2]
    scale = np.mean([np.trace(r.T @ r) for r in mode_rows]) * n_modes / n_basis
    ridge = 1.0e-8 * scale * np.eye(n_basis)
    signs = rng.choice([-1.0, 1.0], size=(k_conf, n_modes))
    prior = np.zeros((n_basis, n_basis))

    def logdet(matrix):
        sgn, value = np.linalg.slogdet(matrix + ridge)
        return value if sgn > 0 else -np.inf

    for k in range(k_conf):
        coeff = signs[k] * amplitudes[k]
        rows = np.einsum("m,mrp->rp", coeff, mode_rows)
        best = logdet(prior + rows.T @ rows)
        for _ in range(sweeps):
            changed = False
            for m in rng.permutation(n_modes):
                new_rows = rows - 2.0 * coeff[m] * mode_rows[m]
                value = logdet(prior + new_rows.T @ new_rows)
                if value > best + 1e-10:
                    coeff[m] *= -1.0
                    rows, best, changed = new_rows, value, True
            if not changed:
                break
        signs[k] = np.sign(coeff) * np.sign(amplitudes[k])
        prior += rows.T @ rows
    return signs


def quantile_amplitudes(k_conf, n_modes, rng):
    """Stratified |N(0,1)| quantiles per mode, randomly permuted across
    configurations and rescaled to unit mean square (exact variance)."""
    if k_conf == 1:
        return np.ones((1, n_modes))
    p = (np.arange(k_conf) + 0.5) / k_conf
    levels = np.sqrt(2.0) * erfinv(p)  # quantiles of |Z|
    levels /= np.sqrt(np.mean(levels ** 2))
    return np.stack([rng.permutation(levels) for _ in range(n_modes)], axis=1)


def design(kind, count, rng, patterns, sigma, mode_rows, phonon=None, temperature=None):
    n_modes = len(sigma)
    if kind == "rattle":
        return paired(thermal_rattle(rng, phonon, temperature, count // 2))
    if kind == "zg":
        _, zp, zs = zg_modes(phonon, temperature)
        return paired(configurations(walsh_signs(len(zs), count // 2), zp, zs))
    if kind == "canonical":
        return gaussian_configs(rng, count, patterns, sigma)
    if kind == "canonical_paired":
        return paired(gaussian_configs(rng, count // 2, patterns, sigma))
    k_conf = count // 2
    if kind == "rademacher_random":
        amp = np.ones((k_conf, n_modes))
        signs = rng.choice([-1.0, 1.0], size=amp.shape)
    elif kind == "sotcd":
        amp = np.ones((k_conf, n_modes))
        signs = dopt_signs(amp, mode_rows, rng)
    elif kind == "sotcdq":
        amp = quantile_amplitudes(k_conf, n_modes, rng)
        signs = dopt_signs(amp, mode_rows, rng)
    elif kind == "canonical_dopt":
        amp = np.abs(rng.standard_normal((k_conf, n_modes)))
        signs = dopt_signs(amp, mode_rows, rng)
    else:
        raise ValueError(kind)
    u = np.einsum("km,m,mia->kia", signs * amp, sigma, patterns)
    return paired(u)


def fit(basis, u, f):
    x = np.concatenate([_design_blocks(ui, basis) for ui in u])
    target = (f - f.mean(axis=1, keepdims=True)).ravel()
    coef, *_ = np.linalg.lstsq(x, target, rcond=None)
    return np.einsum("k,kijab->ijab", coef, basis)


def mesh_frequencies(phonon, fc, qmesh):
    phonon.force_constants = fc
    phonon.run_qpoints(qmesh)
    return np.asarray(phonon.get_qpoints_dict()["frequencies"])


def study(material):
    import phonopy

    temperature, cutoff = SURROGATES[material]
    phonon, fcs, perms, p2s = load_surrogate(material)
    pos = np.asarray(phonon.supercell.positions)
    cell = np.asarray(phonon.supercell.cell)
    symbols = list(phonon.supercell.symbols)
    masses = np.asarray(phonon.supercell.masses)
    harmonic = np.asarray(phonon.force_constants).copy()
    # Designs read the trial from a separate copy: mesh_frequencies() below
    # overwrites the force constants of ``phonon``.
    trial = phonopy.load(ROOT / "validation_results" / material / "real_ifc34"
                         / "phonopy_harmonic.yaml")
    _, patterns, sigma = thermal_modes(harmonic, masses, temperature)
    basis = symfc_fc2_basis(pos, cell, symbols, cutoff)
    mode_rows = np.asarray([sigma[m] * _design_blocks(patterns[m], basis)
                            for m in range(len(sigma))])
    qmesh = mesh_points(10)
    rng = np.random.default_rng(2024)

    # Converged effective reference from paired canonical samples.
    u_ref = paired(gaussian_configs(rng, N_REFERENCE // 2, patterns, sigma))
    u_ref -= u_ref.mean(axis=1, keepdims=True)
    f_ref = surrogate_forces(u_ref, fcs, perms, p2s)
    x_rows = None
    xtx = np.zeros((len(basis), len(basis)))
    xty = np.zeros(len(basis))
    for ui, fi in zip(u_ref, f_ref):
        x_rows = _design_blocks(ui, basis)
        xtx += x_rows.T @ x_rows
        xty += x_rows.T @ (fi - fi.mean(axis=0)).ravel()
    coef = np.linalg.solve(xtx, xty)
    fc_eff = np.einsum("k,kijab->ijab", coef, basis)
    freq_eff = mesh_frequencies(phonon, fc_eff, qmesh)
    half = fit(basis, u_ref[: N_REFERENCE // 2], f_ref[: N_REFERENCE // 2])
    ref_conv = frequency_metrics(freq_eff, mesh_frequencies(phonon, half, qmesh))
    freq_harm = mesh_frequencies(phonon, harmonic, qmesh)

    u_test = gaussian_configs(rng, N_TEST, patterns, sigma)
    u_test -= u_test.mean(axis=1, keepdims=True)
    f_test = surrogate_forces(u_test, fcs, perms, p2s)

    kinds = ("canonical", "canonical_paired", "rademacher_random", "sotcd",
             "sotcdq", "canonical_dopt", "rattle", "zg")
    results = {kind: {} for kind in kinds}
    for kind in kinds:
        for count in COUNTS:
            rows = []
            for rep in range(REPEATS):
                local = np.random.default_rng(10_000 * COUNTS.index(count) + rep
                                              + 1_000_000 * kinds.index(kind))
                u = design(kind, count, local, patterns, sigma, mode_rows,
                           trial, temperature)
                u -= u.mean(axis=1, keepdims=True)
                f = surrogate_forces(u, fcs, perms, p2s)
                fc = fit(basis, u, f)
                freq = mesh_frequencies(phonon, fc, qmesh)
                pred = -np.einsum("ijab,sjb->sia", fc, u_test)
                rows.append({
                    "mesh_rmse_thz": frequency_metrics(freq_eff, freq)["rmse_thz"],
                    "held_out_force_rmse": float(np.sqrt(np.mean((pred - f_test) ** 2))),
                    "unstable_fraction": float(np.mean(freq < -0.05)),
                })
            results[kind][str(count)] = rows
            med = np.median([r["mesh_rmse_thz"] for r in rows])
            print(f"{material} {kind:18s} n={count:2d} median mesh RMSE {med:.4f}",
                  flush=True)
    pred_eff = -np.einsum("ijab,sjb->sia", fc_eff, u_test)
    report = {
        "material": material, "temperature_k": temperature, "cutoff_ang": cutoff,
        "n_basis": len(basis), "n_modes": len(sigma),
        "reference_samples": N_REFERENCE,
        "reference_half_vs_full_mesh": ref_conv,
        "harmonic_vs_effective_mesh": frequency_metrics(freq_eff, freq_harm),
        "effective_held_out_force_rmse": float(np.sqrt(np.mean((pred_eff - f_test) ** 2))),
        "results": results,
    }
    out = ROOT / "validation_results" / material / "surrogate_design"
    out.mkdir(parents=True, exist_ok=True)
    (out / "surrogate_design_study.json").write_text(json.dumps(report, indent=1) + "\n")


def main():
    for material in sys.argv[1:] or list(SURROGATES):
        study(material)


if __name__ == "__main__":
    main()
