#!/usr/bin/env python3
"""Fixed-amplitude bias of SOTC-D-SC on the anharmonic surrogates.

fixed: |q_e| = sigma_e (current SOTC-D); gauss: |q_e| = sigma_e |xi|;
lhs3 / lhs4: |q_e|^2/sigma_e^2 drawn without replacement, per mode and block
of 3 / 4 pairs, from equal-weight values with mean 1 and mean square 3 (the
Gaussian second and fourth moments).
"""
import json
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "codes"))
sys.path.insert(0, str(ROOT / "codes" / "validation"))

from analyze_primary_archives import frequency_metrics, mesh_points, symfc_fc2_basis
from sotc.optimal_design import configurations, information_matrix, sequential_d_optimal
from surrogate_design_study import SURROGATES, load_surrogate, mesh_frequencies, paired, surrogate_forces
from surrogate_self_consistent import fit_coefficients, modes, to_fc

BUDGETS = (16, 32, 64, 128)
REPEATS = 12
LEVELS = {"lhs3": np.array([0.0, 0.0, 3.0]),
          "lhs4": np.array([0.0, 0.0, 2 - np.sqrt(2), 2 + np.sqrt(2)])}


def amplitude_factors(kind, k, n_modes, rng, cache):
    if kind == "fixed":
        return np.ones(n_modes)
    if kind == "gauss":
        return np.abs(rng.standard_normal(n_modes))
    levels = LEVELS[kind]
    b = len(levels)
    if k % b == 0 or "perm" not in cache:
        cache["perm"] = np.array([rng.permutation(b) for _ in range(512)])
    return np.sqrt(levels[cache["perm"][:n_modes, k % b]])


def run(material):
    from phonopy.harmonic.force_constants import compact_fc_to_full_fc
    temperature, cutoff = SURROGATES[material]
    phonon, fcs, perms, p2s = load_surrogate(material)
    harmonic = np.asarray(phonon.force_constants)
    if harmonic.shape[0] != harmonic.shape[1]:
        harmonic = compact_fc_to_full_fc(phonon.primitive, harmonic)
    basis = symfc_fc2_basis(np.asarray(phonon.supercell.positions), np.asarray(phonon.supercell.cell),
                            list(phonon.supercell.symbols), cutoff)
    flat = basis.reshape(len(basis), -1)
    c_harm, *_ = np.linalg.lstsq(flat.T, harmonic.ravel(), rcond=None)
    masses = np.asarray(phonon.supercell.masses)
    qmesh = mesh_points(10)
    fc_sc = np.load(ROOT / "validation_results" / material / "surrogate_design"
                    / "self_consistent_reference.npz")["force_constants"]
    freq_sc = mesh_frequencies(phonon, fc_sc, qmesh)
    forces = lambda u: surrogate_forces(u, fcs, perms, p2s)
    kinds = ["fixed", "gauss", "lhs3", "lhs4"]
    out = {k: {} for k in kinds}
    for budget in BUDGETS:
        for kind in kinds:
            errs = []
            for rep in range(REPEATS):
                rng = np.random.default_rng(500_000 + 1000 * budget + rep)
                cache = {}
                coef, fc = c_harm, to_fc(basis, c_harm)
                us, fs = [], []
                prior = np.zeros((len(basis), len(basis)))
                for k in range(budget // 2):
                    patterns, sigma = modes(fc, masses, temperature)
                    amp = sigma * amplitude_factors(kind, k, len(sigma), rng, cache)
                    signs, _ = sequential_d_optimal(patterns, np.maximum(amp, 1e-12 * sigma.max()), basis, 1,
                                                    seed=int(rng.integers(1 << 30)), prior_information=prior)
                    u = paired(configurations(signs, patterns, amp))
                    u -= u.mean(axis=1, keepdims=True)
                    us.append(u)
                    fs.append(forces(u))
                    prior = prior + information_matrix(u, basis)
                    coef = fit_coefficients(basis, np.concatenate(us), np.concatenate(fs), c_harm)
                    fc = to_fc(basis, coef)
                errs.append(frequency_metrics(freq_sc, mesh_frequencies(phonon, fc, qmesh))["rmse_thz"])
            out[kind][str(budget)] = float(np.median(errs))
        print(material, budget, {k: round(out[k][str(budget)], 4) for k in kinds}, flush=True)
    return out


if __name__ == "__main__":
    m = sys.argv[1]
    (ROOT / "validation_results" / m / "surrogate_design" / "amplitude_bias.json").write_text(
        json.dumps(run(m), indent=1))
