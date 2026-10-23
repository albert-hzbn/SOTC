#!/usr/bin/env python3
"""SOTC-D-SC with DFT (Quantum ESPRESSO) forces in the SOTC cell.

The trial is the same-cell harmonic FD FC2. After every +/- pair, all pairs
so far are refitted in the supercell space-group basis (minimum-norm
correction of the harmonic coefficients while unidentified), and the next
pair is designed from the refitted model, D-optimal given all previous
pairs (``surrogate_self_consistent.sotcd_sc`` with DFT forces). The model
after every pair is stored, so every even budget up to ``--budget`` is
available from one run.

Writes validation_results/<m>/primary_cell/sotcd_sc_dft/s<seed>/history.npz.

Usage: dft_sotcd_sc.py <material> <seed> [--budget 32]
"""

from __future__ import annotations

import argparse
import concurrent.futures
from pathlib import Path
import sys
from types import SimpleNamespace

import numpy as np

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "codes"))
sys.path.insert(0, str(Path(__file__).resolve().parent))

from analyze_primary_archives import PRIMARY, symfc_fc2_basis  # noqa: E402
from run_primary_cell_controls import overrides, primary_spec  # noqa: E402
from run_real_anharmonic_ifcs import run_one_state  # noqa: E402
from sotc.optimal_design import (  # noqa: E402
    configurations, information_matrix, sequential_d_optimal,
)
from surrogate_design_study import paired  # noqa: E402
from surrogate_self_consistent import fit_coefficients, modes, to_fc  # noqa: E402

SEED0 = 77_000


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("material")
    parser.add_argument("seed", type=int)
    parser.add_argument("--budget", type=int, default=32)
    args = parser.parse_args()

    import phonopy
    from phonopy.harmonic.force_constants import compact_fc_to_full_fc

    material = args.material
    base = ROOT / "validation_results" / material / "primary_cell"
    out = base / "sotcd_sc_dft" / f"s{args.seed}"
    out.mkdir(parents=True, exist_ok=True)
    phonon = phonopy.load(base / "fd" / "phonopy_harmonic.yaml")
    with np.load(base / "fd" / "harmonic_reference.npz") as h:
        f0 = np.asarray(h["reference_forces_ev_ang"])
    harmonic = np.asarray(phonon.force_constants)
    if harmonic.shape[0] != harmonic.shape[1]:
        harmonic = compact_fc_to_full_fc(phonon.primitive, harmonic)
    sup = phonon.supercell
    eq = np.asarray(sup.positions)
    cell = np.asarray(sup.cell)
    symbols = list(sup.symbols)
    masses = np.asarray(sup.masses)
    basis = symfc_fc2_basis(eq, cell, symbols, PRIMARY[material]["cutoff"])
    c_harm, *_ = np.linalg.lstsq(basis.reshape(len(basis), -1).T, harmonic.ravel(),
                                 rcond=None)
    temperature = PRIMARY[material]["T"]
    spec = primary_spec(material)
    run_args = SimpleNamespace(material=material, n_parallel=2, tasks_per_qe=32,
                               fit_only=False, force_rerun=False)

    rng = np.random.default_rng(SEED0 + args.seed)
    coef, fc = c_harm, to_fc(basis, c_harm)
    us, fs, coefs, calls = [], [], [], []
    prior = np.zeros((len(basis), len(basis)))
    for k in range(args.budget // 2):
        patterns, sigma = modes(fc, masses, temperature)
        signs, _ = sequential_d_optimal(patterns, sigma, basis, 1,
                                        seed=int(rng.integers(1 << 30)),
                                        prior_information=prior)
        u = paired(configurations(signs, patterns, sigma))
        u -= u.mean(axis=1, keepdims=True)
        with concurrent.futures.ThreadPoolExecutor(max_workers=2) as pool:
            res = list(pool.map(
                lambda j: run_one_state(2 * k + j, eq + u[j], cell, symbols, spec,
                                        run_args, out, overrides(spec)), range(2)))
        df = np.asarray([np.asarray(r["forces_ev_ang"]) - f0 for r in res])
        us.append(u)
        fs.append(df - df.mean(axis=1, keepdims=True))
        prior = prior + information_matrix(u, basis)
        coef = fit_coefficients(basis, np.concatenate(us), np.concatenate(fs), c_harm)
        fc = to_fc(basis, coef)
        coefs.append(coef)
        calls.append(2 * (k + 1))
        np.savez_compressed(out / "history.npz", force_constants=np.asarray(
            [to_fc(basis, c) for c in coefs]), force_calls=np.asarray(calls),
            displacements_ang=np.concatenate(us))
        print(f"{material} SOTC-D-SC seed {args.seed}: {calls[-1]} calls", flush=True)


if __name__ == "__main__":
    main()
