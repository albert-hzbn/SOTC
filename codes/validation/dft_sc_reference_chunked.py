#!/usr/bin/env python3
"""Converged self-consistent FC2 reference (sTDEP, last-iteration fit),
with the configurations of each iteration distributed over several jobs.

Iteration k draws ``--per-iter`` canonical configurations (+-u pairs) from
the FC2 fitted to iteration k-1 alone (iteration 0: the harmonic FD trial),
exactly as ``dft_sc_protocols.py <m> stdep``; the draws are seeded by the
seed and k, so every chunk job generates the same configurations and
computes the subset ``chunk::nchunks``. Jobs of iteration k+1 depend on all
jobs of iteration k (Slurm ``--dependency=afterok``). ``--collect`` writes
<name>/s<seed>/history.npz in the format of ``dft_sc_protocols.py``, so
that ``dft_selfconsistent_compare.py`` reads it as the reference.

Usage: dft_sc_reference_chunked.py <material> <seed> --iter K --chunk C --nchunks N
       [--per-iter 64] [--name stdep_reference] [--n-parallel 1]
       dft_sc_reference_chunked.py <material> <seed> --collect [--name ...]
"""

from __future__ import annotations

import argparse
import concurrent.futures
import os
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
from surrogate_design_study import gaussian_configs, paired  # noqa: E402
from surrogate_self_consistent import fit_coefficients, modes, to_fc  # noqa: E402

SEED0 = 87_000


def setup(material):
    import phonopy
    from phonopy.harmonic.force_constants import compact_fc_to_full_fc

    base = ROOT / "validation_results" / material / "primary_cell"
    phonon = phonopy.load(base / "fd" / "phonopy_harmonic.yaml")
    with np.load(base / "fd" / "harmonic_reference.npz") as h:
        f0 = np.asarray(h["reference_forces_ev_ang"])
    harmonic = np.asarray(phonon.force_constants)
    if harmonic.shape[0] != harmonic.shape[1]:
        harmonic = compact_fc_to_full_fc(phonon.primitive, harmonic)
    sup = phonon.supercell
    eq, cell = np.asarray(sup.positions), np.asarray(sup.cell)
    basis = symfc_fc2_basis(eq, cell, list(sup.symbols), PRIMARY[material]["cutoff"])
    c_harm, *_ = np.linalg.lstsq(basis.reshape(len(basis), -1).T, harmonic.ravel(), rcond=None)
    return base, sup, eq, cell, f0, harmonic, basis, c_harm


def iteration_data(stage, n):
    us, fs = [], []
    for i in range(n):
        with np.load(stage / f"state_{i:04d}" / "result.npz") as r:
            us.append(np.asarray(r["displacements_ang"]) if "displacements_ang" in r.files
                      else None)
            fs.append(np.asarray(r["forces_ev_ang"]))
    return us, fs


def displacements(out, k, rng_seed, fc, masses, temperature, per_iter):
    """Configurations of iteration k (written once; identical in every job)."""
    path = out / f"iter{k:02d}" / "displacements.npz"
    if path.exists():
        return np.load(path)["displacements_ang"]
    rng = np.random.default_rng(rng_seed)
    patterns, sigma = modes(fc, masses, temperature)
    u = paired(gaussian_configs(rng, per_iter // 2, patterns, sigma))
    u -= u.mean(axis=1, keepdims=True)
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(f".{os.getpid()}.npz")
    np.savez_compressed(tmp, displacements_ang=u)
    os.replace(tmp, path)
    return u


def last_fit(out, k, basis, c_harm, f0, per_iter):
    """FC2 fitted to iteration k alone (and its displacements and forces)."""
    stage = out / f"iter{k:02d}"
    u = np.load(stage / "displacements.npz")["displacements_ang"]
    fs = []
    for i in range(per_iter):
        with np.load(stage / f"state_{i:04d}" / "result.npz") as r:
            fs.append(np.asarray(r["forces_ev_ang"]) - f0)
    f = np.asarray(fs)
    f -= f.mean(axis=1, keepdims=True)
    return to_fc(basis, fit_coefficients(basis, u, f, c_harm)), u, f


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("material")
    parser.add_argument("seed", type=int)
    parser.add_argument("--iter", type=int, default=0)
    parser.add_argument("--chunk", type=int, default=0)
    parser.add_argument("--nchunks", type=int, default=1)
    parser.add_argument("--per-iter", type=int, default=64)
    parser.add_argument("--name", default="stdep_reference")
    parser.add_argument("--n-parallel", type=int, default=1)
    parser.add_argument("--collect", action="store_true")
    args = parser.parse_args()

    m = args.material
    base, sup, eq, cell, f0, harmonic, basis, c_harm = setup(m)
    out = base / args.name / f"s{args.seed}"
    out.mkdir(parents=True, exist_ok=True)
    temperature = PRIMARY[m]["T"]
    masses = np.asarray(sup.masses)

    if args.collect:
        lasts, pooled, calls, us, fs = [], [], [], [], []
        k = 0
        while (out / f"iter{k:02d}" / "displacements.npz").exists():
            fc, u, f = last_fit(out, k, basis, c_harm, f0, args.per_iter)
            us.append(u)
            fs.append(f)
            lasts.append(fc)
            pooled.append(to_fc(basis, fit_coefficients(basis, np.concatenate(us),
                                                        np.concatenate(fs), c_harm)))
            calls.append(args.per_iter * (k + 1))
            k += 1
        np.savez_compressed(out / "history.npz", force_constants_last=np.asarray(lasts),
                            force_constants_pooled=np.asarray(pooled),
                            force_calls=np.asarray(calls),
                            displacements_ang=np.concatenate(us))
        print(f"{m} {args.name} s{args.seed}: {k} iterations collected", flush=True)
        return

    fc = harmonic if args.iter == 0 else last_fit(out, args.iter - 1, basis, c_harm, f0,
                                                  args.per_iter)[0]
    u = displacements(out, args.iter, SEED0 + 1000 * args.seed + args.iter, fc, masses,
                      temperature, args.per_iter)
    stage = out / f"iter{args.iter:02d}"
    spec = primary_spec(m)
    run_args = SimpleNamespace(material=m, n_parallel=args.n_parallel,
                               tasks_per_qe=int(os.environ.get("TASKS_PER_QE", "128")),
                               fit_only=False, force_rerun=False)
    todo = [i for i in range(len(u)) if i % args.nchunks == args.chunk
            and not (stage / f"state_{i:04d}" / "result.npz").exists()]
    with concurrent.futures.ThreadPoolExecutor(max_workers=args.n_parallel) as pool:
        list(pool.map(lambda i: run_one_state(i, eq + u[i], cell, list(sup.symbols), spec,
                                              run_args, stage, overrides(spec)), todo))
    print(f"{m} iter {args.iter} chunk {args.chunk}/{args.nchunks}: {len(todo)} states",
          flush=True)


if __name__ == "__main__":
    main()
