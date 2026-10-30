#!/usr/bin/env python3
"""Training data for the MLIP study (Al, Si): extend the SOTC-D and thermal
rattle sets of the SOTC cell to 64 force calls.

SOTC-D continues the nested sequential D-optimal design of the existing
eight pairs (their FC2 information is the prior), so the first 16 calls are
those of the paper; rattle draws further independent configurations with
the canonical per-atom variance. Displacements are fixed before any DFT
call (seeded), so chunk jobs compute disjoint subsets of the same sets.

Writes primary_cell/mlip/<design>_displacements.npz and DFT results in
primary_cell/mlip/<design>_qe/.

Usage: mlip_data.py <material> prepare
       mlip_data.py <material> compute <sotcd|rattle> [--chunk C --nchunks N]
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

N_CALLS = 64


def prepare(material):
    import phonopy
    from analyze_designs import datasets
    from analyze_primary_archives import PRIMARY, symfc_fc2_basis
    from sotc.optimal_design import (configurations, information_matrix,
                                     sequential_d_optimal, thermal_modes, thermal_rattle)
    from surrogate_design_study import paired

    base = ROOT / "validation_results" / material / "primary_cell"
    out = base / "mlip"
    out.mkdir(parents=True, exist_ok=True)
    ph = phonopy.load(base / "fd" / "phonopy_harmonic.yaml")
    sup = ph.supercell
    basis = symfc_fc2_basis(np.asarray(sup.positions), np.asarray(sup.cell), list(sup.symbols),
                            PRIMARY[material]["cutoff"])
    _, patterns, sigma = thermal_modes(np.asarray(ph.force_constants), np.asarray(sup.masses),
                                       PRIMARY[material]["T"])
    data = datasets(material)
    u_old = data["sotcd"][0]
    signs, _ = sequential_d_optimal(patterns, sigma, basis, (N_CALLS - len(u_old)) // 2,
                                    seed=4242, prior_information=information_matrix(u_old, basis))
    u_new = paired(configurations(signs, patterns, sigma))
    u_new -= u_new.mean(axis=1, keepdims=True)
    np.savez_compressed(out / "sotcd_displacements.npz",
                        displacements_ang=np.concatenate([u_old, u_new]), n_existing=len(u_old))
    u_r_old = data["rattle"][0]
    rng = np.random.default_rng(4343)
    u_r_new = thermal_rattle(rng, ph, PRIMARY[material]["T"], N_CALLS - len(u_r_old))
    np.savez_compressed(out / "rattle_displacements.npz",
                        displacements_ang=np.concatenate([u_r_old, np.asarray(u_r_new)]),
                        n_existing=len(u_r_old))
    print(material, "sotcd", N_CALLS, "rattle", N_CALLS, flush=True)


def compute(material, design, chunk, nchunks, n_parallel):
    import phonopy
    from run_primary_cell_controls import overrides, primary_spec
    from run_real_anharmonic_ifcs import run_one_state

    base = ROOT / "validation_results" / material / "primary_cell"
    ph = phonopy.load(base / "fd" / "phonopy_harmonic.yaml")
    eq = np.asarray(ph.supercell.positions)
    cell = np.asarray(ph.supercell.cell)
    symbols = list(ph.supercell.symbols)
    z = np.load(base / "mlip" / f"{design}_displacements.npz")
    u, start = z["displacements_ang"], int(z["n_existing"])
    spec = primary_spec(material)
    run_args = SimpleNamespace(material=material, n_parallel=n_parallel, tasks_per_qe=32,
                               fit_only=False, force_rerun=False)
    stage = base / "mlip" / f"{design}_qe"
    todo = [i for i in range(start, len(u)) if i % nchunks == chunk]
    with concurrent.futures.ThreadPoolExecutor(max_workers=n_parallel) as pool:
        list(pool.map(lambda i: run_one_state(i, eq + u[i], cell, symbols, spec, run_args, stage,
                                              overrides(spec)), todo))


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("material")
    parser.add_argument("stage", choices=["prepare", "compute"])
    parser.add_argument("design", nargs="?")
    parser.add_argument("--chunk", type=int, default=0)
    parser.add_argument("--nchunks", type=int, default=1)
    parser.add_argument("--n-parallel", type=int, default=4)
    args = parser.parse_args()
    if args.stage == "prepare":
        prepare(args.material)
    else:
        compute(args.material, args.design, args.chunk, args.nchunks, args.n_parallel)


if __name__ == "__main__":
    main()
