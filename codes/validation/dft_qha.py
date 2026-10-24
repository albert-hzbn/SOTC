#!/usr/bin/env python3
"""Volume grid of self-consistent SOTC-D runs for thermal expansion (Al).

For one lattice parameter a, in the 32-atom SOTC cell scaled from the
production cell, with the same electronic settings:
* the undisplaced energy E(0);
* the harmonic FD trial (phonopy, +/-0.01 A);
* SOTC-D-SC with ``--budget`` force calls at each design temperature, all
  temperatures run concurrently (as in ``dft_sotcd_sc.py``), keeping the
  displacements, force differences, and energies of every call.

Writes validation_results/<m>/primary_cell/qha/a<a>/{fd.npz,T<T>.npz}.

Usage: dft_qha.py <material> <a_ang> [--temperatures 100 300 500 700] [--budget 8]
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

SEED0 = 91_000


def scaled_phonon(material, a):
    import phonopy
    from phonopy import Phonopy
    from phonopy.structure.atoms import PhonopyAtoms

    base = ROOT / "validation_results" / material / "primary_cell"
    ref = phonopy.load(base / "fd" / "phonopy_harmonic.yaml")
    s = a / PRIMARY[material]["a"]
    unit = ref.unitcell
    cell = PhonopyAtoms(symbols=list(unit.symbols), cell=np.asarray(unit.cell) * s,
                        scaled_positions=np.asarray(unit.scaled_positions),
                        masses=np.asarray(unit.masses))
    return Phonopy(cell, np.eye(3, dtype=int), primitive_matrix=ref.primitive_matrix)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("material")
    parser.add_argument("a", type=float)
    parser.add_argument("--temperatures", type=float, nargs="+", default=[100, 300, 500, 700])
    parser.add_argument("--budget", type=int, default=8)
    args = parser.parse_args()

    material = args.material
    out = ROOT / "validation_results" / material / "primary_cell" / "qha" / f"a{args.a:.4f}"
    out.mkdir(parents=True, exist_ok=True)
    spec = primary_spec(material)
    run_args = SimpleNamespace(material=material, n_parallel=2, tasks_per_qe=16,
                               fit_only=False, force_rerun=False)
    phonon = scaled_phonon(material, args.a)
    sup = phonon.supercell
    eq = np.asarray(sup.positions)
    cell = np.asarray(sup.cell)
    symbols = list(sup.symbols)
    masses = np.asarray(sup.masses)

    # Undisplaced cell and harmonic FD trial.
    phonon.generate_displacements(distance=0.01, is_plusminus=True)
    fd_pos = [eq] + [np.asarray(c.positions) for c in phonon.supercells_with_displacements]
    with concurrent.futures.ThreadPoolExecutor(max_workers=4) as pool:
        res = list(pool.map(lambda i: run_one_state(i, fd_pos[i], cell, symbols, spec, run_args,
                                                    out / "fd_qe", overrides(spec)),
                            range(len(fd_pos))))
    f0 = np.asarray(res[0]["forces_ev_ang"])
    e0 = float(res[0]["energy_ev"])
    phonon.forces = np.asarray([np.asarray(r["forces_ev_ang"]) - f0 for r in res[1:]])
    phonon.produce_force_constants()
    harmonic = np.asarray(phonon.force_constants)
    np.savez_compressed(out / "fd.npz", force_constants=harmonic, energy0_ev=e0,
                        forces0_ev_ang=f0, cell_ang=cell, positions_ang=eq, a_ang=args.a,
                        n_calls=len(fd_pos))
    print(f"{material} a={args.a:.4f}: E0={e0:.6f} eV, FD done", flush=True)

    basis = symfc_fc2_basis(eq, cell, symbols, PRIMARY[material]["cutoff"])
    c_harm, *_ = np.linalg.lstsq(basis.reshape(len(basis), -1).T, harmonic.ravel(), rcond=None)

    def run_temperature(it, temperature):
        rng = np.random.default_rng(SEED0 + 1000 * it + int(round(1000 * args.a)))
        stage = out / f"T{temperature:.0f}_qe"
        coef, fc = c_harm, to_fc(basis, c_harm)
        us, fs, es, coefs = [], [], [], []
        prior = np.zeros((len(basis), len(basis)))
        for k in range(args.budget // 2):
            patterns, sigma = modes(fc, masses, temperature)
            signs, _ = sequential_d_optimal(patterns, sigma, basis, 1,
                                            seed=int(rng.integers(1 << 30)),
                                            prior_information=prior)
            u = paired(configurations(signs, patterns, sigma))
            u -= u.mean(axis=1, keepdims=True)
            res = [run_one_state(2 * k + j, eq + u[j], cell, symbols, spec, run_args, stage,
                                 overrides(spec)) for j in range(2)]
            df = np.asarray([np.asarray(r["forces_ev_ang"]) - f0 for r in res])
            us.append(u)
            fs.append(df - df.mean(axis=1, keepdims=True))
            es.append([float(r["energy_ev"]) - e0 for r in res])
            prior = prior + information_matrix(u, basis)
            coef = fit_coefficients(basis, np.concatenate(us), np.concatenate(fs), c_harm)
            fc = to_fc(basis, coef)
            coefs.append(coef)
        np.savez_compressed(out / f"T{temperature:.0f}.npz", temperature_k=temperature,
                            force_constants=np.asarray([to_fc(basis, c) for c in coefs]),
                            displacements_ang=np.concatenate(us),
                            delta_forces_ev_ang=np.concatenate(fs),
                            delta_energies_ev=np.concatenate(es))
        print(f"{material} a={args.a:.4f} T={temperature:.0f}: {args.budget} calls", flush=True)

    # Two QE runs per temperature at a time (16 MPI tasks each).
    with concurrent.futures.ThreadPoolExecutor(max_workers=len(args.temperatures)) as pool:
        list(pool.map(lambda x: run_temperature(*x), enumerate(args.temperatures)))


if __name__ == "__main__":
    main()
