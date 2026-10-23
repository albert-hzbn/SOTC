#!/usr/bin/env python3
"""Same-cell controls for the nine primary SOTC benchmarks.

All states use exactly the primary SOTC supercell (32 or 64 atoms) and the
primary QE settings (cutoffs, pseudopotentials, smearing, conv_thr=1e-8 Ry,
2x2x2 k for metals and Gamma for rocksalt):

* ``fd``: central finite-displacement harmonic FC2 in the primary cell;
* ``canonical``: 24 +/- pairs of quantum-canonical displacements at the
  primary design temperature, drawn from the same-cell FD model.  Pairs
  0-19 form the training pool for matched random baselines and for the
  converged effective-FC2 reference; pairs 20-23 are held out;
* ``canonical1``/``sotcd1``: the same from the iteration-1 trial (effective
  FC2 of the 40 canonical training snapshots), i.e. one sTDEP update;
* ``boot``: SOTC-D whose trial is updated from its own first pair;
* ``rattle``/``zg``: thermal-amplitude random rattle and Zacharias--Giustino
  type alternating-sign configurations from the harmonic trial.

States are checkpointed individually and can be distributed over several
single-node jobs with ``--chunk``/``--nchunks``.
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

from analyze_primary_archives import PRIMARY, primitive  # noqa: E402
from run_fd_references import MATERIALS  # noqa: E402
from run_real_anharmonic_ifcs import (  # noqa: E402
    harmonic_reference,
    paired_thermal_displacements,
    run_one_state,
)
from sotc.validation import phonopy_atoms  # noqa: E402

N_SNAPSHOTS = 48
N_TRAIN = 40
SEED = 271828


# Primary-cell k meshes of the new structure classes (64-atom cells).
PRIMARY_KPTS = {"si": (2, 2, 2), "mg": (3, 3, 3)}


def primary_spec(material):
    spec = dict(MATERIALS[material])
    spec["kpts"] = PRIMARY_KPTS.get(
        material, (2, 2, 2) if spec["smearing"] is not None else (1, 1, 1))
    spec["conv_thr"] = 1.0e-8
    return spec


def primary_phonon(material):
    from phonopy import Phonopy

    if "supercell_matrix" in PRIMARY[material]:
        from run_real_anharmonic_ifcs import primitive_structure, structure_symbols

        spec = PRIMARY[material]
        cell, positions = primitive_structure(spec)
        unit = phonopy_atoms(structure_symbols(spec, positions), cell, positions)
        return Phonopy(unit, np.asarray(spec["supercell_matrix"]).T)

    data = np.load(
        ROOT / "validation_results" / material / "primary_audit"
        / "primary_force_dataset.npz"
    )
    pos = np.asarray(data["equilibrium_positions_ang"])
    cell = np.asarray(data["cell_ang"])
    symbols = [str(s) for s in data["symbols"]]
    pcell, _ = primitive(PRIMARY[material])
    primitive_matrix = (pcell @ np.linalg.inv(cell)).T
    unit = phonopy_atoms(symbols, cell, pos)
    return Phonopy(unit, np.eye(3, dtype=int), primitive_matrix=primitive_matrix)


def overrides(spec):
    return {"kpts": spec["kpts"], "conv_thr": spec["conv_thr"]}


N_SOTCD = 8
SOTCD_SEED = 1618


def sotcd_design(material, phonon, out):
    """Nested D-optimal thermal-sign design from the same-cell FD model,
    i.e. from exactly the trial model used to draw the canonical baseline."""
    from analyze_primary_archives import symfc_fc2_basis
    from sotc.optimal_design import configurations, sequential_d_optimal, thermal_modes

    archive = out / "sotcd_displacements.npz"
    if archive.exists():
        return np.load(archive)["displacements_ang"]
    spec = PRIMARY[material]
    pos = np.asarray(phonon.supercell.positions, dtype=float)
    cell = np.asarray(phonon.supercell.cell, dtype=float)
    symbols = list(phonon.supercell.symbols)
    masses = np.asarray(phonon.supercell.masses, dtype=float)
    _, patterns, sigma = thermal_modes(
        np.asarray(phonon.force_constants), masses, spec["T"]
    )
    basis = symfc_fc2_basis(pos, cell, symbols, spec["cutoff"])
    signs, report = sequential_d_optimal(
        patterns, sigma, basis, N_SOTCD, seed=SOTCD_SEED
    )
    u = configurations(signs, patterns, sigma)
    u -= np.mean(u, axis=1, keepdims=True)
    paired = np.stack((u, -u), axis=1).reshape(2 * N_SOTCD, len(pos), 3)
    np.savez_compressed(
        archive, displacements_ang=paired, signs=signs, sigma=sigma,
        logdet_by_prefix=np.asarray(report["logdet_by_prefix"]),
        pairing="interleaved u,-u", temperature_k=spec["T"],
    )
    return paired


def run_sotcd(material, phonon, spec, run_args, out, n_parallel):
    u = sotcd_design(material, phonon, out)
    eq = np.asarray(phonon.supercell.positions, dtype=float)
    cell = np.asarray(phonon.supercell.cell, dtype=float)
    symbols = list(phonon.supercell.symbols)
    stage = out / "sotcd_qe"
    stage.mkdir(exist_ok=True)
    with concurrent.futures.ThreadPoolExecutor(max_workers=n_parallel) as pool:
        futures = {
            pool.submit(run_one_state, i, eq + u[i], cell, symbols, spec,
                        run_args, stage, overrides(spec)): i
            for i in range(len(u))
        }
        for done, future in enumerate(concurrent.futures.as_completed(futures), 1):
            future.result()
            print(f"{material} SOTC-D state {futures[future]:04d} "
                  f"({done}/{len(u)})", flush=True)


ITER1_SEED = 314159
RATTLE_SEED = 2718
STDEP_SEEDS = (0, 1, 2)
STDEP_BUDGETS = (8, 16)
STDEP_SEED0 = 4242


def effective_trial(material, phonon, out):
    """Iteration-1 trial: the effective FC2 fitted to the 40 canonical
    training snapshots of iteration 0 (one sTDEP update of the trial)."""
    from analyze_primary_archives import symfc_model
    from analyze_robustness import canonical_dataset

    archive = out / "iter1" / "effective_trial.npz"
    if archive.exists():
        fc = np.load(archive)["force_constants_ev_ang2"]
    else:
        u, df = canonical_dataset(material)
        pos = np.asarray(phonon.supercell.positions, dtype=float)
        cell = np.asarray(phonon.supercell.cell, dtype=float)
        symbols = list(phonon.supercell.symbols)
        fitted, _ = symfc_model(pos, cell, symbols, PRIMARY[material],
                                u[:N_TRAIN], df[:N_TRAIN], PRIMARY[material]["cutoff"])
        fc = np.asarray(fitted.force_constants)
        archive.parent.mkdir(parents=True, exist_ok=True)
        np.savez_compressed(archive, force_constants_ev_ang2=fc)
    import phonopy

    # A fresh object: the harmonic ``phonon`` must stay unmodified for the
    # stages designed from the harmonic trial.
    trial = phonopy.load(out / "fd" / "phonopy_harmonic.yaml")
    trial.force_constants = fc
    return trial


def anchored_update(material, phonon, u, df):
    """Bootstrap trial: the minimum-norm correction of the harmonic FC2 that
    fits the forces of the first SOTC-D pair (exact when identifiable)."""
    from analyze_primary_archives import symfc_fc2_basis

    pos = np.asarray(phonon.supercell.positions, dtype=float)
    cell = np.asarray(phonon.supercell.cell, dtype=float)
    basis = symfc_fc2_basis(pos, cell, list(phonon.supercell.symbols),
                            PRIMARY[material]["cutoff"])
    flat = basis.reshape(len(basis), -1)
    c_h, *_ = np.linalg.lstsq(flat.T, np.asarray(phonon.force_constants).ravel(),
                              rcond=None)
    x = np.stack([-np.einsum("ijab,sjb->sia", t, u).ravel() for t in basis], axis=1)
    y = (df - df.mean(axis=1, keepdims=True)).ravel()
    delta, *_ = np.linalg.lstsq(x, y - x @ c_h, rcond=None)
    return np.einsum("k,kijab->ijab", c_h + delta, basis), basis


def design_stage(material, stage, phonon, out):
    """Displacements (K, N, 3) and output directory of one design stage."""
    from analyze_primary_archives import symfc_fc2_basis
    from analyze_robustness import sotcd_dataset
    from sotc.optimal_design import (
        configurations, information_matrix, sequential_d_optimal,
        thermal_modes, thermal_rattle, walsh_signs, zg_modes,
    )

    spec = PRIMARY[material]
    folder = out / ("iter1" if stage in ("canonical1", "sotcd1", "boot") else "designs")
    folder.mkdir(parents=True, exist_ok=True)
    archive = folder / f"{stage}_displacements.npz"
    if archive.exists():
        return np.load(archive)["displacements_ang"], folder / f"{stage}_qe"
    pos = np.asarray(phonon.supercell.positions, dtype=float)
    cell = np.asarray(phonon.supercell.cell, dtype=float)
    symbols = list(phonon.supercell.symbols)
    masses = np.asarray(phonon.supercell.masses, dtype=float)

    def pair(u):
        u = u - np.mean(u, axis=1, keepdims=True)
        return np.stack((u, -u), axis=1).reshape(2 * len(u), len(pos), 3)

    extra = {}
    if stage == "canonical1":
        trial = effective_trial(material, phonon, out)
        u = paired_thermal_displacements(trial, spec["T"], N_SNAPSHOTS,
                                         ITER1_SEED, archive)
        return u, folder / f"{stage}_qe"
    if stage in ("sotcd1", "boot"):
        basis = symfc_fc2_basis(pos, cell, symbols, spec["cutoff"])
        prior = None
        if stage == "sotcd1":
            fc = np.asarray(effective_trial(material, phonon, out).force_constants)
            n_new = N_SOTCD
        else:
            u0, df0 = sotcd_dataset(material)
            fc, _ = anchored_update(material, phonon, u0[:2], df0[:2])
            prior = information_matrix(u0[:2], basis)
            n_new = N_SOTCD - 1
            extra["force_constants_ev_ang2"] = fc
        _, patterns, sigma = thermal_modes(fc, masses, spec["T"])
        signs, report = sequential_d_optimal(patterns, sigma, basis, n_new,
                                             seed=SOTCD_SEED,
                                             prior_information=prior)
        u = pair(configurations(signs, patterns, sigma))
        if stage == "boot":
            u = np.concatenate([u0[:2], u])
        extra["logdet_by_prefix"] = np.asarray(report["logdet_by_prefix"])
    elif stage == "stdep":
        # Two-iteration sTDEP at budget B: B/2 paired canonical snapshots of
        # the harmonic trial (a random subset of complete pairs of the
        # iteration-0 pool), refit, then B/2 paired canonical snapshots of
        # the refitted model (computed here).  The refit is the minimum-norm
        # correction of the harmonic FC2, as for the bootstrapped SOTC-D.
        from analyze_robustness import canonical_dataset

        uc, fc_ = canonical_dataset(material)
        rows, meta = [], []
        for seed in STDEP_SEEDS:
            for budget in STDEP_BUDGETS:
                rng = np.random.default_rng(STDEP_SEED0 + 100 * seed + budget)
                pairs = rng.choice(N_TRAIN // 2, size=budget // 4, replace=False)
                first = np.ravel(np.stack((2 * pairs, 2 * pairs + 1), axis=1))
                fc, _ = anchored_update(material, phonon, uc[first], fc_[first])
                _, patterns, sigma = thermal_modes(fc, masses, spec["T"])
                q = rng.standard_normal((budget // 4, len(sigma))) * sigma
                rows.append(pair(np.einsum("km,mia->kia", q, patterns)))
                meta.append(np.concatenate([[seed, budget], first]))
        u = np.concatenate(rows)
        width = max(len(m) for m in meta)
        extra["blocks"] = np.asarray([np.pad(m, (0, width - len(m)), constant_values=-1)
                                      for m in meta])
    elif stage == "rattle":
        rng = np.random.default_rng(RATTLE_SEED)
        u = pair(thermal_rattle(rng, phonon, spec["T"], N_SOTCD))
    elif stage == "zg":
        _, patterns, sigma = zg_modes(phonon, spec["T"])
        u = pair(configurations(walsh_signs(len(sigma), N_SOTCD), patterns, sigma))
    else:
        raise ValueError(stage)
    np.savez_compressed(archive, displacements_ang=u, temperature_k=spec["T"],
                        pairing="interleaved u,-u", **extra)
    return u, folder / f"{stage}_qe"


def run_design(material, stage, phonon, spec, run_args, out, args):
    u, stage_dir = design_stage(material, stage, phonon, out)
    eq = np.asarray(phonon.supercell.positions, dtype=float)
    cell = np.asarray(phonon.supercell.cell, dtype=float)
    symbols = list(phonon.supercell.symbols)
    stage_dir.mkdir(parents=True, exist_ok=True)
    # The bootstrap design reuses the first SOTC-D pair already computed.
    first = 2 if stage == "boot" else 0
    indices = [i for i in range(first, len(u)) if i % args.nchunks == args.chunk]
    if args.reverse:
        indices = indices[::-1]
    indices = [i for i in indices
               if not (stage_dir / f"state_{i:04d}" / "result.npz").exists()]
    with concurrent.futures.ThreadPoolExecutor(max_workers=args.n_parallel) as pool:
        futures = {
            pool.submit(run_one_state, i, eq + u[i], cell, symbols, spec,
                        run_args, stage_dir, overrides(spec)): i
            for i in indices
        }
        for done, future in enumerate(concurrent.futures.as_completed(futures), 1):
            future.result()
            print(f"{material} {stage} state {futures[future]:04d} "
                  f"({done}/{len(indices)})", flush=True)


DESIGN_STAGES = ("canonical1", "sotcd1", "boot", "rattle", "zg", "stdep")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("material", choices=sorted(PRIMARY))
    parser.add_argument("stage", choices=["fd", "canonical", "sotcd", *DESIGN_STAGES])
    parser.add_argument("--chunk", type=int, default=0)
    parser.add_argument("--nchunks", type=int, default=1)
    parser.add_argument("--n-parallel", type=int, default=4)
    parser.add_argument("--tasks-per-qe", type=int, default=32)
    parser.add_argument("--reverse", action="store_true",
                        help="process canonical states from the end, so a second "
                             "job can share the work of a running one")
    parser.add_argument("--fit-only", action="store_true")
    parser.add_argument("--force-rerun", action="store_true")
    args = parser.parse_args()
    material = args.material
    spec = primary_spec(material)
    run_args = SimpleNamespace(
        material=material, n_parallel=args.n_parallel,
        tasks_per_qe=args.tasks_per_qe, fit_only=args.fit_only,
        force_rerun=args.force_rerun,
    )
    out = ROOT / "validation_results" / material / "primary_cell"
    out.mkdir(parents=True, exist_ok=True)

    if args.stage == "fd":
        fd = out / "fd"
        fd.mkdir(exist_ok=True)
        result = harmonic_reference(primary_phonon(material), spec, run_args, fd)
        print(f"{material} same-cell FD: {result['n_displaced_states']} displaced, "
              f"min mesh frequency {result['min_mesh_frequency_thz']:.3f} THz",
              flush=True)
        return

    import phonopy

    fd_yaml = out / "fd" / "phonopy_harmonic.yaml"
    phonon = phonopy.load(fd_yaml)
    if args.stage == "sotcd":
        run_sotcd(material, phonon, spec, run_args, out, args.n_parallel)
        return
    if args.stage in DESIGN_STAGES:
        run_design(material, args.stage, phonon, spec, run_args, out, args)
        return
    archive = out / "canonical_displacements.npz"
    u = paired_thermal_displacements(
        phonon, PRIMARY[material]["T"], N_SNAPSHOTS, SEED, archive
    )
    eq = np.asarray(phonon.supercell.positions, dtype=float)
    cell = np.asarray(phonon.supercell.cell, dtype=float)
    symbols = list(phonon.supercell.symbols)
    stage = out / "canonical_qe"
    stage.mkdir(exist_ok=True)
    indices = [i for i in range(N_SNAPSHOTS) if i % args.nchunks == args.chunk]
    if args.reverse:
        indices = indices[::-1]
    stage_dir = out / "canonical_qe"
    indices = [
        i for i in indices
        if not (stage_dir / f"state_{i:04d}" / "result.npz").exists()
    ]
    with concurrent.futures.ThreadPoolExecutor(max_workers=args.n_parallel) as pool:
        futures = {
            pool.submit(
                run_one_state, i, eq + u[i], cell, symbols, spec, run_args,
                stage, overrides(spec),
            ): i for i in indices
        }
        for done, future in enumerate(concurrent.futures.as_completed(futures), 1):
            future.result()
            print(f"{material} canonical state {futures[future]:04d} "
                  f"({done}/{len(indices)})", flush=True)


if __name__ == "__main__":
    main()
