#!/usr/bin/env python3
"""Matched comparison of displacement designs and FC2 bases (primary cells).

Datasets (``run_primary_cell_controls.py``), all in the primary SOTC cell
with the primary QE settings:

* iteration 0 (harmonic same-cell FD trial): ``canonical`` (40 training +
  8 test snapshots), ``sotcd``, ``rattle`` and ``zg``;
* iteration 1 (trial = effective FC2 of the 40 iteration-0 snapshots, one
  sTDEP update): ``canonical1`` (40 + 8) and ``sotcd1``;
* ``boot``: SOTC-D whose trial is updated from its own first pair.

Targets: the effective FC2 fitted to the 40 canonical training snapshots of
iteration 0 or of iteration 1.  Every design is scored against the target of
its trial level, and also against the other target where meaningful (the
cost of skipping the trial update).

Bases: ``supercell`` (symfc FC2 basis of the SOTC supercell, i.e. the
supercell space group) and ``crystal`` (hiPhive cluster space of the full
crystal space group, pair cutoff limited to half the shortest image
distance so that no pair is aliased).

Scores: held-out force RMSE on the 8 test snapshots of the target
iteration; mesh RMSE against the target fitted to the 40-n canonical
snapshots disjoint from the design (median over random complements for the
deterministic designs).
"""

from __future__ import annotations

import io
import contextlib
import itertools
import json
from pathlib import Path
import sys

import numpy as np

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "codes"))
sys.path.insert(0, str(Path(__file__).resolve().parent))

from analyze_primary_archives import (  # noqa: E402
    PRIMARY, frequency_metrics, mesh_points, phonopy_frequencies, symfc_fc2_basis,
)

N_TRAIN = 40
COUNTS = (2, 4, 8, 16)
N_SEEDS = 20
MESH = 12


def load_stage(material, archive, qe_dirs):
    base = ROOT / "validation_results" / material / "primary_cell"
    path = base / archive
    if not path.exists():
        return None
    u = np.load(path)["displacements_ang"]
    with np.load(base / "fd" / "harmonic_reference.npz") as h:
        f0 = np.asarray(h["reference_forces_ev_ang"])
    forces = []
    for i in range(len(u)):
        for directory in qe_dirs:
            result = base / directory / f"state_{i:04d}" / "result.npz"
            if result.exists():
                forces.append(np.load(result)["forces_ev_ang"])
                break
        else:
            return None
    df = np.asarray(forces) - f0
    return u, df - df.mean(axis=1, keepdims=True)


def datasets(material):
    return {
        "canonical": load_stage(material, "canonical_displacements.npz", ["canonical_qe"]),
        "sotcd": load_stage(material, "sotcd_displacements.npz", ["sotcd_qe"]),
        "rattle": load_stage(material, "designs/rattle_displacements.npz",
                             ["designs/rattle_qe"]),
        "zg": load_stage(material, "designs/zg_displacements.npz", ["designs/zg_qe"]),
        "canonical1": load_stage(material, "iter1/canonical1_displacements.npz",
                                 ["iter1/canonical1_qe"]),
        "sotcd1": load_stage(material, "iter1/sotcd1_displacements.npz",
                             ["iter1/sotcd1_qe"]),
        "boot": load_stage(material, "iter1/boot_displacements.npz",
                           ["iter1/boot_qe", "sotcd_qe"]),
        "stdep": load_stage(material, "designs/stdep_displacements.npz",
                            ["designs/stdep_qe"]),
    }


def stdep_blocks(material):
    """(seed, budget, pool indices of the first iteration) of every sTDEP block,
    in the order of the second-iteration states in the archive."""
    path = (ROOT / "validation_results" / material / "primary_cell" / "designs"
            / "stdep_displacements.npz")
    blocks = []
    for row in np.load(path)["blocks"]:
        seed, budget, *first = (int(x) for x in row)
        blocks.append((seed, budget, [i for i in first if i >= 0]))
    return blocks


def min_image_distance(cell):
    return min(np.linalg.norm(np.asarray(n) @ cell)
               for n in itertools.product(range(-2, 3), repeat=3) if any(n))


def crystal_basis(phonon, cutoff):
    """FC2 basis tensors (P, N, N, 3, 3) of the full crystal space group."""
    from ase import Atoms
    from hiphive import ClusterSpace, ForceConstantPotential

    sup, prim = phonon.supercell, phonon.primitive
    cell = np.asarray(sup.cell)
    rc = min(cutoff + 0.01, min_image_distance(cell) / 2.0 - 0.01)
    atoms = Atoms(symbols=list(sup.symbols), positions=np.asarray(sup.positions),
                  cell=cell, pbc=True)
    with contextlib.redirect_stdout(io.StringIO()):
        cs = ClusterSpace(Atoms(symbols=list(prim.symbols),
                                positions=np.asarray(prim.positions),
                                cell=np.asarray(prim.cell), pbc=True), [rc])
        tensors = []
        for k in range(cs.n_dofs):
            p = np.zeros(cs.n_dofs)
            p[k] = 1.0
            tensors.append(ForceConstantPotential(cs, p).get_force_constants(atoms)
                           .get_fc_array(order=2))
    return np.asarray(tensors), rc


class Fitter:
    def __init__(self, basis):
        self.basis = basis

    def __call__(self, u, df):
        x = np.stack([-np.einsum("ijab,sjb->sia", t, u).ravel() for t in self.basis], axis=1)
        coef, *_ = np.linalg.lstsq(x, df.ravel(), rcond=None)
        return np.einsum("k,kijab->ijab", coef, self.basis)


def analyze(material):
    import phonopy

    spec = PRIMARY[material]
    base = ROOT / "validation_results" / material / "primary_cell"
    phonon = phonopy.load(base / "fd" / "phonopy_harmonic.yaml")
    pos = np.asarray(phonon.supercell.positions)
    cell = np.asarray(phonon.supercell.cell)
    symbols = list(phonon.supercell.symbols)
    qmesh = mesh_points(MESH)
    data = datasets(material)
    tensors, rc = crystal_basis(phonon, spec["cutoff"])
    bases = {
        "supercell": symfc_fc2_basis(pos, cell, symbols, spec["cutoff"]),
        "crystal": tensors,
    }

    def mesh(fc):
        phonon.force_constants = fc
        return phonopy_frequencies(phonon, qmesh)

    report = {"material": material, "counts": list(COUNTS),
              "n_parameters": {k: int(len(v)) for k, v in bases.items()},
              "crystal_cutoff_ang": rc, "supercell_cutoff_ang": spec["cutoff"],
              "targets": {}}
    targets = {"iter0": "canonical", "iter1": "canonical1"}
    scored = {
        "iter0": ["sotcd", "rattle", "zg", "boot", "sotcd1"],
        "iter1": ["sotcd1", "boot", "sotcd", "rattle", "zg"],
    }
    for target, pool_name in targets.items():
        pool = data[pool_name]
        if pool is None:
            continue
        u_pool, f_pool = pool
        u_test, f_test = u_pool[N_TRAIN:], f_pool[N_TRAIN:]
        block = {}
        for bname, basis in bases.items():
            fit = Fitter(basis)
            cache = {}

            def disjoint(indices):
                key = tuple(sorted(int(i) for i in indices))
                if key not in cache:
                    cache[key] = mesh(fit(u_pool[list(key)], f_pool[list(key)]))
                return cache[key]

            def held_out(fc):
                pred = -np.einsum("ijab,sjb->sia", fc, u_test)
                return float(np.sqrt(np.mean((pred - f_test) ** 2)))

            full = fit(u_pool[:N_TRAIN], f_pool[:N_TRAIN])
            full_mesh = mesh(full)
            entry = {"reference_held_out_force_rmse": held_out(full),
                     "reference_min_frequency_thz": float(full_mesh.min()),
                     "designs": {}, "canonical_unpaired": {}, "canonical_paired": {},
                     "other_iteration_canonical": {}}
            for name in scored[target]:
                if data[name] is None:
                    continue
                u, df = data[name]
                rows = {}
                for c in COUNTS:
                    fc = fit(u[:c], df[:c])
                    freq = mesh(fc)
                    scores = [frequency_metrics(disjoint(np.random.default_rng(
                        31 * c + s).choice(N_TRAIN, size=N_TRAIN - c, replace=False)),
                        freq)["rmse_thz"] for s in range(N_SEEDS)]
                    rows[str(c)] = {
                        "held_out_force_rmse": held_out(fc),
                        "mesh_vs_disjoint": float(np.median(scores)),
                        "mesh_vs_full_reference": frequency_metrics(full_mesh, freq)["rmse_thz"],
                        "unstable_fraction": float(np.mean(freq < -0.05)),
                    }
                entry["designs"][name] = rows
            for kind in ("canonical_unpaired", "canonical_paired"):
                for c in COUNTS:
                    trials = []
                    for seed in range(N_SEEDS):
                        rng = np.random.default_rng(7000 + 97 * c + seed)
                        if kind == "canonical_unpaired":
                            chosen = rng.choice(N_TRAIN, size=c, replace=False)
                        else:
                            pairs = rng.choice(N_TRAIN // 2, size=c // 2, replace=False)
                            chosen = np.ravel(np.stack((2 * pairs, 2 * pairs + 1), axis=1))
                        fc = fit(u_pool[chosen], f_pool[chosen])
                        freq = mesh(fc)
                        complement = np.setdiff1d(np.arange(N_TRAIN), chosen)
                        trials.append({
                            "held_out_force_rmse": held_out(fc),
                            "mesh_vs_disjoint": frequency_metrics(disjoint(complement), freq)["rmse_thz"],
                            "unstable_fraction": float(np.mean(freq < -0.05)),
                        })
                    entry[kind][str(c)] = trials
            # Two-iteration sTDEP (DFT): first half from the iteration-0 pool,
            # second half drawn from the refitted model; fitted to the last
            # iteration only or to both, scored against the full target.
            if target == "iter1" and data["stdep"] is not None:
                u_st, f_st = data["stdep"]
                u0, f0 = data["canonical"]
                start = 0
                rows = {}
                for seed, budget, first in stdep_blocks(material):
                    n2 = budget // 2
                    u2, f2 = u_st[start:start + n2], f_st[start:start + n2]
                    start += n2
                    for kind, (uu, ff) in (
                            ("stdep_last", (u2, f2)),
                            ("stdep_pool", (np.concatenate([u0[first], u2]),
                                            np.concatenate([f0[first], f2])))):
                        fc = fit(uu, ff)
                        freq = mesh(fc)
                        rows.setdefault(kind, {}).setdefault(str(budget), []).append({
                            "seed": seed,
                            "held_out_force_rmse": held_out(fc),
                            "mesh_vs_full_reference": frequency_metrics(full_mesh, freq)["rmse_thz"],
                            "unstable_fraction": float(np.mean(freq < -0.05)),
                        })
                entry["stdep"] = rows
            # Canonical subsets of the other iteration scored on this target:
            # the error of not updating (or of updating) the trial model.
            other = data["canonical1" if target == "iter0" else "canonical"]
            if other is not None:
                for c in COUNTS:
                    trials = []
                    for seed in range(N_SEEDS):
                        rng = np.random.default_rng(9000 + 97 * c + seed)
                        pairs = rng.choice(N_TRAIN // 2, size=c // 2, replace=False)
                        chosen = np.ravel(np.stack((2 * pairs, 2 * pairs + 1), axis=1))
                        fc = fit(other[0][chosen], other[1][chosen])
                        freq = mesh(fc)
                        trials.append({
                            "held_out_force_rmse": held_out(fc),
                            "mesh_vs_full_reference": frequency_metrics(full_mesh, freq)["rmse_thz"],
                        })
                    entry["other_iteration_canonical"][str(c)] = trials
            block[bname] = entry
            print(f"{material} {target} {bname}: reference F "
                  f"{entry['reference_held_out_force_rmse']:.4f}", flush=True)
        report["targets"][target] = block
    if data["canonical1"] is not None:
        sc, cr = (mesh(Fitter(bases[b])(*[x[:N_TRAIN] for x in data["canonical1"]]))
                  for b in ("supercell", "crystal"))
        report["basis_difference_thz"] = frequency_metrics(sc, cr)["rmse_thz"]
    if data["canonical"] is not None and data["canonical1"] is not None:
        for bname, basis in bases.items():
            fit = Fitter(basis)
            f0 = mesh(fit(*[x[:N_TRAIN] for x in data["canonical"]]))
            f1 = mesh(fit(*[x[:N_TRAIN] for x in data["canonical1"]]))
            report.setdefault("iteration_change_thz", {})[bname] = \
                frequency_metrics(f0, f1)["rmse_thz"]
    out = ROOT / "validation_results" / material / "robustness"
    out.mkdir(parents=True, exist_ok=True)
    (out / "design_comparison.json").write_text(json.dumps(report, indent=1) + "\n")


def main():
    for material in sys.argv[1:] or sorted(PRIMARY):
        analyze(material)


if __name__ == "__main__":
    main()
