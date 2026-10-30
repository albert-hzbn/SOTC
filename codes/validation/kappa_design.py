#!/usr/bin/env python3
"""Lattice thermal conductivity from designed FC3 data (Si, SOTC cell).

Stages (run in this order):

* ``fd3-prepare`` (venv with phono3py): finite-displacement FC3 supercells of
  the 64-atom cell (0.03 A, pair cutoff 5.0 A), written to
  primary_cell/fc3_fd/displacements.npz;
* ``fd3-compute`` (project venv, Slurm): DFT forces of those supercells;
* ``design`` (project venv): eight +-u pairs at the thermal amplitudes of the
  harmonic trial (600 K) whose sign patterns maximize the FC3 information
  given the SOTC-D pairs (``sotc.optimal_design.d_optimal_fc3_pairs``);
* ``design-compute`` (project venv, Slurm): their DFT forces;
* ``fit`` (project venv): joint FC2+FC3 fits (hiPhive cluster space, cutoffs
  5.0/4.0 A, least squares) to canonical pairs, SOTC-D, the FC3 design,
  and SOTC-D plus the FC3 design at increasing numbers of calls; FC3
  written in phono3py format;
* ``kappa`` (venv with phono3py): kappa_L(300 K) in the relaxation-time
  approximation on a 17^3 mesh, with the harmonic FD FC2 of the cell for
  every FC3 so that only the FC3 differs, and for the finite-difference FC3.

Writes validation_results/si/robustness/kappa_design.json.
"""

from __future__ import annotations

import argparse
import concurrent.futures
import json
from pathlib import Path
import sys
from types import SimpleNamespace

import numpy as np

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "codes"))
sys.path.insert(0, str(Path(__file__).resolve().parent))

MATERIAL = "si"
BASE = ROOT / "validation_results" / MATERIAL / "primary_cell"
OUT = BASE / "kappa"
CUTOFFS = (5.0, 4.0)
N_PAIRS = 8
TEMPERATURE_KAPPA = 300.0
MESH = (17, 17, 17)
BUDGETS = (4, 8, 16, 32)


def fd3_prepare():
    import phonopy
    from phono3py import Phono3py

    ph = phonopy.load(BASE / "fd" / "phonopy_harmonic.yaml")
    p3 = Phono3py(ph.unitcell, supercell_matrix=ph.supercell_matrix,
                  primitive_matrix=ph.primitive_matrix)
    p3.generate_displacements(distance=0.03, cutoff_pair_distance=CUTOFFS[0])
    cells = [c for c in p3.supercells_with_displacements if c is not None]
    (BASE / "fc3_fd").mkdir(parents=True, exist_ok=True)
    np.savez_compressed(BASE / "fc3_fd" / "displacements.npz",
                        positions_ang=np.asarray([c.positions for c in cells]))
    p3.save(BASE / "fc3_fd" / "phono3py_disp.yaml")
    print(f"{len(cells)} FD FC3 supercells", flush=True)


def compute(stage_name, positions, chunk, nchunks, n_parallel):
    import phonopy
    from run_primary_cell_controls import overrides, primary_spec
    from run_real_anharmonic_ifcs import run_one_state

    ph = phonopy.load(BASE / "fd" / "phonopy_harmonic.yaml")
    cell = np.asarray(ph.supercell.cell)
    symbols = list(ph.supercell.symbols)
    spec = primary_spec(MATERIAL)
    run_args = SimpleNamespace(material=MATERIAL, n_parallel=n_parallel, tasks_per_qe=32,
                               fit_only=False, force_rerun=False)
    stage = BASE / stage_name
    todo = [i for i in range(len(positions)) if i % nchunks == chunk]
    with concurrent.futures.ThreadPoolExecutor(max_workers=n_parallel) as pool:
        list(pool.map(lambda i: run_one_state(i, positions[i], cell, symbols, spec, run_args,
                                              stage, overrides(spec)), todo))


def cluster_space():
    from ase import Atoms
    from hiphive import ClusterSpace
    import phonopy

    ph = phonopy.load(BASE / "fd" / "phonopy_harmonic.yaml")
    prim = ph.primitive
    cs = ClusterSpace(Atoms(list(prim.symbols), positions=np.asarray(prim.positions),
                            cell=np.asarray(prim.cell), pbc=True), list(CUTOFFS))
    sup = ph.supercell
    atoms = Atoms(list(sup.symbols), positions=np.asarray(sup.positions),
                  cell=np.asarray(sup.cell), pbc=True)
    n2 = ClusterSpace(cs.primitive_structure, [CUTOFFS[0]]).n_dofs
    return cs, atoms, n2, ph


def fit_matrix(cs, atoms, us, fs=None):
    from hiphive import StructureContainer

    sc = StructureContainer(cs)
    for k, u in enumerate(us):
        a = atoms.copy()
        a.new_array("displacements", np.asarray(u))
        a.new_array("forces", np.zeros_like(u) if fs is None else np.asarray(fs[k]))
        sc.add_structure(a)
    return sc.get_fit_data()


def design():
    from analyze_designs import datasets
    from sotc.optimal_design import (configurations, d_optimal_fc3_pairs, thermal_modes)
    from analyze_primary_archives import PRIMARY

    cs, atoms, n2, ph = cluster_space()
    _, patterns, sigma = thermal_modes(np.asarray(ph.force_constants),
                                       np.asarray(ph.supercell.masses), PRIMARY[MATERIAL]["T"])
    rows = lambda u: fit_matrix(cs, atoms, [u])[0][:, n2:]  # noqa: E731
    u_sotcd = datasets(MATERIAL)["sotcd"][0]
    prior = sum(rows(u).T @ rows(u) for u in u_sotcd)
    signs, meta = d_optimal_fc3_pairs(patterns, sigma, rows, N_PAIRS, n_candidates=48,
                                      seed=2026, prior_information=prior)
    u = configurations(signs, patterns, sigma)
    u = np.concatenate([[x, -x] for x in u])
    OUT.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(OUT / "fc3_design.npz", displacements_ang=u, signs=signs,
                        logdet=np.asarray(meta["logdet"]))
    print("FC3 design logdet per pair:", np.round(meta["logdet"], 2), flush=True)


def fit():
    from hiphive import ForceConstantPotential
    from analyze_designs import datasets

    cs, atoms, n2, ph = cluster_space()
    data = datasets(MATERIAL)
    eq = np.asarray(ph.supercell.positions)
    with np.load(BASE / "fd" / "harmonic_reference.npz") as h:
        f0 = np.asarray(h["reference_forces_ev_ang"])
    d = np.load(OUT / "fc3_design.npz")["displacements_ang"]
    fd = []
    for i in range(len(d)):
        with np.load(BASE / "kappa_qe" / f"state_{i:04d}" / "result.npz") as r:
            f = np.asarray(r["forces_ev_ang"]) - f0
            fd.append(f - f.mean(axis=0))
    sets = {"canonical": data["canonical"], "sotcd": data["sotcd"],
            "fc3_design": (d, np.asarray(fd)),
            "sotcd_plus_fc3": (np.concatenate([data["sotcd"][0], d]),
                               np.concatenate([data["sotcd"][1], np.asarray(fd)]))}
    written = []
    for name, (u, f) in sets.items():
        for b in BUDGETS + (40,):
            if b > len(u) or (name == "canonical" and b > 40):
                continue
            a, y = fit_matrix(cs, atoms, u[:b], f[:b])
            params, *_ = np.linalg.lstsq(a, y, rcond=None)
            fcp = ForceConstantPotential(cs, params)
            fcs = fcp.get_force_constants(atoms)
            path = OUT / f"fc3_{name}_{b}.hdf5"
            fcs.write_to_phono3py(str(path), order=3)
            written.append(path.name)
    print("written", written, flush=True)


def kappa():
    import h5py
    import phonopy
    from phono3py import Phono3py

    ph = phonopy.load(BASE / "fd" / "phonopy_harmonic.yaml")
    fc2 = np.asarray(ph.force_constants)

    def run(fc3):
        p3 = Phono3py(ph.unitcell, supercell_matrix=ph.supercell_matrix,
                      primitive_matrix=ph.primitive_matrix, log_level=0)
        p3.fc2 = fc2
        p3.fc3 = fc3
        p3.mesh_numbers = MESH
        p3.init_phph_interaction()
        p3.run_thermal_conductivity(temperatures=[TEMPERATURE_KAPPA], is_isotope=True)
        k = p3.thermal_conductivity.kappa[0, 0]
        return float(np.mean(k[:3]))

    report = {"temperature_k": TEMPERATURE_KAPPA, "mesh": MESH, "kappa_w_mk": {}}
    fd_yaml = BASE / "fc3_fd" / "phono3py_disp.yaml"
    if fd_yaml.exists():
        import phono3py

        p3 = phono3py.load(fd_yaml, produce_fc=False, log_level=0)
        pos = np.load(BASE / "fc3_fd" / "displacements.npz")["positions_ang"]
        with np.load(BASE / "fd" / "harmonic_reference.npz") as h:
            f0 = np.asarray(h["reference_forces_ev_ang"])
        forces = []
        for i in range(len(pos)):
            with np.load(BASE / "fc3_fd_qe" / f"state_{i:04d}" / "result.npz") as r:
                forces.append(np.asarray(r["forces_ev_ang"]) - f0)
        p3.forces = np.asarray(forces)
        p3.produce_fc3()
        report["kappa_w_mk"]["finite_difference"] = run(p3.fc3)
    for path in sorted(OUT.glob("fc3_*_*.hdf5")):
        with h5py.File(path) as hf:
            report["kappa_w_mk"][path.stem[4:]] = run(np.asarray(hf["fc3"]))
        print(path.stem, report["kappa_w_mk"][path.stem[4:]], flush=True)
    out = ROOT / "validation_results" / MATERIAL / "robustness" / "kappa_design.json"
    out.write_text(json.dumps(report, indent=1) + "\n")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("stage", choices=["fd3-prepare", "fd3-compute", "design",
                                          "design-compute", "fit", "kappa"])
    parser.add_argument("--chunk", type=int, default=0)
    parser.add_argument("--nchunks", type=int, default=1)
    parser.add_argument("--n-parallel", type=int, default=4)
    args = parser.parse_args()
    if args.stage == "fd3-prepare":
        fd3_prepare()
    elif args.stage == "fd3-compute":
        pos = np.load(BASE / "fc3_fd" / "displacements.npz")["positions_ang"]
        compute("fc3_fd_qe", pos, args.chunk, args.nchunks, args.n_parallel)
    elif args.stage == "design":
        design()
    elif args.stage == "design-compute":
        import phonopy

        eq = np.asarray(phonopy.load(BASE / "fd" / "phonopy_harmonic.yaml").supercell.positions)
        u = np.load(OUT / "fc3_design.npz")["displacements_ang"]
        compute("kappa_qe", eq + u, args.chunk, args.nchunks, args.n_parallel)
    elif args.stage == "fit":
        fit()
    else:
        kappa()


if __name__ == "__main__":
    main()
