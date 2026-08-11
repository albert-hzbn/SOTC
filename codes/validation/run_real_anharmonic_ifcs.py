#!/usr/bin/env python3
"""Compute real-material FC3/FC4 tensors from finite-temperature QE forces.

The production protocol is independently auditable:

* 3x3x3 primitive supercells at the stated material temperature and volume;
* a DFT finite-displacement harmonic reference;
* 100 quantum-canonical configurations arranged as 50 independent +/- pairs;
* symmetry, permutation and translational-sum-rule constrained joint fits;
* an 80/20 pair-independent train/test split;
* nested FC2, FC2+FC3 and FC2+FC3+FC4 held-out force comparisons;
* snapshot-count, real-space-cutoff and electronic-setting convergence checks;
* restartable, per-configuration raw QE archives and transport-code exports.
"""

from __future__ import annotations

import argparse
import concurrent.futures
import hashlib
import json
import os
import platform
import sys
import time
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "codes"))

from sotc.qe_io import QEForceCalculator
from sotc.anharmonic_io import write_fourphonon_fc4, write_shengbte_fc3
from sotc.symfc_fit import (
    compact_invariance_diagnostics,
    fit_symfc_model,
    force_metrics,
    forces_from_compact_fcs,
    symfc_atoms,
)
from sotc.validation import phonopy_atoms


MATERIALS = {
    "al": {
        "label": "fcc Al", "elements": ["Al"], "masses": [26.9815385],
        "numbers": [13], "a": 4.05, "bravais": "fcc", "temperature_k": 300.0,
        "ecutwfc": 40.0, "ecutrho": 320.0, "kpts": (4, 4, 4),
        "smearing": "marzari-vanderbilt", "degauss": 0.02,
        "pseudos": {"Al": "Al.pbe-n-kjpaw_psl.1.0.0.UPF"},
        # The third-neighbour Al shell is required by the held-out cutoff
        # sensitivity test: extending FC2/FC3 from 4.5 to 5.0 Angstrom lowers
        # the test-force RMSE by about 79%.  FC4 gains no additional basis
        # terms between 3.5 and 4.0 Angstrom, but keep the tested 4.0-Angstrom
        # production value for a directly reproducible cutoff bracket.
        "cutoffs": {2: 5.0, 3: 5.0, 4: 4.0},
    },
    "w": {
        "label": "bcc W", "elements": ["W"], "masses": [183.84],
        "numbers": [74], "a": 3.18, "bravais": "bcc", "temperature_k": 1500.0,
        "ecutwfc": 30.0, "ecutrho": 240.0, "kpts": (4, 4, 4),
        "smearing": "marzari-vanderbilt", "degauss": 0.02,
        "pseudos": {"W": "W_pbe_v1.2.uspp.F.UPF"},
        "cutoffs": {2: 5.0, 3: 4.5, 4: 3.5},
        "extra": {"electrons": {"diagonalization": "david", "diago_david_ndim": 4}},
    },
    "nacl": {
        "label": "rocksalt NaCl", "elements": ["Na", "Cl"],
        "masses": [22.9898, 35.453], "numbers": [11, 17],
        "a": 5.73, "bravais": "fcc", "temperature_k": 700.0,
        "ecutwfc": 60.0, "ecutrho": 480.0, "kpts": (2, 2, 2),
        "smearing": None, "degauss": 0.0,
        "pseudos": {"Na": "na_pbe_v1.5.uspp.F.UPF",
                    "Cl": "cl_pbe_v1.4.uspp.F.UPF"},
        "cutoffs": {2: 4.3, 3: 4.3, 4: 4.0},
    },
}


def json_default(value):
    if isinstance(value, np.ndarray):
        return value.tolist()
    if isinstance(value, (np.floating, np.integer)):
        return value.item()
    if isinstance(value, Path):
        return str(value)
    raise TypeError(type(value).__name__)


def primitive_structure(spec):
    a = float(spec["a"])
    if spec["bravais"] == "fcc":
        cell = 0.5 * a * np.array([[0., 1., 1.], [1., 0., 1.], [1., 1., 0.]])
    elif spec["bravais"] == "bcc":
        cell = 0.5 * a * np.array([[-1., 1., 1.], [1., -1., 1.], [1., 1., -1.]])
    else:
        raise ValueError(spec["bravais"])
    if len(spec["elements"]) == 1:
        positions = np.zeros((1, 3))
    else:
        positions = np.array([[0., 0., 0.], [a / 2., 0., 0.]])
    return cell, positions


def make_phonon(spec):
    from phonopy import Phonopy

    cell, positions = primitive_structure(spec)
    unit = phonopy_atoms(spec["elements"], cell, positions)
    return Phonopy(unit, np.diag([3, 3, 3]))


def file_sha256(path):
    digest = hashlib.sha256()
    with open(path, "rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def calculation_settings(spec, args, symbols, workdir, prefix, overrides=None):
    settings = {
        "species": list(symbols),
        "pseudopotentials": spec["pseudos"],
        "pseudo_dir": Path(os.environ.get("SSSP_DIR", ROOT / "pseudopotentials")),
        "ecutwfc": spec["ecutwfc"], "ecutrho": spec["ecutrho"],
        "kpts": tuple(spec["kpts"]), "smearing": spec["smearing"],
        "degauss": spec["degauss"], "conv_thr": 1.0e-9,
        "pw_cmd": os.environ.get("QE_PW_CMD"),
        "workdir": workdir, "prefix": prefix, "n_parallel": 1,
        "extra_input_data": spec.get("extra"),
    }
    if overrides:
        settings.update(overrides)
    if not settings["pw_cmd"]:
        raise RuntimeError("QE_PW_CMD must be set for DFT stages")
    return settings


def checkpoint_result(path, positions, cell):
    if not path.exists():
        return None
    with np.load(path, allow_pickle=False) as data:
        if not np.allclose(data["positions_ang"], positions, atol=1e-12, rtol=0):
            raise RuntimeError(f"Checkpoint positions differ: {path}")
        if not np.allclose(data["cell_ang"], cell, atol=1e-12, rtol=0):
            raise RuntimeError(f"Checkpoint cell differs: {path}")
        return {name: np.array(data[name]) for name in data.files}


def run_one_state(index, positions, cell, symbols, spec, args, stage, overrides=None):
    state = stage / f"state_{index:04d}"
    result_path = state / "result.npz"
    cached = checkpoint_result(result_path, positions, cell)
    if cached is not None and not args.force_rerun:
        return cached
    if args.fit_only:
        raise FileNotFoundError(f"Missing DFT checkpoint required by --fit-only: {result_path}")
    state.mkdir(parents=True, exist_ok=True)
    calc = QEForceCalculator(**calculation_settings(
        spec, args, symbols, state / "qe_scratch", f"{args.material}_{stage.name}_{index}",
        overrides=overrides,
    ))
    started = time.monotonic()
    forces = calc.compute(positions, cell)
    energy = calc.energy(positions, cell)
    stress = calc.stress(positions, cell)
    elapsed = time.monotonic() - started
    tmp = state / "result.npz.tmp"
    with open(tmp, "wb") as handle:
        np.savez_compressed(
            handle, positions_ang=positions, cell_ang=cell,
            forces_ev_ang=forces, energy_ev=energy,
            stress_ev_ang3=stress, wall_seconds=elapsed,
        )
    os.replace(tmp, result_path)
    return checkpoint_result(result_path, positions, cell)


def run_states(positions, cell, symbols, spec, args, stage, overrides=None):
    stage.mkdir(parents=True, exist_ok=True)
    results = [None] * len(positions)
    missing = []
    for index, pos in enumerate(positions):
        path = stage / f"state_{index:04d}" / "result.npz"
        cached = checkpoint_result(path, pos, cell)
        if cached is not None and not args.force_rerun:
            results[index] = cached
        else:
            missing.append(index)
    print(f"{stage.name}: {len(results)-len(missing)} cached, {len(missing)} to run",
          flush=True)
    if missing:
        with concurrent.futures.ThreadPoolExecutor(max_workers=args.n_parallel) as pool:
            futures = {
                pool.submit(
                    run_one_state, index, positions[index], cell, symbols,
                    spec, args, stage, overrides,
                ): index for index in missing
            }
            for completed, future in enumerate(
                concurrent.futures.as_completed(futures), start=1
            ):
                index = futures[future]
                results[index] = future.result()
                print(f"{stage.name}: completed state {index:04d} "
                      f"({completed}/{len(missing)})", flush=True)
    return results


def harmonic_reference(phonon, spec, args, out):
    phonon.generate_displacements(distance=0.01, is_plusminus=True)
    equilibrium = phonon.supercell
    cell = np.asarray(equilibrium.cell, dtype=float)
    eq_pos = np.asarray(equilibrium.positions, dtype=float)
    symbols = list(equilibrium.symbols)
    displaced = [np.asarray(sc.positions, dtype=float)
                 for sc in phonon.supercells_with_displacements]
    results = run_states(
        [eq_pos] + displaced, cell, symbols, spec, args, out / "harmonic_qe"
    )
    f0 = np.asarray(results[0]["forces_ev_ang"])
    delta = [np.asarray(item["forces_ev_ang"]) - f0 for item in results[1:]]
    phonon.forces = delta
    phonon.produce_force_constants(calculate_full_force_constants=True)
    np.savez_compressed(
        out / "harmonic_reference.npz",
        equilibrium_positions_ang=eq_pos, cell_ang=cell,
        symbols=np.asarray(symbols), reference_forces_ev_ang=f0,
        reference_energy_ev=float(results[0]["energy_ev"]),
        reference_stress_ev_ang3=np.asarray(results[0]["stress_ev_ang3"]),
        displacements_ang=np.asarray([
            np.asarray(sc.positions) - eq_pos
            for sc in phonon.supercells_with_displacements
        ]),
        delta_forces_ev_ang=np.asarray(delta),
        force_constants_ev_ang2=np.asarray(phonon.force_constants),
    )
    phonon.save(
        filename=str(out / "phonopy_harmonic.yaml"),
        settings={"force_constants": True},
    )
    phonon.run_mesh([6, 6, 6])
    frequencies = np.asarray(phonon.get_mesh_dict()["frequencies"])
    return {
        "phonon": phonon, "equilibrium_positions": eq_pos, "cell": cell,
        "symbols": symbols, "reference_forces": f0,
        "reference_stress": np.asarray(results[0]["stress_ev_ang3"]),
        "min_mesh_frequency_thz": float(np.min(frequencies)),
        "fraction_imaginary_mesh_modes": float(np.mean(frequencies < -0.05)),
        "n_displaced_states": len(displaced),
        "wall_seconds_sum": float(sum(float(x["wall_seconds"]) for x in results)),
    }


def paired_thermal_displacements(phonon, temperature, n_snapshots, seed, archive):
    if n_snapshots % 2 or n_snapshots < 4:
        raise ValueError("--snapshots must be even and at least four")
    if archive.exists():
        with np.load(archive, allow_pickle=False) as data:
            u = np.asarray(data["displacements_ang"])
            if len(u) != n_snapshots:
                raise RuntimeError("Archived thermal snapshot count differs")
            return u
    phonon.generate_displacements(
        number_of_snapshots=n_snapshots // 2,
        random_seed=int(seed), temperature=float(temperature),
        cutoff_frequency=0.1,
    )
    eq = np.asarray(phonon.supercell.positions)
    base = np.asarray([
        np.asarray(sc.positions) - eq for sc in phonon.supercells_with_displacements
    ])
    # Remove the numerical translational gauge and interleave exact inversion
    # partners. Pair independence is preserved in the train/test split.
    base -= np.mean(base, axis=1, keepdims=True)
    paired = np.stack((base, -base), axis=1).reshape(n_snapshots, len(eq), 3)
    np.savez_compressed(
        archive, displacements_ang=paired, temperature_k=float(temperature),
        random_seed=int(seed), pairing="interleaved u,-u",
    )
    return paired


def fit_model(supercell, u_train, f_train, u_test, f_test, orders, cutoffs):
    model, design = fit_symfc_model(
        supercell, u_train, f_train, orders, cutoffs,
        batch_size=20, full_force_constants=False,
    )
    basis = model.basis_set[2]
    train_prediction = forces_from_compact_fcs(
        u_train, model.force_constants,
        basis.translation_permutations, basis.p2s_map,
    )
    test_prediction = forces_from_compact_fcs(
        u_test, model.force_constants,
        basis.translation_permutations, basis.p2s_map,
    )
    report = {
        "orders": list(orders), "cutoffs_ang": cutoffs, "design": design,
        "train": force_metrics(f_train, train_prediction),
        "held_out_test": force_metrics(f_test, test_prediction),
        "constraints": compact_invariance_diagnostics(model.force_constants),
    }
    return model, report


def electronic_convergence(spec, args, out, positions, cell, symbols, production):
    selected = [0, args.train_snapshots, args.train_snapshots + 1]
    selected_positions = [positions[i] for i in selected]
    comparisons = {}
    variants = {
        "higher_cutoff": {
            "ecutwfc": 1.25 * spec["ecutwfc"],
            "ecutrho": 1.25 * spec["ecutrho"],
        },
        "denser_kmesh": {
            "kpts": tuple(value + 1 for value in spec["kpts"]),
        },
    }
    reference_forces = np.asarray([
        production[i]["forces_ev_ang"] for i in selected
    ])
    reference_energies = np.asarray([
        production[i]["energy_ev"] for i in selected
    ], dtype=float)
    for name, overrides in variants.items():
        results = run_states(
            selected_positions, cell, symbols, spec, args,
            out / f"electronic_convergence_{name}", overrides=overrides,
        )
        forces = np.asarray([item["forces_ev_ang"] for item in results])
        energies = np.asarray([item["energy_ev"] for item in results], dtype=float)
        comparisons[name] = {
            "settings": overrides,
            "state_indices": selected,
            "force_difference": force_metrics(reference_forces, forces),
            "energy_shift_ev_per_atom": (
                (energies - reference_energies) / len(symbols)
            ).tolist(),
        }
    return comparisons


def write_compact_hdf5(path, force_constants, p2s_map, cutoffs):
    import h5py
    import symfc

    with h5py.File(path, "w") as handle:
        handle.attrs["format"] = "symfc compact force constants"
        handle.attrs["symfc_version"] = symfc.__version__
        handle.create_dataset("p2s_map", data=np.asarray(p2s_map, dtype=int))
        for order, values in sorted(force_constants.items()):
            dataset = handle.create_dataset(
                f"fc{order}", data=np.asarray(values), compression="gzip"
            )
            dataset.attrs["physical_unit"] = f"eV/angstrom^{order}"
            dataset.attrs["cutoff_ang"] = float(cutoffs[order])


def transport_mapping(phonon):
    """Return primitive reference, basis and Cartesian lattice-vector maps."""
    primitive = phonon.primitive
    supercell = phonon.supercell
    references = np.asarray(primitive.p2s_map, dtype=int)
    basis = np.asarray(
        [primitive.p2p_map[index] for index in primitive.s2p_map], dtype=int
    )
    primitive_cell = np.asarray(primitive.cell, dtype=float)
    positions = np.asarray(supercell.positions, dtype=float)
    integer_vectors = []
    for atom, basis_index in enumerate(basis):
        fractional = (
            positions[atom] - positions[references[basis_index]]
        ) @ np.linalg.inv(primitive_cell)
        rounded = np.rint(fractional)
        if np.max(np.abs(fractional - rounded)) > 1e-8:
            raise RuntimeError("Failed to map supercell atom to a primitive translation")
        integer_vectors.append(rounded)
    return references, basis, np.asarray(integer_vectors) @ primitive_cell


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("material", choices=sorted(MATERIALS))
    parser.add_argument("--snapshots", type=int, default=100)
    parser.add_argument("--train-snapshots", type=int, default=80)
    parser.add_argument("--seed", type=int, default=314159)
    parser.add_argument("--n-parallel", type=int, default=8)
    parser.add_argument("--tasks-per-qe", type=int, default=16)
    parser.add_argument("--output", type=Path)
    parser.add_argument("--basis-only", action="store_true")
    parser.add_argument("--fit-only", action="store_true")
    parser.add_argument("--force-rerun", action="store_true")
    parser.add_argument("--skip-electronic-convergence", action="store_true")
    args = parser.parse_args()
    if args.train_snapshots >= args.snapshots or args.train_snapshots % 2:
        parser.error("--train-snapshots must be even and smaller than --snapshots")
    spec = MATERIALS[args.material]
    out = args.output or (
        ROOT / "validation_results" / args.material / "real_ifc34"
    )
    out.mkdir(parents=True, exist_ok=True)

    phonon = make_phonon(spec)
    supercell = phonon.supercell
    sf_atoms = symfc_atoms(
        supercell.numbers, supercell.scaled_positions, supercell.cell
    )
    from symfc import Symfc
    import symfc as symfc_package
    import phonopy as phonopy_package
    estimator = Symfc(sf_atoms, cutoff=spec["cutoffs"], use_mkl=False, log_level=0)
    estimated = estimator.estimate_basis_size(orders=[2, 3, 4])
    basis_summary = {
        "material": args.material, "n_supercell_atoms": len(supercell),
        "supercell_matrix": np.diag([3, 3, 3]).tolist(),
        "cutoffs_ang": spec["cutoffs"],
        "estimated_basis_parameters_by_order": estimated,
    }
    (out / "basis_estimate.json").write_text(json.dumps(
        basis_summary, indent=2, default=json_default, allow_nan=False
    ))
    print(json.dumps(basis_summary, indent=2, default=json_default), flush=True)
    if args.basis_only:
        return

    started = time.monotonic()
    harmonic = harmonic_reference(phonon, spec, args, out)
    u = paired_thermal_displacements(
        phonon, spec["temperature_k"], args.snapshots, args.seed,
        out / "thermal_displacements.npz",
    )
    eq = harmonic["equilibrium_positions"]
    cell = harmonic["cell"]
    symbols = harmonic["symbols"]
    thermal_results = run_states(
        [eq + snapshot for snapshot in u], cell, symbols,
        spec, args, out / "thermal_qe",
    )
    raw_forces = np.asarray([item["forces_ev_ang"] for item in thermal_results])
    delta_raw = raw_forces - harmonic["reference_forces"][None, :, :]
    force_drift = np.mean(delta_raw, axis=1)
    delta_forces = delta_raw - force_drift[:, None, :]
    np.savez_compressed(
        out / "thermal_force_dataset.npz",
        displacements_ang=u, raw_forces_ev_ang=raw_forces,
        reference_forces_ev_ang=harmonic["reference_forces"],
        delta_forces_before_drift_ev_ang=delta_raw,
        translational_force_drift_ev_ang=force_drift,
        delta_forces_ev_ang=delta_forces,
        energies_ev=np.asarray([x["energy_ev"] for x in thermal_results]),
        stresses_ev_ang3=np.asarray([x["stress_ev_ang3"] for x in thermal_results]),
        train_indices=np.arange(args.train_snapshots),
        test_indices=np.arange(args.train_snapshots, args.snapshots),
    )

    train = slice(0, args.train_snapshots)
    test = slice(args.train_snapshots, args.snapshots)
    nested = {}
    for orders in ([2], [2, 3], [2, 3, 4]):
        name = "".join(str(order) for order in orders)
        print(f"Fitting held-out model orders={orders}", flush=True)
        model, report = fit_model(
            sf_atoms, u[train], delta_forces[train],
            u[test], delta_forces[test], orders, spec["cutoffs"],
        )
        nested[name] = report

    snapshot_convergence = []
    for count in (20, 40, 60):
        if count >= args.train_snapshots:
            continue
        print(f"Fitting snapshot convergence prefix={count}", flush=True)
        _, report = fit_model(
            sf_atoms, u[:count], delta_forces[:count],
            u[test], delta_forces[test], [2, 3, 4], spec["cutoffs"],
        )
        report["n_training_snapshots"] = count
        snapshot_convergence.append(report)
    full_train_report = dict(nested["234"])
    full_train_report["n_training_snapshots"] = args.train_snapshots
    snapshot_convergence.append(full_train_report)

    cutoff_sensitivity = []
    for shift in (-0.5, 0.5):
        cutoffs = {
            order: max(2.5, float(value) + shift)
            for order, value in spec["cutoffs"].items()
        }
        print(f"Fitting cutoff sensitivity {cutoffs}", flush=True)
        _, report = fit_model(
            sf_atoms, u[train], delta_forces[train],
            u[test], delta_forces[test], [2, 3, 4], cutoffs,
        )
        cutoff_sensitivity.append(report)
    cutoff_sensitivity.insert(1, nested["234"])

    print("Fitting final production FC2+FC3+FC4 model on all snapshots", flush=True)
    final_model, final_design = fit_symfc_model(
        sf_atoms, u, delta_forces, [2, 3, 4], spec["cutoffs"],
        batch_size=20, full_force_constants=False,
    )
    basis = final_model.basis_set[2]
    final_dir = out / "final_force_constants"
    final_dir.mkdir(exist_ok=True)
    for order, values in final_model.force_constants.items():
        np.save(final_dir / f"fc{order}_compact.npy", np.asarray(values))
    np.save(final_dir / "p2s_map.npy", np.asarray(basis.p2s_map))
    np.save(
        final_dir / "translation_permutations.npy",
        np.asarray(basis.translation_permutations),
    )
    write_compact_hdf5(
        final_dir / "fc234_compact.hdf5", final_model.force_constants,
        basis.p2s_map, spec["cutoffs"],
    )
    from phono3py.file_IO import write_fc3_to_hdf5
    fc3 = np.asarray(final_model.force_constants[3])
    fc3_nonzero = (
        np.max(np.abs(fc3), axis=(3, 4, 5)) > 1e-14
    ).astype(np.byte)
    write_fc3_to_hdf5(
        fc3, fc3_nonzero_indices=fc3_nonzero,
        filename=str(final_dir / "fc3.hdf5"),
        p2s_map=np.asarray(basis.p2s_map, dtype=np.int64),
        fc3_cutoff=spec["cutoffs"][3],
    )
    references, basis_indices, lattice_vectors = transport_mapping(phonon)
    n_shengbte_blocks = write_shengbte_fc3(
        final_dir / "FORCE_CONSTANTS_3RD",
        final_model.force_constants[3],
        references, basis_indices, lattice_vectors,
        threshold_ev_ang3=1e-12,
    )
    n_fourphonon_blocks = write_fourphonon_fc4(
        final_dir / "FORCE_CONSTANTS_4TH",
        final_model.force_constants[4],
        references, basis_indices, lattice_vectors,
        threshold_ev_ang4=1e-12,
    )

    electronic = {}
    if not args.skip_electronic_convergence:
        electronic = electronic_convergence(
            spec, args, out,
            [eq + snapshot for snapshot in u], cell, symbols, thermal_results,
        )

    pseudo_dir = Path(os.environ.get("SSSP_DIR", ROOT / "pseudopotentials"))
    provenance = {
        "quantum_espresso_command": os.environ.get("QE_PW_CMD"),
        "phonopy_version": phonopy_package.__version__,
        "symfc_version": symfc_package.__version__,
        "python_version": platform.python_version(),
        "pseudopotentials": {
            element: {
                "filename": filename,
                "sha256": file_sha256(pseudo_dir / filename),
            } for element, filename in spec["pseudos"].items()
        },
    }
    summary = {
        "status": "complete",
        "material": spec["label"], "temperature_k": spec["temperature_k"],
        "lattice_parameter_ang": spec["a"],
        "thermodynamic_condition": "fixed stated lattice parameter; quantum-canonical ionic ensemble",
        "supercell_matrix": np.diag([3, 3, 3]),
        "n_supercell_atoms": len(symbols),
        "n_snapshots": args.snapshots,
        "n_training_snapshots": args.train_snapshots,
        "n_strictly_held_out_snapshots": args.snapshots - args.train_snapshots,
        "sampling": {
            "random_seed": args.seed,
            "distribution": "phonopy quantum harmonic canonical",
            "pairing": "50 independent draws, each paired with its exact inversion",
            "split_rule": "complete +/- pairs; no pair crosses train/test",
        },
        "dft_settings": {
            "ecutwfc_ry": spec["ecutwfc"], "ecutrho_ry": spec["ecutrho"],
            "k_mesh_supercell": spec["kpts"], "smearing": spec["smearing"],
            "degauss_ry": spec["degauss"], "scf_conv_thr_ry": 1e-9,
        },
        "ifc_fit": {
            "backend": "symfc joint dense solver",
            "orders": [2, 3, 4], "cutoffs_ang": spec["cutoffs"],
            "constraints": [
                "space-group symmetry", "index permutation symmetry",
                "lattice translation symmetry", "translational acoustic sum rules",
            ],
            "regularization": "none",
            "final_all_snapshot_design": final_design,
            "final_constraints": compact_invariance_diagnostics(
                final_model.force_constants
            ),
        },
        "harmonic_reference": {
            key: value for key, value in harmonic.items()
            if key not in {"phonon", "equilibrium_positions", "cell", "symbols",
                           "reference_forces", "reference_stress"}
        },
        "reference_force": {
            "rms_component_ev_ang": float(np.sqrt(np.mean(
                harmonic["reference_forces"] ** 2
            ))),
            "max_atom_norm_ev_ang": float(np.max(np.linalg.norm(
                harmonic["reference_forces"], axis=1
            ))),
            "stress_ev_ang3": harmonic["reference_stress"],
        },
        "force_drift_projection": {
            "method": "subtract F(0), then remove each snapshot's uniform net-force drift",
            "rms_component_ev_ang": float(np.sqrt(np.mean(force_drift ** 2))),
            "max_component_ev_ang": float(np.max(np.abs(force_drift))),
        },
        "nested_held_out_order_comparison": nested,
        "snapshot_count_convergence": snapshot_convergence,
        "cutoff_sensitivity": cutoff_sensitivity,
        "electronic_convergence": electronic,
        "force_call_accounting": {
            "harmonic_reference_and_displacements": harmonic["n_displaced_states"] + 1,
            "thermal_snapshots": args.snapshots,
            "electronic_convergence": 0 if args.skip_electronic_convergence else 6,
            "total": harmonic["n_displaced_states"] + 1 + args.snapshots
                     + (0 if args.skip_electronic_convergence else 6),
        },
        "timing": {
            "thermal_qe_wall_seconds_sum": float(sum(
                float(x["wall_seconds"]) for x in thermal_results
            )),
            "estimated_qe_core_hours": float(
                (harmonic["wall_seconds_sum"] + sum(
                    float(x["wall_seconds"]) for x in thermal_results
                )) * args.tasks_per_qe / 3600.0
            ),
            "workflow_elapsed_seconds": float(time.monotonic() - started),
        },
        "provenance": provenance,
        "artifacts": {
            "raw_dataset": "thermal_force_dataset.npz",
            "harmonic_reference": "harmonic_reference.npz",
            "compact_fc234_hdf5": "final_force_constants/fc234_compact.hdf5",
            "phono3py_fc3": "final_force_constants/fc3.hdf5",
            "compact_fc4_npy": "final_force_constants/fc4_compact.npy",
            "shengbte_fc3": {
                "path": "final_force_constants/FORCE_CONSTANTS_3RD",
                "tensor_blocks": n_shengbte_blocks,
            },
            "fourphonon_fc4": {
                "path": "final_force_constants/FORCE_CONSTANTS_4TH",
                "tensor_blocks": n_fourphonon_blocks,
            },
        },
    }
    (out / "ifc34_summary.json").write_text(json.dumps(
        summary, indent=2, default=json_default, allow_nan=False
    ))
    print(f"COMPLETE: {out / 'ifc34_summary.json'}", flush=True)


if __name__ == "__main__":
    main()
