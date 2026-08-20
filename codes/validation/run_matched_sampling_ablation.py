#!/usr/bin/env python3
"""Matched SOTC-versus-canonical FC2 sampling ablation.

This calculation reuses the equilibrium structure, conventional finite-
displacement reference, 80/20 canonical train/test pool, pseudopotentials, and
electronic settings of ``run_real_anharmonic_ifcs.py``. The only new DFT states
are SOTC-designed displacements in those same 3x3x3 supercells.

Three Debye initializations (0.75, 1.00, and 1.25 times the nominal Debye
temperature) test initialization sensitivity. Every QE state is individually
checkpointed, so interrupted Slurm jobs can be restarted safely.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys
from types import SimpleNamespace

import numpy as np
import phonopy

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "codes"))

from sotc.cell_design import DisplacementOptimizer
from sotc.correlators import DebyeCorrelator
from sotc.ifc_extractor import IFCExtractor
from sotc.phonons import PhononCalculator

from run_real_anharmonic_ifcs import MATERIALS as DFT_MATERIALS, run_states


DESIGN = {
    "al": {"debye_k": 390.0, "cutoff_ang": 4.5, "r_max_ang": 8.0},
    "w": {"debye_k": 300.0, "cutoff_ang": 5.0, "r_max_ang": 9.0},
    "nacl": {"debye_k": 310.0, "cutoff_ang": 4.3, "r_max_ang": 9.0},
}
TD_SCALES = (0.75, 1.0, 1.25)
COUNTS = (1, 2, 4, 8, 16)


def serial(value):
    if isinstance(value, np.ndarray):
        return value.tolist()
    if isinstance(value, (np.floating, np.integer)):
        return value.item()
    if isinstance(value, Path):
        return str(value)
    raise TypeError(type(value).__name__)


def frequency_metrics(reference, prediction):
    reference = np.sort(np.asarray(reference, dtype=float), axis=1)
    prediction = np.sort(np.asarray(prediction, dtype=float), axis=1)
    residual = prediction - reference
    return {
        "mae_thz": float(np.mean(np.abs(residual))),
        "rmse_thz": float(np.sqrt(np.mean(residual ** 2))),
        "max_abs_thz": float(np.max(np.abs(residual))),
        "unstable_fraction_below_minus_0p05_thz": float(
            np.mean(prediction < -0.05)
        ),
    }


def force_metrics(reference, prediction):
    residual = np.asarray(prediction) - np.asarray(reference)
    return {
        "mae_ev_ang": float(np.mean(np.abs(residual))),
        "rmse_ev_ang": float(np.sqrt(np.mean(residual ** 2))),
        "max_abs_ev_ang": float(np.max(np.abs(residual))),
    }


def build_model(positions, cell, masses, basis, cutoff, u, forces):
    model = IFCExtractor(
        positions,
        cell,
        masses,
        basis_indices=basis,
        r_cutoff=cutoff,
        ridge_alpha=1.0e-3,
        symmetrize_bonds=True,
    )
    model.fit(list(np.asarray(u)), list(np.asarray(forces)))
    return model


def design_displacements(material, reference_phonon, out, n_members):
    spec = DESIGN[material]
    positions = np.asarray(reference_phonon.supercell.positions, dtype=float)
    cell = np.asarray(reference_phonon.supercell.cell, dtype=float)
    masses = np.asarray(reference_phonon.supercell.masses, dtype=float)
    _, basis = np.unique(
        np.asarray(reference_phonon.primitive.s2p_map), return_inverse=True
    )
    n_basis = len(reference_phonon.primitive)
    mean_mass = float(np.mean(reference_phonon.primitive.masses))
    volume_per_atom = (
        abs(float(np.linalg.det(reference_phonon.primitive.cell))) / n_basis
    )

    design_path = out / "designed_displacements.npz"
    if design_path.exists():
        with np.load(design_path, allow_pickle=False) as data:
            if int(data["n_members"]) != n_members:
                raise RuntimeError(
                    f"Archived design has {int(data['n_members'])} members, "
                    f"requested {n_members}"
                )
            return {
                str(label): np.asarray(data[f"u_{label}"])
                for label in data["labels"].astype(str)
            }

    archive = {
        "labels": np.asarray([f"td_{scale:.2f}" for scale in TD_SCALES]),
        "td_scales": np.asarray(TD_SCALES),
        "n_members": np.asarray(n_members),
        "shell_tolerance_ang": np.asarray(0.05),
        "xi_w_ang": np.asarray(max(spec["cutoff_ang"] / 2.0, 1.0)),
        "r_max_ang": np.asarray(spec["r_max_ang"]),
    }
    designs = {}
    for index, scale in enumerate(TD_SCALES):
        label = f"td_{scale:.2f}"
        corr = DebyeCorrelator(
            T_D=scale * spec["debye_k"],
            M_amu=mean_mass,
            V_per_atom_ang3=volume_per_atom,
        )
        optimizer = DisplacementOptimizer(
            corr,
            DFT_MATERIALS[material]["temperature_k"],
            xi_T=max(spec["cutoff_ang"] / 2.0, 1.0),
            r_max=spec["r_max_ang"],
            basis_indices=basis,
            masses_amu=masses,
            shell_tolerance_ang=0.05,
        )
        values = np.asarray(optimizer.generate_ensemble(
            positions,
            cell,
            masses,
            n_members=n_members,
            seed=42000 + 1000 * index,
            verbose=False,
        ))
        # A rigid translation contains no IFC information and is absent from
        # the paired canonical archive. Remove it before any DFT evaluation.
        values -= np.mean(values, axis=1, keepdims=True)
        shells, targets = optimizer.find_shells(positions, cell)
        objectives = np.asarray([
            optimizer.quality(u, positions, cell, shells, targets) for u in values
        ])
        archive[f"u_{label}"] = values
        archive[f"objective_{label}_ang4"] = objectives
        archive[f"target_debye_k_{label}"] = np.asarray(scale * spec["debye_k"])
        designs[label] = values
        print(
            f"designed {material} {label}: median Q={np.median(objectives):.6e}",
            flush=True,
        )
    np.savez_compressed(design_path, **archive)
    return designs


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("material", choices=sorted(DESIGN))
    parser.add_argument("--members", type=int, default=16)
    parser.add_argument("--canonical-seeds", type=int, default=20)
    parser.add_argument("--n-parallel", type=int, default=1)
    parser.add_argument("--tasks-per-qe", type=int, default=128)
    parser.add_argument("--design-only", action="store_true")
    parser.add_argument("--fit-only", action="store_true")
    parser.add_argument("--force-rerun", action="store_true")
    args = parser.parse_args()
    if args.members < max(COUNTS):
        parser.error(f"--members must be at least {max(COUNTS)}")

    material = args.material
    dft_spec = DFT_MATERIALS[material]
    design_spec = DESIGN[material]
    campaign = ROOT / "validation_results" / material / "real_ifc34"
    out = ROOT / "validation_results" / material / "matched_sampling"
    out.mkdir(parents=True, exist_ok=True)
    reference_phonon = phonopy.load(campaign / "phonopy_harmonic.yaml")
    designs = design_displacements(material, reference_phonon, out, args.members)
    if args.design_only:
        return

    positions = np.asarray(reference_phonon.supercell.positions, dtype=float)
    cell = np.asarray(reference_phonon.supercell.cell, dtype=float)
    masses = np.asarray(reference_phonon.supercell.masses, dtype=float)
    symbols = list(reference_phonon.supercell.symbols)
    _, basis = np.unique(
        np.asarray(reference_phonon.primitive.s2p_map), return_inverse=True
    )
    primitive_positions = np.asarray(reference_phonon.primitive.positions)
    primitive_cell = np.asarray(reference_phonon.primitive.cell)
    primitive_masses = np.asarray(reference_phonon.primitive.masses)

    with np.load(campaign / "thermal_force_dataset.npz") as data:
        canonical_u = np.asarray(data["displacements_ang"])
        canonical_f = np.asarray(data["delta_forces_ev_ang"])
        reference_force = np.asarray(data["reference_forces_ev_ang"])
        train_indices = np.asarray(data["train_indices"], dtype=int)
        test_indices = np.asarray(data["test_indices"], dtype=int)

    run_args = SimpleNamespace(
        material=material,
        n_parallel=args.n_parallel,
        tasks_per_qe=args.tasks_per_qe,
        fit_only=args.fit_only,
        force_rerun=args.force_rerun,
    )
    designed_forces = {}
    for label, displacements in designs.items():
        states = positions[None, :, :] + displacements
        results = run_states(
            states,
            cell,
            symbols,
            dft_spec,
            run_args,
            out / f"qe_{label}",
        )
        raw = np.asarray([row["forces_ev_ang"] for row in results])
        delta_before_drift = raw - reference_force[None, :, :]
        drift = np.mean(delta_before_drift, axis=1, keepdims=True)
        delta = delta_before_drift - drift
        designed_forces[label] = delta
        np.savez_compressed(
            out / f"force_dataset_{label}.npz",
            displacements_ang=displacements,
            raw_forces_ev_ang=raw,
            reference_forces_ev_ang=reference_force,
            delta_forces_before_drift_ev_ang=delta_before_drift,
            translational_force_drift_ev_ang=drift[:, 0, :],
            delta_forces_ev_ang=delta,
        )

    band_path = ROOT / "examples" / f"sotc_{material}_qe_run" / "phonon_bandstructure.npz"
    with np.load(band_path) as band:
        q_points = np.asarray(band["q_points"])
    reference_phonon.run_qpoints(q_points)
    reference_frequencies = np.asarray(
        reference_phonon.get_qpoints_dict()["frequencies"]
    )

    report = {
        "material": material,
        "scope": (
            "same 3x3x3 cell, equilibrium structure, QE settings, force "
            "subtraction, force-drift projection, IFC priors, held-out set, "
            "and q path; displacement selection is the controlled variable"
        ),
        "temperature_k": dft_spec["temperature_k"],
        "supercell_atoms": len(positions),
        "supercell_cell_ang": cell,
        "electronic_settings": {
            "ecutwfc_ry": dft_spec["ecutwfc"],
            "ecutrho_ry": dft_spec["ecutrho"],
            "supercell_kmesh": dft_spec["kpts"],
            "conv_thr_ry": 1.0e-9,
            "smearing": dft_spec["smearing"],
            "degauss_ry": dft_spec["degauss"],
        },
        "fit_settings": {
            "fc2_cutoff_ang": design_spec["cutoff_ang"],
            "ridge_alpha": 1.0e-3,
            "central_force_projection": True,
            "acoustic_sum_rule": True,
        },
        "counts": list(COUNTS),
        "canonical_seed_count": args.canonical_seeds,
        "test_snapshot_count": len(test_indices),
        "displacement_amplitudes_ang": {
            "definition": "RMS over atoms and Cartesian components per configuration",
            "canonical_training_pool": {
                "median": float(np.median(np.sqrt(np.mean(
                    canonical_u[train_indices] ** 2, axis=(1, 2)
                )))),
                "p10": float(np.percentile(np.sqrt(np.mean(
                    canonical_u[train_indices] ** 2, axis=(1, 2)
                )), 10)),
                "p90": float(np.percentile(np.sqrt(np.mean(
                    canonical_u[train_indices] ** 2, axis=(1, 2)
                )), 90)),
            },
            "sotc_by_initialization": {
                label: np.sqrt(np.mean(values ** 2, axis=(1, 2))).tolist()
                for label, values in designs.items()
            },
        },
        "sotc_initialization_series": {},
        "canonical_series": [],
    }

    for label, u_values in designs.items():
        rows = []
        f_values = designed_forces[label]
        for count in COUNTS:
            model = build_model(
                positions, cell, masses, basis, design_spec["cutoff_ang"],
                u_values[:count], f_values[:count],
            )
            predicted_test = np.asarray([
                model.predict_forces(u) for u in canonical_u[test_indices]
            ])
            calculator = PhononCalculator(
                model, primitive_positions, primitive_cell, primitive_masses
            )
            frequencies = calculator.band_structure(q_points)[1]
            rows.append({
                "n_configurations": count,
                "dispersion": frequency_metrics(reference_frequencies, frequencies),
                "held_out_force": force_metrics(
                    canonical_f[test_indices], predicted_test
                ),
                "design": model.diagnostics(),
            })
            if count == max(COUNTS):
                model.save(str(out / f"ifc_{label}_n{count}.npz"))
        report["sotc_initialization_series"][label] = rows

    for seed in range(args.canonical_seeds):
        rng = np.random.default_rng(93000 + seed)
        ordering = rng.permutation(train_indices)
        rows = []
        for count in COUNTS:
            chosen = ordering[:count]
            model = build_model(
                positions, cell, masses, basis, design_spec["cutoff_ang"],
                canonical_u[chosen], canonical_f[chosen],
            )
            predicted_test = np.asarray([
                model.predict_forces(u) for u in canonical_u[test_indices]
            ])
            calculator = PhononCalculator(
                model, primitive_positions, primitive_cell, primitive_masses
            )
            frequencies = calculator.band_structure(q_points)[1]
            rows.append({
                "n_configurations": count,
                "chosen_training_indices": chosen.tolist(),
                "dispersion": frequency_metrics(reference_frequencies, frequencies),
                "held_out_force": force_metrics(
                    canonical_f[test_indices], predicted_test
                ),
                "design": model.diagnostics(),
            })
        report["canonical_series"].append({"seed": seed, "counts": rows})

    with np.load(campaign / "harmonic_reference.npz") as harmonic:
        n_fd = len(harmonic["displacements_ang"])
    report["force_call_accounting"] = {
        "new_sotc_displaced_calls": len(TD_SCALES) * args.members,
        "new_reference_calls": 0,
        "reused_canonical_calls": len(canonical_u),
        "reused_finite_displacement_calls": n_fd,
    }
    destination = out / "matched_sampling_summary.json"
    destination.write_text(
        json.dumps(report, indent=2, default=serial, allow_nan=False) + "\n"
    )
    print(f"matched sampling complete: {destination}", flush=True)


if __name__ == "__main__":
    main()
