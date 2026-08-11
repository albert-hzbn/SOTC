"""Validation utilities for matched DFT phonon comparisons.

This module contains no benchmark numbers.  Every metric is generated from
archived force calls and a common q path.  Phonopy is used only for the
symmetry-reduced finite-displacement reference and canonical harmonic sampling.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Dict, Iterable, List, Sequence, Tuple

import numpy as np

from .ifc_extractor import IFCExtractor
from .mock_forces import mock_vasp_results
from .phonons import PhononCalculator


def phonopy_atoms(elements, cell, cart_positions):
    from phonopy.structure.atoms import PhonopyAtoms

    scaled = np.asarray(cart_positions) @ np.linalg.inv(np.asarray(cell))
    return PhonopyAtoms(
        symbols=list(elements), cell=np.asarray(cell), scaled_positions=scaled
    )


def build_finite_displacement_reference(
    elements: Sequence[str],
    primitive_cell: np.ndarray,
    primitive_positions: np.ndarray,
    supercell_matrix: np.ndarray,
    force_calculator,
    output_dir: str | Path,
    displacement_distance_ang: float = 0.01,
    plus_minus: bool = True,
):
    """Run a symmetry-reduced, central finite-displacement reference.

    The undisplaced force is evaluated once and subtracted from every displaced
    force.  The exact force arrays, displacement dataset, force constants,
    stress, settings, and pseudopotential hashes are archived.
    """
    from phonopy import Phonopy
    import phonopy

    out = Path(output_dir)
    out.mkdir(parents=True, exist_ok=True)
    unit = phonopy_atoms(elements, primitive_cell, primitive_positions)
    phonon = Phonopy(unit, np.asarray(supercell_matrix, dtype=int))
    phonon.generate_displacements(
        distance=float(displacement_distance_ang),
        is_plusminus=bool(plus_minus),
    )
    equilibrium = phonon.supercell
    eq_pos = np.asarray(equilibrium.positions, dtype=float)
    eq_cell = np.asarray(equilibrium.cell, dtype=float)
    reference = mock_vasp_results(eq_pos, eq_cell, force_calculator)
    f0 = np.asarray(reference["forces_ev_ang"], dtype=float)

    delta_forces = []
    displacements = []
    for displaced in phonon.supercells_with_displacements:
        pos = np.asarray(displaced.positions, dtype=float)
        result = mock_vasp_results(pos, np.asarray(displaced.cell), force_calculator)
        delta_forces.append(np.asarray(result["forces_ev_ang"]) - f0)
        displacements.append(_minimum_image_displacement(pos - eq_pos, eq_cell))
    phonon.forces = delta_forces
    phonon.produce_force_constants(calculate_full_force_constants=True)

    np.savez_compressed(
        out / "finite_displacement_data.npz",
        equilibrium_positions=eq_pos,
        supercell=eq_cell,
        displacements=np.asarray(displacements),
        delta_forces=np.asarray(delta_forces),
        reference_forces=f0,
        reference_stress_kbar=np.asarray(reference["stress_kbar"]),
        force_constants=np.asarray(phonon.force_constants),
        supercell_matrix=np.asarray(supercell_matrix, dtype=int),
    )
    metadata = {
        "reference_method": "phonopy symmetry-reduced central finite displacement",
        "phonopy_version": phonopy.__version__,
        "displacement_distance_ang": float(displacement_distance_ang),
        "plus_minus": bool(plus_minus),
        "n_displaced_force_calls": len(delta_forces),
        "n_reference_force_calls": 1,
        "max_reference_force_ev_ang": float(
            np.max(np.linalg.norm(f0, axis=1))
        ),
        "rms_reference_force_component_ev_ang": float(np.sqrt(np.mean(f0**2))),
        "reference_stress_kbar": np.asarray(reference["stress_kbar"]).tolist(),
        "supercell_matrix": np.asarray(supercell_matrix, dtype=int).tolist(),
    }
    if hasattr(force_calculator, "provenance"):
        metadata["dft_provenance"] = force_calculator.provenance()
    (out / "reference_metadata.json").write_text(
        json.dumps(metadata, indent=2, allow_nan=False)
    )
    return phonon


def canonical_random_displacements(
    reference_phonon,
    temperature_k: float,
    n_snapshots: int,
    seed: int,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray, List[str]]:
    """Draw harmonic canonical snapshots from the converged reference IFCs."""
    reference_phonon.generate_displacements(
        number_of_snapshots=int(n_snapshots),
        random_seed=int(seed),
        temperature=float(temperature_k),
        cutoff_frequency=0.1,
    )
    equilibrium = reference_phonon.supercell
    eq_pos = np.asarray(equilibrium.positions, dtype=float)
    cell = np.asarray(equilibrium.cell, dtype=float)
    displacements = []
    for snapshot in reference_phonon.supercells_with_displacements:
        displacements.append(
            _minimum_image_displacement(
                np.asarray(snapshot.positions) - eq_pos, cell
            )
        )
    return (
        np.asarray(displacements),
        eq_pos,
        cell,
        list(equilibrium.symbols),
    )


def reorder_displacements_to_target(
    source_equilibrium: np.ndarray,
    source_symbols: Sequence[str],
    source_displacements: np.ndarray,
    target_equilibrium: np.ndarray,
    target_symbols: Sequence[str],
    cell: np.ndarray,
    tolerance_ang: float = 1.0e-5,
) -> np.ndarray:
    """Reorder phonopy snapshots into the SOTC supercell atom order."""
    source_equilibrium = np.asarray(source_equilibrium)
    target_equilibrium = np.asarray(target_equilibrium)
    used = set()
    mapping = []
    inv_cell = np.linalg.inv(cell)
    for target_pos, target_symbol in zip(target_equilibrium, target_symbols):
        best = None
        best_distance = np.inf
        for i, (source_pos, source_symbol) in enumerate(
            zip(source_equilibrium, source_symbols)
        ):
            if i in used or source_symbol != target_symbol:
                continue
            delta = source_pos - target_pos
            frac = delta @ inv_cell
            frac -= np.round(frac)
            distance = np.linalg.norm(frac @ cell)
            if distance < best_distance:
                best, best_distance = i, distance
        if best is None or best_distance > tolerance_ang:
            raise ValueError(
                f"Could not match target atom {target_symbol} at {target_pos}; "
                f"nearest distance={best_distance}"
            )
        used.add(best)
        mapping.append(best)
    return np.asarray(source_displacements)[:, mapping, :]


def evaluate_displacement_forces(
    equilibrium_positions: np.ndarray,
    cell: np.ndarray,
    displacements: np.ndarray,
    force_calculator,
) -> Tuple[np.ndarray, Dict]:
    """Evaluate Delta F for a supplied displacement ensemble with one F(0)."""
    ref = mock_vasp_results(equilibrium_positions, cell, force_calculator)
    f0 = np.asarray(ref["forces_ev_ang"])
    delta = []
    for u in np.asarray(displacements):
        result = mock_vasp_results(equilibrium_positions + u, cell, force_calculator)
        delta.append(np.asarray(result["forces_ev_ang"]) - f0)
    evidence = {
        "max_reference_force_ev_ang": float(np.max(np.linalg.norm(f0, axis=1))),
        "rms_reference_force_component_ev_ang": float(np.sqrt(np.mean(f0**2))),
        "reference_stress_kbar": np.asarray(ref["stress_kbar"]).tolist(),
    }
    return np.asarray(delta), evidence


def fit_snapshot_prefix(
    equilibrium_positions: np.ndarray,
    cell: np.ndarray,
    masses_amu: np.ndarray,
    basis_indices: np.ndarray,
    displacements: np.ndarray,
    delta_forces: np.ndarray,
    n_snapshots: int,
    r_cutoff: float,
    ridge_alpha: float,
    symmetrize_bonds: bool,
) -> IFCExtractor:
    extractor = IFCExtractor(
        equilibrium_positions,
        cell,
        masses_amu,
        basis_indices=basis_indices,
        r_cutoff=r_cutoff,
        ridge_alpha=ridge_alpha,
        symmetrize_bonds=symmetrize_bonds,
    )
    extractor.fit(
        list(np.asarray(displacements)[:n_snapshots]),
        list(np.asarray(delta_forces)[:n_snapshots]),
    )
    return extractor


def common_q_path(calc: PhononCalculator, points_per_segment: int = 50):
    q_path, labels, label_positions = calc._auto_high_symmetry_path(
        points_per_segment
    )
    return q_path, labels, label_positions


def phonopy_frequencies_on_path(reference_phonon, q_path: np.ndarray) -> np.ndarray:
    return np.asarray([
        np.sort(np.asarray(reference_phonon.get_frequencies(q), dtype=float))
        for q in q_path
    ])


def dispersion_metrics(
    model_frequencies_thz: np.ndarray,
    reference_frequencies_thz: np.ndarray,
    q_path: np.ndarray,
    reciprocal_lattice_rad_ang: np.ndarray,
    unstable_tolerance_thz: float = 0.05,
) -> Dict:
    """Return branch-sorted common-q dispersion and sound-velocity metrics."""
    model = np.sort(np.asarray(model_frequencies_thz, dtype=float), axis=1)
    reference = np.sort(np.asarray(reference_frequencies_thz, dtype=float), axis=1)
    if model.shape != reference.shape:
        raise ValueError(f"Frequency shapes differ: {model.shape} vs {reference.shape}")
    error = model - reference
    abs_error = np.abs(error)
    metrics = {
        "frequency_mae_thz": float(np.mean(abs_error)),
        "frequency_rmse_thz": float(np.sqrt(np.mean(error**2))),
        "frequency_max_abs_error_thz": float(np.max(abs_error)),
        "model_unstable_fraction": float(np.mean(model < -unstable_tolerance_thz)),
        "reference_unstable_fraction": float(
            np.mean(reference < -unstable_tolerance_thz)
        ),
        "n_qpoints": int(model.shape[0]),
        "n_branches": int(model.shape[1]),
    }
    # First Gamma-originating segment: fit the three lowest branches over the
    # first five nonzero q points.  v = d(2*pi*nu)/dq.
    gamma_indices = np.where(np.linalg.norm(q_path, axis=1) < 1.0e-12)[0]
    if len(gamma_indices):
        start = int(gamma_indices[0])
        stop = min(start + 6, len(q_path))
        q_cart = (q_path[start:stop] - q_path[start]) @ reciprocal_lattice_rad_ang
        distance = np.linalg.norm(q_cart, axis=1)
        if len(distance) >= 3 and np.any(distance[1:] > 0):
            velocities_model = []
            velocities_reference = []
            for branch in range(min(3, model.shape[1])):
                slope_m = np.polyfit(distance[1:], model[start+1:stop, branch], 1)[0]
                slope_r = np.polyfit(distance[1:], reference[start+1:stop, branch], 1)[0]
                velocities_model.append(float(2.0 * np.pi * 100.0 * slope_m))
                velocities_reference.append(float(2.0 * np.pi * 100.0 * slope_r))
            metrics["sound_velocity_model_m_s"] = velocities_model
            metrics["sound_velocity_reference_m_s"] = velocities_reference
            metrics["sound_velocity_mae_m_s"] = float(np.mean(np.abs(
                np.asarray(velocities_model) - np.asarray(velocities_reference)
            )))
    return metrics


def _minimum_image_displacement(displacement, cell):
    inv = np.linalg.inv(np.asarray(cell))
    frac = np.asarray(displacement) @ inv
    frac -= np.round(frac)
    return frac @ np.asarray(cell)



def compare_anharmonic_orders(
    displacements: np.ndarray,
    delta_forces: np.ndarray,
    output_dir: str | Path,
    max_modes: int = 3,
    test_fraction: float = 0.2,
    random_seed: int = 2718,
) -> Dict:
    """Compare nested effective order-2/3/4 force models on held-out DFT data.

    A PCA basis is learned from training displacements only. All three models
    use exactly that basis and exactly the same split. This is a reduced-space
    force-prediction diagnostic, not a substitute for a symmetry/locality-
    complete real-space fc3/fc4 transport calculation.
    """
    from math import comb
    from .anharmonic_ifc import ModeSpaceAnharmonicIFC

    u = np.asarray(displacements, dtype=float)
    forces = np.asarray(delta_forces, dtype=float)
    if u.shape != forces.shape or u.ndim != 3 or u.shape[2] != 3:
        raise ValueError("displacements and delta_forces must have shape (n,N,3)")
    if len(u) < 10:
        raise ValueError("At least ten snapshots are required for order comparison")
    if not 0.0 < test_fraction < 0.5:
        raise ValueError("test_fraction must lie between zero and 0.5")

    # Remove the rigid-translation components that IFCs cannot identify.
    u = u - np.mean(u, axis=1, keepdims=True)
    forces = forces - np.mean(forces, axis=1, keepdims=True)
    rng = np.random.default_rng(random_seed)
    permutation = rng.permutation(len(u))
    n_test = max(2, int(round(test_fraction * len(u))))
    test_indices = np.sort(permutation[:n_test])
    train_indices = np.sort(permutation[n_test:])
    u_train = u[train_indices]
    f_train = forces[train_indices]
    u_test = u[test_indices]
    f_test = forces[test_indices]

    _, singular_values, right_vectors = np.linalg.svd(
        u_train.reshape((len(u_train), -1)), full_matrices=False
    )
    displacement_rank = int(np.count_nonzero(
        singular_values > 1.0e-10 * singular_values[0]
    ))
    n_modes = min(int(max_modes), displacement_rank)
    # Each snapshot provides n_modes projected force equations. Ensure the
    # quartic model has more training equations than unknown coefficients.
    while n_modes > 0:
        n_parameters_order4 = sum(
            comb(n_modes + order - 1, order) for order in range(2, 5)
        )
        if n_parameters_order4 < len(u_train) * n_modes:
            break
        n_modes -= 1
    if n_modes == 0:
        raise ValueError("Not enough independent snapshots for even one quartic mode")
    basis = right_vectors[:n_modes].T
    # Re-orthogonalize after removing numerical translation leakage.
    n_atoms = u.shape[1]
    for axis in range(3):
        translation = np.zeros(3 * n_atoms)
        translation[axis::3] = 1.0 / np.sqrt(n_atoms)
        basis -= np.outer(translation, translation @ basis)
    basis, _ = np.linalg.qr(basis)

    output = Path(output_dir)
    output.mkdir(parents=True, exist_ok=True)
    report: Dict = {
        "scope": (
            "nested reduced-mode energy-derived force models; not complete "
            "real-space transport IFCs"
        ),
        "n_snapshots": len(u),
        "n_training_snapshots": len(train_indices),
        "n_test_snapshots": len(test_indices),
        "training_indices": train_indices.tolist(),
        "test_indices": test_indices.tolist(),
        "pca_modes": n_modes,
        "training_displacement_singular_values_ang": singular_values.tolist(),
        "orders": {},
    }
    previous_rmse = None
    for order in (2, 3, 4):
        model = ModeSpaceAnharmonicIFC(
            n_atoms=n_atoms, max_order=order, mode_basis=basis,
            ridge_by_order={2: 0.0, 3: 0.0, 4: 0.0},
            rcond=1.0e-10,
        )
        try:
            model.fit(
                u_train, f_train, require_full_rank=True,
                validation_fraction=0.2, random_seed=random_seed + order,
            )
        except ValueError as error:
            report["orders"][str(order)] = {
                "identifiable": False, "error": str(error)
            }
            continue
        prediction = model.predict_forces(u_test)
        projected_prediction = model.project_forces(prediction)
        projected_reference = model.project_forces(f_test)
        projected_error = projected_prediction - projected_reference
        cartesian_error = prediction - f_test
        projected_rmse = float(np.sqrt(np.mean(projected_error**2)))
        order_report = model.diagnostics()
        order_report.pop("singular_values", None)
        order_report.pop("validation_indices", None)
        order_report.update({
            "external_projected_force_rmse_ev_ang": projected_rmse,
            "external_cartesian_force_rmse_ev_ang": float(
                np.sqrt(np.mean(cartesian_error**2))
            ),
            "external_projected_force_mae_ev_ang": float(
                np.mean(np.abs(projected_error))
            ),
        })
        if previous_rmse is not None:
            order_report["projected_rmse_improvement_vs_lower_order_percent"] = (
                100.0 * (previous_rmse - projected_rmse) / previous_rmse
                if previous_rmse > 0 else 0.0
            )
        previous_rmse = projected_rmse
        model.save(str(output / f"mode_ifc_order{order}.npz"))
        report["orders"][str(order)] = order_report

    order3 = report["orders"].get("3", {})
    order4 = report["orders"].get("4", {})
    report["cubic_supported_by_external_cv"] = bool(
        order3.get("identifiable", False)
        and order3.get("projected_rmse_improvement_vs_lower_order_percent", 0.0) > 2.0
    )
    report["quartic_supported_by_external_cv"] = bool(
        order4.get("identifiable", False)
        and order4.get("projected_rmse_improvement_vs_lower_order_percent", 0.0) > 2.0
    )
    (output / "order_comparison.json").write_text(
        json.dumps(report, indent=2, allow_nan=False)
    )
    np.savez_compressed(
        output / "order_comparison_data.npz",
        displacements=u,
        delta_forces=forces,
        mode_basis=basis,
        train_indices=train_indices,
        test_indices=test_indices,
    )
    return report
