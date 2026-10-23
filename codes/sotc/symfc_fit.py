"""Symmetry-constrained FC2/FC3/FC4 fitting and validation utilities.

The fitting backend is symfc. This module centralizes the production
pieces: translational sum rules are imposed in the basis, nested models are
compared on an untouched test set, and the normal-equation rank is reported
after column scaling so IFC orders with different units remain comparable.
"""

from __future__ import annotations

from typing import Dict, Iterable, Sequence

import numpy as np


def symfc_atoms(numbers, scaled_positions, cell):
    """Create a ``SymfcAtoms`` without requiring symfc at package import."""
    from symfc.utils.utils import SymfcAtoms

    return SymfcAtoms(
        numbers=np.asarray(numbers, dtype=int),
        scaled_positions=np.asarray(scaled_positions, dtype=float),
        cell=np.asarray(cell, dtype=float),
    )


def forces_from_fcs(
    displacements: np.ndarray,
    force_constants: Dict[int, np.ndarray],
) -> np.ndarray:
    """Evaluate ``F=-Phi2*u-Phi3*u*u/2-Phi4*u*u*u/6``."""
    u = np.asarray(displacements, dtype=float)
    if u.ndim == 2:
        u = u[None, ...]
        squeeze = True
    elif u.ndim == 3:
        squeeze = False
    else:
        raise ValueError("displacements must have shape (N,3) or (S,N,3)")

    prediction = np.zeros_like(u)
    if 2 in force_constants:
        prediction -= np.einsum(
            "ijab,sjb->sia", np.asarray(force_constants[2]), u, optimize=True
        )
    if 3 in force_constants:
        prediction -= 0.5 * np.einsum(
            "ijkabc,sjb,skc->sia",
            np.asarray(force_constants[3]), u, u, optimize=True,
        )
    if 4 in force_constants:
        prediction -= (1.0 / 6.0) * np.einsum(
            "ijklabcd,sjb,skc,sld->sia",
            np.asarray(force_constants[4]), u, u, u, optimize=True,
        )
    return prediction[0] if squeeze else prediction


def forces_from_compact_fcs(
    displacements: np.ndarray,
    force_constants: Dict[int, np.ndarray],
    translation_permutations: np.ndarray,
    p2s_map: np.ndarray,
    zero_tolerance: float = 1.0e-14,
) -> np.ndarray:
    """Evaluate compact translationally reduced FCs without full expansion.

    Only nonzero atom-cluster blocks are contracted. This is important for
    FC4, for which a 54-atom full tensor would occupy several gigabytes even
    though the cutoff-constrained compact tensor is sparse.
    """
    u = np.asarray(displacements, dtype=float)
    if u.ndim == 2:
        u = u[None, ...]
        squeeze = True
    elif u.ndim == 3:
        squeeze = False
    else:
        raise ValueError("displacements must have shape (N,3) or (S,N,3)")
    permutations = np.asarray(translation_permutations, dtype=int)
    representatives = np.asarray(p2s_map, dtype=int)
    n_atoms = u.shape[1]
    if permutations.ndim != 2 or permutations.shape[1] != n_atoms:
        raise ValueError("translation_permutations has inconsistent shape")

    atom_map = []
    for atom in range(n_atoms):
        matches = np.argwhere(permutations[:, representatives] == atom)
        if len(matches) != 1:
            raise ValueError(f"Could not uniquely translate a representative to atom {atom}")
        translation, primitive_index = matches[0]
        atom_map.append((int(primitive_index), int(translation)))

    prediction = np.zeros_like(u)
    tol = float(zero_tolerance)
    for atom, (primitive_index, translation) in enumerate(atom_map):
        perm = permutations[translation]
        if 2 in force_constants:
            fc2 = np.asarray(force_constants[2])[primitive_index]
            nonzero = np.where(np.max(np.abs(fc2), axis=(1, 2)) > tol)[0]
            if len(nonzero):
                prediction[:, atom] -= np.einsum(
                    "qab,sqb->sa", fc2[nonzero], u[:, perm[nonzero]], optimize=True
                )
        if 3 in force_constants:
            fc3 = np.asarray(force_constants[3])[primitive_index]
            nonzero = np.argwhere(np.max(np.abs(fc3), axis=(2, 3, 4)) > tol)
            if len(nonzero):
                j, k = nonzero.T
                prediction[:, atom] -= 0.5 * np.einsum(
                    "qabc,sqb,sqc->sa", fc3[j, k], u[:, perm[j]], u[:, perm[k]],
                    optimize=True,
                )
        if 4 in force_constants:
            fc4 = np.asarray(force_constants[4])[primitive_index]
            nonzero = np.argwhere(np.max(np.abs(fc4), axis=(3, 4, 5, 6)) > tol)
            if len(nonzero):
                j, k, ell = nonzero.T
                prediction[:, atom] -= (1.0 / 6.0) * np.einsum(
                    "qabcd,sqb,sqc,sqd->sa",
                    fc4[j, k, ell], u[:, perm[j]], u[:, perm[k]], u[:, perm[ell]],
                    optimize=True,
                )
    return prediction[0] if squeeze else prediction


def force_metrics(reference: np.ndarray, prediction: np.ndarray) -> Dict[str, float]:
    """Return component-wise force errors in eV/angstrom."""
    actual = np.asarray(reference, dtype=float).ravel()
    fitted = np.asarray(prediction, dtype=float).ravel()
    if actual.shape != fitted.shape:
        raise ValueError("reference and prediction shapes differ")
    residual = fitted - actual
    ss_res = float(residual @ residual)
    centered = actual - np.mean(actual)
    ss_tot = float(centered @ centered)
    return {
        "rmse_force_component_ev_ang": float(np.sqrt(np.mean(residual**2))),
        "mae_force_component_ev_ang": float(np.mean(np.abs(residual))),
        "max_abs_force_component_ev_ang": float(np.max(np.abs(residual))),
        "r2_force_components": float(1.0 - ss_res / ss_tot) if ss_tot > 0 else 0.0,
    }


def _normal_equation(symfc_object, orders: Sequence[int], batch_size: int):
    """Return symfc's exact normal equation for design diagnostics."""
    d = np.asarray(symfc_object.displacements).reshape(
        len(symfc_object.displacements), -1
    )
    f = np.asarray(symfc_object.forces).reshape(len(symfc_object.forces), -1)
    basis = symfc_object.basis_set
    key = tuple(sorted(orders))
    if key == (2,):
        from symfc.solvers.solver_O2 import prepare_normal_equation_O2

        return prepare_normal_equation_O2(
            d, f, basis[2], batch_size=batch_size, use_mkl=False, verbose=False
        )
    if key == (2, 3):
        from symfc.solvers.solver_O2O3 import prepare_normal_equation_O2O3

        return prepare_normal_equation_O2O3(
            d, f, basis[2], basis[3], batch_size=batch_size,
            use_mkl=False, verbose=False,
        )
    if key == (2, 3, 4):
        from symfc.solvers.solver_O2O3O4 import prepare_normal_equation_O2O3O4

        return prepare_normal_equation_O2O3O4(
            d, f, basis[2], basis[3], basis[4], batch_size=batch_size,
            use_mkl=False, verbose=False,
        )
    raise ValueError(f"Unsupported joint orders: {key}")


def design_diagnostics(normal_matrix: np.ndarray) -> Dict[str, float | int]:
    """Diagnose rank after normalizing every design-matrix column."""
    gram = np.asarray(normal_matrix, dtype=float)
    diagonal = np.clip(np.diag(gram), 0.0, None)
    scale = np.sqrt(diagonal)
    active = scale > np.finfo(float).eps * max(float(np.max(scale)), 1.0)
    normalized = gram[np.ix_(active, active)] / np.outer(scale[active], scale[active])
    eig = np.clip(np.linalg.eigvalsh(0.5 * (normalized + normalized.T)), 0.0, None)
    threshold = max(len(eig), 1) * np.finfo(float).eps * max(float(np.max(eig)), 1.0)
    positive = eig[eig > threshold]
    raw_eig = np.clip(np.linalg.eigvalsh(0.5 * (gram + gram.T)), 0.0, None)
    raw_threshold = max(len(raw_eig), 1) * np.finfo(float).eps * max(
        float(np.max(raw_eig)), 1.0
    )
    raw_positive = raw_eig[raw_eig > raw_threshold]
    return {
        "n_parameters": int(gram.shape[0]),
        "normalized_design_rank": int(len(positive)),
        "normalized_design_condition_number": float(positive[-1] / positive[0])
        if len(positive) else float("inf"),
        "raw_normal_matrix_rank": int(len(raw_positive)),
        "raw_normal_matrix_condition_number": float(raw_positive[-1] / raw_positive[0])
        if len(raw_positive) else float("inf"),
        "min_normalized_eigenvalue": float(positive[0]) if len(positive) else 0.0,
        "max_normalized_eigenvalue": float(positive[-1]) if len(positive) else 0.0,
    }


def fit_symfc_model(
    supercell,
    displacements: np.ndarray,
    forces: np.ndarray,
    orders: Iterable[int],
    cutoffs: Dict[int, float],
    batch_size: int = 25,
    full_force_constants: bool = True,
):
    """Fit one joint, symmetry- and sum-rule-constrained symfc model."""
    from symfc import Symfc

    selected = tuple(sorted(int(order) for order in orders))
    model = Symfc(
        supercell,
        cutoff={int(k): float(v) for k, v in cutoffs.items()},
        use_mkl=False,
        log_level=0,
    )
    model.displacements = np.asarray(displacements, dtype=float)
    model.forces = np.asarray(forces, dtype=float)
    # Symfc.run() disables sum rules for regression data, so construct the
    # constrained basis explicitly before solving.
    try:
        model.compute_basis_set(orders=list(selected), apply_sum_rule=True)
    except TypeError:
        # symfc >= 1.6 always builds the sum rules into the basis.
        model.compute_basis_set(orders=list(selected))
    try:
        xtx, _ = _normal_equation(model, selected, batch_size)
        design = design_diagnostics(xtx)
    except TypeError:
        # The internal normal-equation helpers changed signature in
        # symfc 1.6; the solve below does not depend on them.
        design = {}
    design["basis_parameters_by_order"] = {
        str(order): int(model.basis_set[order].blocked_basis_set.shape[1])
        for order in selected
    }
    model.solve(
        orders=list(selected),
        is_compact_fc=not full_force_constants,
        batch_size=batch_size,
    )
    return model, design


def invariance_diagnostics(force_constants: Dict[int, np.ndarray]) -> Dict[str, float]:
    """Return acoustic-sum-rule and adjacent permutation residuals."""
    result: Dict[str, float] = {}
    for order, values in sorted(force_constants.items()):
        fc = np.asarray(values)
        asr = max(
            float(np.max(np.abs(np.sum(fc, axis=axis)))) for axis in range(order)
        )
        permutation = 0.0
        for axis in range(order - 1):
            swapped = np.swapaxes(
                np.swapaxes(fc, axis, axis + 1), order + axis, order + axis + 1
            )
            permutation = max(permutation, float(np.max(np.abs(fc - swapped))))
        result[f"fc{order}_max_asr_residual"] = asr
        result[f"fc{order}_max_adjacent_permutation_residual"] = permutation
    return result


def compact_invariance_diagnostics(
    force_constants: Dict[int, np.ndarray],
) -> Dict[str, float]:
    """Return sum-rule and trailing-index permutation residuals for compact FCs."""
    result: Dict[str, float] = {}
    for order, values in sorted(force_constants.items()):
        fc = np.asarray(values)
        asr = max(
            float(np.max(np.abs(np.sum(fc, axis=axis))))
            for axis in range(1, order)
        )
        permutation = 0.0
        for axis in range(1, order - 1):
            swapped = np.swapaxes(
                np.swapaxes(fc, axis, axis + 1), order + axis, order + axis + 1
            )
            permutation = max(permutation, float(np.max(np.abs(fc - swapped))))
        result[f"compact_fc{order}_max_asr_residual"] = asr
        result[f"compact_fc{order}_max_trailing_permutation_residual"] = permutation
    return result
