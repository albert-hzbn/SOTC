"""Energy-consistent mode-space second-, third-, and fourth-order IFC fits.

The coefficients describe a polynomial Born--Oppenheimer surface in a
controlled internal-coordinate subspace. This is deliberately distinct from
a complete real-space high-order cluster expansion.
"""

from __future__ import annotations

from collections import Counter
from dataclasses import dataclass
from itertools import combinations_with_replacement, permutations, product
from math import factorial
from typing import Callable

import numpy as np


@dataclass(frozen=True)
class Monomial:
    """One unique symmetric tensor component."""

    indices: tuple[int, ...]

    @property
    def order(self) -> int:
        return len(self.indices)


class ModeSpaceAnharmonicIFC:
    """Fit an energy-derived polynomial force model through fourth order.

    mode_basis is an optional orthonormal, translation-free Cartesian basis
    with shape (3*n_atoms, n_modes). If absent, the complete translation-free
    basis is used. Ridge penalties stabilize a solve but never count toward the
    reported data rank.
    """

    def __init__(
        self,
        n_atoms: int,
        max_order: int = 4,
        mode_basis: np.ndarray | None = None,
        ridge_by_order: dict[int, float] | None = None,
        rcond: float = 1.0e-10,
    ):
        self.n_atoms = int(n_atoms)
        if self.n_atoms < 2:
            raise ValueError("At least two atoms are needed to remove translations")
        self.max_order = int(max_order)
        if self.max_order not in (2, 3, 4):
            raise ValueError("max_order must be 2, 3, or 4")
        self.rcond = float(rcond)
        if self.rcond <= 0:
            raise ValueError("rcond must be positive")

        self.mode_basis = self._prepare_basis(mode_basis)
        self.n_modes = self.mode_basis.shape[1]
        self.monomials = [
            Monomial(tuple(indices))
            for order in range(2, self.max_order + 1)
            for indices in combinations_with_replacement(range(self.n_modes), order)
        ]
        penalties = {2: 0.0, 3: 0.0, 4: 0.0}
        if ridge_by_order is not None:
            penalties.update({int(k): float(v) for k, v in ridge_by_order.items()})
        if any(value < 0 for value in penalties.values()):
            raise ValueError("Ridge penalties must be non-negative")
        self.ridge_by_order = penalties
        self.coefficients = np.zeros(len(self.monomials), dtype=float)
        self._diagnostics: dict[str, object] = {}

    def _prepare_basis(self, supplied: np.ndarray | None) -> np.ndarray:
        ndof = 3 * self.n_atoms
        translations = np.zeros((ndof, 3))
        for atom in range(self.n_atoms):
            translations[3 * atom : 3 * atom + 3, :] = np.eye(3)
        translations /= np.sqrt(self.n_atoms)

        if supplied is None:
            projector = np.eye(ndof) - translations @ translations.T
            eigenvalues, eigenvectors = np.linalg.eigh(projector)
            basis = eigenvectors[:, eigenvalues > 0.5]
        else:
            basis = np.asarray(supplied, dtype=float)
            if basis.ndim != 2 or basis.shape[0] != ndof or basis.shape[1] == 0:
                raise ValueError("mode_basis must have shape (3*n_atoms, n_modes)")
            if not np.allclose(basis.T @ basis, np.eye(basis.shape[1]), atol=1e-10):
                raise ValueError("mode_basis columns must be orthonormal")
            if not np.allclose(translations.T @ basis, 0.0, atol=1e-10):
                raise ValueError("mode_basis must be orthogonal to translations")
        return basis

    @staticmethod
    def _monomial_value(x: np.ndarray, indices: tuple[int, ...]) -> float:
        return float(np.prod(x[np.asarray(indices, dtype=int)]))

    @staticmethod
    def _derivative(x: np.ndarray, indices: tuple[int, ...], coordinate: int) -> float:
        multiplicity = indices.count(coordinate)
        if multiplicity == 0:
            return 0.0
        remaining = list(indices)
        remaining.remove(coordinate)
        return float(multiplicity * np.prod(x[remaining])) if remaining else float(multiplicity)

    def project_displacements(self, displacements: np.ndarray) -> np.ndarray:
        values = np.asarray(displacements, dtype=float)
        if values.shape[-2:] != (self.n_atoms, 3):
            raise ValueError("displacements must end in (n_atoms, 3)")
        return values.reshape((-1, 3 * self.n_atoms)) @ self.mode_basis

    def project_forces(self, forces: np.ndarray) -> np.ndarray:
        values = np.asarray(forces, dtype=float)
        if values.shape[-2:] != (self.n_atoms, 3):
            raise ValueError("forces must end in (n_atoms, 3)")
        return values.reshape((-1, 3 * self.n_atoms)) @ self.mode_basis

    def design_matrix(self, displacements: np.ndarray) -> np.ndarray:
        x_values = self.project_displacements(displacements)
        matrix = np.zeros((len(x_values) * self.n_modes, len(self.monomials)))
        for snapshot, x in enumerate(x_values):
            for coordinate in range(self.n_modes):
                row = snapshot * self.n_modes + coordinate
                for column, monomial in enumerate(self.monomials):
                    matrix[row, column] = -self._derivative(
                        x, monomial.indices, coordinate
                    )
        return matrix

    def _rank_diagnostics(self, matrix: np.ndarray) -> dict[str, object]:
        singular_values = np.linalg.svd(matrix, compute_uv=False)
        largest = float(singular_values[0]) if singular_values.size else 0.0
        tolerance = self.rcond * largest
        rank = int(np.count_nonzero(singular_values > tolerance))
        nonzero = singular_values[singular_values > tolerance]
        condition = float(nonzero[0] / nonzero[-1]) if nonzero.size > 1 else 1.0
        return {
            "data_rank": rank,
            "parameter_count": len(self.monomials),
            "rank_deficiency": len(self.monomials) - rank,
            "rank_tolerance": tolerance,
            "condition_number_identifiable_subspace": condition,
            "singular_values": singular_values,
            "identifiable": rank == len(self.monomials),
        }

    def _solve(self, matrix: np.ndarray, target: np.ndarray) -> np.ndarray:
        regularization = np.asarray(
            [self.ridge_by_order[item.order] for item in self.monomials], dtype=float
        )
        augmented_matrix = np.vstack((matrix, np.diag(np.sqrt(regularization))))
        augmented_target = np.concatenate((target, np.zeros(len(regularization))))
        return np.linalg.lstsq(augmented_matrix, augmented_target, rcond=self.rcond)[0]

    def fit(
        self,
        displacements: np.ndarray,
        forces: np.ndarray,
        require_full_rank: bool = True,
        validation_fraction: float = 0.2,
        random_seed: int = 0,
    ) -> "ModeSpaceAnharmonicIFC":
        u = np.asarray(displacements, dtype=float)
        f = np.asarray(forces, dtype=float)
        if u.shape != f.shape or u.ndim != 3 or u.shape[1:] != (self.n_atoms, 3):
            raise ValueError("displacements and forces must have shape (n, n_atoms, 3)")
        if len(u) < 2:
            raise ValueError("At least two snapshots are required")
        if not 0.0 < validation_fraction < 1.0:
            raise ValueError("validation_fraction must lie between zero and one")

        matrix = self.design_matrix(u)
        target = self.project_forces(f).reshape(-1)
        rank = self._rank_diagnostics(matrix)
        if require_full_rank and not rank["identifiable"]:
            raise ValueError(
                "High-order IFC fit is not identifiable: "
                f"rank {rank['data_rank']}/{rank['parameter_count']}"
            )
        self.coefficients = self._solve(matrix, target)
        residual = matrix @ self.coefficients - target

        n_validation = min(len(u) - 1, max(1, int(round(validation_fraction * len(u)))))
        rng = np.random.default_rng(random_seed)
        validation_indices = np.sort(rng.choice(len(u), n_validation, replace=False))
        training_indices = np.setdiff1d(np.arange(len(u)), validation_indices)
        training_rows = np.concatenate([
            np.arange(index * self.n_modes, (index + 1) * self.n_modes)
            for index in training_indices
        ])
        validation_rows = np.concatenate([
            np.arange(index * self.n_modes, (index + 1) * self.n_modes)
            for index in validation_indices
        ])
        train_coefficients = self._solve(matrix[training_rows], target[training_rows])
        validation_residual = (
            matrix[validation_rows] @ train_coefficients - target[validation_rows]
        )
        training_rank = self._rank_diagnostics(matrix[training_rows])

        self._diagnostics = {
            **rank,
            "n_snapshots": len(u),
            "n_modes": self.n_modes,
            "max_order": self.max_order,
            "n_training_snapshots": len(training_indices),
            "n_validation_snapshots": len(validation_indices),
            "training_subset_identifiable": training_rank["identifiable"],
            "fit_force_rmse": float(np.sqrt(np.mean(residual**2))),
            "validation_force_rmse": float(np.sqrt(np.mean(validation_residual**2))),
            "validation_indices": validation_indices,
            "ridge_by_order": dict(self.ridge_by_order),
            "coefficient_convention": "U=sum c_m product(x_i)",
            "complete_real_space_ifcs": False,
        }
        return self

    def energy(self, displacements: np.ndarray) -> np.ndarray:
        x_values = self.project_displacements(displacements)
        result = np.zeros(len(x_values))
        for snapshot, x in enumerate(x_values):
            result[snapshot] = sum(
                coefficient * self._monomial_value(x, monomial.indices)
                for coefficient, monomial in zip(self.coefficients, self.monomials)
            )
        return result

    def predict_forces(self, displacements: np.ndarray) -> np.ndarray:
        values = np.asarray(displacements, dtype=float)
        projected = (self.design_matrix(values) @ self.coefficients).reshape(
            (-1, self.n_modes)
        )
        return (projected @ self.mode_basis.T).reshape((-1, self.n_atoms, 3))

    def symmetric_tensor_components(self, order: int) -> dict[tuple[int, ...], float]:
        """Return unique Phi^(order) in U=(1/order!) Phi x... convention."""
        if order < 2 or order > self.max_order:
            raise ValueError("Requested order was not fitted")
        components = {}
        for coefficient, monomial in zip(self.coefficients, self.monomials):
            if monomial.order != order:
                continue
            counts = [monomial.indices.count(index) for index in set(monomial.indices)]
            components[monomial.indices] = float(
                coefficient * np.prod([factorial(count) for count in counts])
            )
        return components

    def dense_cartesian_tensor(
        self, order: int, max_elements: int = 50_000_000
    ) -> np.ndarray:
        """Expand a fitted mode tensor into atom-major Cartesian form.

        The returned shapes are (N,N,3,3), (N,N,N,3,3,3), or
        (N,N,N,N,3,3,3,3). Expansion is refused before allocating an array
        larger than max_elements; practical large-cell transport fits require
        locality and space-group reduction, as used by phono3py/symfc.
        """
        if order < 2 or order > self.max_order:
            raise ValueError("Requested order was not fitted")
        n_cart = 3 * self.n_atoms
        n_elements = n_cart ** order
        if n_elements > int(max_elements):
            raise MemoryError(
                f"Dense order-{order} tensor has {n_elements} elements; "
                "use a symmetry/locality-reduced real-space solver"
            )
        mode_shape = (self.n_modes,) * order
        tensor_mode = np.zeros(mode_shape, dtype=float)
        from itertools import permutations

        for indices, value in self.symmetric_tensor_components(order).items():
            for permuted in set(permutations(indices)):
                tensor_mode[permuted] = value

        if order == 2:
            cartesian = np.einsum(
                "am,bn,mn->ab", self.mode_basis, self.mode_basis,
                tensor_mode, optimize=True
            )
        elif order == 3:
            cartesian = np.einsum(
                "am,bn,co,mno->abc", self.mode_basis, self.mode_basis,
                self.mode_basis, tensor_mode, optimize=True
            )
        else:
            cartesian = np.einsum(
                "am,bn,co,dp,mnop->abcd", self.mode_basis, self.mode_basis,
                self.mode_basis, self.mode_basis, tensor_mode, optimize=True
            )
        interleaved_shape = tuple(
            value for _ in range(order) for value in (self.n_atoms, 3)
        )
        interleaved = cartesian.reshape(interleaved_shape)
        atom_axes = tuple(range(0, 2 * order, 2))
        cartesian_axes = tuple(range(1, 2 * order, 2))
        return interleaved.transpose(atom_axes + cartesian_axes)

    def save(self, path: str) -> None:
        """Archive coefficients, basis, conventions, and fit diagnostics."""
        import json

        indices = np.full(
            (len(self.monomials), self.max_order), -1, dtype=int
        )
        orders = np.zeros(len(self.monomials), dtype=int)
        for row, monomial in enumerate(self.monomials):
            orders[row] = monomial.order
            indices[row, :monomial.order] = monomial.indices
        diagnostics = {}
        for key, value in self._diagnostics.items():
            if isinstance(value, np.ndarray):
                diagnostics[key] = value.tolist()
            elif isinstance(value, np.generic):
                diagnostics[key] = value.item()
            else:
                diagnostics[key] = value
        np.savez_compressed(
            path,
            n_atoms=self.n_atoms,
            max_order=self.max_order,
            rcond=self.rcond,
            mode_basis=self.mode_basis,
            coefficients=self.coefficients,
            monomial_indices=indices,
            monomial_orders=orders,
            ridge_orders=np.array([2, 3, 4], dtype=int),
            ridge_values=np.array(
                [self.ridge_by_order[order] for order in (2, 3, 4)]
            ),
            diagnostics_json=json.dumps(diagnostics),
            coefficient_convention="U=sum c_m product(x_i)",
        )

    def diagnostics(self) -> dict[str, object]:
        result = dict(self._diagnostics)
        for key in ("singular_values", "validation_indices"):
            if key in result:
                result[key] = np.asarray(result[key]).copy()
        return result


class FiniteDifferenceAnharmonicIFC:
    """Complete small-cell real-space fc3/fc4 finite differences.

    ``force_function`` receives one Cartesian displacement array with shape
    ``(n_atoms, 3)`` and returns the corresponding force array in eV/angstrom.
    The implementation evaluates central mixed derivatives of the forces,

    ``Phi^(n) = -d^(n-1) F / du^(n-1)``,

    caches repeated force calls, and optionally projects the result onto full
    permutation symmetry.  Its cost scales as ``(3N)^(n-1)``; it is intended
    for reference cells and independent validation, while the mode-space model
    above is the practical reduced representation for larger cells.
    """

    _STENCILS = {
        1: ((-1, -0.5), (1, 0.5)),
        2: ((-1, 1.0), (0, -2.0), (1, 1.0)),
        3: ((-2, -0.5), (-1, 1.0), (1, -1.0), (2, 0.5)),
    }

    def __init__(
        self,
        n_atoms: int,
        force_function: Callable[[np.ndarray], np.ndarray],
        displacement_step_ang: float = 0.01,
    ):
        self.n_atoms = int(n_atoms)
        if self.n_atoms < 1:
            raise ValueError("n_atoms must be positive")
        if not callable(force_function):
            raise TypeError("force_function must be callable")
        self.force_function = force_function
        self.step = float(displacement_step_ang)
        if not np.isfinite(self.step) or self.step <= 0.0:
            raise ValueError("displacement_step_ang must be positive and finite")
        self.n_cart = 3 * self.n_atoms
        self._cache: dict[tuple[int, ...], np.ndarray] = {}
        self._diagnostics: dict[int, dict[str, float | int | bool]] = {}

    @property
    def force_evaluations(self) -> int:
        return len(self._cache)

    def _evaluate(self, integer_offsets: np.ndarray) -> np.ndarray:
        key = tuple(map(int, integer_offsets))
        if key not in self._cache:
            displacement = (
                self.step * np.asarray(integer_offsets, dtype=float)
            ).reshape((self.n_atoms, 3))
            forces = np.asarray(self.force_function(displacement), dtype=float)
            if forces.shape != (self.n_atoms, 3):
                raise ValueError(
                    "force_function must return shape "
                    f"({self.n_atoms}, 3), got {forces.shape}"
                )
            if not np.all(np.isfinite(forces)):
                raise ValueError("force_function returned non-finite forces")
            self._cache[key] = forces.reshape(-1).copy()
        return self._cache[key]

    def _force_derivative(self, coordinates: tuple[int, ...]) -> np.ndarray:
        counts = Counter(coordinates)
        axes = sorted(counts)
        stencil_choices = [self._STENCILS[counts[axis]] for axis in axes]
        derivative = np.zeros(self.n_cart, dtype=float)
        for stencil_points in product(*stencil_choices):
            offsets = np.zeros(self.n_cart, dtype=int)
            weight = 1.0
            for axis, (offset, coefficient) in zip(axes, stencil_points):
                offsets[axis] = offset
                weight *= coefficient
            derivative += weight * self._evaluate(offsets)
        return derivative / self.step ** len(coordinates)

    def _to_atom_major(self, flat_tensor: np.ndarray, order: int) -> np.ndarray:
        interleaved = flat_tensor.reshape((self.n_atoms, 3) * order)
        atom_axes = tuple(range(0, 2 * order, 2))
        cartesian_axes = tuple(range(1, 2 * order, 2))
        return interleaved.transpose(atom_axes + cartesian_axes)

    def extract(self, order: int, symmetrize: bool = True) -> np.ndarray:
        """Return full atom-major fc3 or fc4 in eV/angstrom**order."""
        order = int(order)
        if order not in (3, 4):
            raise ValueError("Only third- and fourth-order IFCs are supported")
        evaluations_before = self.force_evaluations
        flat = np.zeros((self.n_cart,) * order, dtype=float)
        for coordinates in combinations_with_replacement(
            range(self.n_cart), order - 1
        ):
            values = -self._force_derivative(coordinates)
            for permuted in set(permutations(coordinates)):
                flat[(slice(None),) + permuted] = values

        permutation_error = 0.0
        for axes in permutations(range(order)):
            permutation_error = max(
                permutation_error,
                float(np.max(np.abs(flat - flat.transpose(axes)))),
            )
        if symmetrize:
            flat = sum(
                flat.transpose(axes) for axes in permutations(range(order))
            ) / factorial(order)

        asr_by_axis = [
            float(np.max(np.abs(np.sum(flat, axis=axis))))
            for axis in range(order)
        ]
        self._diagnostics[order] = {
            "order": order,
            "displacement_step_ang": self.step,
            "force_evaluations_total": self.force_evaluations,
            "force_evaluations_added": self.force_evaluations - evaluations_before,
            "permutation_max_abs_before_symmetrization": permutation_error,
            "permutation_symmetrized": bool(symmetrize),
            "acoustic_sum_rule_max_abs": max(asr_by_axis),
        }
        return self._to_atom_major(flat, order)

    def extract_fc3_fc4(
        self, symmetrize: bool = True
    ) -> tuple[np.ndarray, np.ndarray]:
        """Extract both tensors while reusing the force-evaluation cache."""
        return self.extract(3, symmetrize), self.extract(4, symmetrize)

    def diagnostics(self, order: int | None = None) -> dict:
        if order is None:
            return {key: dict(value) for key, value in self._diagnostics.items()}
        if int(order) not in self._diagnostics:
            raise ValueError(f"Order {order} has not been extracted")
        return dict(self._diagnostics[int(order)])
