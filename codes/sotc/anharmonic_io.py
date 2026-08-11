"""Interoperability writers for real-space third- and fourth-order IFCs."""

from __future__ import annotations

from itertools import product
from pathlib import Path

import numpy as np


def write_phono3py_fc3(path: str | Path, fc3: np.ndarray) -> None:
    """Write a full real-space fc3 array in phono3py's HDF5 convention."""
    import h5py

    values = np.asarray(fc3, dtype=float)
    if values.ndim != 6 or values.shape[3:] != (3, 3, 3):
        raise ValueError("fc3 must have shape (N, N, N, 3, 3, 3)")
    if len(set(values.shape[:3])) != 1:
        raise ValueError("The first three fc3 dimensions must be equal")
    target = Path(path)
    target.parent.mkdir(parents=True, exist_ok=True)
    with h5py.File(target, "w") as handle:
        dataset = handle.create_dataset("fc3", data=values, compression="gzip")
        dataset.attrs["physical_unit"] = "eV/angstrom^3"
        dataset.attrs["format"] = "full"


def _validate_mapping(
    n_atoms: int,
    reference_atom_indices: np.ndarray,
    basis_indices: np.ndarray,
    lattice_vectors_ang: np.ndarray,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    references = np.asarray(reference_atom_indices, dtype=int)
    basis = np.asarray(basis_indices, dtype=int)
    vectors = np.asarray(lattice_vectors_ang, dtype=float)
    if basis.shape != (n_atoms,):
        raise ValueError("basis_indices must have length N")
    if vectors.shape != (n_atoms, 3):
        raise ValueError("lattice_vectors_ang must have shape (N, 3)")
    if references.ndim != 1 or len(references) == 0:
        raise ValueError("reference_atom_indices must be a nonempty vector")
    if np.any(references < 0) or np.any(references >= n_atoms):
        raise ValueError("reference atom index outside the supercell")
    reference_basis = basis[references]
    if len(np.unique(reference_basis)) != len(references):
        raise ValueError("Reference atoms must map one-to-one onto primitive basis atoms")
    return references, basis, vectors


def _blocks_fc3(
    fc3: np.ndarray,
    references: np.ndarray,
    threshold: float,
):
    n_atoms = fc3.shape[1]
    compact = fc3.shape[0] != n_atoms
    for reference_slot, i in enumerate(references):
        first = reference_slot if compact else i
        for j in range(n_atoms):
            for k in range(n_atoms):
                block = fc3[first, j, k]
                if np.max(np.abs(block)) > threshold:
                    yield int(i), j, k, block


def write_shengbte_fc3(
    path: str | Path,
    fc3: np.ndarray,
    reference_atom_indices: np.ndarray,
    basis_indices: np.ndarray,
    lattice_vectors_ang: np.ndarray,
    threshold_ev_ang3: float = 0.0,
) -> int:
    """Write sparse FORCE_CONSTANTS_3RD input for ShengBTE.

    lattice_vectors_ang gives the primitive-lattice translation of every
    supercell atom relative to its basis site. Only terms above threshold are
    emitted. The return value is the number of tensor blocks written.
    """
    values = np.asarray(fc3, dtype=float)
    if values.ndim != 6 or values.shape[3:] != (3, 3, 3):
        raise ValueError("fc3 must have shape (N, N, N, 3, 3, 3)")
    n_atoms = values.shape[1]
    if values.shape[2] != n_atoms:
        raise ValueError("fc3 trailing atom dimensions must be equal")
    references, basis, vectors = _validate_mapping(
        n_atoms, reference_atom_indices, basis_indices, lattice_vectors_ang
    )
    if values.shape[0] not in (n_atoms, len(references)):
        raise ValueError(
            "fc3 first atom dimension must be N (full) or len(reference_atom_indices) "
            "(compact)"
        )
    threshold = float(threshold_ev_ang3)
    if threshold < 0:
        raise ValueError("threshold must be non-negative")
    blocks = list(_blocks_fc3(values, references, threshold))
    lines = [str(len(blocks))]
    for block_number, (i, j, k, block) in enumerate(blocks, start=1):
        lines.extend([
            "",
            str(block_number),
            " ".join(f"{value:.16e}" for value in vectors[j]),
            " ".join(f"{value:.16e}" for value in vectors[k]),
            f"{basis[i] + 1:d} {basis[j] + 1:d} {basis[k] + 1:d}",
        ])
        for alpha, beta, gamma in product(range(3), repeat=3):
            lines.append(
                f"{alpha + 1:d} {beta + 1:d} {gamma + 1:d} "
                f"{block[alpha, beta, gamma]:.16e}"
            )
    target = Path(path)
    target.parent.mkdir(parents=True, exist_ok=True)
    target.write_text("\n".join(lines) + "\n")
    return len(blocks)


def _blocks_fc4(
    fc4: np.ndarray,
    references: np.ndarray,
    threshold: float,
):
    n_atoms = fc4.shape[1]
    compact = fc4.shape[0] != n_atoms
    for reference_slot, i in enumerate(references):
        first = reference_slot if compact else i
        for j in range(n_atoms):
            for k in range(n_atoms):
                for ell in range(n_atoms):
                    block = fc4[first, j, k, ell]
                    if np.max(np.abs(block)) > threshold:
                        yield int(i), j, k, ell, block


def write_fourphonon_fc4(
    path: str | Path,
    fc4: np.ndarray,
    reference_atom_indices: np.ndarray,
    basis_indices: np.ndarray,
    lattice_vectors_ang: np.ndarray,
    threshold_ev_ang4: float = 0.0,
) -> int:
    """Write sparse FORCE_CONSTANTS_4TH input for the FourPhonon extension."""
    values = np.asarray(fc4, dtype=float)
    if values.ndim != 8 or values.shape[4:] != (3, 3, 3, 3):
        raise ValueError("fc4 must have shape (N, N, N, N, 3, 3, 3, 3)")
    n_atoms = values.shape[1]
    if values.shape[2:4] != (n_atoms, n_atoms):
        raise ValueError("fc4 trailing atom dimensions must be equal")
    references, basis, vectors = _validate_mapping(
        n_atoms, reference_atom_indices, basis_indices, lattice_vectors_ang
    )
    if values.shape[0] not in (n_atoms, len(references)):
        raise ValueError(
            "fc4 first atom dimension must be N (full) or len(reference_atom_indices) "
            "(compact)"
        )
    threshold = float(threshold_ev_ang4)
    if threshold < 0:
        raise ValueError("threshold must be non-negative")
    blocks = list(_blocks_fc4(values, references, threshold))
    lines = [str(len(blocks))]
    for block_number, (i, j, k, ell, block) in enumerate(blocks, start=1):
        lines.extend([
            "",
            str(block_number),
            " ".join(f"{value:.16e}" for value in vectors[j]),
            " ".join(f"{value:.16e}" for value in vectors[k]),
            " ".join(f"{value:.16e}" for value in vectors[ell]),
            f"{basis[i] + 1:d} {basis[j] + 1:d} {basis[k] + 1:d} {basis[ell] + 1:d}",
        ])
        for alpha, beta, gamma, delta in product(range(3), repeat=4):
            lines.append(
                f"{alpha + 1:d} {beta + 1:d} {gamma + 1:d} {delta + 1:d} "
                f"{block[alpha, beta, gamma, delta]:.16e}"
            )
    target = Path(path)
    target.parent.mkdir(parents=True, exist_ok=True)
    target.write_text("\n".join(lines) + "\n")
    return len(blocks)
