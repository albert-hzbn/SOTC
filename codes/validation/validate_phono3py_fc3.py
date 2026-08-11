#!/usr/bin/env python3
"""Independent synthetic fc2/fc3 convention test against phono3py."""

from __future__ import annotations

import argparse
import json
import os
import platform
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "codes"))

from sotc.anharmonic_ifc import ModeSpaceAnharmonicIFC
from sotc.anharmonic_io import write_phono3py_fc3, write_shengbte_fc3


def metrics(candidate: np.ndarray, reference: np.ndarray) -> dict[str, float]:
    error = np.asarray(candidate) - np.asarray(reference)
    return {
        "rmse": float(np.sqrt(np.mean(error**2))),
        "max_abs": float(np.max(np.abs(error))),
        "reference_max_abs": float(np.max(np.abs(reference))),
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--output", type=Path,
        default=ROOT / "validation_results" / "synthetic_phono3py",
    )
    parser.add_argument("--seed", type=int, default=5)
    parser.add_argument("--distance", type=float, default=0.01)
    args = parser.parse_args()

    import phono3py
    from phono3py import Phono3py
    from phonopy import __version__ as phonopy_version
    from phonopy.structure.atoms import PhonopyAtoms

    unitcell = PhonopyAtoms(
        symbols=["Na", "Cl"],
        cell=4.0 * np.eye(3),
        scaled_positions=[[0.0, 0.0, 0.0], [0.5, 0.5, 0.5]],
    )
    reference = Phono3py(
        unitcell,
        supercell_matrix=np.eye(3, dtype=int),
        primitive_matrix=np.eye(3),
        is_symmetry=False,
        log_level=0,
    )
    reference.generate_displacements(
        distance=args.distance, is_plusminus=True, is_diagonal=False
    )

    model = ModeSpaceAnharmonicIFC(n_atoms=2, max_order=3, rcond=1e-12)
    rng = np.random.default_rng(args.seed)
    model.coefficients = rng.normal(0.0, 0.5, len(model.coefficients))
    equilibrium = np.asarray(reference.supercell.positions)
    displacements = []
    forces = []
    for displaced_cell in reference.supercells_with_displacements:
        displacement = np.asarray(displaced_cell.positions) - equilibrium
        displacements.append(displacement)
        forces.append(model.predict_forces(displacement[None])[0])
    reference.forces = np.asarray(forces)
    reference.produce_fc3(is_compact_fc=False, fc_calculator="traditional")

    fc2_expected = model.dense_cartesian_tensor(2)
    fc3_expected = model.dense_cartesian_tensor(3)
    result = {
        "test": "analytic polynomial forces -> phono3py finite differences",
        "python": platform.python_version(),
        "numpy": np.__version__,
        "phonopy": phonopy_version,
        "phono3py": phono3py.__version__,
        "n_displaced_supercells": len(displacements),
        "displacement_distance_ang": args.distance,
        "is_symmetry": False,
        "fc2_units": "eV/angstrom^2",
        "fc3_units": "eV/angstrom^3",
        "fc2": metrics(reference.fc2, fc2_expected),
        "fc3": metrics(reference.fc3, fc3_expected),
    }
    tolerance = 1.0e-9
    result["passed"] = (
        result["fc2"]["max_abs"] < tolerance
        and result["fc3"]["max_abs"] < tolerance
    )
    args.output.mkdir(parents=True, exist_ok=True)
    write_phono3py_fc3(args.output / "fc3.hdf5", fc3_expected)
    write_shengbte_fc3(
        args.output / "FORCE_CONSTANTS_3RD",
        fc3_expected,
        reference_atom_indices=np.array([0, 1]),
        basis_indices=np.array([0, 1]),
        lattice_vectors_ang=np.zeros((2, 3)),
    )
    np.savez_compressed(
        args.output / "comparison_arrays.npz",
        displacements=np.asarray(displacements),
        forces=np.asarray(forces),
        fc2_sotc=fc2_expected,
        fc2_phono3py=reference.fc2,
        fc3_sotc=fc3_expected,
        fc3_phono3py=reference.fc3,
    )
    (args.output / "validation.json").write_text(json.dumps(result, indent=2))
    print(json.dumps(result, indent=2))
    if not result["passed"]:
        raise SystemExit("phono3py tensor-convention validation failed")


if __name__ == "__main__":
    main()
