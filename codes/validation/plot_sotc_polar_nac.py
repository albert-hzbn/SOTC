#!/usr/bin/env python3
"""Apply QE-DFPT tensors to selected production SOTC FC2 fits."""

from __future__ import annotations

import json
from pathlib import Path
import sys

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from phonopy import Phonopy
from phonopy.structure.atoms import PhonopyAtoms

try:
    from phonopy.physical_units import get_physical_units
    _units = get_physical_units()
    NAC_FACTOR = _units.Hartree * _units.Bohr
except ModuleNotFoundError:
    from phonopy import units as _units
    NAC_FACTOR = _units.Hartree * _units.Bohr

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "codes"))

from sotc.ifc_extractor import IFCExtractor
from sotc.phonons import PhononCalculator
from sotc.postprocessor import _parse_qe_pwo


MATERIALS = {
    "nacl": {"label": "NaCl", "elements": ["Na", "Cl"],
             "masses": [22.9898, 35.453]},
    "mgo": {"label": "MgO", "elements": ["Mg", "O"],
            "masses": [24.305, 15.999]},
    "pbte": {"label": "PbTe", "elements": ["Pb", "Te"],
             "masses": [207.2, 127.60]},
}


def signed_frequencies(phonon, qpoint, q_direction=None):
    if q_direction is None:
        phonon.dynamical_matrix.run(qpoint)
    else:
        phonon.dynamical_matrix.run(qpoint, q_direction=q_direction)
    eigenvalues = np.linalg.eigvalsh(
        phonon.dynamical_matrix.dynamical_matrix
    ).real
    return (np.sign(eigenvalues) * np.sqrt(np.abs(eigenvalues))
            * phonon.unit_conversion_factor)


def minimum_image(values, cell):
    fractional = np.asarray(values) @ np.linalg.inv(cell)
    fractional -= np.round(fractional)
    return fractional @ cell


def atom_mapping(source_positions, source_symbols, target_positions,
                 target_symbols, cell):
    """Map target atom order to source atom order using species and PBC."""
    mapping, used = [], set()
    for target_pos, target_symbol in zip(target_positions, target_symbols):
        candidates = []
        for index, (source_pos, source_symbol) in enumerate(
            zip(source_positions, source_symbols)
        ):
            if index in used or source_symbol != target_symbol:
                continue
            distance = np.linalg.norm(
                minimum_image(source_pos - target_pos, cell)
            )
            candidates.append((distance, index))
        if not candidates:
            raise RuntimeError(f"No source atom available for {target_symbol}")
        distance, index = min(candidates)
        if distance > 2.0e-5:
            raise RuntimeError(
                f"Atom-order mapping residual {distance:.3e} Angstrom"
            )
        used.add(index)
        mapping.append(index)
    return np.asarray(mapping, dtype=int)


def full_force_constants(model):
    values = np.zeros((model.N, model.N, 3, 3), dtype=float)
    for (i, j), tensor in model._Phi.items():
        values[i, j] = tensor
    return values


def fit_selected_iteration(material):
    spec = MATERIALS[material]
    run_dir = ROOT / "examples" / f"sotc_{material}_qe_run"
    saved = json.loads((run_dir / "sotc_results.json").read_text())
    selected = int(saved["selected_iteration"])
    n_ensemble = int(saved["n_ensemble"])
    block = n_ensemble + 1
    equilibrium_index = (selected - 1) * block
    displaced_indices = list(range(
        equilibrium_index + 1, equilibrium_index + 1 + n_ensemble
    ))
    scratch = run_dir / "qe_scratch"
    paths = sorted(scratch.glob("snap_*/espresso.pwo"))
    if displaced_indices[-1] >= len(paths):
        raise RuntimeError(
            f"Selected iteration {selected} is incomplete in {scratch}"
        )
    eq_pos, f0, cell = _parse_qe_pwo(paths[equilibrium_index])
    displacements, forces = [], []
    for index in displaced_indices:
        pos, force, state_cell = _parse_qe_pwo(paths[index])
        if not np.allclose(state_cell, cell, atol=1.0e-8, rtol=0):
            raise RuntimeError("QE snapshot cell changed within iteration")
        displacements.append(minimum_image(pos - eq_pos, cell))
        forces.append(force - f0)
    displacements = np.asarray(displacements)
    forces = np.asarray(forces)
    n_basis = len(spec["elements"])
    basis = np.arange(len(eq_pos), dtype=int) % n_basis
    masses = np.asarray(spec["masses"])[basis]
    model = IFCExtractor(
        eq_pos, cell, masses, basis_indices=basis,
        r_cutoff=float(saved["r_cutoff"]),
        ridge_alpha=float(saved["ridge_alpha"]),
        symmetrize_bonds=bool(saved["symmetrize_bonds"]),
    )
    model.fit(list(displacements), list(forces))
    return model, saved, eq_pos, cell, displacements, forces, {
        "equilibrium_snapshot_index": equilibrium_index,
        "displaced_snapshot_indices": displaced_indices,
    }


def analyze_material(material):
    spec = MATERIALS[material]
    model, saved, eq_pos, sc_cell, displacements, forces, indices = (
        fit_selected_iteration(material)
    )
    production_dfpt = (ROOT / "validation_results" / material
                       / "dfpt_nac_production_volume" / "dfpt_summary.json")
    dfpt_path = production_dfpt if production_dfpt.exists() else (
        ROOT / "validation_results" / material / "dfpt_nac"
        / "dfpt_summary.json"
    )
    dfpt = json.loads(dfpt_path.read_text())
    dfpt_cell = np.asarray(dfpt["primitive_cell_ang"], dtype=float)
    approximate_matrix = np.rint(sc_cell @ np.linalg.inv(dfpt_cell)).astype(int)
    n_primitive = abs(int(round(np.linalg.det(approximate_matrix))))
    production_a = (4.0 * abs(np.linalg.det(sc_cell)) / n_primitive) ** (1.0 / 3.0)
    prim_cell = 0.5 * production_a * np.array(
        [[0., 1., 1.], [1., 0., 1.], [1., 1., 0.]]
    )
    prim_pos = np.array([[0., 0., 0.], [production_a / 2., 0., 0.]])
    masses = np.asarray(spec["masses"], dtype=float)
    native = PhononCalculator(model, prim_pos, prim_cell, masses)

    matrix_float = sc_cell @ np.linalg.inv(prim_cell)
    matrix = np.rint(matrix_float).astype(int)
    # QE lattice vectors are printed to finite precision (residual below 3e-6).
    if not np.allclose(matrix_float, matrix, atol=1.0e-5, rtol=0):
        raise RuntimeError(f"Non-integral production supercell matrix: {matrix_float}")
    unit = PhonopyAtoms(
        symbols=spec["elements"], cell=prim_cell,
        scaled_positions=prim_pos @ np.linalg.inv(prim_cell), masses=masses,
    )
    # Phonopy uses the transpose of the SOTC row-vector convention.
    no_nac = Phonopy(unit, matrix.T)
    target_pos = np.asarray(no_nac.supercell.positions)
    target_symbols = list(no_nac.supercell.symbols)
    source_symbols = [spec["elements"][i % len(spec["elements"])]
                      for i in range(len(eq_pos))]
    mapping = atom_mapping(
        eq_pos, source_symbols, target_pos, target_symbols, sc_cell
    )
    force_constants = full_force_constants(model)[mapping][:, mapping]
    no_nac.force_constants = force_constants
    with_nac = Phonopy(unit, matrix.T)
    with_nac.force_constants = force_constants
    born = np.asarray(dfpt["born"], dtype=float)
    dielectric = np.asarray(dfpt["dielectric"], dtype=float)
    with_nac.nac_params = {
        "born": born, "dielectric": dielectric, "factor": NAC_FACTOR, "method": "wang",
    }

    band_path = (ROOT / "examples" / f"sotc_{material}_qe_run"
                 / "phonon_bandstructure.npz")
    with np.load(band_path) as band:
        qpoints = np.asarray(band["q_points"])
        distances = np.asarray(band["distances"])
        labels = np.asarray(band["labels"])
        label_positions = np.asarray(band["label_positions"])
        archived = np.asarray(band["frequencies_thz"])
    native_frequency = native.band_structure(qpoints)[1]
    phonopy_no, phonopy_yes = [], []
    for index, qpoint in enumerate(qpoints):
        if np.linalg.norm(qpoint) < 1.0e-12:
            if index + 1 < len(qpoints) and np.linalg.norm(qpoints[index + 1]) > 0:
                direction = qpoints[index + 1] - qpoint
            elif index and np.linalg.norm(qpoints[index - 1]) > 0:
                direction = qpoint - qpoints[index - 1]
            else:
                direction = np.array([1.0, 0.0, 0.0])
            phonopy_yes.append(signed_frequencies(
                with_nac, qpoint, q_direction=direction
            ))
        else:
            phonopy_yes.append(signed_frequencies(with_nac, qpoint))
        phonopy_no.append(signed_frequencies(no_nac, qpoint))
    phonopy_no = np.asarray(phonopy_no)
    phonopy_yes = np.asarray(phonopy_yes)

    gamma_direction = qpoints[1] - qpoints[0]
    gamma_to = signed_frequencies(no_nac, np.zeros(3))
    gamma_nac = signed_frequencies(
        with_nac, np.zeros(3), q_direction=gamma_direction
    )
    to_value = float(np.sort(gamma_to)[3])
    lo_value = float(np.max(gamma_nac))

    output = ROOT / "validation_results" / material / "production_sotc_nac"
    output.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(
        output / "production_sotc_nac_bands.npz",
        qpoints_fractional=qpoints, distances=distances, labels=labels,
        label_positions=label_positions,
        frequencies_sotc_native_no_nac_thz=native_frequency,
        frequencies_phonopy_no_nac_thz=phonopy_no,
        frequencies_phonopy_with_nac_thz=phonopy_yes,
        born=born, dielectric=dielectric,
        full_force_constants_ev_ang2=force_constants,
        supercell_matrix=matrix, atom_mapping_target_to_source=mapping,
    )
    return {
        "material": material, "label": spec["label"],
        "selected_iteration": int(saved["selected_iteration"]),
        "n_ensemble": int(saved["n_ensemble"]),
        "snapshot_indices": indices, "supercell_matrix": matrix,
        "supercell_atoms": len(eq_pos),
        "fit_report": model.fit_report(list(displacements), list(forces)),
        "native_vs_archived_no_nac_rmse_thz": float(np.sqrt(np.mean(
            (np.sort(native_frequency, axis=1) - np.sort(archived, axis=1)) ** 2
        ))),
        "phonopy_vs_native_no_nac_rmse_thz": float(np.sqrt(np.mean(
            (np.sort(phonopy_no, axis=1) - np.sort(native_frequency, axis=1)) ** 2
        ))),
        "gamma_to_without_nac_thz": to_value,
        "gamma_lo_with_nac_thz": lo_value,
        "gamma_lo_to_split_thz": lo_value - to_value,
        "production_lattice_parameter_ang": production_a,
        "dfpt_lattice_parameter_ang": float(dfpt["lattice_parameter_ang"]),
        "dfpt_summary": str(dfpt_path),
        "born": born, "dielectric": dielectric,
        "output_npz": str(output / "production_sotc_nac_bands.npz"),
        "plot": {"distances": distances, "labels": labels,
                 "label_positions": label_positions,
                 "without": phonopy_no, "with": phonopy_yes},
    }


def serial(value):
    if isinstance(value, np.ndarray):
        return value.tolist()
    if isinstance(value, (np.floating, np.integer)):
        return value.item()
    raise TypeError(type(value).__name__)


def main():
    results = {material: analyze_material(material) for material in MATERIALS}
    figure, axes = plt.subplots(1, 3, figsize=(7.2, 2.75), constrained_layout=True)
    for axis, material in zip(axes, MATERIALS):
        result = results[material]
        plot = result.pop("plot")
        for branch in range(plot["without"].shape[1]):
            axis.plot(plot["distances"], plot["without"][:, branch],
                      color="0.55", linestyle="--", linewidth=0.85,
                      label="production SOTC, no NAC" if branch == 0 else None)
            axis.plot(plot["distances"], plot["with"][:, branch],
                      color="#0057b8", linewidth=1.05,
                      label="production SOTC + DFPT NAC" if branch == 0 else None)
        for position in plot["label_positions"]:
            axis.axvline(position, color="0.85", linewidth=0.5)
        axis.axhline(0.0, color="0.7", linewidth=0.5)
        axis.set_xticks(plot["label_positions"])
        axis.set_xticklabels([
            "$\\Gamma$" if str(label) in ("G", "Gamma") else str(label)
            for label in plot["labels"]
        ], fontsize=7)
        axis.set_xlim(plot["distances"][0], plot["distances"][-1])
        axis.set_title(result["label"], fontsize=9)
        axis.tick_params(axis="y", labelsize=7)
        axis.text(0.03, 0.97,
                  f"TO={result['gamma_to_without_nac_thz']:.3f} THz\n"
                  f"LO={result['gamma_lo_with_nac_thz']:.3f} THz\n"
                  f"split={result['gamma_lo_to_split_thz']:.3f} THz",
                  transform=axis.transAxes, va="top", fontsize=6.5,
                  bbox={"facecolor": "white", "edgecolor": "none", "alpha": 0.8})
    axes[0].set_ylabel("Frequency (THz)", fontsize=8)
    handles, legend_labels = axes[-1].get_legend_handles_labels()
    figure.legend(handles, legend_labels, loc="upper center", ncol=2,
                  bbox_to_anchor=(0.5, 1.08), frameon=False, fontsize=7)
    destination = ROOT / "manuscript" / "figures" / "fig_polar_sotc_nac"
    figure.savefig(destination.with_suffix(".pdf"), bbox_inches="tight")
    figure.savefig(destination.with_suffix(".png"), dpi=240, bbox_inches="tight")
    plt.close(figure)
    summary = {
        "scope": ("QE-DFPT tensors applied to FC2 refits of the actual selected "
                  "production SOTC iteration for each polar benchmark"),
        "materials": results, "figure": str(destination.with_suffix(".pdf")),
    }
    summary_path = ROOT / "validation_results" / "production_sotc_nac_summary.json"
    summary_path.write_text(
        json.dumps(summary, indent=2, default=serial, allow_nan=False) + "\n"
    )
    print(summary_path)
    print(destination.with_suffix(".pdf"))


if __name__ == "__main__":
    main()
