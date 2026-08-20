#!/usr/bin/env python3
"""Build reviewer-requested diagnostics from archived QE-only calculations.

The analysis is deliberately separated into quantities that can be established
from the existing archive:

* corrected one-component 300 K displacement parameters for Al, Au, and Cu;
* reference-force and stress magnitudes for Al, NaCl, and W;
* direct SOTC-versus-conventional-finite-displacement dispersion errors; and
* distributions for FC2 fits to random quantum-canonical subsets using the
  primary SOTC cutoff, ridge coefficient, central-force projection, and ASR.

The random-subset comparison uses the independent 3x3x3 validation campaigns.
It is a cross-archive control because the primary SOTC cells, displacement
ensembles, and electronic meshes are not identical to those campaigns.  The output records this caveat
and must not be described as a matched proof of SOTC sampling superiority.
"""

from __future__ import annotations

import json
from pathlib import Path
import sys

import matplotlib.pyplot as plt
import numpy as np
import phonopy

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "codes"))

from sotc.ifc_extractor import IFCExtractor
from sotc.phonons import PhononCalculator


MATERIALS = {
    "al": {"label": "Al", "cutoff": 4.5, "sotc_displaced_calls": 10},
    "w": {"label": "W", "cutoff": 5.0, "sotc_displaced_calls": 12},
    "nacl": {"label": "NaCl", "cutoff": 4.3, "sotc_displaced_calls": 15},
}


def serial(value):
    if isinstance(value, np.ndarray):
        return value.tolist()
    if isinstance(value, (np.floating, np.integer)):
        return value.item()
    raise TypeError(type(value).__name__)


def frequency_metrics(reference, prediction):
    reference = np.sort(np.asarray(reference, dtype=float), axis=1)
    prediction = np.sort(np.asarray(prediction, dtype=float), axis=1)
    residual = prediction - reference
    return {
        "mae_thz": float(np.mean(np.abs(residual))),
        "rmse_thz": float(np.sqrt(np.mean(residual**2))),
        "max_abs_thz": float(np.max(np.abs(residual))),
        "unstable_fraction_below_minus_0p05_thz": float(
            np.mean(prediction < -0.05)
        ),
    }


def corrected_msd():
    result = {}
    experimental = {
        "al": [0.010, 0.011],
        "au": [0.013, 0.017],
        "cu": [0.008, 0.010],
    }
    for material in ("al", "au", "cu"):
        path = (
            ROOT / "examples" / f"sotc_{material}_qe_run" / "postproc"
            / "thermal_summary.json"
        )
        data = json.loads(path.read_text())
        row = next(item for item in data["temperatures"] if item["T_K"] == 300.0)
        total = float(row["MSD_ang2"])
        u_iso = total / 3.0
        low, high = experimental[material]
        result[material] = {
            "temperature_k": 300.0,
            "archived_total_msd_ang2": total,
            "u_iso_ang2": u_iso,
            "debye_waller_b_ang2": 8.0 * np.pi**2 * u_iso,
            "experimental_u_iso_range_ang2": [low, high],
            "relative_to_experimental_low_percent": 100.0 * (u_iso / low - 1.0),
            "relative_to_experimental_high_percent": 100.0 * (u_iso / high - 1.0),
        }
    return result


def reference_force_audit(material):
    directory = ROOT / "validation_results" / material / "real_ifc34"
    with np.load(directory / "thermal_force_dataset.npz") as data:
        reference = np.asarray(data["reference_forces_ev_ang"])
        induced = np.asarray(data["delta_forces_ev_ang"])
    with np.load(directory / "harmonic_reference.npz") as data:
        stress = np.asarray(data["reference_stress_ev_ang3"])
    reference_rms = float(np.sqrt(np.mean(reference**2)))
    induced_rms = float(np.sqrt(np.mean(induced**2)))
    return {
        "reference_force_rms_component_ev_ang": reference_rms,
        "reference_force_max_atom_norm_ev_ang": float(
            np.max(np.linalg.norm(reference, axis=1))
        ),
        "induced_force_rms_component_ev_ang": induced_rms,
        "induced_to_reference_rms_ratio": induced_rms / reference_rms,
        "reference_stress_ev_ang3": stress,
        "hydrostatic_reference_stress_gpa": float(
            np.trace(stress) / 3.0 * 160.21766208
        ),
        "force_model": "Delta F = F(u) - F(0)",
    }


def material_analysis(material, n_seeds=10):
    spec = MATERIALS[material]
    band_file = ROOT / "examples" / f"sotc_{material}_qe_run" / "phonon_bandstructure.npz"
    campaign = ROOT / "validation_results" / material / "real_ifc34"
    with np.load(band_file) as band:
        q_points = np.asarray(band["q_points"])
        distances = np.asarray(band["distances"])
        sotc_frequencies = np.asarray(band["frequencies_thz"])
        labels = np.asarray(band["labels"])
        label_positions = np.asarray(band["label_positions"])

    reference_phonon = phonopy.load(campaign / "phonopy_harmonic.yaml")
    reference_phonon.run_qpoints(q_points)
    reference_frequencies = np.asarray(
        reference_phonon.get_qpoints_dict()["frequencies"]
    )

    with np.load(campaign / "thermal_force_dataset.npz") as data:
        displacements = np.asarray(data["displacements_ang"])
        forces = np.asarray(data["delta_forces_ev_ang"])
        training_pool = np.asarray(data["train_indices"], dtype=int)
        test_indices = np.asarray(data["test_indices"], dtype=int)

    supercell_positions = np.asarray(reference_phonon.supercell.positions)
    supercell_cell = np.asarray(reference_phonon.supercell.cell)
    supercell_masses = np.asarray(reference_phonon.supercell.masses)
    primitive_positions = np.asarray(reference_phonon.primitive.positions)
    primitive_cell = np.asarray(reference_phonon.primitive.cell)
    primitive_masses = np.asarray(reference_phonon.primitive.masses)
    _, basis_indices = np.unique(
        np.asarray(reference_phonon.primitive.s2p_map), return_inverse=True
    )

    equivalent_random_count = spec["sotc_displaced_calls"]
    counts = sorted(set([1, 2, 4, 8, 16, 32, 64, 80, equivalent_random_count]))
    convergence = []
    for count in counts:
        seed_rows = []
        for seed in range(n_seeds):
            rng = np.random.default_rng(1000 + seed)
            chosen = rng.choice(training_pool, size=count, replace=False)
            model = IFCExtractor(
                supercell_positions,
                supercell_cell,
                supercell_masses,
                basis_indices=basis_indices,
                r_cutoff=spec["cutoff"],
                ridge_alpha=1.0e-3,
                symmetrize_bonds=True,
            )
            model.fit(
                list(displacements[chosen]), list(forces[chosen])
            )
            prediction = np.asarray([
                model.predict_forces(u) for u in displacements[test_indices]
            ])
            residual = prediction - forces[test_indices]
            test_force = {
                "rmse_ev_ang": float(np.sqrt(np.mean(residual**2))),
                "mae_ev_ang": float(np.mean(np.abs(residual))),
                "max_abs_ev_ang": float(np.max(np.abs(residual))),
            }
            calculator = PhononCalculator(
                model,
                primitive_positions,
                primitive_cell,
                primitive_masses,
            )
            fitted_frequencies = calculator.band_structure(q_points)[1]
            seed_rows.append({
                "seed": seed,
                "design": model.diagnostics(),
                "held_out_force": test_force,
                "dispersion": frequency_metrics(
                    reference_frequencies, fitted_frequencies
                ),
            })
        convergence.append({"n_canonical_snapshots": count, "seeds": seed_rows})

    return {
        "label": spec["label"],
        "sotc_displaced_calls": spec["sotc_displaced_calls"],
        "equivalent_random_snapshot_count": equivalent_random_count,
        "sotc_vs_conventional_fd": frequency_metrics(
            reference_frequencies, sotc_frequencies
        ),
        "conventional_fd": {
            "minimum_frequency_thz": float(np.min(reference_frequencies)),
            "maximum_frequency_thz": float(np.max(reference_frequencies)),
            "unstable_fraction_below_minus_0p05_thz": float(
                np.mean(reference_frequencies < -0.05)
            ),
        },
        "random_canonical_fc2_convergence": convergence,
        "plot_data": {
            "distances": distances,
            "sotc_frequencies": sotc_frequencies,
            "reference_frequencies": reference_frequencies,
            "labels": labels,
            "label_positions": label_positions,
        },
    }


def percentile_series(rows, key):
    values = np.asarray([
        [seed["dispersion"][key] for seed in item["seeds"]] for item in rows
    ])
    return (
        np.median(values, axis=1),
        np.percentile(values, 10, axis=1),
        np.percentile(values, 90, axis=1),
    )


def make_figure(material_results, output):
    # Match the native width of a two-column REVTeX figure so labels are not
    # reduced by roughly one third when LaTeX scales the graphic to text width.
    fig, axes = plt.subplots(2, 3, figsize=(7.2, 5.0), constrained_layout=True)
    lower_ylim = {"al": (0.0, 0.6), "w": (0.0, 0.4), "nacl": (0.0, 1.6)}
    for column, material in enumerate(("al", "w", "nacl")):
        result = material_results[material]
        plot = result.pop("plot_data")
        ax = axes[0, column]
        for branch in range(plot["reference_frequencies"].shape[1]):
            ax.plot(
                plot["distances"], plot["reference_frequencies"][:, branch],
                color="0.35", lw=1.15,
                label="QE finite displacement" if branch == 0 else None,
            )
            ax.plot(
                plot["distances"], plot["sotc_frequencies"][:, branch],
                color="#1764ab", lw=1.0, ls="--",
                label="SOTC FC2" if branch == 0 else None,
            )
        for position in plot["label_positions"]:
            ax.axvline(position, color="0.82", lw=0.55)
        ax.axhline(0.0, color="0.65", lw=0.55)
        ax.set_xticks(plot["label_positions"])
        ax.set_xticklabels(["$\\Gamma$" if x == "G" else x for x in plot["labels"]])
        ax.set_xlim(plot["distances"][0], plot["distances"][-1])
        ax.set_title(result["label"], fontsize=10, pad=4)
        ax.tick_params(axis="both", labelsize=8)
        metrics = result["sotc_vs_conventional_fd"]
        ax.text(
            0.03, 0.97,
            "MAE={:.3f}\nRMSE={:.3f}\nmax={:.3f} THz\nunstable={:.1f}%".format(
                metrics["mae_thz"], metrics["rmse_thz"],
                metrics["max_abs_thz"],
                100.0 * metrics["unstable_fraction_below_minus_0p05_thz"],
            ),
            transform=ax.transAxes, va="top", ha="left", fontsize=6.5,
            bbox={"facecolor": "white", "alpha": 0.82, "edgecolor": "none", "pad": 1.5},
        )
        if column == 0:
            ax.set_ylabel("Frequency (THz)", fontsize=9)

        ax = axes[1, column]
        rows = result["random_canonical_fc2_convergence"]
        counts = np.asarray([item["n_canonical_snapshots"] for item in rows])
        median, low, high = percentile_series(rows, "rmse_thz")
        ax.fill_between(counts, low, high, color="#d95f02", alpha=0.20,
                        label="canonical FC2, 10--90%")
        ax.plot(counts, median, "o-", color="#d95f02", ms=3.5,
                label="canonical FC2 median")
        sotc = result["sotc_vs_conventional_fd"]["rmse_thz"]
        equivalent = result["equivalent_random_snapshot_count"]
        ax.plot(equivalent, sotc, marker="D", color="#1764ab", ms=6,
                linestyle="none", label="SOTC archive")
        ax.set_xscale("log", base=2)
        ax.set_xlim(0.8, 100)
        ax.set_ylim(*lower_ylim[material])
        ax.set_xlabel("Displaced QE configurations", fontsize=9)
        ax.tick_params(axis="both", labelsize=8)
        if column == 0:
            ax.set_ylabel("Dispersion RMSE (THz)", fontsize=9)
        ax.grid(alpha=0.2)

    top_handles, top_labels = axes[0, 2].get_legend_handles_labels()
    bottom_handles, bottom_labels = axes[1, 2].get_legend_handles_labels()
    fig.legend(top_handles, top_labels, loc="upper center",
               bbox_to_anchor=(0.5, 1.055), ncol=2, frameon=False, fontsize=8)
    fig.legend(bottom_handles, bottom_labels, loc="lower center",
               bbox_to_anchor=(0.5, -0.055), ncol=3, frameon=False, fontsize=7)
    fig.savefig(output.with_suffix(".pdf"), bbox_inches="tight")
    fig.savefig(output.with_suffix(".png"), dpi=220, bbox_inches="tight")
    plt.close(fig)


def au_instability_audit():
    band_path = ROOT / "examples" / "sotc_au_qe_run" / "phonon_bandstructure.npz"
    result_path = ROOT / "examples" / "sotc_au_qe_run" / "sotc_results.json"
    with np.load(band_path) as band:
        frequencies = np.asarray(band["frequencies_thz"])
        minimum_index = np.unravel_index(np.argmin(frequencies), frequencies.shape)
        unstable_q_indices = np.where(np.any(frequencies < -0.05, axis=1))[0]
        audit = {
            "minimum_frequency_thz": float(frequencies[minimum_index]),
            "minimum_q_index": int(minimum_index[0]),
            "minimum_sorted_branch_index": int(minimum_index[1]),
            "minimum_q_point_reduced": np.asarray(
                band["q_points"][minimum_index[0]]
            ),
            "minimum_path_distance": float(band["distances"][minimum_index[0]]),
            "unstable_mode_count_below_minus_0p05_thz": int(
                np.sum(frequencies < -0.05)
            ),
            "total_path_mode_count": int(frequencies.size),
            "unstable_fraction_below_minus_0p05_thz": float(
                np.mean(frequencies < -0.05)
            ),
            "unstable_q_index_first": int(unstable_q_indices[0]),
            "unstable_q_index_last": int(unstable_q_indices[-1]),
        }
    history = json.loads(result_path.read_text())["iteration_log"]
    audit["iteration_mesh_unstable_fractions"] = [
        {"iteration": int(row["iteration"]), "unstable_fraction": row["unstable_fraction"]}
        for row in history
    ]
    return audit


def main():
    material_results = {
        material: material_analysis(material) for material in MATERIALS
    }
    output_base = ROOT / "manuscript" / "figures" / "fig_dispersion_controls"
    make_figure(material_results, output_base)
    report = {
        "scope": "postprocessing of archived Quantum ESPRESSO calculations only",
        "msd_convention": (
            "U_iso = <|u|^2>/3; B = 8*pi^2*U_iso. The archived postprocessing "
            "field MSD_ang2 is the total three-dimensional displacement."
        ),
        "corrected_300k_displacements": corrected_msd(),
        "reference_force_audit": {
            material: reference_force_audit(material) for material in MATERIALS
        },
        "dispersion_controls": material_results,
        "au_instability_audit": au_instability_audit(),
        "comparison_caveat": (
            "The canonical-subset controls refit independent 3x3x3 QE campaigns with "
            "the primary cutoff, ridge coefficient, central-force projection, and ASR, "
            "whereas the primary SOTC runs use different cells, displacement ensembles, "
            "and k meshes. "
            "The comparison tests the same dispersion observable against each "
            "campaign's conventional finite-displacement reference, but is not "
            "a cell-by-cell matched proof of sampling superiority."
        ),
        "random_seed_count": 10,
    }
    destination = ROOT / "validation_results" / "reviewer_evidence_summary.json"
    destination.write_text(json.dumps(report, indent=2, default=serial) + "\n")
    print(destination)
    print(output_base.with_suffix(".pdf"))


if __name__ == "__main__":
    main()
