#!/usr/bin/env python3
"""Aggregate and audit the real-material FC2/FC3/FC4 validation campaigns."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import h5py
import numpy as np


ROOT = Path(__file__).resolve().parents[2]
MATERIALS = ("al", "nacl", "w")
FINAL_ARTIFACTS = (
    "fc2_compact.npy",
    "fc3_compact.npy",
    "fc4_compact.npy",
    "fc234_compact.hdf5",
    "fc3.hdf5",
    "FORCE_CONSTANTS_3RD",
    "FORCE_CONSTANTS_4TH",
)


def improvement(before: float, after: float) -> float:
    return 100.0 * (before - after) / before


def force_metrics(reference: np.ndarray, prediction: np.ndarray) -> dict:
    residual = np.asarray(prediction) - np.asarray(reference)
    centered = np.asarray(reference) - np.mean(reference)
    return {
        "rmse_force_component_ev_ang":
            float(np.sqrt(np.mean(residual**2))),
        "mae_force_component_ev_ang": float(np.mean(np.abs(residual))),
        "max_abs_force_component_ev_ang":
            float(np.max(np.abs(residual))),
        "r2_force_components":
            float(1.0 - np.sum(residual**2) / np.sum(centered**2)),
    }


def audit_material(root: Path, material: str) -> dict:
    campaign = root / material / "real_ifc34"
    summary = json.loads((campaign / "ifc34_summary.json").read_text())
    with np.load(campaign / "thermal_force_dataset.npz") as dataset:
        displacements = np.asarray(dataset["displacements_ang"])
        forces = np.asarray(dataset["delta_forces_ev_ang"])
        pair_error = float(np.max(np.abs(
            displacements[0::2] + displacements[1::2]
        )))
        finite_dataset = bool(
            np.isfinite(displacements).all() and np.isfinite(forces).all()
        )
        displacement_rms = float(np.sqrt(np.mean(displacements**2)))
        force_rms = float(np.sqrt(np.mean(forces**2)))

    with np.load(campaign / "harmonic_reference.npz") as harmonic:
        finite_displacement_fc2 = np.asarray(
            harmonic["force_constants_ev_ang2"]
        )
        harmonic_displacements = np.asarray(harmonic["displacements_ang"])
        harmonic_forces = np.asarray(harmonic["delta_forces_ev_ang"])
    test_start = summary["n_training_snapshots"]
    fd_prediction = -np.einsum(
        "ijab,sjb->sia", finite_displacement_fc2,
        displacements[test_start:], optimize=True,
    )
    fd_metrics = force_metrics(forces[test_start:], fd_prediction)
    fd_self_prediction = -np.einsum(
        "ijab,sjb->sia", finite_displacement_fc2,
        harmonic_displacements, optimize=True,
    )
    fd_contraction_rmse = force_metrics(
        harmonic_forces, fd_self_prediction
    )["rmse_force_component_ev_ang"]

    final_dir = campaign / "final_force_constants"
    artifact_presence = {
        name: (final_dir / name).is_file() for name in FINAL_ARTIFACTS
    }
    with h5py.File(final_dir / "fc234_compact.hdf5", "r") as handle:
        hdf5_shapes = {
            order: list(handle[order].shape) for order in ("fc2", "fc3", "fc4")
        }
        finite_tensors = all(
            np.isfinite(handle[order][...]).all()
            for order in ("fc2", "fc3", "fc4")
        )

    comparisons = summary["nested_held_out_order_comparison"]
    rmse2 = comparisons["2"]["held_out_test"]["rmse_force_component_ev_ang"]
    rmse23 = comparisons["23"]["held_out_test"]["rmse_force_component_ev_ang"]
    rmse234 = comparisons["234"]["held_out_test"]["rmse_force_component_ev_ang"]
    design = summary["ifc_fit"]["final_all_snapshot_design"]
    constraints = summary["ifc_fit"]["final_constraints"]
    max_asr = max(
        value for key, value in constraints.items() if "asr_residual" in key
    )
    snapshot_rmse = {
        str(item["n_training_snapshots"]):
            item["held_out_test"]["rmse_force_component_ev_ang"]
        for item in summary["snapshot_count_convergence"]
    }
    cutoff_rmse = [
        {
            "cutoffs_ang": item["cutoffs_ang"],
            "held_out_rmse_force_component_ev_ang":
                item["held_out_test"]["rmse_force_component_ev_ang"],
        }
        for item in summary["cutoff_sensitivity"]
    ]
    checks = {
        "summary_complete": summary["status"] == "complete",
        "all_final_artifacts_present": all(artifact_presence.values()),
        "finite_dataset": finite_dataset,
        "finite_final_tensors": bool(finite_tensors),
        "exact_inversion_pairing": pair_error < 1.0e-12,
        "finite_displacement_contraction_rmse_below_1e-3":
            fd_contraction_rmse < 1.0e-3,
        "full_normalized_design_rank":
            design["normalized_design_rank"] == design["n_parameters"],
        "acoustic_sum_rule_residual_below_1e-10": max_asr < 1.0e-10,
    }
    return {
        "material_key": material,
        "material": summary["material"],
        "temperature_k": summary["temperature_k"],
        "n_supercell_atoms": summary["n_supercell_atoms"],
        "n_snapshots": summary["n_snapshots"],
        "n_training_snapshots": summary["n_training_snapshots"],
        "n_held_out_snapshots": summary["n_strictly_held_out_snapshots"],
        "cutoffs_ang": summary["ifc_fit"]["cutoffs_ang"],
        "basis_parameters_by_order":
            design["basis_parameters_by_order"],
        "n_parameters": design["n_parameters"],
        "normalized_design_rank": design["normalized_design_rank"],
        "normalized_design_condition_number":
            design["normalized_design_condition_number"],
        "held_out_force_metrics": {
            "finite_displacement_fc2": fd_metrics,
            "temperature_fitted_fc2": comparisons["2"]["held_out_test"],
            "temperature_fitted_fc2_fc3": comparisons["23"]["held_out_test"],
            "temperature_fitted_fc2_fc3_fc4":
                comparisons["234"]["held_out_test"],
        },
        "finite_displacement_contraction_rmse_ev_ang":
            fd_contraction_rmse,
        "held_out_rmse_force_component_ev_ang": {
            "finite_displacement_fc2":
                fd_metrics["rmse_force_component_ev_ang"],
            "fc2": rmse2,
            "fc2_fc3": rmse23,
            "fc2_fc3_fc4": rmse234,
        },
        "held_out_r2_fc2_fc3_fc4":
            comparisons["234"]["held_out_test"]["r2_force_components"],
        "thermal_fc2_vs_finite_displacement_rmse_reduction_percent":
            improvement(fd_metrics["rmse_force_component_ev_ang"], rmse2),
        "fc3_rmse_reduction_percent": improvement(rmse2, rmse23),
        "fc4_incremental_rmse_reduction_percent":
            improvement(rmse23, rmse234),
        "overall_rmse_reduction_percent": improvement(rmse2, rmse234),
        "overall_vs_finite_displacement_rmse_reduction_percent":
            improvement(fd_metrics["rmse_force_component_ev_ang"], rmse234),
        "snapshot_count_convergence": snapshot_rmse,
        "cutoff_sensitivity": cutoff_rmse,
        "electronic_convergence_force_rmse_ev_ang": {
            name: item["force_difference"]["rmse_force_component_ev_ang"]
            for name, item in summary["electronic_convergence"].items()
        },
        "max_compact_asr_residual": max_asr,
        "displacement_rms_ang": displacement_rms,
        "force_rms_ev_ang": force_rms,
        "max_inversion_pair_error_ang": pair_error,
        "hdf5_tensor_shapes": hdf5_shapes,
        "artifact_presence": artifact_presence,
        "checks": checks,
    }


def markdown_report(rows: list[dict]) -> str:
    lines = [
        "# Real-material FC3/FC4 campaign audit",
        "",
        "Generated from the machine-readable campaign archives by "
        "codes/validation/summarize_real_anharmonic_ifcs.py.",
        "",
        "| Material | T (K) | atoms | parameters/rank | condition | "
        "FD FC2 | T-fit FC2 | T-fit FC2+3 | T-fit FC2+3+4 | R2 |",
        "|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|",
    ]
    for row in rows:
        rmse = row["held_out_rmse_force_component_ev_ang"]
        lines.append(
            f"| {row['material']} | {row['temperature_k']:.0f} | "
            f"{row['n_supercell_atoms']} | {row['n_parameters']}/"
            f"{row['normalized_design_rank']} | "
            f"{row['normalized_design_condition_number']:.1f} | "
            f"{rmse['finite_displacement_fc2']:.6f} | "
            f"{rmse['fc2']:.6f} | {rmse['fc2_fc3']:.6f} | "
            f"{rmse['fc2_fc3_fc4']:.6f} | "
            f"{row['held_out_r2_fc2_fc3_fc4']:.6f} |"
        )
    lines.extend([
        "",
        "RMSE values are per Cartesian force component in eV/angstrom on 20 "
        "strictly held-out snapshots. FD FC2 is the conventional central "
        "finite-displacement phonopy model generated with matched QE settings; "
        "the other columns are temperature-fitted symfc models. Complete "
        "inversion pairs remain wholly inside either the 80-snapshot training "
        "set or the test set.",
        "",
        "All campaigns passed the archive, finiteness, exact-pairing, "
        "symmetry-reduced full-rank, and acoustic-sum-rule checks. The "
        "higher-order workflow uses 100 canonical snapshots and is independent "
        "of the original few-configuration SOTC sampling claim.",
        "",
        "Interpretive cautions:",
        "",
        "- Al required the 5.0-angstrom FC2/FC3 cutoff selected by its held-out "
        "cutoff test; its denser-k-mesh force shift remains larger than the "
        "final model RMSE.",
        "- NaCl has the largest canonical displacements and the weakest final "
        "held-out score; the model is useful evidence for higher-order force "
        "content but not a claim of converged transport properties.",
        "- Transport-ready file formatting and tensor conventions are tested, "
        "but thermal conductivity, linewidths, and free energies were not "
        "computed in this campaign.",
        "",
    ])
    return "\n".join(lines)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--root", type=Path, default=ROOT / "validation_results"
    )
    args = parser.parse_args()
    rows = [audit_material(args.root, material) for material in MATERIALS]
    failed = [
        f"{row['material_key']}:{name}"
        for row in rows for name, passed in row["checks"].items() if not passed
    ]
    aggregate = {
        "status": "complete" if not failed else "failed",
        "protocol": {
            "sampling": "paired quantum-canonical",
            "train_test_split": "80/20 snapshots; inversion-pair independent",
            "fit_backend": "symfc joint FC2/FC3/FC4 dense solver",
        },
        "materials": rows,
        "failed_checks": failed,
    }
    json_path = args.root / "ifc34_campaign_summary.json"
    md_path = args.root / "ifc34_campaign_report.md"
    json_path.write_text(json.dumps(aggregate, indent=2, allow_nan=False) + "\n")
    md_path.write_text(markdown_report(rows))
    print(json_path)
    print(md_path)
    if failed:
        raise SystemExit("Campaign audit failed: " + ", ".join(failed))


if __name__ == "__main__":
    main()
