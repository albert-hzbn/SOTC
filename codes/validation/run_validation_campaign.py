#!/usr/bin/env python3
"""Run matched SOTC, finite-displacement, and canonical-random DFT validation."""

from __future__ import annotations

import argparse
import json
import os
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "codes"))

from sotc import SOTCRunner
from sotc.cell_design import DisplacementOptimizer
from sotc.correlators import PhononCorrelator
from sotc.ifc_extractor import IFCExtractor
from sotc.phonons import PhononCalculator
from sotc.qe_io import QEForceCalculator
from sotc.validation import (
    build_finite_displacement_reference,
    canonical_random_displacements,
    common_q_path,
    compare_anharmonic_orders,
    dispersion_metrics,
    evaluate_displacement_forces,
    fit_snapshot_prefix,
    phonopy_frequencies_on_path,
    reorder_displacements_to_target,
)


MATERIALS = {
    "al": {
        "elements": ["Al"], "masses": [26.9815385], "a": 4.05,
        "bravais": "fcc", "T": 300.0, "T_D": 390.0, "n_atoms": 32,
        "r_cutoff": 4.5, "r_max": 8.0, "ecutwfc": 40.0,
        "ecutrho": 320.0, "kpts": (2, 2, 2), "smearing": "marzari-vanderbilt",
        "degauss": 0.02, "pseudos": {"Al": "Al.pbe-n-kjpaw_psl.1.0.0.UPF"},
    },
    "w": {
        "elements": ["W"], "masses": [183.84], "a": 3.18,
        "bravais": "bcc", "T": 1500.0, "T_D": 300.0, "n_atoms": 32,
        "r_cutoff": 5.0, "r_max": 9.0, "ecutwfc": 30.0,
        "ecutrho": 240.0, "kpts": (2, 2, 2), "smearing": "marzari-vanderbilt",
        "degauss": 0.02, "pseudos": {"W": "W_pbe_v1.2.uspp.F.UPF"},
        "extra": {"electrons": {"diagonalization": "david", "diago_david_ndim": 4}},
    },
    "nacl": {
        "elements": ["Na", "Cl"], "masses": [22.9898, 35.453], "a": 5.73,
        "bravais": "fcc", "T": 700.0, "T_D": 310.0, "n_atoms": 54,
        "r_cutoff": 4.3, "r_max": 9.0, "ecutwfc": 60.0,
        "ecutrho": 480.0, "kpts": (1, 1, 1), "smearing": None,
        "degauss": 0.0,
        "pseudos": {"Na": "na_pbe_v1.5.uspp.F.UPF", "Cl": "cl_pbe_v1.4.uspp.F.UPF"},
    },
}


def structure(spec):
    a = spec["a"]
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


def calculator(spec, workdir, prefix):
    return QEForceCalculator(
        species=spec["elements"],
        pseudopotentials=spec["pseudos"],
        pseudo_dir=Path(os.environ.get("SSSP_DIR", ROOT / "pseudopotentials")),
        ecutwfc=spec["ecutwfc"], ecutrho=spec["ecutrho"],
        kpts=spec["kpts"], smearing=spec["smearing"],
        degauss=spec["degauss"], conv_thr=1e-8,
        pw_cmd=os.environ["QE_PW_CMD"], workdir=workdir,
        prefix=prefix, n_parallel=1,
        extra_input_data=spec.get("extra"),
    )


def json_default(value):
    if isinstance(value, np.ndarray): return value.tolist()
    if isinstance(value, (np.floating, np.integer)): return value.item()
    raise TypeError(type(value).__name__)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("material", choices=sorted(MATERIALS))
    parser.add_argument("--random-seeds", type=int, default=5)
    parser.add_argument("--snapshots-per-seed", type=int, default=5)
    parser.add_argument("--iterations", type=int, default=6)
    parser.add_argument(
        "--anharmonic-modes", type=int, default=3,
        help="PCA modes in the matched order-2/3/4 held-out force comparison",
    )
    args = parser.parse_args()

    spec = MATERIALS[args.material]
    cell, prim_pos = structure(spec)
    out = ROOT / "validation_results" / args.material
    out.mkdir(parents=True, exist_ok=True)
    sotc_dir = out / "sotc"
    qe_sotc = calculator(spec, sotc_dir / "qe_scratch", f"{args.material}_sotc")
    runner = SOTCRunner(
        element=spec["elements"][0], mass_amu=spec["masses"][0],
        elements=spec["elements"], masses_amu=spec["masses"],
        prim_cell=cell, prim_positions=prim_pos,
        T=spec["T"], T_D=spec["T_D"], n_atoms_sc=spec["n_atoms"],
        force_calculator=qe_sotc, r_cutoff=spec["r_cutoff"],
        r_max_corr=spec["r_max"], n_ensemble=args.snapshots_per_seed,
        work_dir=sotc_dir, ridge_alpha=1e-3, symmetrize_bonds=True,
        verbosity=1,
    )
    result = runner.run(
        n_sc_iterations=args.iterations, min_iterations=4,
        epsilon_conv=0.003, mixing=0.3,
        T_values=np.array([spec["T"]]), q_mesh_cv=(8, 8, 8),
    )
    H = np.asarray(result["supercell_matrix"], dtype=int)
    eq_target = np.asarray(result["equilibrium_positions_ang"])
    basis = np.asarray(result["basis_indices"], dtype=int)
    masses_sc = np.asarray(spec["masses"])[basis]
    target_symbols = [spec["elements"][i] for i in basis]

    qe_ref = calculator(spec, out / "reference" / "qe_scratch", f"{args.material}_ref")
    reference = build_finite_displacement_reference(
        spec["elements"], cell, prim_pos, H, qe_ref, out / "reference",
        displacement_distance_ang=0.01, plus_minus=True,
    )

    selected_calc = result["phonon_calculator"]
    q_path, labels, label_positions = common_q_path(selected_calc, 40)
    reference_freq = phonopy_frequencies_on_path(reference, q_path)
    selected_freq = selected_calc.band_structure(q_path)[1]
    summary = {
        "material": args.material,
        "temperature_K": spec["T"],
        "common_path_labels": labels,
        "supercell_matrix": H,
        "selected_iteration": result["selected_iteration"],
        "selected_sotc": dispersion_metrics(
            selected_freq, reference_freq, q_path, selected_calc._rec_lat
        ),
        "sotc_iterations": [], "random_controls": [], "prior_ablations": [],
    }

    # Direct convergence of each self-consistency iteration against one reference.
    for iteration in range(1, result["n_iterations"] + 1):
        archive = sotc_dir / "iterations" / f"iter_{iteration:02d}"
        ifc = IFCExtractor.load(str(archive / "ifc_extractor.npz"))
        calc = PhononCalculator(ifc, prim_pos, cell, np.asarray(spec["masses"]))
        metric = dispersion_metrics(
            calc.band_structure(q_path)[1], reference_freq, q_path, calc._rec_lat
        )
        metric.update(ifc.diagnostics())
        metric["iteration"] = iteration
        summary["sotc_iterations"].append(metric)

    # Reference-target objective for SOTC and every canonical random snapshot.
    corr = PhononCorrelator(reference, q_mesh=(8, 8, 8))
    objective = DisplacementOptimizer(
        corr, spec["T"], xi_T=max(spec["r_cutoff"] / 2.0, 1.0),
        r_max=spec["r_max"], basis_indices=basis, masses_amu=masses_sc,
    )
    shells, targets = objective.find_shells(eq_target, np.asarray(result["supercell_cell_ang"]))
    selected_archive = sotc_dir / "iterations" / f"iter_{result['selected_iteration']:02d}"
    selected_data = np.load(selected_archive / "force_displacement_data.npz")
    summary["selected_sotc"]["reference_objective_ang4"] = [
        float(objective.quality(u, eq_target, result["supercell_cell_ang"], shells, targets))
        for u in selected_data["displacements"]
    ]

    random_sets = []
    for seed in range(args.random_seeds):
        source_u, source_eq, source_cell, source_symbols = canonical_random_displacements(
            reference, spec["T"], args.snapshots_per_seed, 1000 + seed
        )
        random_sets.append(reorder_displacements_to_target(
            source_eq, source_symbols, source_u,
            eq_target, target_symbols, np.asarray(result["supercell_cell_ang"]),
        ))
    random_all = np.concatenate(random_sets, axis=0)
    qe_random = calculator(spec, out / "random" / "qe_scratch", f"{args.material}_random")
    random_forces, random_reference = evaluate_displacement_forces(
        eq_target, np.asarray(result["supercell_cell_ang"]), random_all, qe_random
    )
    np.savez_compressed(
        out / "random" / "canonical_random_data.npz",
        displacements=random_all, delta_forces=random_forces,
        seeds=np.repeat(np.arange(args.random_seeds), args.snapshots_per_seed),
    )

    for seed in range(args.random_seeds):
        lo = seed * args.snapshots_per_seed
        hi = lo + args.snapshots_per_seed
        u_seed = random_all[lo:hi]
        f_seed = random_forces[lo:hi]
        for n in range(1, args.snapshots_per_seed + 1):
            ifc = fit_snapshot_prefix(
                eq_target, result["supercell_cell_ang"], masses_sc, basis,
                u_seed, f_seed, n, spec["r_cutoff"], 1e-3, True,
            )
            calc = PhononCalculator(ifc, prim_pos, cell, np.asarray(spec["masses"]))
            metric = dispersion_metrics(
                calc.band_structure(q_path)[1], reference_freq, q_path, calc._rec_lat
            )
            metric.update(ifc.fit_report(list(u_seed[:n]), list(f_seed[:n])))
            metric.update({"seed": seed, "n_snapshots": n})
            metric["reference_objective_ang4"] = [
                float(objective.quality(u, eq_target, result["supercell_cell_ang"], shells, targets))
                for u in u_seed[:n]
            ]
            summary["random_controls"].append(metric)

    # No-extra-DFT ablations isolate ridge, central-force projection, and cutoff.
    u_sel = selected_data["displacements"]
    f_sel = selected_data["delta_forces"]
    variants = []
    for ridge in (0.0, 1e-5, 1e-3, 1e-1):
        variants.append((spec["r_cutoff"], ridge, True))
    variants.extend([
        (spec["r_cutoff"], 1e-3, False),
        (0.8 * spec["r_cutoff"], 1e-3, True),
        (1.2 * spec["r_cutoff"], 1e-3, True),
    ])
    for cutoff, ridge, central in variants:
        ifc = fit_snapshot_prefix(
            eq_target, result["supercell_cell_ang"], masses_sc, basis,
            u_sel, f_sel, len(u_sel), cutoff, ridge, central,
        )
        calc = PhononCalculator(ifc, prim_pos, cell, np.asarray(spec["masses"]))
        metric = dispersion_metrics(
            calc.band_structure(q_path)[1], reference_freq, q_path, calc._rec_lat
        )
        metric.update(ifc.fit_report(list(u_sel), list(f_sel)))
        metric.update({"r_cutoff_ang": cutoff, "ridge_alpha": ridge,
                       "central_force_projection": central})
        summary["prior_ablations"].append(metric)

    # This uses no additional DFT calls: compare nested energy-derived
    # order-2/3/4 models on the union of selected SOTC and random snapshots.
    # It is a reduced-mode force test, not an fc3/fc4 transport calculation.
    summary["anharmonic_order_comparison"] = compare_anharmonic_orders(
        np.concatenate((u_sel, random_all), axis=0),
        np.concatenate((f_sel, random_forces), axis=0),
        out / "anharmonic_order_comparison",
        max_modes=args.anharmonic_modes,
        test_fraction=0.2,
        random_seed=2718,
    )

    summary["random_reference_state"] = random_reference
    summary["force_call_accounting"] = {
        "sotc": result["force_call_accounting"] if "force_call_accounting" in result else {
            "reference_calls": 1,
            "displaced_calls": result["n_iterations"] * args.snapshots_per_seed,
        },
        "finite_displacement_reference": json.loads(
            (out / "reference" / "reference_metadata.json").read_text()
        ),
        "random": {"reference_calls": 1, "displaced_calls": len(random_all)},
    }
    np.savez_compressed(
        out / "common_path_frequencies.npz",
        q_path=q_path, reference_frequencies_thz=reference_freq,
        selected_sotc_frequencies_thz=selected_freq,
        labels=np.asarray(labels), label_positions=label_positions,
    )
    (out / "validation_summary.json").write_text(
        json.dumps(summary, indent=2, default=json_default, allow_nan=False)
    )
    print(f"Validation complete: {out / 'validation_summary.json'}")


if __name__ == "__main__":
    main()
