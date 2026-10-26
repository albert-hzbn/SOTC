#!/usr/bin/env python3
"""Plot the NaCl phonon dispersion with and without the QE DFPT NAC."""

from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import phonopy

try:
    from phonopy.physical_units import get_physical_units
    _phonopy_units = get_physical_units()
    NAC_FACTOR = _phonopy_units.Hartree * _phonopy_units.Bohr
except ModuleNotFoundError:  # phonopy < 2.26
    from phonopy import units as _phonopy_units
    NAC_FACTOR = _phonopy_units.Hartree * _phonopy_units.Bohr


ROOT = Path(__file__).resolve().parents[2]
DEFAULT_PHONOPY = (
    ROOT / "validation_results" / "nacl" / "real_ifc34"
    / "phonopy_harmonic.yaml"
)
DEFAULT_DFPT = (
    ROOT / "validation_results" / "nacl" / "dfpt_nac"
    / "dfpt_summary.json"
)
DEFAULT_DATA_DIR = ROOT / "validation_results" / "nacl" / "dfpt_nac"
DEFAULT_FIGURE = ROOT / "manuscript" / "jap" / "figures" / "fig_nacl_loto"
DEFAULT_NEUTRON = ROOT / "data" / "nacl_neutron_80k.csv"

# Zone-center optical limits reported in NBS Special Publication 301
# (Carter and Rush, eds., 1969), converted using the exact speed of light.
CM1_TO_THZ = 0.0299792458
LITERATURE_TO_CM1 = 164.0
LITERATURE_LO_CM1 = 268.0


POINTS = {
    "Gamma": np.array([0.0, 0.0, 0.0]),
    "X": np.array([0.5, 0.5, 0.0]),
    "W": np.array([0.5, 0.75, 0.25]),
    "K": np.array([0.375, 0.75, 0.375]),
    "L": np.array([0.5, 0.5, 0.5]),
}
PATH = ("Gamma", "X", "W", "K", "Gamma", "L")


def frequencies(phonon, qpoint, q_direction=None):
    if q_direction is None:
        phonon.dynamical_matrix.run(qpoint)
    else:
        phonon.dynamical_matrix.run(qpoint, q_direction=q_direction)
    eigenvalues = np.linalg.eigvalsh(
        phonon.dynamical_matrix.dynamical_matrix
    ).real
    return (
        np.sign(eigenvalues)
        * np.sqrt(np.abs(eigenvalues))
        * phonon.unit_conversion_factor
    )


def make_path(cell, points_per_segment):
    reciprocal = np.linalg.inv(np.asarray(cell, dtype=float)).T
    qpoints = []
    distances = []
    directions = []
    ticks = [0.0]
    cursor = 0.0
    for segment, (start_label, stop_label) in enumerate(
        zip(PATH[:-1], PATH[1:])
    ):
        start = POINTS[start_label]
        stop = POINTS[stop_label]
        direction = stop - start
        segment_length = float(np.linalg.norm(direction @ reciprocal))
        values = np.linspace(0.0, 1.0, points_per_segment)
        if segment:
            values = values[1:]
        for value in values:
            qpoints.append(start + value * direction)
            distances.append(cursor + value * segment_length)
            directions.append(direction)
        cursor += segment_length
        ticks.append(cursor)
    return (
        np.asarray(qpoints),
        np.asarray(distances),
        np.asarray(directions),
        np.asarray(ticks),
    )



def load_neutron_table(path, ticks):
    """Load tabulated 80 K neutron points and map them to this plot path."""
    rows = []
    with path.open() as handle:
        reader = csv.DictReader(line for line in handle if not line.startswith("#"))
        for row in reader:
            direction = row["direction"]
            fraction = float(row["lambda"])
            if direction == "100":
                path_distance = ticks[0] + fraction * (ticks[1] - ticks[0])
            elif direction == "111":
                path_distance = ticks[-2] + fraction * (ticks[-1] - ticks[-2])
            else:
                raise ValueError(f"Unsupported neutron direction: {direction}")
            rows.append((
                direction,
                fraction,
                row["branch"],
                float(row["omega_1e13_rad_s"]),
                path_distance,
            ))
    omega = np.asarray([row[3] for row in rows])
    return rows, omega * 10.0 / (2.0 * np.pi)
def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--phonopy-yaml", type=Path, default=DEFAULT_PHONOPY)
    parser.add_argument("--dfpt-summary", type=Path, default=DEFAULT_DFPT)
    parser.add_argument("--data-dir", type=Path, default=DEFAULT_DATA_DIR)
    parser.add_argument("--figure", type=Path, default=DEFAULT_FIGURE)
    parser.add_argument("--neutron-table", type=Path, default=DEFAULT_NEUTRON)
    parser.add_argument("--points-per-segment", type=int, default=81)
    args = parser.parse_args()

    no_nac = phonopy.load(args.phonopy_yaml)
    with_nac = phonopy.load(args.phonopy_yaml)
    dfpt = json.loads(args.dfpt_summary.read_text())
    born = np.asarray(dfpt["born"], dtype=float)
    dielectric = np.asarray(dfpt["dielectric"], dtype=float)
    with_nac.nac_params = {
        "born": born,
        "dielectric": dielectric,
        "factor": NAC_FACTOR,
        "method": "wang",
    }

    qpoints, distance, directions, ticks = make_path(
        no_nac.primitive.cell, args.points_per_segment
    )
    frequencies_no_nac = []
    frequencies_with_nac = []
    for qpoint, direction in zip(qpoints, directions):
        frequencies_no_nac.append(frequencies(no_nac, qpoint))
        if np.linalg.norm(qpoint) < 1.0e-12:
            frequencies_with_nac.append(
                frequencies(with_nac, qpoint, q_direction=direction)
            )
        else:
            frequencies_with_nac.append(frequencies(with_nac, qpoint))
    frequencies_no_nac = np.asarray(frequencies_no_nac)
    frequencies_with_nac = np.asarray(frequencies_with_nac)

    gamma_to = frequencies(no_nac, np.zeros(3))[-1]
    gamma_nac = frequencies(
        with_nac, np.zeros(3), q_direction=POINTS["X"]
    )
    gamma_lo = gamma_nac[-1]
    split = gamma_lo - gamma_to
    literature_to = LITERATURE_TO_CM1 * CM1_TO_THZ
    literature_lo = LITERATURE_LO_CM1 * CM1_TO_THZ
    literature_split = literature_lo - literature_to
    neutron_rows, neutron_frequency_thz = load_neutron_table(
        args.neutron_table, ticks
    )
    neutron_distance = np.asarray([row[4] for row in neutron_rows])
    neutron_direction = np.asarray([row[0] for row in neutron_rows])
    neutron_fraction = np.asarray([row[1] for row in neutron_rows])
    neutron_branch = np.asarray([row[2] for row in neutron_rows])
    neutron_omega = np.asarray([row[3] for row in neutron_rows])

    args.data_dir.mkdir(parents=True, exist_ok=True)
    args.figure.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(
        args.data_dir / "nacl_loto_dispersion.npz",
        qpoints_fractional=qpoints,
        path_distance=distance,
        tick_positions=ticks,
        labels=np.asarray(PATH),
        frequencies_without_nac_thz=frequencies_no_nac,
        frequencies_with_nac_thz=frequencies_with_nac,
        born=born,
        dielectric=dielectric,
        neutron_direction=neutron_direction,
        neutron_reduced_coordinate=neutron_fraction,
        neutron_branch=neutron_branch,
        neutron_omega_1e13_rad_s=neutron_omega,
        neutron_path_distance=neutron_distance,
        neutron_frequency_thz=neutron_frequency_thz,
    )
    plot_summary = {
        "backend": "QE pw.x + ph.x DFPT tensors applied with phonopy NAC",
        "phonopy_yaml": str(args.phonopy_yaml),
        "dfpt_summary": str(args.dfpt_summary),
        "path": list(PATH),
        "gamma_to_without_nac_thz": float(gamma_to),
        "gamma_lo_with_nac_thz": float(gamma_lo),
        "gamma_lo_to_split_thz": float(split),
        "neutron_reference": (
            "Raunio, Almqvist, and Stedman, Phys. Rev. 178, 1496 (1969), Table I"
        ),
        "neutron_temperature_k": 80.0,
        "neutron_table": str(args.neutron_table),
        "neutron_point_count": len(neutron_rows),
        "literature_reference": "NBS Special Publication 301 (1969)",
        "literature_gamma_to_cm1": LITERATURE_TO_CM1,
        "literature_gamma_lo_cm1": LITERATURE_LO_CM1,
        "literature_gamma_to_thz": float(literature_to),
        "literature_gamma_lo_thz": float(literature_lo),
        "literature_gamma_split_thz": float(literature_split),
        "gamma_to_error_percent": float(
            100.0 * (gamma_to / literature_to - 1.0)
        ),
        "gamma_lo_error_percent": float(
            100.0 * (gamma_lo / literature_lo - 1.0)
        ),
        "gamma_split_error_percent": float(
            100.0 * (split / literature_split - 1.0)
        ),
        "lo_to_ratio_error_percent": float(
            100.0
            * ((gamma_lo / gamma_to) / (literature_lo / literature_to) - 1.0)
        ),
        "born_effective_charges": born.tolist(),
        "dielectric_tensor": dielectric.tolist(),
    }
    (args.data_dir / "nacl_loto_plot_summary.json").write_text(
        json.dumps(plot_summary, indent=2, allow_nan=False) + "\n"
    )

    figure, axes = plt.subplots(
        1, 2, figsize=(10.0, 4.0),
        gridspec_kw={"width_ratios": [1.65, 1.0]},
    )
    ax = axes[0]
    for branch in range(frequencies_no_nac.shape[1]):
        label_no = "without NAC" if branch == 0 else None
        label_yes = "with QE-DFPT NAC" if branch == 0 else None
        ax.plot(
            distance, frequencies_no_nac[:, branch],
            color="0.55", linestyle="--", linewidth=1.0, label=label_no,
        )
        ax.plot(
            distance, frequencies_with_nac[:, branch],
            color="#0057b8", linewidth=1.35, label=label_yes,
        )
    ax.scatter(
        neutron_distance,
        neutron_frequency_thz,
        marker="x",
        s=18,
        linewidths=0.8,
        color="#b21f2d",
        zorder=5,
        label="80 K neutron (Raunio et al.)",
    )
    for tick in ticks:
        ax.axvline(tick, color="0.82", linewidth=0.7)
    ax.axhline(0.0, color="black", linewidth=0.7)
    ax.set_xlim(ticks[0], ticks[-1])
    ax.set_ylim(-0.3, max(8.2, float(frequencies_with_nac.max()) + 0.3))
    ax.set_xticks(ticks)
    ax.set_xticklabels(
        [r"$\Gamma$", "X", "W", "K", r"$\Gamma$", "L"]
    )
    ax.set_ylabel("Frequency (THz)")
    ax.set_title("(a) NaCl phonon dispersion")

    zoom = axes[1]
    n_zoom = max(8, args.points_per_segment // 4)
    zoom_slice = slice(0, n_zoom)
    q_fraction = np.linspace(0.0, 1.0, args.points_per_segment)[:n_zoom]
    for branch in range(3, 6):
        zoom.plot(
            q_fraction, frequencies_no_nac[zoom_slice, branch],
            color="0.55", linestyle="--", linewidth=1.2,
        )
        zoom.plot(
            q_fraction, frequencies_with_nac[zoom_slice, branch],
            color="#0057b8", linewidth=1.6,
        )
    zoom.scatter([0.0], [gamma_to], color="0.35", marker="s", zorder=5)
    zoom.scatter([0.0], [gamma_lo], color="#c23b22", marker="o", zorder=6)
    zoom.scatter(
        [0.0, 0.0], [literature_to, literature_lo], marker="D", s=43,
        facecolors="white", edgecolors="#6a1b9a", linewidths=1.25,
        zorder=7, label="optical $\\Gamma$ limits (NBS, 1969)",
    )
    zoom.annotate(
        f"TO = {gamma_to:.3f} THz", (0.0, gamma_to),
        xytext=(0.025, gamma_to - 0.35), fontsize=9,
    )
    zoom.annotate(
        f"LO = {gamma_lo:.3f} THz", (0.0, gamma_lo),
        xytext=(0.025, gamma_lo + 0.15), fontsize=9,
    )
    zoom.annotate(
        f"lit. TO = {literature_to:.3f}", (0.0, literature_to),
        xytext=(0.025, literature_to + 0.10), fontsize=8.5,
        color="#6a1b9a",
    )
    zoom.annotate(
        f"lit. LO = {literature_lo:.3f}", (0.0, literature_lo),
        xytext=(0.025, literature_lo + 0.10), fontsize=8.5,
        color="#6a1b9a",
    )
    zoom.set_xlim(0.0, float(q_fraction[-1]))
    zoom.set_ylim(3.7, 8.45)
    zoom.set_xlabel(r"Reduced coordinate along $\Gamma$--X")
    zoom.set_ylabel("Frequency (THz)")
    zoom.set_title(
        rf"(b) Near-$\Gamma$ splitting: {split:.3f} THz"
    )
    zoom.grid(alpha=0.2)

    handles, labels = [], []
    for axis in axes:
        axis_handles, axis_labels = axis.get_legend_handles_labels()
        handles.extend(axis_handles)
        labels.extend(axis_labels)
    figure.legend(
        handles, labels, loc="upper center", bbox_to_anchor=(0.5, 1.015),
        ncol=3, frameon=False, fontsize=9,
    )
    figure.tight_layout(rect=[0.0, 0.0, 1.0, 0.90])
    figure.savefig(args.figure.with_suffix(".pdf"), bbox_inches="tight")
    figure.savefig(args.figure.with_suffix(".png"), dpi=300, bbox_inches="tight")
    plt.close(figure)
    print(json.dumps(plot_summary, indent=2))


if __name__ == "__main__":
    main()
