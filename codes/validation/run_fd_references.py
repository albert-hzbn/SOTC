#!/usr/bin/env python3
"""Conventional finite-displacement FC2 references for all nine benchmarks.

Three independent harmonic references are produced per material in 3x3x3
primitive supercells with the electronic settings of the primary SOTC run:

* ``production``: the lattice parameter used by the primary SOTC benchmark;
* ``pbe``: the PBE equilibrium lattice parameter from a primitive-cell
  Birch--Murnaghan equation of state (sensitivity to the volume choice);
* ``room``: the experimental room-temperature lattice parameter (Al, Cu, Au
  only), used for temperature-consistent 300 K displacement parameters.

Every QE state is checkpointed by ``run_real_anharmonic_ifcs.run_states``.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys
from types import SimpleNamespace

import numpy as np

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "codes"))
sys.path.insert(0, str(Path(__file__).resolve().parent))

from run_real_anharmonic_ifcs import (  # noqa: E402
    MATERIALS as CAMPAIGN_MATERIALS,
    harmonic_reference,
    json_default,
    primitive_structure,
    structure_symbols,
    run_states,
)
from sotc.validation import phonopy_atoms  # noqa: E402


# Electronic settings follow codes/benchmarks/run_sotc_<material>_qe.py.
# Supercell k meshes follow the independent 3x3x3 campaigns: 4x4x4 for metals
# and 2x2x2 for insulators.
EXTRA_MATERIALS = {
    "ag": {
        "label": "fcc Ag", "elements": ["Ag"], "masses": [107.8682],
        "a": 4.12, "bravais": "fcc", "temperature_k": 1000.0,
        "ecutwfc": 50.0, "ecutrho": 200.0, "kpts": (4, 4, 4),
        "smearing": "marzari-vanderbilt", "degauss": 0.02,
        "pseudos": {"Ag": "Ag_ONCV_PBE-1.0.oncvpsp.upf"},
    },
    "au": {
        "label": "fcc Au", "elements": ["Au"], "masses": [196.966569],
        "a": 4.13, "bravais": "fcc", "temperature_k": 1200.0,
        "ecutwfc": 45.0, "ecutrho": 180.0, "kpts": (4, 4, 4),
        "smearing": "marzari-vanderbilt", "degauss": 0.02,
        "pseudos": {"Au": "Au_ONCV_PBE-1.0.oncvpsp.upf"},
    },
    "cu": {
        "label": "fcc Cu", "elements": ["Cu"], "masses": [63.546],
        "a": 3.67, "bravais": "fcc", "temperature_k": 1200.0,
        "ecutwfc": 45.0, "ecutrho": 360.0, "kpts": (4, 4, 4),
        "smearing": "marzari-vanderbilt", "degauss": 0.02,
        "pseudos": {"Cu": "Cu.paw.z_11.ld1.psl.v1.0.0-low.upf"},
    },
    "mo": {
        "label": "bcc Mo", "elements": ["Mo"], "masses": [95.95],
        "a": 3.16, "bravais": "bcc", "temperature_k": 1500.0,
        "ecutwfc": 40.0, "ecutrho": 160.0, "kpts": (4, 4, 4),
        "smearing": "marzari-vanderbilt", "degauss": 0.02,
        "pseudos": {"Mo": "Mo_ONCV_PBE-1.0.oncvpsp.upf"},
    },
    "mgo": {
        "label": "rocksalt MgO", "elements": ["Mg", "O"],
        "masses": [24.305, 15.999],
        "a": 4.26, "bravais": "fcc", "temperature_k": 1500.0,
        "ecutwfc": 80.0, "ecutrho": 640.0, "kpts": (2, 2, 2),
        "smearing": None, "degauss": 0.0,
        "pseudos": {"Mg": "Mg.pbe-n-kjpaw_psl.0.3.0.UPF",
                    "O": "O.pbe-n-kjpaw_psl.0.1.UPF"},
    },
    "pbte": {
        "label": "rocksalt PbTe", "elements": ["Pb", "Te"],
        "masses": [207.2, 127.60],
        "a": 6.51, "bravais": "fcc", "temperature_k": 700.0,
        "ecutwfc": 44.0, "ecutrho": 352.0, "kpts": (2, 2, 2),
        "smearing": None, "degauss": 0.0,
        "pseudos": {"Pb": "Pb.pbe-dn-kjpaw_psl.0.2.2.UPF",
                    "Te": "Te_pbe_v1.uspp.F.UPF"},
    },
}
# Additional structure classes (covalent diamond, non-cubic hcp metal) with
# no archived SOTC run; they are benchmarked with the same-cell controls.
NEW_CLASS_MATERIALS = {
    "si": {
        "label": "diamond Si", "elements": ["Si"], "masses": [28.0855],
        "a": 5.431, "bravais": "diamond", "temperature_k": 600.0,
        "ecutwfc": 30.0, "ecutrho": 120.0, "kpts": (4, 4, 4),
        "smearing": None, "degauss": 0.0,
        "pseudos": {"Si": "Si_ONCV_PBE-1.1.upf"},
    },
    "mg": {
        "label": "hcp Mg", "elements": ["Mg"], "masses": [24.305],
        "a": 3.209, "c": 5.211, "bravais": "hcp", "temperature_k": 600.0,
        "ecutwfc": 40.0, "ecutrho": 320.0, "kpts": (4, 4, 3),
        "smearing": "marzari-vanderbilt", "degauss": 0.02,
        "pseudos": {"Mg": "Mg.pbe-n-kjpaw_psl.0.3.0.UPF"},
    },
}
EXTRA_MATERIALS.update(NEW_CLASS_MATERIALS)
for _key, _spec in EXTRA_MATERIALS.items():
    _spec["key"] = _key

MATERIALS = {**CAMPAIGN_MATERIALS, **EXTRA_MATERIALS}

# Experimental room-temperature lattice parameters (Angstrom).
ROOM_LATTICE = {"al": 4.0495, "cu": 3.6149, "au": 4.0782}

# Primitive-cell k meshes for the equation of state.
EOS_KMESH = {"metal": (16, 16, 16), "insulator": (8, 8, 8)}
EOS_STRAINS = np.linspace(-0.03, 0.03, 7)


def make_phonon(spec, a, size=3):
    from phonopy import Phonopy

    local = dict(spec, a=float(a))
    if "c" in spec:
        local["c"] = float(spec["c"]) * float(a) / float(spec["a"])
    cell, positions = primitive_structure(local)
    unit = phonopy_atoms(structure_symbols(spec, positions), cell, positions)
    return Phonopy(unit, np.diag(np.broadcast_to(np.asarray(size), 3)))


# Large-cell references: 5x5x5 primitive (125 atoms) for the monatomic metals
# and 4x4x4 (128 atoms) for rocksalt, with supercell k meshes chosen to keep
# the primitive-equivalent sampling at least that of the 3x3x3 campaigns.
LARGE_SIZE = {"metal": 5, "insulator": 4}
LARGE_KMESH = {"metal": (3, 3, 3), "insulator": (2, 2, 2)}


# Davidson fails with a Cholesky error in the 128-atom PbTe reference, and
# plain mixing sloshes in the 26-Angstrom cell; the large-cell PbTe states use
# ParO diagonalization with local Thomas--Fermi mixing instead.
LARGE_EXTRA = {"pbte": {"electrons": {
    "diagonalization": "paro", "mixing_mode": "local-TF",
    "mixing_beta": 0.2, "mixing_ndim": 12,
}}}


# Non-cubic cells use an anisotropic large supercell (hcp Mg: 150 atoms).
LARGE_SIZE_OVERRIDE = {"mg": (5, 5, 3)}


def large_settings(spec):
    kind = "metal" if spec["smearing"] is not None else "insulator"
    size = LARGE_SIZE_OVERRIDE.get(spec.get("key"), LARGE_SIZE[kind])
    return size, LARGE_KMESH[kind]


def run_single_state(material, spec, args, out, a, size, index):
    """Evaluate one displaced (or reference) state for job-array execution."""
    from run_real_anharmonic_ifcs import run_one_state

    phonon = make_phonon(spec, a, size)
    phonon.generate_displacements(distance=0.01, is_plusminus=True)
    cell = np.asarray(phonon.supercell.cell, dtype=float)
    states = [np.asarray(phonon.supercell.positions, dtype=float)] + [
        np.asarray(sc.positions, dtype=float)
        for sc in phonon.supercells_with_displacements
    ]
    if index >= len(states):
        print(f"state {index} does not exist ({len(states)} states)")
        return
    stage = out / "harmonic_qe"
    stage.mkdir(parents=True, exist_ok=True)
    local = dict(spec, a=float(a))
    if material in LARGE_EXTRA:
        local["extra"] = LARGE_EXTRA[material]
    run_one_state(
        index, states[index], cell, list(phonon.supercell.symbols),
        local, args, stage,
        overrides={"kpts": large_settings(spec)[1]} if np.any(np.asarray(size) != 3) else None,
    )


def birch_murnaghan(volumes, energies):
    """Third-order Birch--Murnaghan fit; returns V0, E0, B0 (GPa), B0'."""
    from scipy.optimize import curve_fit

    def model(v, e0, v0, b0, bp):
        eta = (v0 / v) ** (2.0 / 3.0)
        return e0 + 9.0 * v0 * b0 / 16.0 * (
            (eta - 1.0) ** 3 * bp + (eta - 1.0) ** 2 * (6.0 - 4.0 * eta)
        )

    coeff = np.polyfit(volumes, energies, 2)
    v_guess = -coeff[1] / (2.0 * coeff[0])
    b_guess = 2.0 * coeff[0] * v_guess
    popt, _ = curve_fit(
        model, volumes, energies,
        p0=[float(np.min(energies)), v_guess, b_guess, 4.5], maxfev=20000,
    )
    residual = energies - model(volumes, *popt)
    return {
        "E0_ev": float(popt[0]), "V0_ang3": float(popt[1]),
        "B0_gpa": float(popt[2] * 160.21766208), "B0_prime": float(popt[3]),
        "fit_rms_ev": float(np.sqrt(np.mean(residual ** 2))),
    }


def pbe_lattice(material, spec, args, out):
    metal = spec["smearing"] is not None
    kpts = EOS_KMESH["metal" if metal else "insulator"]
    lattice = spec["a"] * (1.0 + EOS_STRAINS)
    states, cells = [], []
    for a in lattice:
        cell, positions = primitive_structure(dict(spec, a=float(a)))
        cells.append(cell)
        states.append(positions)
    energies, pressures = [], []
    for index, (cell, positions) in enumerate(zip(cells, states)):
        result = run_states(
            [positions], cell, structure_symbols(spec, positions), spec, args,
            out / "eos" / f"a_{index:02d}", overrides={"kpts": kpts},
        )[0]
        energies.append(float(result["energy_ev"]))
        stress = np.asarray(result["stress_ev_ang3"])
        pressures.append(float(-np.trace(stress) / 3.0 * 160.21766208))
    volumes = np.asarray([abs(np.linalg.det(c)) for c in cells])
    fit = birch_murnaghan(volumes, np.asarray(energies))
    factor = 4.0 if spec["bravais"] in ("fcc", "diamond") else 2.0
    a0 = float((factor * fit["V0_ang3"]) ** (1.0 / 3.0))
    report = {
        "kpts": kpts, "lattice_ang": lattice, "volumes_ang3": volumes,
        "energies_ev": energies, "pressures_gpa": pressures, "fit": fit,
        "a_pbe_ang": a0, "a_production_ang": spec["a"],
        "relative_difference": (a0 - spec["a"]) / spec["a"],
    }
    (out / "eos_summary.json").write_text(
        json.dumps(report, indent=2, default=json_default) + "\n"
    )
    return a0, report


def reference_summary(material, spec, label, a, result, out):
    phonon = result["phonon"]
    temperatures = np.array([300.0, spec["temperature_k"]])
    phonon.run_mesh([20, 20, 20], with_eigenvectors=True, is_mesh_symmetry=False)
    phonon.run_thermal_displacements(temperatures=temperatures, freq_min=0.05)
    thermal = phonon.get_thermal_displacements_dict()
    # phonopy reports <u_alpha^2> per atom and Cartesian component, flattened
    # as (temperature, 3 * n_primitive_atoms) in phonopy 2.x.
    values = np.asarray(thermal["thermal_displacements"])
    u_iso = np.mean(values.reshape(len(temperatures), -1, 3), axis=2)
    phonon.run_mesh([20, 20, 20])
    phonon.run_thermal_properties(temperatures=temperatures)
    tp = phonon.get_thermal_properties_dict()
    mesh_freq = np.asarray(phonon.get_mesh_dict()["frequencies"])
    return {
        "material": material, "variant": label, "lattice_ang": a,
        "supercell": "3x3x3 primitive", "n_atoms": len(phonon.supercell),
        "kpts": spec["kpts"], "ecutwfc_ry": spec["ecutwfc"],
        "ecutrho_ry": spec["ecutrho"],
        "min_mesh_frequency_thz": result["min_mesh_frequency_thz"],
        "fraction_imaginary_mesh_modes": result["fraction_imaginary_mesh_modes"],
        "n_displaced_states": result["n_displaced_states"],
        "reference_max_force_ev_ang": float(
            np.max(np.linalg.norm(result["reference_forces"], axis=1))
        ),
        "reference_pressure_gpa": float(
            -np.trace(result["reference_stress"]) / 3.0 * 160.21766208
        ),
        "temperatures_k": temperatures,
        "u_iso_ang2_by_atom": u_iso,
        "cv_j_per_k_mol_cell": tp["heat_capacity"],
        "entropy_j_per_k_mol_cell": tp["entropy"],
        "free_energy_kj_per_mol_cell": tp["free_energy"],
        "max_mesh_frequency_thz": float(np.max(mesh_freq)),
        "dir": out,
    }


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("material", choices=sorted(MATERIALS))
    parser.add_argument("--variants", default="production,pbe,room")
    parser.add_argument("--n-parallel", type=int, default=1)
    parser.add_argument("--tasks-per-qe", type=int, default=128)
    parser.add_argument("--large", action="store_true",
                        help="use the large converged-reference supercell")
    parser.add_argument("--state", type=int,
                        help="run only this QE state index and exit")
    parser.add_argument("--fit-only", action="store_true")
    parser.add_argument("--force-rerun", action="store_true")
    args = parser.parse_args()
    material = args.material
    spec = MATERIALS[material]
    run_args = SimpleNamespace(
        material=material, n_parallel=args.n_parallel,
        tasks_per_qe=args.tasks_per_qe, fit_only=args.fit_only,
        force_rerun=args.force_rerun,
    )
    base = ROOT / "validation_results" / material / "fd_references"
    base.mkdir(parents=True, exist_ok=True)
    variants = [v.strip() for v in args.variants.split(",") if v.strip()]
    summary_path = base / "fd_reference_summary.json"
    summaries = (
        json.loads(summary_path.read_text()) if summary_path.exists() else {}
    )

    for label in variants:
        if label == "production":
            a = spec["a"]
        elif label == "pbe":
            a, eos = pbe_lattice(material, spec, run_args, base)
        elif label == "room":
            if material not in ROOM_LATTICE:
                continue
            a = ROOM_LATTICE[material]
        else:
            raise ValueError(label)
        size = large_settings(spec)[0] if args.large else 3
        key = f"{label}_large" if args.large else label
        out = base / key
        out.mkdir(parents=True, exist_ok=True)
        local = dict(spec, a=float(a))
        if args.large:
            local["kpts"] = large_settings(spec)[1]
            if material in LARGE_EXTRA:
                local["extra"] = LARGE_EXTRA[material]
        if args.state is not None:
            run_single_state(material, spec, run_args, out, a, size, args.state)
            continue
        result = harmonic_reference(make_phonon(spec, a, size), local, run_args, out)
        summaries[key] = reference_summary(material, local, key, a, result, out)
        summaries[key]["supercell"] = "x".join(
            str(int(n)) for n in np.broadcast_to(np.asarray(size), 3)) + " primitive"
        if label == "pbe":
            summaries[key]["eos"] = eos
        summary_path.write_text(
            json.dumps(summaries, indent=2, default=json_default) + "\n"
        )
        print(f"{material} {label}: a={a:.4f} A, min freq "
              f"{result['min_mesh_frequency_thz']:.3f} THz", flush=True)


if __name__ == "__main__":
    main()
