#!/usr/bin/env python3
"""Run primitive-cell QE DFPT for Born charges and epsilon-infinity."""

from __future__ import annotations

import argparse
import json
import os
from pathlib import Path
import shlex
import sys

import numpy as np

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "codes"))

from sotc.constants import AMU_TO_KG, EV_TO_J, RAD_S_TO_THZ
from sotc.phonons import PhononCalculator
from sotc.qe_dfpt import QEDFPTNAC, parse_ph_gamma_frequencies


MATERIALS = {
    "nacl": {
        "elements": ["Na", "Cl"], "masses": [22.9898, 35.453], "a": 5.73,
        "pseudos": {
            "Na": "na_pbe_v1.5.uspp.F.UPF",
            "Cl": "cl_pbe_v1.4.uspp.F.UPF",
        },
        "ecutwfc": 60.0, "ecutrho": 480.0, "kpts": (8, 8, 8),
    },
    "mgo": {
        "elements": ["Mg", "O"], "masses": [24.305, 15.999], "a": 4.21,
        "pseudos": {
            "Mg": "Mg.pbe-n-kjpaw_psl.0.3.0.UPF",
            "O": "O.pbe-n-kjpaw_psl.0.1.UPF",
        },
        "ecutwfc": 60.0, "ecutrho": 480.0, "kpts": (8, 8, 8),
    },
    "pbte": {
        "elements": ["Pb", "Te"], "masses": [207.2, 127.60], "a": 6.46,
        "pseudos": {
            "Pb": "Pb.pbe-dn-kjpaw_psl.0.2.2.UPF",
            "Te": "Te_pbe_v1.uspp.F.UPF",
        },
        "ecutwfc": 60.0, "ecutrho": 480.0, "kpts": (8, 8, 8),
        "note": "scalar-relativistic PBE validation; SOC is a separate sensitivity test",
    },
}


def primitive_rocksalt(a: float) -> tuple[np.ndarray, np.ndarray]:
    cell = 0.5 * a * np.array([
        [0.0, 1.0, 1.0],
        [1.0, 0.0, 1.0],
        [1.0, 1.0, 0.0],
    ])
    positions = np.array([[0.0, 0.0, 0.0], [a / 2.0, 0.0, 0.0]])
    return cell, positions


def derive_ph_command(pw_command: str) -> str:
    tokens = shlex.split(pw_command)
    matches = [
        index for index, token in enumerate(tokens)
        if Path(token).name == "pw.x"
    ]
    if len(matches) != 1:
        raise ValueError(
            "Set QE_PH_CMD explicitly; QE_PW_CMD must contain exactly one pw.x"
        )
    index = matches[0]
    tokens[index] = str(Path(tokens[index]).with_name("ph.x"))
    return shlex.join(tokens)


def gamma_loto_audit(spec, cell, positions, result):
    """Apply real DFPT tensors to a spring model matching QE's Gamma TO mode."""
    gamma = parse_ph_gamma_frequencies(result.ph_output.read_text(errors="replace"))
    if len(gamma) < 6:
        raise ValueError("Expected six Gamma frequencies for rocksalt primitive cell")
    sorted_gamma = np.sort(gamma[-6:])
    to_frequency = float(np.mean(sorted_gamma[-3:]))
    to_spread = float(np.ptp(sorted_gamma[-3:]))
    masses = np.asarray(spec["masses"], dtype=float)
    omega_to = to_frequency / RAD_S_TO_THZ
    conversion = EV_TO_J / (1.0e-20 * AMU_TO_KG)
    spring_ev_ang2 = omega_to**2 / (
        conversion * np.sum(1.0 / masses)
    )

    class GammaSpringIFC:
        N = 2
        _pair_vecs = {(0, 1): np.asarray(positions[1] - positions[0])}
        _Phi = {
            (0, 0): spring_ev_ang2 * np.eye(3),
            (0, 1): -spring_ev_ang2 * np.eye(3),
            (1, 0): -spring_ev_ang2 * np.eye(3),
            (1, 1): spring_ev_ang2 * np.eye(3),
        }

    calculator = PhononCalculator(
        GammaSpringIFC(), positions, cell, masses,
        born_charges=result.born, dielectric_tensor=result.dielectric,
    )
    directions = {}
    for label, direction in {
        "100": np.array([1.0, 0.0, 0.0]),
        "010": np.array([0.0, 1.0, 0.0]),
        "001": np.array([0.0, 0.0, 1.0]),
    }.items():
        audit = calculator.loto_frequencies_near_gamma(direction)
        without = np.asarray(audit["without_nac_thz"])
        with_nac = np.asarray(audit["with_nac_thz"])
        directions[label] = {
            "without_nac_thz": without.tolist(),
            "with_nac_thz": with_nac.tolist(),
            "lo_frequency_thz": float(with_nac[-1]),
            "to_frequency_thz": float(without[-1]),
            "lo_to_split_thz": float(with_nac[-1] - without[-1]),
        }
    return {
        "scope": (
            "real QE Born charges and epsilon-infinity applied by "
            "PhononCalculator to an isotropic Gamma spring fitted to QE's TO mode"
        ),
        "qe_gamma_frequencies_thz": gamma.tolist(),
        "qe_to_triplet_mean_thz": to_frequency,
        "qe_to_triplet_spread_thz": to_spread,
        "fitted_gamma_spring_ev_ang2": float(spring_ev_ang2),
        "directions": directions,
        "directional_lo_spread_thz": float(np.ptp([
            item["lo_frequency_thz"] for item in directions.values()
        ])),
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("material", choices=sorted(MATERIALS))
    parser.add_argument("--no-charge-neutrality-projection", action="store_true")
    parser.add_argument("--reuse-existing", action="store_true")
    parser.add_argument("--lattice", type=float, help="override lattice parameter [Angstrom]")
    parser.add_argument("--output", type=Path, help="output directory override")
    args = parser.parse_args()
    spec = dict(MATERIALS[args.material])
    if args.lattice is not None:
        spec["a"] = float(args.lattice)
    cell, positions = primitive_rocksalt(spec["a"])
    pw_cmd = os.environ["QE_PW_CMD"]
    ph_cmd = os.environ.get("QE_PH_CMD") or derive_ph_command(pw_cmd)
    output = args.output or (ROOT / "validation_results" / args.material / "dfpt_nac")
    driver = QEDFPTNAC(
        species=spec["elements"],
        pseudopotentials=spec["pseudos"],
        pseudo_dir=os.environ.get("SSSP_DIR", ROOT / "pseudopotentials"),
        ecutwfc=spec["ecutwfc"],
        ecutrho=spec["ecutrho"],
        kpts=spec["kpts"],
        pw_cmd=pw_cmd,
        ph_cmd=ph_cmd,
        workdir=output,
        prefix=f"{args.material}_nac",
        smearing=None,
        conv_thr=1.0e-10,
        tr2_ph=1.0e-14,
    )
    result = driver.run(
        positions, cell,
        enforce_charge_neutrality=not args.no_charge_neutrality_projection,
        reuse_existing=args.reuse_existing,
    )
    summary = {
        "material": args.material,
        "lattice_parameter_ang": spec["a"],
        "primitive_cell_ang": cell.tolist(),
        "primitive_positions_ang": positions.tolist(),
        "dielectric": result.dielectric.tolist(),
        "born": result.born.tolist(),
        "charge_neutrality_residual_raw": (
            result.charge_neutrality_residual_raw.tolist()
        ),
        "charge_neutrality_residual_corrected": (
            result.charge_neutrality_residual_corrected.tolist()
        ),
        "gamma_loto_audit": gamma_loto_audit(
            spec, cell, positions, result
        ),
        "note": spec.get("note"),
    }
    (output / "dfpt_summary.json").write_text(json.dumps(summary, indent=2))
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
