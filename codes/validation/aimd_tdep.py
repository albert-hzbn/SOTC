#!/usr/bin/env python3
"""Ab initio molecular dynamics in the SOTC cell, for an AIMD-TDEP baseline.

Runs Born--Oppenheimer MD with Quantum ESPRESSO (Verlet, stochastic velocity
rescaling at T_des) in the same cell and with the same electronic settings
as all other calculations, starting from a canonical configuration. Every
MD step is one force evaluation. Positions and forces of all steps are
stored, so that a TDEP-type least-squares fit (``Fitter`` of
analyze_designs.py) can be made to any number of steps.

Writes validation_results/<m>/primary_cell/aimd/traj<k>/trajectory.npz.

Usage: aimd_tdep.py <material> <traj> [--nstep 1500] [--dt-au 80]
"""

from __future__ import annotations

import argparse
import os
from pathlib import Path
import subprocess
import sys

import numpy as np

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "codes"))
sys.path.insert(0, str(Path(__file__).resolve().parent))

from analyze_primary_archives import PRIMARY  # noqa: E402
from run_primary_cell_controls import primary_spec  # noqa: E402


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("material")
    parser.add_argument("traj", type=int)
    parser.add_argument("--nstep", type=int, default=1500)
    parser.add_argument("--dt-au", type=float, default=80.0)  # Rydberg a.u. (4.84e-17 s)
    parser.add_argument("--parse-only", action="store_true",
                        help="parse the steps already in md.pwo (e.g. after a time limit)")
    args = parser.parse_args()

    from ase import Atoms
    from ase.io import read, write

    material = args.material
    base = ROOT / "validation_results" / material / "primary_cell"
    out = base / "aimd" / f"traj{args.traj}"
    out.mkdir(parents=True, exist_ok=True)
    with np.load(base / "fd" / "harmonic_reference.npz") as h:
        eq = np.asarray(h["equilibrium_positions_ang"])
        cell = np.asarray(h["cell_ang"])
        symbols = [str(s) for s in h["symbols"]]
        f0 = np.asarray(h["reference_forces_ev_ang"])
    # Start from an independent canonical configuration of the iteration-0 pool.
    u = np.load(base / "canonical_displacements.npz")["displacements_ang"]
    atoms = Atoms(symbols=symbols, positions=eq + u[2 * args.traj], cell=cell, pbc=True)
    spec = primary_spec(material)
    system = {"ecutwfc": spec["ecutwfc"], "ecutrho": spec["ecutrho"]}
    if spec["smearing"] is not None:
        system.update(occupations="smearing", smearing=spec["smearing"],
                      degauss=spec["degauss"])
    input_data = {
        "control": {"calculation": "md", "dt": args.dt_au, "nstep": args.nstep,
                    "tprnfor": True, "disk_io": "low", "prefix": f"md{args.traj}",
                    "outdir": str(out / "scratch"),
                    "pseudo_dir": os.environ.get("SSSP_DIR", str(ROOT / "pseudopotentials"))},
        "system": system,
        "electrons": {"conv_thr": spec["conv_thr"], "mixing_beta": 0.4,
                      "electron_maxstep": 200},
        "ions": {"ion_dynamics": "verlet", "ion_temperature": "svr",
                 "tempw": float(PRIMARY[material]["T"]), "nraise": 20,
                 "pot_extrapolation": "second_order",
                 "wfc_extrapolation": "second_order"},
    }
    for section, extra in (spec.get("extra") or {}).items():
        input_data.setdefault(section, {}).update(extra)
    pwi, pwo = out / "md.pwi", out / "md.pwo"
    write(pwi, atoms, format="espresso-in", input_data=input_data,
          pseudopotentials=spec["pseudos"], kpts=tuple(spec["kpts"]))
    if not args.parse_only and (not pwo.exists()
                                or "JOB DONE" not in pwo.read_text(errors="ignore")):
        with open(pwi) as fin, open(pwo, "w") as fout:
            subprocess.run(os.environ["QE_PW_CMD"].split(), stdin=fin, stdout=fout,
                           check=True, cwd=out)
    frames = read(pwo, index=":", format="espresso-out")
    pos = np.asarray([f.get_positions() for f in frames])
    forces = np.asarray([f.get_forces() for f in frames])
    energies = np.asarray([f.get_potential_energy() for f in frames])
    d = pos - eq
    frac = d @ np.linalg.inv(cell)
    d = (frac - np.rint(frac)) @ cell
    df = forces - f0
    np.savez_compressed(out / "trajectory.npz", displacements_ang=d,
                        delta_forces_ev_ang=df - df.mean(axis=1, keepdims=True),
                        energies_ev=energies, dt_fs=args.dt_au * 4.8378e-2)
    print(f"{material} AIMD trajectory {args.traj}: {len(frames)} steps, "
          f"T_kin(last 500) from u_rms {np.sqrt((d[-500:] ** 2).sum(-1).mean()):.3f} A",
          flush=True)


if __name__ == "__main__":
    main()
