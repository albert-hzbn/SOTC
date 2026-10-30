#!/usr/bin/env python3
"""Do designed training sets give better machine-learned potentials per DFT call?

Separate study note (not part of the SOTC manuscript). For Al (300 K) and
Si (600 K) in the SOTC cells, small MACE models are trained on the first N
configurations (N = 4, 8, 16, 32, 64) of four designs with DFT forces:

* ``canonical``: the paired quantum-canonical pool (48 configurations);
* ``sotcd``: SOTC-D pairs (nested D-optimal for the FC2, 64 calls);
* ``rattle``: thermal rattle (independent atoms, canonical variance);
* ``aimd``: every 20th production step of AIMD trajectories 0--2.

Evaluation: force RMSE on 100 held-out snapshots of AIMD trajectory 3,
harmonic phonons of the potential (finite differences in the same cell)
against the DFT finite-difference FC2, and the stability of 2-ps Langevin
MD at the design temperature (fraction of runs without an atom moving more
than 1 A from its site or the temperature exceeding 5 T).

Stages (run with the MLIP venv): export <m>; train <m> <design> <N> [--seed]
[--steps S]; evaluate <m>. With ``--steps`` every model gets the same number
of optimizer steps (epochs = S * batch / N, directory suffix ``_st``), so
that small training sets are not also under-trained. Writes validation_results/<m>/mlip_study/ and
validation_results/summaries/<m>/robustness/mlip_study.json.
"""

from __future__ import annotations

import argparse
import json
import os
from pathlib import Path
import subprocess
import sys

import numpy as np

ROOT = Path(__file__).resolve().parents[2]
TEMPS = {"al": 300.0, "si": 600.0}
NS = (4, 8, 16, 32, 64)
DESIGNS = ("canonical", "sotcd", "rattle", "aimd")
N_EQ = 250


def base(m):
    return ROOT / "validation_results" / m / "primary_cell"


def states(path, indices=None):
    out = []
    for p in sorted(path.glob("state_*/result.npz")):
        i = int(p.parent.name.split("_")[1])
        if indices is None or i in indices:
            with np.load(p) as r:
                out.append((i, np.asarray(r["positions_ang"]), np.asarray(r["forces_ev_ang"]),
                            float(r["energy_ev"])))
    return [x[1:] for x in sorted(out)]


def atoms_list(m, rows):
    import phonopy
    from ase import Atoms

    sup = phonopy.load(base(m) / "fd" / "phonopy_harmonic.yaml").supercell
    out = []
    for pos, f, e in rows:
        a = Atoms(list(sup.symbols), positions=pos, cell=np.asarray(sup.cell), pbc=True)
        a.info["REF_energy"] = e
        a.arrays["REF_forces"] = f
        out.append(a)
    return out


def aimd_frames(m, traj, stride, start=N_EQ):
    from ase.io import read

    frames = read(base(m) / "aimd" / f"traj{traj}" / "md.pwo", index=":", format="espresso-out")
    rows = []
    for a in frames[start::stride]:
        rows.append((a.get_positions(), a.get_forces(), a.get_potential_energy()))
    return rows


def export(m):
    from ase.io import write

    out = ROOT / "validation_results" / m / "mlip_study"
    out.mkdir(parents=True, exist_ok=True)
    b = base(m)
    sets = {
        "canonical": states(b / "canonical_qe"),
        "sotcd": states(b / "sotcd_qe") + states(b / "mlip" / "sotcd_qe"),
        "rattle": states(b / "designs" / "rattle_qe") + states(b / "mlip" / "rattle_qe"),
    }
    # AIMD training: every 20th production step, interleaved over trajectories 0-2.
    trajs = [aimd_frames(m, t, 20) for t in range(3)]
    sets["aimd"] = [x for group in zip(*trajs) for x in group]
    for name, rows in sets.items():
        write(out / f"train_{name}.xyz", atoms_list(m, rows), format="extxyz")
        print(m, name, len(rows), flush=True)
    test = aimd_frames(m, 3, max(1, (len(aimd_frames(m, 3, 1)) // 100)))[:100]
    write(out / "test_aimd.xyz", atoms_list(m, test), format="extxyz")
    print(m, "test", len(test), flush=True)


def train(m, design, n, seed, steps=None):
    from ase.io import read, write

    out = ROOT / "validation_results" / m / "mlip_study"
    configs = read(out / f"train_{design}.xyz", index=":")
    if len(configs) < n:
        print(f"{m} {design}: only {len(configs)} configurations", flush=True)
        return
    name = f"{design}_N{n}_s{seed}" + ("_st" if steps else "")
    epochs = 150 if not steps else int(np.ceil(steps * 2 / n))
    work = out / name
    work.mkdir(exist_ok=True)
    write(work / "train.xyz", configs[:n], format="extxyz")
    cmd = [str(Path(sys.executable).parent / "mace_run_train"),
           f"--name={name}", f"--train_file={work / 'train.xyz'}",
           f"--valid_file={work / 'train.xyz'}", "--energy_key=REF_energy",
           "--forces_key=REF_forces", "--model=MACE", "--num_channels=32", "--max_L=0",
           "--correlation=2", "--r_max=5.0", "--batch_size=2", f"--max_num_epochs={epochs}",
           "--energy_weight=1", "--forces_weight=100", "--E0s=average", "--device=cpu",
           "--default_dtype=float64", f"--seed={seed}", f"--eval_interval={max(1, epochs // 3)}",
           f"--model_dir={work}", f"--checkpoints_dir={work / 'ckpt'}",
           f"--results_dir={work / 'results'}", f"--log_dir={work / 'logs'}"]
    res = subprocess.run(cmd, cwd=work, capture_output=True, text=True)
    (work / "train.log").write_text(res.stdout[-20000:] + res.stderr[-20000:])
    print(name, "exit", res.returncode, flush=True)


def evaluate(m):
    import phonopy
    from ase import units
    from ase.io import read
    from ase.md.langevin import Langevin
    from ase.md.velocitydistribution import MaxwellBoltzmannDistribution
    from mace.calculators import MACECalculator

    out = ROOT / "validation_results" / m / "mlip_study"
    test = read(out / "test_aimd.xyz", index=":")
    ph = phonopy.load(base(m) / "fd" / "phonopy_harmonic.yaml")
    sup = ph.supercell
    dft_fc = np.asarray(ph.force_constants)
    qmesh = np.array([[i, j, k] for i in range(6) for j in range(6) for k in range(6)]) / 6.0

    def mesh_freqs(fc):
        ph.force_constants = fc
        ph.run_qpoints(qmesh)
        return np.sort(np.asarray(ph.get_qpoints_dict()["frequencies"]), axis=1)

    ref_freq = mesh_freqs(dft_fc)
    report = {"material": m, "temperature_k": TEMPS[m], "n_test": len(test), "models": {}}
    for model in sorted(out.glob("*_N*_s*/*.model")):
        if "_compiled" in model.name or "swa" in model.name:
            continue
        name = model.parent.name
        calc = MACECalculator(model_paths=str(model), device="cpu", default_dtype="float64")
        err = []
        for a in test:
            a = a.copy()
            a.calc = calc
            err.append(a.get_forces() - a.arrays["REF_forces"])
        f_rmse = float(np.sqrt(np.mean(np.square(err))))
        # harmonic phonons of the potential (central differences, 0.01 A)
        from ase import Atoms
        ph2 = phonopy.Phonopy(ph.unitcell, ph.supercell_matrix, ph.primitive_matrix)
        ph2.generate_displacements(distance=0.01, is_plusminus=True)
        forces = []
        for c in ph2.supercells_with_displacements:
            a = Atoms(list(c.symbols), positions=c.positions, cell=c.cell, pbc=True)
            a.calc = calc
            forces.append(a.get_forces())
        ph2.forces = np.asarray(forces)
        ph2.produce_force_constants()
        freq = mesh_freqs(np.asarray(ph2.force_constants))
        ph_rmse = float(np.sqrt(np.mean((freq - ref_freq) ** 2)))
        # MD stability: three 2-ps Langevin runs
        stable = 0
        for r in range(3):
            a = Atoms(list(sup.symbols), positions=np.asarray(sup.positions),
                      cell=np.asarray(sup.cell), pbc=True)
            a.calc = calc
            MaxwellBoltzmannDistribution(a, temperature_K=TEMPS[m],
                                         rng=np.random.default_rng(r))
            dyn = Langevin(a, 2.0 * units.fs, temperature_K=TEMPS[m], friction=0.02,
                           rng=np.random.default_rng(100 + r))
            ok = True
            for _ in range(100):
                dyn.run(10)
                d = a.get_positions() - np.asarray(sup.positions)
                frac = d @ np.linalg.inv(np.asarray(sup.cell))
                d = (frac - np.rint(frac)) @ np.asarray(sup.cell)
                if np.abs(d).max() > 1.0 or a.get_temperature() > 5 * TEMPS[m]:
                    ok = False
                    break
            stable += ok
        report["models"][name] = {"force_rmse_ev_ang": f_rmse, "phonon_rmse_thz": ph_rmse,
                                  "md_stable_runs": stable}
        print(name, report["models"][name], flush=True)
    path = ROOT / "validation_results" / "summaries" / m / "robustness" / "mlip_study.json"
    path.write_text(json.dumps(report, indent=1) + "\n")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("stage", choices=["export", "train", "evaluate"])
    parser.add_argument("material")
    parser.add_argument("design", nargs="?")
    parser.add_argument("n", nargs="?", type=int)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--steps", type=int)
    args = parser.parse_args()
    if args.stage == "export":
        export(args.material)
    elif args.stage == "train":
        train(args.material, args.design, args.n, args.seed, args.steps)
    else:
        evaluate(args.material)


if __name__ == "__main__":
    main()
