#!/usr/bin/env python3
"""Cross-check of our least-squares FC2 fit against the TDEP code.

The same displacement--force data are fitted (i) by the regression used for
every protocol in this work (symfc space-group basis of the SOTC supercell,
ordinary least squares; FC2 alone, and jointly with FC3 for MD data) and
(ii) by ``extract_forceconstants`` of TDEP (Hellman et al.), with the same
pair cutoff, both with TDEP's default rotational and Huang invariances and
without them. Data sets: the AIMD production steps of ``aimd_tdep.py`` and
the 40-snapshot iteration-0 canonical pool. Every model is compared on the
same q mesh with the converged self-consistent reference of
``dft_selfconsistent_compare.py``.

Writes validation_results/<m>/robustness/tdep_compare.json.

Usage: tdep_compare.py <material> [--tdep-bin DIR] [--nproc 8]
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
sys.path.insert(0, str(ROOT / "codes"))
sys.path.insert(0, str(Path(__file__).resolve().parent))

from aimd_tdep_compare import FC3_CUTOFF, N_EQ  # noqa: E402
from analyze_designs import N_TRAIN, Fitter, crystal_basis, datasets  # noqa: E402
from analyze_primary_archives import (  # noqa: E402
    PRIMARY, frequency_metrics, mesh_points, symfc_fc2_basis,
)
from sotc.symfc_fit import fit_symfc_model, symfc_atoms  # noqa: E402
from surrogate_design_study import mesh_frequencies  # noqa: E402

TDEP_BIN = Path("/viper/ptmp1/alli/sotc_validation_work/tdep_build/tdep/bin")
TDEP_ENV = ("source /etc/profile.d/modules.sh; module load gcc/14 impi/2021.17 mkl/2025.2 "
            "hdf5-mpi/1.14.1 fftw-mpi/3.3.10; export I_MPI_FABRICS=shm "
            "I_MPI_HYDRA_BOOTSTRAP=fork OMP_NUM_THREADS=1; ")


def write_poscar(path, cell, symbols, frac):
    species = list(dict.fromkeys(symbols))
    order = [i for s in species for i in range(len(symbols)) if symbols[i] == s]
    lines = ["generated", "1.0"] + [" ".join(f"{x:.12f}" for x in v) for v in cell]
    lines += [" ".join(species), " ".join(str(symbols.count(s)) for s in species), "direct"]
    lines += [" ".join(f"{x:.12f}" for x in frac[i]) for i in order]
    path.write_text("\n".join(lines) + "\n")
    return order


def run_tdep(work, phonon, u, f, rc2, rc3=None, constraints=True, tdep_bin=TDEP_BIN, nproc=8):
    """Write TDEP inputs for displacements u and forces f (eV/A) and fit."""
    work.mkdir(parents=True, exist_ok=True)
    prim, sup = phonon.primitive, phonon.supercell
    sym_s = list(sup.symbols)
    order = write_poscar(work / "infile.ssposcar", np.asarray(sup.cell), sym_s,
                         np.asarray(sup.scaled_positions))
    write_poscar(work / "infile.ucposcar", np.asarray(prim.cell), list(prim.symbols),
                 np.asarray(prim.scaled_positions))
    inv = np.linalg.inv(np.asarray(sup.cell))
    eq = np.asarray(sup.positions)
    frac = ((eq[None] + u) @ inv) % 1.0
    with open(work / "infile.positions", "w") as fp:
        for c in frac[:, order]:
            fp.write("\n".join(" ".join(f"{x:.12f}" for x in r) for r in c) + "\n")
    with open(work / "infile.forces", "w") as fp:
        for c in f[:, order]:
            fp.write("\n".join(" ".join(f"{x:.12e}" for x in r) for r in c) + "\n")
    with open(work / "infile.stat", "w") as fp:
        for k in range(len(u)):
            fp.write(f"{k + 1} {k:.1f} 0 0 0 0 0 0 0 0 0 0 0\n")
    (work / "infile.meta").write_text(f"{len(sym_s)}\n{len(u)}\n1.0\n{0}\n")
    args = f"-rc2 {rc2}" + (f" -rc3 {rc3}" if rc3 else "")
    if not constraints:
        args += " --norotational --nohuang"
    cmd = f"{TDEP_ENV} cd {work} && mpirun -np {nproc} {tdep_bin}/extract_forceconstants {args}"
    res = subprocess.run(["bash", "-c", cmd], capture_output=True, text=True)
    (work / "tdep.log").write_text(res.stdout + res.stderr)
    if res.returncode != 0:
        raise RuntimeError(f"TDEP failed in {work}:\n{res.stderr[-2000:]}")
    return read_tdep_fc(work / "outfile.forceconstant", phonon, order)


def tdep_cutoff(work):
    return float((work / "outfile.forceconstant").read_text().splitlines()[1].split()[0])


def read_tdep_fc(path, phonon, order):
    """Map TDEP's pair list (unit-cell atoms, fractional lattice vectors) onto
    the full supercell FC2 in phonopy atom order."""
    prim, sup = phonon.primitive, phonon.supercell
    # every line starts with numbers and may end with a text comment
    lines = [ln.split() for ln in path.read_text().splitlines() if ln.strip()]
    pos = 0

    def nxt(k=1):
        nonlocal pos
        pos += 1
        return lines[pos - 1][:k]

    n_uc = int(nxt()[0])
    nxt()  # cutoff
    pcell = np.asarray(prim.cell)
    tau = np.asarray(prim.positions)
    scell = np.asarray(sup.cell)
    sinv = np.linalg.inv(scell)
    spos = np.asarray(sup.positions)
    s2p = np.asarray(prim.s2p_map)
    p2s = list(prim.p2s_map)
    n = len(spos)
    fc = np.zeros((n, n, 3, 3))
    uc_species = list(prim.symbols)
    uc_order = [i for s in dict.fromkeys(uc_species) for i in range(n_uc) if uc_species[i] == s]
    pairs = {}
    for a_t in range(n_uc):
        a = uc_order[a_t]
        rows = []
        for _ in range(int(nxt()[0])):
            b = uc_order[int(nxt()[0]) - 1]
            lv = np.array(nxt(3), float)
            m = np.array([nxt(3) for _ in range(3)], float)
            rows.append((b, lv @ pcell + tau[b] - tau[a], m))
        pairs[a] = rows
    for i in range(n):
        a = p2s.index(s2p[i])
        for b, vec, m in pairs[a]:
            target = spos[i] + vec
            d = (target[None] - spos) @ sinv
            d -= np.rint(d)
            j = int(np.argmin(np.linalg.norm(d @ scell, axis=1)))
            fc[i, j] += m
    return fc


def main():
    import phonopy

    parser = argparse.ArgumentParser()
    parser.add_argument("material")
    parser.add_argument("--tdep-bin", default=str(TDEP_BIN))
    parser.add_argument("--nproc", type=int, default=8)
    args = parser.parse_args()
    m = args.material
    base = ROOT / "validation_results" / m / "primary_cell"
    phonon = phonopy.load(base / "fd" / "phonopy_harmonic.yaml")
    sup = phonon.supercell
    rc2 = PRIMARY[m]["cutoff"]
    basis = symfc_fc2_basis(np.asarray(sup.positions), np.asarray(sup.cell),
                            list(sup.symbols), rc2)
    flat = basis.reshape(len(basis), -1)
    fit2 = Fitter(basis)

    def project(fc):
        coef, *_ = np.linalg.lstsq(flat.T, np.asarray(fc).ravel(), rcond=None)
        return np.einsum("k,kijab->ijab", coef, basis)

    def fit23(u, f):
        model, _ = fit_symfc_model(symfc_atoms(sup.numbers, sup.scaled_positions, sup.cell),
                                   u, f, [2, 3], {2: rc2, 3: FC3_CUTOFF}, batch_size=20)
        return project(model.force_constants[2])

    qmesh = mesh_points(10)
    mesh = lambda fc: np.array(mesh_frequencies(phonon, fc, qmesh))  # noqa: E731
    rms = lambda a, b: frequency_metrics(a, b)["rmse_thz"]  # noqa: E731
    sc_runs = sorted((base / "stdep_reference").glob("s*/history.npz"))
    ref = mesh(np.mean([np.load(p)["force_constants_last"][-1] for p in sc_runs], axis=0))

    data = {"canonical": [x[:N_TRAIN] for x in datasets(m)["canonical"]]}
    us, fs = [], []
    for path in sorted((base / "aimd").glob("traj*/trajectory.npz")):
        t = np.load(path)
        us.append(t["displacements_ang"][N_EQ:])
        fs.append(t["delta_forces_ev_ang"][N_EQ:])
    if us:
        data["aimd"] = [np.concatenate(us), np.concatenate(fs)]

    report = {"material": m, "rc2_ang": rc2, "rc3_ang": FC3_CUTOFF, "sets": {}}
    work0 = ROOT / "validation_results" / m / "tdep_runs"
    for name, (u, f) in data.items():
        models = {"ours_fc2": mesh(fit2(u, f))}
        models["tdep_fc2"] = mesh(run_tdep(work0 / f"{name}_fc2", phonon, u, f, rc2,
                                           tdep_bin=Path(args.tdep_bin), nproc=args.nproc))
        # TDEP limits rc2 to the Wigner--Seitz radius of the supercell; refit our
        # model at the cutoff TDEP actually used, so both span the same pairs.
        rc_eff = tdep_cutoff(work0 / f"{name}_fc2")
        report["tdep_effective_rc2_ang"] = rc_eff
        basis_eff = symfc_fc2_basis(np.asarray(sup.positions), np.asarray(sup.cell),
                                    list(sup.symbols), rc_eff + 0.01)
        models["ours_fc2_tdep_rc"] = mesh(Fitter(basis_eff)(u, f))
        # Same pairs and the full crystal space group, as in TDEP.
        models["ours_crystal_tdep_rc"] = mesh(Fitter(crystal_basis(phonon, rc_eff)[0])(u, f))
        models["tdep_fc2_noinv"] = mesh(run_tdep(work0 / f"{name}_fc2_noinv", phonon, u, f, rc2,
                                                 constraints=False, tdep_bin=Path(args.tdep_bin),
                                                 nproc=args.nproc))
        if name == "aimd":
            models["ours_fc23"] = mesh(fit23(u, f))
            models["tdep_fc23"] = mesh(run_tdep(work0 / f"{name}_fc23", phonon, u, f, rc2,
                                                rc3=FC3_CUTOFF, tdep_bin=Path(args.tdep_bin),
                                                nproc=args.nproc))
        names = list(models)
        entry = {"n_configurations": int(len(u)),
                 "vs_reference_thz": {k: rms(ref, v) for k, v in models.items()},
                 "mean_frequency_thz": {k: float(v.mean()) for k, v in models.items()},
                 "pairwise_thz": {f"{a}|{b}": rms(models[a], models[b])
                                  for i, a in enumerate(names) for b in names[i + 1:]}}
        report["sets"][name] = entry
        print(name, json.dumps(entry, indent=1), flush=True)
    out = ROOT / "validation_results" / m / "robustness" / "tdep_compare.json"
    out.write_text(json.dumps(report, indent=1) + "\n")


if __name__ == "__main__":
    main()
