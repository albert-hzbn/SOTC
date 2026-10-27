#!/usr/bin/env python3
"""Quantum vs classical vs non-Gaussian contributions to the AIMD-TDEP gap.

Effective FC2 models of the SOTC cell at T (same basis and regression):
* quantum Gaussian: the 40-snapshot iteration-0 canonical pool;
* classical Gaussian: ``dft_classical_gaussian.py`` (same harmonic trial,
  classical amplitudes);
* AIMD: the fit to all production steps of ``aimd_tdep.py``, with FC2
  alone and jointly with FC3 (``aimd_fc23``; see ``aimd_tdep_compare.py``).
Mesh RMSE between each pair and mean frequencies. The quantum--classical
difference is the effect of zero-point amplitudes; the classical-Gaussian--
AIMD difference is that of the non-Gaussian distribution (and of the trial,
which the AIMD does not use).

Writes validation_results/<m>/robustness/classical_gaussian.json.
"""

from __future__ import annotations

import json
from pathlib import Path
import sys

import numpy as np

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "codes"))
sys.path.insert(0, str(Path(__file__).resolve().parent))

from aimd_tdep_compare import FC3_CUTOFF, N_EQ  # noqa: E402
from analyze_designs import N_TRAIN, Fitter, datasets  # noqa: E402
from analyze_primary_archives import (  # noqa: E402
    PRIMARY, frequency_metrics, mesh_points, symfc_fc2_basis,
)
from surrogate_design_study import mesh_frequencies  # noqa: E402
from sotc.symfc_fit import fit_symfc_model, symfc_atoms  # noqa: E402


def main(material="al"):
    import phonopy

    base = ROOT / "validation_results" / material / "primary_cell"
    phonon = phonopy.load(base / "fd" / "phonopy_harmonic.yaml")
    harmonic = np.array(phonon.force_constants)
    sup = phonon.supercell
    basis = symfc_fc2_basis(np.asarray(sup.positions), np.asarray(sup.cell),
                            list(sup.symbols), PRIMARY[material]["cutoff"])
    fit = Fitter(basis)
    qmesh = mesh_points(10)
    mesh = lambda fc: np.array(mesh_frequencies(phonon, fc, qmesh))  # noqa: E731
    c = np.load(base / "classical_gaussian" / "configs.npz")
    u_aimd, f_aimd = [], []
    for path in sorted((base / "aimd").glob("traj*/trajectory.npz")):
        t = np.load(path)
        u_aimd.append(t["displacements_ang"][N_EQ:])
        f_aimd.append(t["delta_forces_ev_ang"][N_EQ:])
    u_q, f_q = [x[:N_TRAIN] for x in datasets(material)["canonical"]]
    model, _ = fit_symfc_model(symfc_atoms(sup.numbers, sup.scaled_positions, sup.cell),
                               np.concatenate(u_aimd), np.concatenate(f_aimd), [2, 3],
                               {2: PRIMARY[material]["cutoff"], 3: FC3_CUTOFF}, batch_size=20)
    flat = basis.reshape(len(basis), -1)
    coef, *_ = np.linalg.lstsq(flat.T, np.asarray(model.force_constants[2]).ravel(),
                               rcond=None)
    models = {
        "harmonic": mesh(harmonic),
        "quantum_gaussian": mesh(fit(u_q, f_q)),
        "classical_gaussian": mesh(fit(c["displacements_ang"], c["delta_forces_ev_ang"])),
        "aimd": mesh(fit(np.concatenate(u_aimd), np.concatenate(f_aimd))),
        "aimd_fc23": mesh(np.einsum("k,kijab->ijab", coef, basis)),
    }
    names = list(models)
    report = {"material": material,
              "u_rms_ang": {"quantum_gaussian": float(np.sqrt((u_q ** 2).sum(-1).mean())),
                            "classical_gaussian": float(c["u_rms_classical"]),
                            "aimd": float(np.sqrt((np.concatenate(u_aimd) ** 2).sum(-1).mean()))},
              "mean_frequency_thz": {k: float(np.mean(m)) for k, m in models.items()},
              "rmse_thz": {f"{a}|{b}": frequency_metrics(models[a], models[b])["rmse_thz"]
                           for i, a in enumerate(names) for b in names[i + 1:]}}
    out = ROOT / "validation_results" / material / "robustness" / "classical_gaussian.json"
    out.write_text(json.dumps(report, indent=1) + "\n")
    print(json.dumps(report, indent=1))


if __name__ == "__main__":
    main(sys.argv[1] if len(sys.argv) > 1 else "al")
