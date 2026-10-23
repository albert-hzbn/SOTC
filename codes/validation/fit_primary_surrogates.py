#!/usr/bin/env python3
"""FC2+FC3+FC4 surrogates of diamond Si and hcp Mg from primary-cell DFT data.

All same-cell QE force calculations of ``run_primary_cell_controls.py``
(canonical and SOTC-D sets of both trial levels, bootstrapped SOTC-D, rattle,
ZG, sTDEP) are pooled; the 8+8 canonical test snapshots of the two trial
levels are held out.  The joint symmetry- and sum-rule-constrained fit is
written in the layout of ``run_real_anharmonic_ifcs.py`` so that the
surrogate studies (``surrogate_self_consistent.py``, ``surrogate_sscha.py``)
can use it unchanged.
"""

from __future__ import annotations

import json
from pathlib import Path
import shutil
import sys

import numpy as np

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "codes"))
sys.path.insert(0, str(Path(__file__).resolve().parent))

from analyze_designs import N_TRAIN, datasets  # noqa: E402
from run_real_anharmonic_ifcs import write_compact_hdf5  # noqa: E402
from sotc.symfc_fit import fit_symfc_model, forces_from_compact_fcs, symfc_atoms  # noqa: E402

CUTOFFS = {"si": {2: 5.0, 3: 4.0, 4: 2.5}, "mg": {2: 4.6, 3: 3.5, 4: 3.3}}


def fit(material):
    import phonopy

    cutoffs = CUTOFFS[material]
    base = ROOT / "validation_results" / material / "primary_cell"
    phonon = phonopy.load(base / "fd" / "phonopy_harmonic.yaml")
    sup = phonon.supercell
    data = {k: v for k, v in datasets(material).items() if v is not None}
    train_u, train_f, test_u, test_f = [], [], [], []
    for name, (u, f) in data.items():
        if name in ("canonical", "canonical1"):
            train_u.append(u[:N_TRAIN]), train_f.append(f[:N_TRAIN])
            test_u.append(u[N_TRAIN:]), test_f.append(f[N_TRAIN:])
        elif name == "boot":
            train_u.append(u[2:]), train_f.append(f[2:])  # first pair is in sotcd
        else:
            train_u.append(u), train_f.append(f)
    u, f = np.concatenate(train_u), np.concatenate(train_f)
    ut, ft = np.concatenate(test_u), np.concatenate(test_f)
    atoms = symfc_atoms(sup.numbers, sup.scaled_positions, sup.cell)
    report = {"material": material, "cutoffs_ang": cutoffs, "n_train": len(u),
              "n_test": len(ut), "held_out": {}}
    for orders in ([2], [2, 3, 4]):
        model, design = fit_symfc_model(atoms, u, f, orders, cutoffs, batch_size=20,
                                        full_force_constants=False)
        basis = model.basis_set[2]
        pred = forces_from_compact_fcs(ut, model.force_constants,
                                       basis.translation_permutations, basis.p2s_map)
        rmse = float(np.sqrt(np.mean((pred - ft) ** 2)))
        r2 = float(1.0 - np.sum((pred - ft) ** 2) / np.sum((ft - ft.mean()) ** 2))
        report["held_out"]["".join(map(str, orders))] = {
            "force_rmse_ev_ang": rmse, "r2": r2,
            "n_parameters": {str(k): int(v) for k, v in
                             (design.get("n_parameters_by_order") or {}).items()}}
        print(material, orders, f"held-out RMSE {rmse:.4f} eV/A, R2 {r2:.4f}", flush=True)
    out = ROOT / "validation_results" / material / "real_ifc34"
    final = out / "final_force_constants"
    final.mkdir(parents=True, exist_ok=True)
    shutil.copy(base / "fd" / "phonopy_harmonic.yaml", out / "phonopy_harmonic.yaml")
    np.save(final / "p2s_map.npy", np.asarray(basis.p2s_map))
    np.save(final / "translation_permutations.npy", np.asarray(basis.translation_permutations))
    write_compact_hdf5(final / "fc234_compact.hdf5", model.force_constants,
                       basis.p2s_map, cutoffs)
    (out / "surrogate_fit_summary.json").write_text(json.dumps(report, indent=1) + "\n")


def main():
    for material in sys.argv[1:] or list(CUTOFFS):
        fit(material)


if __name__ == "__main__":
    main()
