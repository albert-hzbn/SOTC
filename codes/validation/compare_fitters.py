#!/usr/bin/env python3
"""hiPhive/trainstation regression frameworks on the primary-cell DFT data.

The same displacement--force data (canonical subsets and SOTC-D prefixes of
the primary SOTC cell, Sec. ``run_primary_cell_controls.py``) are fitted with
the hiPhive cluster-space FC2 model (same pair cutoff as the symfc basis)
and the trainstation fit methods used in hiPhive workflows: ordinary least
squares, ridge, Bayesian ridge, automatic relevance determination (ARDR),
recursive feature elimination (RFE), orthogonal matching pursuit (OMP),
LASSO, adaptive LASSO and elastic net (cross-validated hyperparameters).  Scores: held-out canonical force RMSE and mesh RMSE
against an effective FC2 fitted (symfc) to the canonical snapshots disjoint
from the training data.
"""

from __future__ import annotations

import json
from pathlib import Path
import sys
import warnings

import numpy as np

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "codes"))
sys.path.insert(0, str(Path(__file__).resolve().parent))

from analyze_primary_archives import (  # noqa: E402
    PRIMARY, frequency_metrics, mesh_points, phonopy_frequencies, symfc_model,
)
from analyze_designs import min_image_distance  # noqa: E402
from analyze_robustness import N_TRAIN, canonical_dataset, sotcd_dataset  # noqa: E402

FIT_METHODS = ("least-squares", "ridge", "bayesian-ridge", "ardr", "rfe", "omp",
               "lasso", "adaptive-lasso", "elasticnet")
COUNTS = (2, 4, 8, 16)
N_SEEDS = 10


def main():
    import phonopy
    from ase import Atoms
    from hiphive import ClusterSpace, ForceConstantPotential, StructureContainer
    from trainstation import fit

    warnings.filterwarnings("ignore")
    for material in sys.argv[1:] or sorted(PRIMARY):
        spec = PRIMARY[material]
        base = ROOT / "validation_results" / material / "primary_cell"
        phonon = phonopy.load(base / "fd" / "phonopy_harmonic.yaml")
        sup = phonon.supercell
        pos, cell = np.asarray(sup.positions), np.asarray(sup.cell)
        symbols = list(sup.symbols)
        atoms = Atoms(symbols=symbols, positions=pos, cell=cell, pbc=True)
        prim = phonon.primitive
        prim_atoms = Atoms(symbols=list(prim.symbols), positions=np.asarray(prim.positions),
                           cell=np.asarray(prim.cell), pbc=True)
        # Alias-free pair cutoff: at most half the shortest image distance.
        rc = min(spec["cutoff"] + 0.01, min_image_distance(cell) / 2.0 - 0.01)
        cs = ClusterSpace(prim_atoms, [rc])
        uc, fc_ = canonical_dataset(material)
        us, fs = sotcd_dataset(material)
        u_all = np.concatenate([uc, us])
        f_all = np.concatenate([fc_, fs])
        container = StructureContainer(cs)
        for u, f in zip(u_all, f_all):
            a = atoms.copy()
            a.new_array("displacements", u)
            a.new_array("forces", f)
            container.add_structure(a)
        a_mat, y_vec = container.get_fit_data()
        rows = 3 * len(atoms)
        u_test, f_test = uc[N_TRAIN:], fc_[N_TRAIN:]
        qmesh = mesh_points(12)
        cache = {}

        def disjoint(indices):
            key = tuple(sorted(int(i) for i in indices))
            if key not in cache:
                model, _ = symfc_model(pos, cell, symbols, spec, uc[list(key)],
                                       fc_[list(key)], spec["cutoff"])
                cache[key] = phonopy_frequencies(model, qmesh)
            return cache[key]

        def evaluate(structures, method, reference):
            sel = np.concatenate([np.arange(s * rows, (s + 1) * rows) for s in structures])
            try:
                result = fit(a_mat[sel], y_vec[sel], method)
            except Exception as error:  # noqa: BLE001
                return {"error": repr(error)[:120]}
            fcp = ForceConstantPotential(cs, result["parameters"])
            fc2 = fcp.get_force_constants(atoms).get_fc_array(order=2)
            phonon.force_constants = fc2
            freq = phonopy_frequencies(phonon, qmesh)
            pred = -np.einsum("ijab,sjb->sia", fc2, u_test)
            return {
                "held_out_force_rmse": float(np.sqrt(np.mean((pred - f_test) ** 2))),
                "mesh_vs_disjoint": frequency_metrics(reference, freq)["rmse_thz"],
                "unstable_fraction": float(np.mean(freq < -0.05)),
            }

        report = {"material": material, "n_parameters": int(cs.n_dofs), "cutoff_ang": rc,
                  "counts": list(COUNTS), "fit_methods": list(FIT_METHODS),
                  "canonical": {}, "sotcd": {}}
        for method in FIT_METHODS:
            report["canonical"][method] = {}
            report["sotcd"][method] = {}
            for c in COUNTS:
                trials = []
                for seed in range(N_SEEDS):
                    rng = np.random.default_rng(7000 + 97 * c + seed)
                    chosen = rng.choice(N_TRAIN, size=c, replace=False)
                    ref = disjoint(np.setdiff1d(np.arange(N_TRAIN), chosen))
                    trials.append(evaluate(list(chosen), method, ref))
                report["canonical"][method][str(c)] = trials
                design = list(range(len(uc), len(uc) + c))
                refs = [disjoint(np.random.default_rng(31 * c + s).choice(
                    N_TRAIN, size=N_TRAIN - c, replace=False)) for s in range(N_SEEDS)]
                scores = [evaluate(design, method, r) for r in refs[:3]]
                ok = [s for s in scores if "error" not in s]
                report["sotcd"][method][str(c)] = ({
                    k: float(np.median([s[k] for s in ok])) for k in ok[0]
                } if ok else scores[0])
                med = np.median([t["held_out_force_rmse"] for t in trials if "error" not in t] or [np.nan])
                print(f"{material} {method:15s} n={c:2d} canonical F {med:.4f} "
                      f"sotcd F {report['sotcd'][method][str(c)].get('held_out_force_rmse', np.nan):.4f}",
                      flush=True)
        out = base.parent / "robustness"
        out.mkdir(parents=True, exist_ok=True)
        (out / "fitter_comparison.json").write_text(json.dumps(report, indent=1) + "\n")


if __name__ == "__main__":
    main()
