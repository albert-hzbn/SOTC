#!/usr/bin/env python3
"""Near-neighbour force constants against neutron Born--von Karman models.

Experimental 1NN and 2NN longitudinal and mean 1NN transverse force
constants (N/m) of Born--von Karman fits to inelastic neutron data, as
compiled by Kresch (Caltech PhD thesis, 2009, Tables 9.2, 14.2, 14.3), are
compared with the same invariants of our FC2 models: the converged
large-cell harmonic FD reference, the 40-snapshot iteration-1 effective FC2
of the SOTC cell at T_des, and SOTC-D with eight calls from the updated
trial and bootstrapped from the harmonic one. K_L = -r.Phi.r and the
transverse constants are the eigenvalues of -Phi perpendicular to the bond,
averaged over all pairs of the shell whose nearest periodic image is unique
within the cutoff.

Writes validation_results/summaries/experimental_bvk.json.
"""

from __future__ import annotations

import itertools
import json
from pathlib import Path
import sys

import numpy as np

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "codes"))
sys.path.insert(0, str(Path(__file__).resolve().parent))

from analyze_designs import N_TRAIN, Fitter, datasets  # noqa: E402
from analyze_primary_archives import PRIMARY, symfc_fc2_basis  # noqa: E402

EV_A2_TO_N_M = 16.0217663
# (T in K): (K1, K2, <K_T1>) in N/m; Kresch (2009), Tables 14.2 and 14.3.
EXPERIMENT = {
    "al": {10: (21.26, 2.45, -1.20), 300: (20.09, 2.22, -1.36)},
    "cu": {49: (27.91, -0.04, -1.35), 298: (27.75, 0.53, -1.50),
           973: (26.07, 1.55, -2.06), 1336: (25.37, 0.24, -1.86)},
    "ag": {296: (23.03, 0.06, -1.68)},
    "au": {296: (36.36, 4.04, -5.02)},
    "mo": {296: (40.07, 44.57, 4.73)},
    "w": {298: (59.90, 45.70, 3.20)},
}


def shell_constants(positions, cell, fc, shells):
    """Mean K_L for each shell distance and mean transverse for the first."""
    images = [np.asarray(n) @ cell for n in itertools.product(range(-2, 3), repeat=3)]
    out = {}
    for k, d in enumerate(shells):
        kl, kt = [], []
        for i in range(len(positions)):
            for j in range(len(positions)):
                vecs = [positions[j] + t - positions[i] for t in images]
                dist = np.linalg.norm(vecs, axis=1)
                order = np.argsort(dist)
                if abs(dist[order[0]] - d) > 0.05 or dist[order[1]] < 4.6:
                    continue
                r = vecs[order[0]] / dist[order[0]]
                phi = -np.asarray(fc[i, j]) * EV_A2_TO_N_M
                kl.append(r @ phi @ r)
                if k == 0:
                    # Orthonormal basis of the plane perpendicular to the bond.
                    perp = np.linalg.svd(np.eye(3) - np.outer(r, r))[0][:, :2]
                    kt.append(np.mean(np.linalg.eigvalsh(perp.T @ phi @ perp)))
        out[f"K{k + 1}"] = float(np.mean(kl)) if kl else None
        if k == 0:
            out["KT1"] = float(np.mean(kt)) if kt else None
        out[f"n_pairs_{k + 1}"] = len(kl)
    return out


def main():
    import phonopy

    report = {"source": "Kresch, Caltech PhD thesis (2009), Tables 14.2/14.3",
              "experiment": {m: {str(t): v for t, v in e.items()} for m, e in EXPERIMENT.items()},
              "models": {}}
    for m in EXPERIMENT:
        spec = PRIMARY[m]
        a = spec["a"]
        shells = ([a / np.sqrt(2), a] if spec["bravais"] == "fcc"
                  else [a * np.sqrt(3) / 2, a])
        vr = ROOT / "validation_results" / m
        large = phonopy.load(vr / "fd_references" / "production_large" / "phonopy_harmonic.yaml")
        phonon = phonopy.load(vr / "primary_cell" / "fd" / "phonopy_harmonic.yaml")
        sup = phonon.supercell
        pos, cell = np.asarray(sup.positions), np.asarray(sup.cell)
        fit = Fitter(symfc_fc2_basis(pos, cell, list(sup.symbols), spec["cutoff"]))
        data = datasets(m)
        lsup = large.supercell
        models = {"harmonic_converged": shell_constants(
            np.asarray(lsup.positions), np.asarray(lsup.cell),
            np.asarray(large.force_constants), shells),
            "harmonic_cell": shell_constants(pos, cell, np.asarray(phonon.force_constants), shells)}
        for name, key, n in (("effective_iter1", "canonical1", N_TRAIN),
                             ("sotcd_updated_8", "sotcd1", 8), ("boot_8", "boot", 8)):
            if data[key] is None:
                continue
            fc = fit(*[x[:n] for x in data[key]])
            models[name] = shell_constants(pos, cell, fc, shells)
        report["models"][m] = {"T_des": spec["T"], "a": a, **models}
        print(m, spec["T"], {k: {q: (round(x, 2) if isinstance(x, float) else x)
                                 for q, x in v.items()} for k, v in models.items()},
              "expt", EXPERIMENT[m], flush=True)
    out = ROOT / "validation_results" / "summaries" / "experimental_bvk.json"
    out.write_text(json.dumps(report, indent=1) + "\n")


if __name__ == "__main__":
    main()
