#!/usr/bin/env python3
"""Thermal expansion of Al from the ``dft_qha.py`` volume grid.

At every design temperature T, the free energy per atom on the volume grid is

    F(V) = E_st(V) + F_h[Phi(V, T)](T) + dU(V, T),

with E_st the third-order Birch--Murnaghan fit of the primitive-cell PBE
energies (``run_fd_references.py``, 16^3 k mesh), F_h the quantum harmonic
free energy of the self-consistent SOTC-D FC2 (Fourier-interpolated from
the 32-atom cell, 20^3 mesh), and dU = <E - E(0) - u.Phi.u/2> the
anharmonic energy term of the SCHA free energy, averaged over the second
half of the designed configurations and fitted linearly in V. Results are
given after 8 and 16 force calls per volume and temperature.  Because the SCHA free energy
is stationary in Phi, the entropy is the harmonic entropy S_h[Phi](T), and

    alpha = (dS/dV)_T / (3 B_T),   B_T = V d^2F/dV^2,

both at the volume minimizing F.  The QHA with the harmonic FD FC2 (dU = 0), in
the same cell and in the converged 125-atom cell (``dft_qha_large.py``), is
evaluated with the same formula. F(V) is fitted by a cubic and S(V) by a
quadratic polynomial over the seven volumes.  The spread over dU (fitted,
raw, omitted) is reported as a sensitivity.

Writes validation_results/<m>/robustness/qha_thermal_expansion[_common].json.

Usage: dft_qha_analyze.py <material> [qha|qha_common]
"""

from __future__ import annotations

import json
from pathlib import Path
import sys

import numpy as np

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "codes"))
sys.path.insert(0, str(Path(__file__).resolve().parent))

from dft_qha import scaled_phonon  # noqa: E402

EV_PER_KJMOL = 1.0 / 96.48533212
# Averages of measured alpha (1e-6/K) and a (A), Nakashima, arXiv:2002.01562, Table 2.
EXPERIMENT_ALPHA = {50: (3.7, 0.4), 273: (22.7, 0.4), 298: (23.2, 0.2), 400: (25.1, 0.2),
                    923: (36.0, 1.0)}
EXPERIMENT_A = {0: 4.0317, 298: 4.0497, 400: 4.059, 923: 4.1237}
CALLS = (8, 16)


def birch_murnaghan(v, e0, v0, b0, bp):
    eta = (v0 / v) ** (2.0 / 3.0)
    return e0 + 9.0 * v0 * b0 / 16.0 * ((eta - 1.0) ** 3 * bp + (eta - 1.0) ** 2 * (6.0 - 4.0 * eta))


def main(material="al", grid_dir="qha"):
    import phonopy
    from scipy.optimize import curve_fit

    base = ROOT / "validation_results" / material
    eos = json.loads((base / "fd_references" / "eos_summary.json").read_text())
    vols, ens = np.asarray(eos["volumes_ang3"]), np.asarray(eos["energies_ev"])
    p0 = (ens.min(), vols[np.argmin(ens)], 0.5, 4.0)
    popt, _ = curve_fit(birch_murnaghan, vols, ens, p0=p0, maxfev=20000)
    e_static = lambda v: birch_murnaghan(v, *popt)  # noqa: E731

    grid = []
    for d in sorted((base / "primary_cell" / grid_dir).glob("a*")):
        temps = sorted(float(p.stem[1:]) for p in d.glob("T*.npz"))
        if not (base / "primary_cell" / "qha" / d.name / "fd.npz").exists() or not temps:
            continue
        fd = np.load(base / "primary_cell" / "qha" / d.name / "fd.npz")
        a = float(fd["a_ang"])
        n = len(fd["positions_ang"])
        entry = {"a": a, "volume": a ** 3 / 4.0, "fd": np.asarray(fd["force_constants"]),
                 "temps": np.asarray(temps), "phi": {}, "du": {}}
        for t in temps:
            r = np.load(d / f"T{t:.0f}.npz")
            n_calls = 2 * len(r["force_constants"])
            for c in CALLS:
                if c > n_calls:
                    continue
                # FC2 after c calls; energy term over the second half of them.
                phi = r["force_constants"][c // 2 - 1]
                u = r["displacements_ang"][c // 2:c]
                de = r["delta_energies_ev"][c // 2:c]
                harm = 0.5 * np.einsum("sia,ijab,sjb->s", u, phi, u)
                entry["phi"].setdefault(c, []).append(phi)
                entry["du"].setdefault(c, []).append(float(np.mean(de - harm)) / n)
        grid.append(entry)
    grid.sort(key=lambda g: g["a"])
    design_t = grid[0]["temps"]
    assert all(np.array_equal(g["temps"], design_t) for g in grid)
    calls = [c for c in CALLS if all(len(g["phi"].get(c, [])) == len(design_t) for g in grid)]
    primary = max(calls)

    def thermal(phonon, fc, t):
        phonon.force_constants = fc
        phonon.run_mesh([20, 20, 20])
        phonon.run_thermal_properties(temperatures=[t])
        props = phonon.get_thermal_properties_dict()
        return (float(props["free_energy"][0]) * EV_PER_KJMOL,
                float(props["entropy"][0]) * EV_PER_KJMOL / 1000.0)

    def min_frequency(phonon, fc):
        # Full (unsymmetrized) mesh, the convention of the other minimum frequencies.
        phonon.force_constants = fc
        phonon.run_mesh([20, 20, 20], is_mesh_symmetry=False)
        return float(np.min(phonon.get_mesh_dict()["frequencies"]))

    phonons = [scaled_phonon(material, g["a"]) for g in grid]
    large = []
    for g in grid:
        path = (base / "fd_references" / "qha_large" / f"a{g['a']:.4f}" / "phonopy_harmonic.yaml")
        large.append(phonopy.load(path) if path.exists() else None)
    if any(ph is None for ph in large):
        large = None
    v = np.asarray([g["volume"] for g in grid])
    e_st = np.asarray([e_static(x) for x in v])

    def expansion(f, s):
        # Cubic polynomial F(V) and quadratic S(V) over the seven volumes.
        c = np.polyfit(v, f, 3)
        roots = np.roots(np.polyder(c))
        roots = roots[np.isreal(roots)].real
        roots = roots[(np.polyval(np.polyder(c, 2), roots) > 0)]
        v0 = float(roots[np.argmin(abs(roots - v.mean()))])
        bulk = v0 * np.polyval(np.polyder(c, 2), v0)
        ds = np.polyval(np.polyder(np.polyfit(v, s, 2)), v0)
        return {"a_ang": (4.0 * v0) ** (1.0 / 3.0), "B_gpa": bulk * 160.21766,
                "alpha_1e6": 1e6 * ds / (3.0 * bulk),
                "inside_grid": bool(v.min() <= v0 <= v.max())}

    report = {"material": material, "grid": grid_dir,
              "design_temperatures_k": design_t.tolist(),
              "lattice_grid_ang": [g["a"] for g in grid], "calls": calls,
              "primary_calls": primary,
              "eos_fit": {"E0": popt[0], "V0": popt[1], "B0_gpa": popt[2] * 160.21766,
                          "Bp": popt[3]},
              "experiment_alpha_1e6": {str(k): x for k, x in EXPERIMENT_ALPHA.items()},
              "experiment_a_ang": {str(k): x for k, x in EXPERIMENT_A.items()},
              "min_frequency_thz": {
                  "harmonic_cell": [min_frequency(ph, g["fd"]) for g, ph in zip(grid, phonons)],
                  "sotcd_sc": {f"{t:.0f}": [min_frequency(ph, g["phi"][primary][it])
                                            for g, ph in zip(grid, phonons)]
                               for it, t in enumerate(design_t)}},
              "temperatures": {}}
    for it, t in enumerate(design_t):
        fq, sq = map(np.asarray, zip(*[thermal(ph, g["fd"], t) for g, ph in zip(grid, phonons)]))
        entry = {"harmonic_qha_cell": expansion(e_st + fq, sq)}
        if large is not None:
            fl, sl = map(np.asarray, zip(*[thermal(ph, np.asarray(ph.force_constants), t)
                                           for ph in large]))
            entry["harmonic_qha_converged"] = expansion(e_st + fl, sl)
        for c in calls:
            fh, sh = map(np.asarray, zip(*[thermal(ph, g["phi"][c][it], t)
                                           for g, ph in zip(grid, phonons)]))
            du = np.asarray([g["du"][c][it] for g in grid])
            du_fit = np.polyval(np.polyfit(v, du, 1), v)
            entry[f"du_mev_per_atom_{c}"] = (1000 * du).tolist()
            entry[f"sotcd_sc_{c}"] = expansion(e_st + fh + du_fit, sh)
            entry[f"sotcd_sc_{c}_raw_du"] = expansion(e_st + fh + du, sh)
            entry[f"sotcd_sc_{c}_no_du"] = expansion(e_st + fh, sh)
        report["temperatures"][f"{t:.0f}"] = entry
        print(f"T={t:.0f}: dU({primary})={np.round(entry[f'du_mev_per_atom_{primary}'], 2)} meV/atom")
        for key, e in entry.items():
            if isinstance(e, dict):
                print(f"   {key:24s} a={e['a_ang']:.4f} B={e['B_gpa']:.1f} GPa "
                      f"alpha={e['alpha_1e6']:.1f}e-6/K inside={e['inside_grid']}")
    suffix = "" if grid_dir == "qha" else "_" + grid_dir.split("_", 1)[1]
    out = base / "robustness" / f"qha_thermal_expansion{suffix}.json"
    out.write_text(json.dumps(report, indent=1) + "\n")
    print("min frequency:", {k: np.round(v_, 2).tolist() if isinstance(v_, list) else
                             {t: np.round(x, 2).tolist() for t, x in v_.items()}
                             for k, v_ in report["min_frequency_thz"].items()})
    print("experiment alpha:", EXPERIMENT_ALPHA, "a:", EXPERIMENT_A)


if __name__ == "__main__":
    main(*sys.argv[1:3])
