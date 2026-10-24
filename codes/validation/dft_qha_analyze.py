#!/usr/bin/env python3
"""Thermal expansion of Al from the ``dft_qha.py`` volume grid.

At every design temperature T, the free energy per atom on the volume grid is

    F(V) = E_st(V) + F_h[Phi(V, T)](T) + dU(V, T),

with E_st the third-order Birch--Murnaghan fit of the primitive-cell PBE
energies (``run_fd_references.py``, 16^3 k mesh), F_h the quantum harmonic
free energy of the self-consistent SOTC-D FC2 (Fourier-interpolated from
the 32-atom cell, 20^3 mesh), and dU = <E - E(0) - u.Phi.u/2> the
anharmonic energy term of the SCHA free energy, averaged over the last two
designed +/- pairs and fitted linearly in V.  Because the SCHA free energy
is stationary in Phi, the entropy is the harmonic entropy S_h[Phi](T), and

    alpha = (dS/dV)_T / (3 B_T),   B_T = V d^2F/dV^2,

both at the volume minimizing F.  The harmonic QHA in the same cell (FD FC2,
dU = 0) is evaluated with the same formula.  The spread over dU (fitted,
raw, omitted) is reported as a sensitivity.

Writes validation_results/<m>/robustness/qha_thermal_expansion.json.
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


def birch_murnaghan(v, e0, v0, b0, bp):
    eta = (v0 / v) ** (2.0 / 3.0)
    return e0 + 9.0 * v0 * b0 / 16.0 * ((eta - 1.0) ** 3 * bp + (eta - 1.0) ** 2 * (6.0 - 4.0 * eta))


def main(material="al"):
    from scipy.optimize import curve_fit

    base = ROOT / "validation_results" / material
    eos = json.loads((base / "fd_references" / "eos_summary.json").read_text())
    vols, ens = np.asarray(eos["volumes_ang3"]), np.asarray(eos["energies_ev"])
    p0 = (ens.min(), vols[np.argmin(ens)], 0.5, 4.0)
    popt, _ = curve_fit(birch_murnaghan, vols, ens, p0=p0, maxfev=20000)
    e_static = lambda v: birch_murnaghan(v, *popt)  # noqa: E731

    grid = []
    for d in sorted((base / "primary_cell" / "qha").glob("a*")):
        temps = sorted(float(p.stem[1:]) for p in d.glob("T*.npz"))
        if not (d / "fd.npz").exists() or not temps:
            continue
        fd = np.load(d / "fd.npz")
        a = float(fd["a_ang"])
        n = len(fd["positions_ang"])
        phis, dus = [], []
        for t in temps:
            r = np.load(d / f"T{t:.0f}.npz")
            phi = r["force_constants"][-1]
            u, de = r["displacements_ang"][-4:], r["delta_energies_ev"][-4:]
            harm = 0.5 * np.einsum("sia,ijab,sjb->s", u, phi, u)
            phis.append(phi)
            dus.append(float(np.mean(de - harm)) / n)
        grid.append({"a": a, "volume": a ** 3 / 4.0, "fd": np.asarray(fd["force_constants"]),
                     "temps": np.asarray(temps), "phi": np.asarray(phis), "du": np.asarray(dus)})
    grid.sort(key=lambda g: g["a"])
    design_t = grid[0]["temps"]
    assert all(np.array_equal(g["temps"], design_t) for g in grid)

    def thermal(phonon, fc, t):
        phonon.force_constants = fc
        phonon.run_mesh([20, 20, 20])
        phonon.run_thermal_properties(temperatures=[t])
        props = phonon.get_thermal_properties_dict()
        return (float(props["free_energy"][0]) * EV_PER_KJMOL,
                float(props["entropy"][0]) * EV_PER_KJMOL / 1000.0)

    phonons = [scaled_phonon(material, g["a"]) for g in grid]
    v = np.asarray([g["volume"] for g in grid])
    e_st = np.asarray([e_static(x) for x in v])

    def expansion(f, s):
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

    report = {"material": material, "design_temperatures_k": design_t.tolist(),
              "lattice_grid_ang": [g["a"] for g in grid],
              "eos_fit": {"E0": popt[0], "V0": popt[1], "B0_gpa": popt[2] * 160.21766,
                          "Bp": popt[3]},
              "experiment_alpha_1e6": {str(k): x for k, x in EXPERIMENT_ALPHA.items()},
              "experiment_a_ang": {str(k): x for k, x in EXPERIMENT_A.items()},
              "temperatures": {}}
    for it, t in enumerate(design_t):
        fh, sh, fq, sq = [], [], [], []
        for g, ph in zip(grid, phonons):
            f_, s_ = thermal(ph, g["phi"][it], t)
            fh.append(f_)
            sh.append(s_)
            f_, s_ = thermal(ph, g["fd"], t)
            fq.append(f_)
            sq.append(s_)
        du = np.asarray([g["du"][it] for g in grid])
        du_fit = np.polyval(np.polyfit(v, du, 1), v)
        fh, sh, fq, sq = map(np.asarray, (fh, sh, fq, sq))
        entry = {"du_mev_per_atom": (1000 * du).tolist(),
                 "sotcd_sc": expansion(e_st + fh + du_fit, sh),
                 "sotcd_sc_raw_du": expansion(e_st + fh + du, sh),
                 "sotcd_sc_no_du": expansion(e_st + fh, sh),
                 "harmonic_qha": expansion(e_st + fq, sq)}
        report["temperatures"][f"{t:.0f}"] = entry
        print(f"T={t:.0f}: dU={np.round(1000 * du, 2)} meV/atom")
        for key in ("sotcd_sc", "sotcd_sc_raw_du", "sotcd_sc_no_du", "harmonic_qha"):
            e = entry[key]
            print(f"   {key:16s} a={e['a_ang']:.4f} B={e['B_gpa']:.1f} GPa "
                  f"alpha={e['alpha_1e6']:.1f}e-6/K inside={e['inside_grid']}")
    out = base / "robustness" / "qha_thermal_expansion.json"
    out.write_text(json.dumps(report, indent=1) + "\n")
    print("experiment alpha:", EXPERIMENT_ALPHA, "a:", EXPERIMENT_A)


if __name__ == "__main__":
    main(sys.argv[1] if len(sys.argv) > 1 else "al")
