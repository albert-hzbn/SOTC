#!/usr/bin/env python3
"""Dynamically stabilized bcc Zr (1188 K, 54-atom cell): how many DFT force
calls do the self-consistent protocols need?

For every FC2 in the histories of SOTC-D-SC (``sotcd_sc_dft``), sTDEP
(``stdep_dft``, last and pooled), and SSCHA (``sscha_dft/conv_N*``):

* the lowest frequency on a 10^3 mesh (Gamma excluded) and the call count
  after which every later FC2 of the run is stable (lowest frequency above
  -0.05 THz), on the mesh and separately at H, N, P, and L 2/3;
* the mesh RMSE against the chunked sTDEP reference (``stdep_reference``,
  4 x 64 calls), pooled over all its iterations: the single-iteration fits
  still scatter by 0.1-0.3 THz (mesh RMSE between consecutive iterations,
  reported), the noise of the overdamped soft modes, so the pooled fit is
  the best estimate; and the calls to reach 0.1 and 0.05 THz (the first
  call count after which the error stays below the threshold);
* the frequencies at H, N, P, and L 2/3(1,1,1), compared with the inelastic
  neutron scattering energies of Heiming et al., PRB 43, 10948 (1991),
  Table III (measured at 895-960 C; H at 915 C = 1188 K).

Double well: the omega (L 2/3(1,1,1), longitudinal) and H displacement
patterns are commensurate with the cell. The generalized force along the
omega coordinate Q of every DFT configuration is fitted to
F = -a Q - c Q^2 - b Q^3 (the cubic invariant is allowed because 3q is a
reciprocal lattice vector); a < 0 means the bcc site is a local maximum
along Q, i.e. a double (here asymmetric) well.

Usage: zr_stabilization.py [zr|zr_64]. The 64-atom 4x4x4 primitive cell
contains N and P (where the harmonic trial is unstable) but not L 2/3(1,1,1),
so the omega double-well fit is made only in the 54-atom cell. A reference that
is itself unstable is rejected: the sampler gives trial-unstable modes zero
amplitude, so a chunked run whose first fit leaves such modes imaginary cannot
leave them (zr_64, 4 x 32 calls from the harmonic trial; the count per iteration
is reported). Runs are then scored against the mean pooled sTDEP fixed point of
the seeds that stay stable over the second half of the budget.

Close contacts: the shortest interatomic distance of every configuration of
the SOTC-D-SC and sTDEP runs, and the final pooled fit repeated without the
pairs that contain a distance below CONTACT_CUT (the nearest-neighbor distance
is 3.1 A). This refit is not self-consistent: the remaining configurations were
drawn from trials updated with the excluded pairs.
Writes validation_results/summaries/<material>/robustness/zr_stabilization.json.
"""

from __future__ import annotations

import json
from pathlib import Path
import sys

import numpy as np

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "codes"))
sys.path.insert(0, str(Path(__file__).resolve().parent))

from analyze_primary_archives import PRIMARY, mesh_points, symfc_fc2_basis  # noqa: E402
from surrogate_design_study import mesh_frequencies  # noqa: E402
from surrogate_self_consistent import fit_coefficients, to_fc  # noqa: E402

MEV_TO_THZ = 0.241799
# Heiming et al. (1991), Table III: energy (meV), uncertainty, temperature (C)
INS = {"L2/3": (7.5, 0.5, 960), "H": (19.8, 0.3, 915), "N_T1": (4.48, 0.11, 895),
       "N_T2": (14.0, 0.5, 960), "N_L": (19.5, 0.2, 960)}
# cubic reciprocal coordinates (units of 2 pi / a)
QPOINTS = {"H": (1.0, 0.0, 0.0), "N": (0.5, 0.5, 0.0), "P": (0.5, 0.5, 0.5),
           "L2/3": (2 / 3, 2 / 3, 2 / 3)}
STABLE_TOL = -0.05
THRESHOLDS = (0.1, 0.05)
CONTACT_CUT = 2.0  # Angstrom


def calls_to(calls, values, ok):
    """First call count after which ok(value) holds for the rest of the run."""
    good = [ok(v) for v in values]
    for k in range(len(good)):
        if all(good[k:]):
            return int(calls[k])
    return None


def main(material="zr"):
    import phonopy

    base = ROOT / "validation_results" / material / "primary_cell"
    phonon = phonopy.load(base / "fd" / "phonopy_harmonic.yaml")
    a = PRIMARY[material]["a"]
    prim = np.asarray(phonon.primitive.cell)
    reduced = {k: prim @ (np.asarray(q) / a) for k, q in QPOINTS.items()}
    qmesh = mesh_points(10)
    qmesh = qmesh[np.linalg.norm(qmesh, axis=1) > 1e-8]
    names = list(reduced)
    qall = np.concatenate([qmesh, np.array([reduced[k] for k in names])])

    def spectrum(fc):
        f = mesh_frequencies(phonon, fc, qall)
        pts = {k: np.sort(f[len(qmesh) + i]).tolist() for i, k in enumerate(names)}
        return np.sort(f[:len(qmesh)], axis=1), pts

    runs = {}
    for path in sorted((base / "sotcd_sc_dft").glob("s*/history.npz")):
        h = np.load(path)
        runs[f"sotcd_sc/{path.parent.name}"] = (h["force_calls"], h["force_constants"])
    for path in sorted((base / "stdep_dft").glob("s*/history.npz")):
        h = np.load(path)
        for kind in ("last", "pooled"):
            runs[f"stdep_{kind}/{path.parent.name}"] = (h["force_calls"],
                                                        h[f"force_constants_{kind}"])
    for path in sorted((base / "sscha_dft").glob("conv_N*_s*/history.npz")):
        h = np.load(path)
        runs[f"sscha/{path.parent.name}"] = (h["force_calls"], h["force_constants"])

    ref_path = base / "stdep_reference" / "s0" / "history.npz"
    reference = None
    report = {"material": material, "temperature_k": PRIMARY[material]["T"],
              "ins_heiming1991_thz": {k: [v[0] * MEV_TO_THZ, v[1] * MEV_TO_THZ, v[2]]
                                      for k, v in INS.items()},
              "stable_tolerance_thz": STABLE_TOL, "runs": {}}
    harm_mesh, harm_pts = spectrum(np.asarray(phonon.force_constants))
    report["harmonic"] = {"min_thz": float(harm_mesh.min()), "points_thz": harm_pts}
    if ref_path.exists():
        h = np.load(ref_path)
        reference = h["force_constants_pooled"][-1]
        ref_mesh, ref_pts = spectrum(reference)
        lasts = [spectrum(f)[0] for f in h["force_constants_last"]]
        report["reference"] = {
            "calls_per_iteration": int(h["force_calls"][0]),
            "iterations": len(h["force_calls"]),
            "min_thz": float(ref_mesh.min()), "points_thz": ref_pts,
            "last_iteration_change_thz": [float(np.sqrt(np.mean((x - y) ** 2)))
                                          for x, y in zip(lasts[1:], lasts[:-1])],
            "iteration_points_thz": [spectrum(f)[1] for f in h["force_constants_last"]],
            "trial_unstable_modes": [unstable_modes(phonon, f) for f in
                                     [np.asarray(phonon.force_constants),
                                      *h["force_constants_last"][:-1]]]}
        if ref_mesh.min() < STABLE_TOL:
            report["reference"]["rejected"] = "unstable: trial-unstable modes are never sampled"
            reference = None
    for name, (calls, fcs) in runs.items():
        mins, rmses, pts = [], [], []
        for fc in fcs:
            m, p = spectrum(fc)
            mins.append(float(m.min()))
            pts.append(p)
            if reference is not None:
                rmses.append(float(np.sqrt(np.mean((m - ref_mesh) ** 2))))
        entry = {"calls": [int(c) for c in calls], "min_thz": mins,
                 "calls_to_stable": calls_to(calls, mins, lambda v: v > STABLE_TOL),
                 "calls_to_stable_point": {
                     k: calls_to(calls, [p[k][0] for p in pts], lambda v: v > STABLE_TOL)
                     for k in names},
                 "soft_n_thz": [p["N"][0] for p in pts],
                 "final_points_thz": pts[-1]}
        if rmses:
            entry["rmse_thz"] = rmses
            for t in THRESHOLDS:
                entry[f"calls_to_{t}"] = calls_to(calls, rmses, lambda v, t=t: v < t)
        report["runs"][name] = entry
        print(name, entry["calls_to_stable"], entry.get("calls_to_0.1"),
              entry.get("calls_to_0.05"), f"final min {mins[-1]:.3f}",
              f"rmse {rmses[-1]:.3f}" if rmses else "", flush=True)
    if reference is None:
        score_against_consensus(report, runs, spectrum)
    report["close_contacts"] = close_contacts(base, phonon, material, spectrum)
    if material == "zr":
        report["double_well"] = omega_double_well(base, phonon, a)
    out = ROOT / "validation_results" / "summaries" / material / "robustness"
    out.mkdir(parents=True, exist_ok=True)
    (out / "zr_stabilization.json").write_text(json.dumps(report, indent=1) + "\n")


def unstable_modes(phonon, fc):
    """Imaginary supercell modes of fc, which the sampler displaces by zero."""
    masses = np.asarray(phonon.supercell.masses)
    n = len(masses)
    weighted = fc / np.sqrt(np.outer(masses, masses))[:, :, None, None]
    dyn = weighted.transpose(0, 2, 1, 3).reshape(3 * n, 3 * n)
    w = np.linalg.eigvalsh((dyn + dyn.T) / 2)
    return int((np.sign(w) * np.sqrt(np.abs(w)) * 15.633 < STABLE_TOL).sum())


def score_against_consensus(report, runs, spectrum):
    """Mesh RMSE against the mean fixed point of the stable pooled sTDEP seeds."""
    finals = []
    stable = []
    for name, (calls, fcs) in runs.items():
        half = [f for f, c in zip(fcs, calls) if c > calls[-1] / 2]
        if name.startswith("stdep_pooled/") and all(spectrum(f)[0].min() > STABLE_TOL
                                                    for f in half):
            finals.append(np.mean(half, axis=0))
            stable.append(name)
    consensus, consensus_pts = spectrum(np.mean(finals, axis=0))
    report["consensus"] = {"seeds": stable, "points_thz": consensus_pts,
                           "seed_spread_thz": float(np.sqrt(np.mean(
                               (spectrum(finals[0])[0] - spectrum(finals[-1])[0]) ** 2)))}
    for name, (calls, fcs) in runs.items():
        rmses = [float(np.sqrt(np.mean((spectrum(f)[0] - consensus) ** 2))) for f in fcs]
        entry = report["runs"][name]
        entry["rmse_vs_consensus_thz"] = rmses
        for t in THRESHOLDS:
            entry[f"calls_to_{t}"] = calls_to(calls, rmses, lambda v, t=t: v < t)
        print(name, "vs consensus", [round(x, 3) for x in rmses], flush=True)


def close_contacts(base, phonon, material, spectrum):
    """Shortest distances per configuration and the fit without close pairs."""
    sup = phonon.supercell
    eq, cell = np.asarray(sup.positions), np.asarray(sup.cell)
    inverse = np.linalg.inv(cell)
    with np.load(base / "fd" / "harmonic_reference.npz") as h:
        f0 = np.asarray(h["reference_forces_ev_ang"])
    harmonic = np.asarray(phonon.force_constants)
    fc2_basis = symfc_fc2_basis(eq, cell, list(sup.symbols), PRIMARY[material]["cutoff"])
    c_harm, *_ = np.linalg.lstsq(fc2_basis.reshape(len(fc2_basis), -1).T, harmonic.ravel(),
                                 rcond=None)

    def shortest(u):
        pos = eq + u
        frac = (pos[:, None] - pos[None]) @ inverse
        dist = np.linalg.norm((frac - np.rint(frac)) @ cell, axis=-1)
        np.fill_diagonal(dist, np.inf)
        return float(dist.min())

    out = {}
    for proto in ("sotcd_sc", "stdep"):
        for path in sorted((base / f"{proto}_dft").glob("s*/history.npz")):
            u = np.load(path)["displacements_ang"]
            forces = []
            for k in range(len(u)):
                with np.load(path.parent / f"state_{k:04d}" / "result.npz") as r:
                    f = np.asarray(r["forces_ev_ang"]) - f0
                forces.append(f - f.mean(axis=0))
            forces = np.asarray(forces)
            dmin = np.array([shortest(x) for x in u])
            # the +u and -u of a pair have consecutive indices
            keep = np.array([min(dmin[k], dmin[k ^ 1]) >= CONTACT_CUT for k in range(len(u))])
            mesh, pts = spectrum(to_fc(fc2_basis, fit_coefficients(
                fc2_basis, u[keep], forces[keep], c_harm)))
            out[f"{proto}/{path.parent.name}"] = {
                "shortest_ang": float(dmin.min()),
                "close_configurations": [int(k) for k in np.flatnonzero(dmin < CONTACT_CUT)],
                "n_kept": int(keep.sum()), "refit_min_thz": float(mesh.min()),
                "refit_points_thz": pts}
            print(proto, path.parent.name, f"shortest {dmin.min():.2f}",
                  np.flatnonzero(dmin < CONTACT_CUT).tolist(), "refit N",
                  np.round(pts["N"], 2).tolist(), flush=True)
    return out


def omega_double_well(base, phonon, a):
    """Force along the omega coordinate in all DFT configurations of the cell."""
    sup = phonon.supercell
    eq = np.asarray(sup.positions)
    masses = np.asarray(sup.masses)
    with np.load(base / "fd" / "harmonic_reference.npz") as h:
        f0 = np.asarray(h["reference_forces_ev_ang"])
    q = 2 * np.pi * np.array([2, 2, 2]) / (3 * a)
    e = np.ones(3) / np.sqrt(3)
    # omega pattern: longitudinal [111] displacement with phase cos(q.R), unit
    # norm in mass-weighted coordinates
    pattern = np.cos(eq @ q)[:, None] * e[None, :] * np.sqrt(masses)[:, None]
    pattern /= np.linalg.norm(pattern)
    us, fs = [], []
    for res in base.rglob("state_*/result.npz"):
        with np.load(res) as r:
            if "positions_ang" not in r.files:
                continue
            pos = np.asarray(r["positions_ang"])
            if pos.shape != eq.shape:
                continue
            d = pos - eq
            frac = d @ np.linalg.inv(np.asarray(sup.cell))
            d = (frac - np.rint(frac)) @ np.asarray(sup.cell)
            us.append(d - d.mean(axis=0))
            fs.append(np.asarray(r["forces_ev_ang"]) - f0)
    if not us:
        return None
    u, f = np.asarray(us), np.asarray(fs)
    Q = np.einsum("nia,ia->n", u * np.sqrt(masses)[None, :, None], pattern)
    F = np.einsum("nia,ia->n", f / np.sqrt(masses)[None, :, None], pattern)
    design = -np.stack([Q, Q ** 2, Q ** 3], axis=1)
    coef, *_ = np.linalg.lstsq(design, F, rcond=None)
    resid = F - design @ coef
    cov = np.linalg.inv(design.T @ design) * resid.var(ddof=3)
    lin, quad, cub = coef
    # potential V(Q) = a/2 Q^2 + c/3 Q^3 + b/4 Q^4 (mass-weighted, eV/(amu A^2))
    roots = np.roots([cub, quad, lin])
    minima = sorted(float(r.real) for r in roots if abs(r.imag) < 1e-12)
    pot = lambda x: lin / 2 * x ** 2 + quad / 3 * x ** 3 + cub / 4 * x ** 4  # noqa: E731
    out = {"n_configurations": int(len(Q)),
           "q_rms_amu_half_ang": float(np.sqrt(np.mean(Q ** 2))),
           "q_range": [float(Q.min()), float(Q.max())],
           "a_ev_amu_ang2": float(lin), "a_se": float(np.sqrt(cov[0, 0])),
           # sign(a) sqrt|a| as a frequency (1 eV/(amu A^2) -> 15.633 THz)
           "a_as_frequency_thz": float(np.sign(lin) * np.sqrt(abs(lin)) * 15.633),
           "c": float(quad), "c_se": float(np.sqrt(cov[1, 1])),
           "b": float(cub), "b_se": float(np.sqrt(cov[2, 2])),
           "stationary_points_Q": minima,
           "well_depth_ev": [float(pot(x)) for x in minima],
           "double_well": bool(lin < 0 and cub > 0)}
    print("omega coordinate:", out, flush=True)
    return out


if __name__ == "__main__":
    main(*sys.argv[1:])
