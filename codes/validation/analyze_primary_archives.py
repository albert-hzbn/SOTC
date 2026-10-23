#!/usr/bin/env python3
"""Identifiability and fitting-prior audit of the nine primary SOTC archives.

The primary runs archived every QE input/output (``examples/sotc_*_qe_run/
qe_scratch/snap_NNNN``).  This script reconstructs the per-iteration
force--displacement data from those files, verifies that the published
effective-FC2 bands are reproduced, and then refits the same data with

* the published prior (ridge 1e-3, axial central-force projection),
* a ridge sweep, no projection, and cutoff brackets, and
* a space-group/sum-rule-constrained symfc FC2 basis without ridge.

Every model is evaluated against the conventional finite-displacement QE
references produced by ``run_fd_references.py`` on the archived band path
and on a uniform q mesh.  No new DFT calculation is performed here.
"""

from __future__ import annotations

import json
from pathlib import Path
import sys

import numpy as np

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "codes"))

from sotc.ifc_extractor import IFCExtractor  # noqa: E402
from sotc.phonons import PhononCalculator  # noqa: E402
from sotc.symfc_fit import symfc_atoms  # noqa: E402
from sotc.validation import phonopy_atoms  # noqa: E402


PRIMARY = {
    "al": dict(a=4.05, bravais="fcc", elements=["Al"], masses=[26.9815385],
               T=300.0, cutoff=4.5, n_iter=2, n_ens=5),
    "ag": dict(a=4.12, bravais="fcc", elements=["Ag"], masses=[107.8682],
               T=1000.0, cutoff=4.5, n_iter=4, n_ens=5),
    "au": dict(a=4.13, bravais="fcc", elements=["Au"], masses=[196.966569],
               T=1200.0, cutoff=4.5, n_iter=2, n_ens=1),
    "cu": dict(a=3.67, bravais="fcc", elements=["Cu"], masses=[63.546],
               T=1200.0, cutoff=4.5, n_iter=4, n_ens=8),
    "mo": dict(a=3.16, bravais="bcc", elements=["Mo"], masses=[95.95],
               T=1500.0, cutoff=5.0, n_iter=2, n_ens=1),
    "w": dict(a=3.18, bravais="bcc", elements=["W"], masses=[183.84],
              T=1500.0, cutoff=5.0, n_iter=4, n_ens=3),
    "nacl": dict(a=5.73, bravais="fcc", elements=["Na", "Cl"],
                 masses=[22.98976928, 35.453], T=700.0, cutoff=4.3,
                 n_iter=5, n_ens=3),
    "mgo": dict(a=4.26, bravais="fcc", elements=["Mg", "O"],
                masses=[24.305, 15.999], T=1500.0, cutoff=3.2,
                n_iter=4, n_ens=3),
    "pbte": dict(a=6.51, bravais="fcc", elements=["Pb", "Te"],
                 masses=[207.2, 127.60], T=700.0, cutoff=5.0,
                 n_iter=4, n_ens=5),
    # New structure classes without an archived SOTC run: the primary cell is
    # built from the supercell matrix (64 atoms each) and only the same-cell
    # controls and designs are evaluated.
    "si": dict(a=5.431, bravais="diamond", elements=["Si"], masses=[28.0855],
               T=600.0, cutoff=5.0, n_iter=0, n_ens=0,
               supercell_matrix=[[-2, 2, 2], [2, -2, 2], [2, 2, -2]]),
    "mg": dict(a=3.209, c=5.211, bravais="hcp", elements=["Mg"], masses=[24.305],
               T=600.0, cutoff=4.6, n_iter=0, n_ens=0,
               supercell_matrix=[[4, 0, 0], [0, 4, 0], [0, 0, 2]]),
}
RIDGES = (0.0, 1.0e-4, 1.0e-3, 1.0e-2, 1.0e-1)
MESH = 12
THZ_TO_K = 47.99243073366221
KB_EV = 8.617333262e-5
H_EV_S = 4.135667696e-15


def primitive(spec):
    a = spec["a"]
    if spec["bravais"] == "diamond":
        cell = 0.5 * a * np.array([[0., 1., 1.], [1., 0., 1.], [1., 1., 0.]])
        return cell, np.array([[0., 0., 0.], [a / 4., a / 4., a / 4.]])
    if spec["bravais"] == "hcp":
        c = spec["c"]
        cell = np.array([[a, 0., 0.], [-a / 2., a * np.sqrt(3.) / 2., 0.], [0., 0., c]])
        frac = np.array([[1. / 3., 2. / 3., 0.25], [2. / 3., 1. / 3., 0.75]])
        return cell, frac @ cell
    if spec["bravais"] == "fcc":
        cell = 0.5 * a * np.array([[0., 1., 1.], [1., 0., 1.], [1., 1., 0.]])
    else:
        cell = 0.5 * a * np.array([[-1., 1., 1.], [1., -1., 1.], [1., 1., -1.]])
    pos = np.zeros((1, 3)) if len(spec["elements"]) == 1 else np.array(
        [[0., 0., 0.], [a / 2., 0., 0.]]
    )
    return cell, pos


def read_snapshots(material, spec):
    from ase.io import read

    scratch = ROOT / "examples" / f"sotc_{material}_qe_run" / "qe_scratch"
    outputs = sorted(scratch.glob("snap_*/espresso.pwo"))
    n_force = spec["n_iter"] * (spec["n_ens"] + 1)
    # Older aborted runs left stale, lower-mtime snapshot directories with
    # higher indices; the production run is the newest n_force outputs.
    newest = sorted(outputs, key=lambda p: p.stat().st_mtime)[-n_force:]
    newest = sorted(newest, key=lambda p: int(p.parent.name.split("_")[1]))
    indices = [int(p.parent.name.split("_")[1]) for p in newest]
    if indices != list(range(n_force)):
        raise RuntimeError(f"{material}: unexpected snapshot indices {indices}")
    frames = []
    for path in newest:
        atoms = read(path, format="espresso-out", index=-1)
        frames.append({
            "positions": atoms.get_positions(),
            "cell": np.asarray(atoms.cell[:]),
            "forces": atoms.get_forces(),
            "symbols": atoms.get_chemical_symbols(),
            "path": str(path.relative_to(ROOT)),
        })
    return frames


def minimum_image(delta, cell):
    frac = delta @ np.linalg.inv(cell)
    frac -= np.round(frac)
    return frac @ cell


def group_iterations(frames, spec):
    """Split the sequential archive into [reference, displaced...] blocks."""
    block = spec["n_ens"] + 1
    reference = frames[0]
    cell = reference["cell"]
    iterations = []
    for start in range(0, len(frames), block):
        chunk = frames[start:start + block]
        amplitudes = [
            float(np.max(np.abs(minimum_image(
                f["positions"] - reference["positions"], cell
            )))) for f in chunk
        ]
        ref_local = int(np.argmin(amplitudes))
        if amplitudes[ref_local] > 1.0e-6:
            raise RuntimeError("iteration block lacks an undisplaced reference")
        f0 = chunk[ref_local]["forces"]
        u, df = [], []
        for k, frame in enumerate(chunk):
            if k == ref_local:
                continue
            u.append(minimum_image(frame["positions"] - reference["positions"], cell))
            df.append(frame["forces"] - f0)
        iterations.append({
            "u": np.asarray(u), "df": np.asarray(df),
            "reference_rms_force": float(np.sqrt(np.mean(f0 ** 2))),
            "reference_index": start + ref_local,
        })
    return reference["positions"], cell, reference["symbols"], iterations


def snap_to_lattice(pos, cell, spec):
    """Replace printed-precision QE geometry by the exact ideal supercell."""
    pcell, ppos = primitive(spec)
    hnf = np.rint(cell @ np.linalg.inv(pcell))
    exact_cell = hnf @ pcell
    snapped = []
    for r in pos:
        best = None
        for b in ppos:
            frac = (r - b) @ np.linalg.inv(pcell)
            candidate = np.rint(frac) @ pcell + b
            if best is None or np.linalg.norm(candidate - r) < np.linalg.norm(best - r):
                best = candidate
        snapped.append(best)
    snapped = np.asarray(snapped)
    if np.max(np.abs(snapped - pos)) > 1.0e-3:
        raise RuntimeError("reference geometry is not an ideal lattice")
    return snapped, exact_cell


def thermo_from_mesh(freqs_thz, T, n_atoms_prim):
    """Harmonic C_V (J/mol-atom/K), ZPE (eV/atom), spectral T_D (K)."""
    f = np.asarray(freqs_thz).ravel()
    stable = f[f > 0.05]
    x = stable * 1.0e12 * H_EV_S / (KB_EV * T)
    weight = 1.0 / (len(f) / (3 * n_atoms_prim)) / n_atoms_prim
    cv = 8.314462618 * np.sum(x ** 2 * np.exp(x) / np.expm1(x) ** 2) * weight
    zpe = 0.5 * np.sum(stable * 1.0e12 * H_EV_S) * weight
    td = THZ_TO_K * np.sqrt(5.0 * np.mean(f.clip(min=0.0) ** 2) / 3.0)
    return {"cv_j_mol_atom_k": float(cv), "zpe_ev_atom": float(zpe),
            "t_debye_spectral_k": float(td)}


def mesh_points(n):
    grid = (np.arange(n) + 0.5) / n - 0.5
    return np.array(np.meshgrid(grid, grid, grid, indexing="ij")).reshape(3, -1).T


def frequency_metrics(reference, prediction):
    reference = np.sort(np.asarray(reference), axis=1)
    prediction = np.sort(np.asarray(prediction), axis=1)
    residual = prediction - reference
    return {
        "mae_thz": float(np.mean(np.abs(residual))),
        "rmse_thz": float(np.sqrt(np.mean(residual ** 2))),
        "max_abs_thz": float(np.max(np.abs(residual))),
    }


def stability(freqs):
    f = np.asarray(freqs)
    return {"min_frequency_thz": float(np.min(f)),
            "unstable_fraction": float(np.mean(f < -0.05))}


def extractor_model(pos, cell, masses, basis, cutoff, u, df, ridge, central):
    model = IFCExtractor(
        pos, cell, masses, r_cutoff=cutoff, symmetrise=True,
        ridge_alpha=ridge, symmetrize_bonds=central, basis_indices=basis,
    )
    model.fit(list(u), list(df))
    return model


def extractor_frequencies(model, spec, qpoints):
    pcell, ppos = primitive(spec)
    calc = PhononCalculator(model, ppos, pcell, np.asarray(spec["masses"]))
    return np.asarray(calc.band_structure(qpoints)[1])


def supercell_phonopy(pos, cell, symbols, spec):
    from phonopy import Phonopy

    pcell, _ = primitive(spec)
    primitive_matrix = (pcell @ np.linalg.inv(cell)).T
    unit = phonopy_atoms(symbols, cell, pos)
    return Phonopy(unit, np.eye(3, dtype=int), primitive_matrix=primitive_matrix)


_SYMFC_BASIS = {}


def symfc_fc2_basis(pos, cell, symbols, cutoff):
    """Space-group, permutation and sum-rule adapted FC2 basis tensors."""
    from symfc import Symfc

    key = (pos.tobytes(), cell.tobytes(), tuple(symbols), float(cutoff))
    if key not in _SYMFC_BASIS:
        numbers = {s: i + 1 for i, s in enumerate(dict.fromkeys(symbols))}
        atoms = symfc_atoms(
            [numbers[s] for s in symbols], pos @ np.linalg.inv(cell), cell
        )
        model = Symfc(atoms, cutoff={2: float(cutoff)}, use_mkl=False)
        model.compute_basis_set(orders=[2])
        basis = model.basis_set[2]
        n_basis = basis.blocked_basis_set.shape[1]
        n = len(pos)
        _SYMFC_BASIS[key] = np.asarray([
            np.asarray(
                basis.compression_matrix
                @ basis.blocked_basis_set.dot(np.eye(n_basis)[k])
            ).reshape(n, n, 3, 3)
            for k in range(n_basis)
        ])
    return _SYMFC_BASIS[key]


def symfc_model(pos, cell, symbols, spec, u, df, cutoff):
    """Unregularized least-squares FC2 fit in the symmetry-adapted basis.

    The explicit design X_k = -Phi_k u reproduces symfc's own solver to
    machine precision and exposes the rank and conditioning of the actual
    force--parameter map, normalized column by column.
    """
    basis = symfc_fc2_basis(pos, cell, symbols, cutoff)
    u = np.asarray(u)
    target = np.asarray(df) - np.mean(df, axis=1, keepdims=True)
    design = np.stack(
        [-np.einsum("ijab,sjb->sia", tensor, u).ravel() for tensor in basis],
        axis=1,
    )
    coefficients, *_ = np.linalg.lstsq(design, target.ravel(), rcond=None)
    norms = np.linalg.norm(design, axis=0)
    active = norms > 0.0
    sv = np.linalg.svd(design[:, active] / norms[active], compute_uv=False)
    rank = int(np.sum(sv > sv[0] * len(sv) * np.finfo(float).eps))
    diagnostics = {
        "n_parameters": int(len(basis)),
        "n_force_observations": int(design.shape[0]),
        "normalized_design_rank": rank,
        "normalized_design_condition_number": (
            float(sv[0] / sv[rank - 1]) if rank else float("inf")
        ),
        "full_rank": bool(rank == len(basis)),
        "train_rmse_ev_ang": float(np.sqrt(np.mean(
            (design @ coefficients - target.ravel()) ** 2
        ))),
    }
    phonon = supercell_phonopy(pos, cell, symbols, spec)
    phonon.force_constants = np.einsum("k,kijab->ijab", coefficients, basis)
    return phonon, diagnostics


def phonopy_frequencies(phonon, qpoints):
    phonon.run_qpoints(qpoints)
    return np.asarray(phonon.get_qpoints_dict()["frequencies"])


def load_reference(material):
    import phonopy

    candidates = [
        ROOT / "validation_results" / material / "fd_references" / "production_large"
        / "phonopy_harmonic.yaml",
        ROOT / "validation_results" / material / "fd_references" / "production"
        / "phonopy_harmonic.yaml",
        ROOT / "validation_results" / material / "real_ifc34" / "phonopy_harmonic.yaml",
    ]
    for path in candidates:
        if path.exists():
            return phonopy.load(path), str(path.relative_to(ROOT))
    return None, None


def evaluate(freq_path, freq_mesh, ref_path, ref_mesh, spec, n_prim):
    row = {"path": stability(freq_path), "mesh": stability(freq_mesh),
           "thermo": thermo_from_mesh(freq_mesh, spec["T"], n_prim)}
    if ref_path is not None:
        row["dispersion_vs_fd"] = frequency_metrics(ref_path, freq_path)
        row["mesh_vs_fd"] = frequency_metrics(ref_mesh, freq_mesh)
    return row


def audit(material):
    spec = PRIMARY[material]
    frames = read_snapshots(material, spec)
    pos, cell, symbols, iterations = group_iterations(frames, spec)
    pos, cell = snap_to_lattice(pos, cell, spec)
    n_prim = len(spec["elements"])
    basis = np.arange(len(pos)) % n_prim
    masses = np.tile(spec["masses"], len(pos) // n_prim)
    band = np.load(
        ROOT / "examples" / f"sotc_{material}_qe_run" / "phonon_bandstructure.npz"
    )
    qpath = np.asarray(band["q_points"])
    archived = np.asarray(band["frequencies_thz"])
    qmesh = mesh_points(MESH)
    results_json = json.loads(
        (ROOT / "examples" / f"sotc_{material}_qe_run" / "sotc_results.json").read_text()
    )
    selected = int(results_json["selected_iteration"])

    reference, reference_source = load_reference(material)
    ref_path = ref_mesh = None
    if reference is not None:
        ref_path = phonopy_frequencies(reference, qpath)
        ref_mesh = phonopy_frequencies(reference, qmesh)

    report = {
        "material": material, "settings": spec, "n_atoms": len(pos),
        "selected_iteration": selected, "fd_reference": reference_source,
        "snapshot_files": [f["path"] for f in frames],
        "reference_rms_force_by_iteration": [
            it["reference_rms_force"] for it in iterations
        ],
        "fd_reference_metrics": None,
        "iterations": [], "pooled": {},
    }
    if reference is not None:
        report["fd_reference_metrics"] = evaluate(
            ref_path, ref_mesh, None, None, spec, n_prim
        )
        report["fd_reference_metrics"]["thermo"] = thermo_from_mesh(
            ref_mesh, spec["T"], n_prim
        )

    def run_variants(u, df, label):
        variants = {}
        published = extractor_model(
            pos, cell, masses, basis, spec["cutoff"], u, df, 1.0e-3, True
        )
        fp = extractor_frequencies(published, spec, qpath)
        fm = extractor_frequencies(published, spec, qmesh)
        variants["published_prior"] = evaluate(fp, fm, ref_path, ref_mesh, spec, n_prim)
        variants["published_prior"]["diagnostics"] = published.diagnostics()
        if label == f"iteration_{selected}":
            variants["published_prior"]["reproduces_archive_rms_thz"] = float(
                np.sqrt(np.mean((np.sort(fp, axis=1) - np.sort(archived, axis=1)) ** 2))
            )
        for ridge in RIDGES:
            if ridge == 1.0e-3:
                continue
            model = extractor_model(
                pos, cell, masses, basis, spec["cutoff"], u, df, ridge, True
            )
            variants[f"central_ridge_{ridge:g}"] = evaluate(
                extractor_frequencies(model, spec, qpath),
                extractor_frequencies(model, spec, qmesh),
                ref_path, ref_mesh, spec, n_prim,
            )
        model = extractor_model(
            pos, cell, masses, basis, spec["cutoff"], u, df, 1.0e-3, False
        )
        variants["no_projection_ridge_0.001"] = evaluate(
            extractor_frequencies(model, spec, qpath),
            extractor_frequencies(model, spec, qmesh),
            ref_path, ref_mesh, spec, n_prim,
        )
        for shift in (-0.5, 0.5):
            model = extractor_model(
                pos, cell, masses, basis, spec["cutoff"] + shift, u, df, 1.0e-3, True
            )
            variants[f"central_cutoff_{spec['cutoff'] + shift:.1f}"] = evaluate(
                extractor_frequencies(model, spec, qpath),
                extractor_frequencies(model, spec, qmesh),
                ref_path, ref_mesh, spec, n_prim,
            )
        try:
            phonon, design = symfc_model(pos, cell, symbols, spec, u, df, spec["cutoff"])
            variants["space_group_symfc"] = evaluate(
                phonopy_frequencies(phonon, qpath),
                phonopy_frequencies(phonon, qmesh),
                ref_path, ref_mesh, spec, n_prim,
            )
            variants["space_group_symfc"]["design"] = design
        except Exception as exc:  # report, never hide, a failed basis
            variants["space_group_symfc"] = {"error": repr(exc)}
        return variants

    for index, it in enumerate(iterations, start=1):
        report["iterations"].append({
            "iteration": index, "n_displaced": len(it["u"]),
            "rms_displacement_ang": float(np.sqrt(np.mean(it["u"] ** 2))),
            "variants": run_variants(it["u"], it["df"], f"iteration_{index}"),
        })
        print(f"{material}: iteration {index} done", flush=True)
    u_all = np.concatenate([it["u"] for it in iterations])
    df_all = np.concatenate([it["df"] for it in iterations])
    report["pooled"] = {
        "n_displaced": len(u_all),
        "variants": run_variants(u_all, df_all, "pooled"),
    }
    out = ROOT / "validation_results" / material / "primary_audit"
    out.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(
        out / "primary_force_dataset.npz",
        equilibrium_positions_ang=pos, cell_ang=cell, symbols=np.asarray(symbols),
        iteration=np.concatenate([
            np.full(len(it["u"]), k + 1) for k, it in enumerate(iterations)
        ]),
        displacements_ang=u_all, delta_forces_ev_ang=df_all,
    )
    (out / "primary_audit.json").write_text(
        json.dumps(report, indent=2, default=float) + "\n"
    )
    return report


def main():
    materials = sys.argv[1:] or list(PRIMARY)
    for material in materials:
        audit(material)


if __name__ == "__main__":
    main()
