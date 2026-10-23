#!/usr/bin/env python3
"""Matched robustness analysis of the nine primary SOTC benchmarks.

Inputs (all produced by the drivers in this directory):

* archived primary SOTC force data (``analyze_primary_archives.py``);
* converged large-cell FD references (``run_fd_references.py --large``);
* 3x3x3 FD references at production, PBE and room-temperature lattices;
* same-cell FD and 48 canonical snapshots at T_des in the primary cell
  (``run_primary_cell_controls.py``).

For every material the archived SOTC FC2 (selected iteration and all
iterations pooled) is compared with random canonical subsets of exactly the
same size, fitted with the same constrained regression, in the same cell
and electronic settings.  References are (i) the converged harmonic FD FC2
and (ii) the effective FC2 fitted to all 40 canonical training snapshots.
Held-out canonical forces provide a reference-free predictive test.
"""

from __future__ import annotations

import json
from pathlib import Path
import sys

import numpy as np

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "codes"))
sys.path.insert(0, str(Path(__file__).resolve().parent))

from analyze_primary_archives import (  # noqa: E402
    PRIMARY,
    extractor_frequencies,
    extractor_model,
    frequency_metrics,
    mesh_points,
    phonopy_frequencies,
    primitive,
    stability,
    supercell_phonopy,
    symfc_model,
)
from sotc.phonons import PhononCalculator  # noqa: E402

N_TRAIN = 40
N_SEEDS = 20
MESH = 12
U_MESH = 24
HBAR = 1.054571817e-34
KB = 1.380649e-23
AMU = 1.66053906660e-27
R_GAS = 8.314462618


def thermo(freqs_thz, T, n_prim):
    """Harmonic per-atom C_V, S, F_vib (J/mol K, J/mol K, kJ/mol), ZPE (meV)."""
    f = np.asarray(freqs_thz).ravel()
    weight = 1.0 / (len(f) / (3 * n_prim)) / n_prim
    w = f[f > 0.05] * 1.0e12 * 2.0 * np.pi
    x = HBAR * w / (KB * T)
    cv = R_GAS * np.sum(x ** 2 * np.exp(x) / np.expm1(x) ** 2) * weight
    s = R_GAS * np.sum(x / np.expm1(x) - np.log1p(-np.exp(-x))) * weight
    fvib = R_GAS * T * np.sum(np.log(2.0 * np.sinh(x / 2.0))) * weight / 1000.0
    zpe = 0.5 * np.sum(HBAR * w) / 1.602176634e-19 * weight * 1000.0
    td = 47.99243073366221 * np.sqrt(5.0 * np.mean(f.clip(min=0.0) ** 2) / 3.0)
    return {"cv_j_mol_atom_k": float(cv), "s_j_mol_atom_k": float(s),
            "fvib_kj_mol_atom": float(fvib), "zpe_mev_atom": float(zpe),
            "t_debye_spectral_k": float(td)}


def phonopy_uiso(phonon, temperatures):
    phonon.run_mesh([U_MESH] * 3, is_gamma_center=False,
                    with_eigenvectors=True, is_mesh_symmetry=False)
    phonon.run_thermal_displacements(temperatures=temperatures, freq_min=0.05)
    values = np.asarray(phonon.get_thermal_displacements_dict()["thermal_displacements"])
    return values.reshape(len(temperatures), -1, 3).mean(axis=2)


def extractor_uiso(model, spec, temperatures):
    pcell, ppos = primitive(spec)
    calc = PhononCalculator(model, ppos, pcell, np.asarray(spec["masses"]))
    return np.asarray([
        calc.mean_square_displacements(T, q_mesh=(U_MESH,) * 3)["u_iso_ang2_by_atom"]
        for T in temperatures
    ])


def load(path):
    import phonopy

    return phonopy.load(path) if Path(path).exists() else None


def canonical_dataset(material):
    base = ROOT / "validation_results" / material / "primary_cell"
    archive = base / "canonical_displacements.npz"
    if not archive.exists():
        return None
    u = np.load(archive)["displacements_ang"]
    with np.load(base / "fd" / "harmonic_reference.npz") as h:
        f0 = np.asarray(h["reference_forces_ev_ang"])
    forces = []
    for i in range(len(u)):
        path = base / "canonical_qe" / f"state_{i:04d}" / "result.npz"
        if not path.exists():
            return None
        forces.append(np.load(path)["forces_ev_ang"])
    df = np.asarray(forces) - f0
    df -= np.mean(df, axis=1, keepdims=True)
    return u, df


def evaluate_model(label, freq_path, freq_mesh, predict, refs, test, spec, n_prim):
    row = {"model": label, "path": stability(freq_path),
           "mesh": stability(freq_mesh),
           "thermo_T_des": thermo(freq_mesh, spec["T"], n_prim)}
    for name, (rpath, rmesh) in refs.items():
        if rpath is None:
            continue
        row[f"path_vs_{name}"] = frequency_metrics(rpath, freq_path)
        row[f"mesh_vs_{name}"] = frequency_metrics(rmesh, freq_mesh)
    if test is not None and predict is not None:
        u_test, f_test = test
        prediction = np.asarray([predict(u) for u in u_test])
        residual = prediction - f_test
        row["held_out_force_rmse_ev_ang"] = float(np.sqrt(np.mean(residual ** 2)))
        row["held_out_force_r2"] = float(
            1.0 - np.sum(residual ** 2) / np.sum((f_test - f_test.mean()) ** 2)
        )
    return row


COUNTS = (2, 4, 8, 16)
FITTERS = ("prior", "symfc")


def sotcd_dataset(material):
    base = ROOT / "validation_results" / material / "primary_cell"
    archive = base / "sotcd_displacements.npz"
    if not archive.exists():
        return None
    u = np.load(archive)["displacements_ang"]
    with np.load(base / "fd" / "harmonic_reference.npz") as h:
        f0 = np.asarray(h["reference_forces_ev_ang"])
    forces = []
    for i in range(len(u)):
        path = base / "sotcd_qe" / f"state_{i:04d}" / "result.npz"
        if not path.exists():
            return None
        forces.append(np.load(path)["forces_ev_ang"])
    df = np.asarray(forces) - f0
    df -= np.mean(df, axis=1, keepdims=True)
    return u, df


def fit_and_score(fitter, u, df, spec, pos, cell, symbols, masses, basis,
                  qpath, qmesh, refs, test, n_prim, label):
    if fitter == "prior":
        model = extractor_model(pos, cell, masses, basis, spec["cutoff"],
                                u, df, 1.0e-3, True)
        freq_mesh = extractor_frequencies(model, spec, qmesh)
        row = evaluate_model(
            label, extractor_frequencies(model, spec, qpath), freq_mesh,
            model.predict_forces, refs, test, spec, n_prim,
        )
        row["_mesh_frequencies"] = freq_mesh
        return row
    phonon, diag = symfc_model(pos, cell, symbols, spec, u, df, spec["cutoff"])
    fc = np.asarray(phonon.force_constants)
    freq_mesh = phonopy_frequencies(phonon, qmesh)
    row = evaluate_model(
        label, phonopy_frequencies(phonon, qpath), freq_mesh,
        lambda x, fc=fc: -np.einsum("ijab,jb->ia", fc, x), refs, test, spec, n_prim,
    )
    row["design"] = diag
    row["_mesh_frequencies"] = freq_mesh
    return row


def compact(row, disjoint=None):
    keep = {"model": row["model"]}
    if disjoint is not None:
        keep["mesh_vs_disjoint_effective"] = disjoint
    for key in ("mesh_vs_effective", "mesh_vs_fd_large", "path_vs_fd_large"):
        if key in row:
            keep[key] = row[key]["rmse_thz"]
    keep["held_out_force_rmse_ev_ang"] = row.get("held_out_force_rmse_ev_ang")
    keep["unstable_fraction"] = row["mesh"]["unstable_fraction"]
    keep["thermo_T_des"] = row["thermo_T_des"]
    if "design" in row:
        keep["rank"] = row["design"]["normalized_design_rank"]
        keep["n_parameters"] = row["design"]["n_parameters"]
    return keep


def count_series(material, spec, pos, cell, symbols, masses, basis, qpath, qmesh,
                 refs, test, u_train, f_train, n_prim):
    """Error versus number of force calls for SOTC-D and canonical baselines.

    Canonical baselines: random unpaired subsets of the 40 training snapshots,
    and random sets of complete +/- pairs (the pool is pair-interleaved).
    """
    series = {"counts": list(COUNTS), "disjoint_reference": (
        "effective symfc FC2 fitted to 40-n canonical snapshots that share no "
        "configuration with the scored design: the complement of a canonical "
        "subset, or a random (40-n)-subset of the pool for SOTC-D"
    )}
    common = (spec, pos, cell, symbols, masses, basis, qpath, qmesh, refs, test, n_prim)
    cache = {}

    def disjoint_mesh(indices):
        key = tuple(sorted(int(i) for i in indices))
        if key not in cache:
            phonon, _ = symfc_model(pos, cell, symbols, spec, u_train[list(key)],
                                    f_train[list(key)], spec["cutoff"])
            cache[key] = phonopy_frequencies(phonon, qmesh)
        return cache[key]

    def disjoint_score(freq_mesh, indices):
        return frequency_metrics(disjoint_mesh(indices), freq_mesh)["rmse_thz"]

    sotcd = sotcd_dataset(material)
    for fitter in FITTERS:
        block = {}
        if sotcd is not None:
            u, df = sotcd
            rows = []
            for c in COUNTS:
                row = fit_and_score(fitter, u[:c], df[:c], *common, f"sotcd_{c}")
                freq = row.pop("_mesh_frequencies")
                scores = [
                    disjoint_score(freq, np.random.default_rng(31 * c + s).choice(
                        N_TRAIN, size=N_TRAIN - c, replace=False))
                    for s in range(N_SEEDS)
                ]
                rows.append(compact(row, float(np.median(scores))))
            block["sotcd"] = rows
        for kind in ("canonical_unpaired", "canonical_paired"):
            rows = []
            for c in COUNTS:
                trials = []
                for seed in range(N_SEEDS):
                    rng = np.random.default_rng(7000 + 97 * c + seed)
                    if kind == "canonical_unpaired":
                        chosen = rng.choice(N_TRAIN, size=c, replace=False)
                    else:
                        pairs = rng.choice(N_TRAIN // 2, size=c // 2, replace=False)
                        chosen = np.ravel(np.stack((2 * pairs, 2 * pairs + 1), axis=1))
                    row = fit_and_score(fitter, u_train[chosen], f_train[chosen],
                                        *common, f"{kind}_{c}_seed{seed}")
                    freq = row.pop("_mesh_frequencies")
                    complement = np.setdiff1d(np.arange(N_TRAIN), chosen)
                    trials.append(compact(row, disjoint_score(freq, complement)))
                rows.append(trials)
            block[kind] = rows
        series[fitter] = block
    return series


def analyze(material):
    spec = PRIMARY[material]
    n_prim = len(spec["elements"])
    data = np.load(
        ROOT / "validation_results" / material / "primary_audit"
        / "primary_force_dataset.npz"
    )
    pos = np.asarray(data["equilibrium_positions_ang"])
    cell = np.asarray(data["cell_ang"])
    symbols = [str(s) for s in data["symbols"]]
    u_all = np.asarray(data["displacements_ang"])
    df_all = np.asarray(data["delta_forces_ev_ang"])
    iteration = np.asarray(data["iteration"])
    results_json = json.loads(
        (ROOT / "examples" / f"sotc_{material}_qe_run" / "sotc_results.json").read_text()
    )
    selected = int(results_json["selected_iteration"])
    sel = iteration == selected
    basis = np.arange(len(pos)) % n_prim
    masses = np.tile(spec["masses"], len(pos) // n_prim)
    band = np.load(
        ROOT / "examples" / f"sotc_{material}_qe_run" / "phonon_bandstructure.npz"
    )
    qpath = np.asarray(band["q_points"])
    qmesh = mesh_points(MESH)
    vr = ROOT / "validation_results" / material

    refs = {}
    fd_large = load(vr / "fd_references" / "production_large" / "phonopy_harmonic.yaml")
    fd_small = load(vr / "fd_references" / "production" / "phonopy_harmonic.yaml")
    fd_cell = load(vr / "primary_cell" / "fd" / "phonopy_harmonic.yaml")
    for name, ph in (("fd_large", fd_large), ("fd_3x3x3", fd_small),
                     ("fd_same_cell", fd_cell)):
        refs[name] = (
            (phonopy_frequencies(ph, qpath), phonopy_frequencies(ph, qmesh))
            if ph is not None else (None, None)
        )

    canonical = canonical_dataset(material)
    test = None
    rows = []
    report = {"material": material, "selected_iteration": selected,
              "n_atoms": len(pos), "n_sotc_selected": int(np.sum(sel)),
              "n_sotc_pooled": int(len(u_all)),
              "canonical_available": canonical is not None}

    if canonical is not None:
        u_can, df_can = canonical
        u_train, f_train = u_can[:N_TRAIN], df_can[:N_TRAIN]
        test = (u_can[N_TRAIN:], df_can[N_TRAIN:])
        eff, eff_diag = symfc_model(pos, cell, symbols, spec, u_train, f_train,
                                    spec["cutoff"])
        refs["effective"] = (phonopy_frequencies(eff, qpath),
                             phonopy_frequencies(eff, qmesh))
        half, _ = symfc_model(pos, cell, symbols, spec, u_train[:N_TRAIN // 2],
                              f_train[:N_TRAIN // 2], spec["cutoff"])
        report["effective_reference"] = {
            "design": eff_diag,
            "convergence_20_vs_40_mesh": frequency_metrics(
                refs["effective"][1], phonopy_frequencies(half, qmesh)
            ),
            "canonical_rms_displacement_ang": float(np.sqrt(np.mean(u_can ** 2))),
            "sotc_rms_displacement_ang": float(np.sqrt(np.mean(u_all ** 2))),
        }
        fc_eff = np.asarray(eff.force_constants)
        rows.append(evaluate_model(
            "effective_reference_symfc_40", refs["effective"][0],
            refs["effective"][1],
            lambda u, fc=fc_eff: -np.einsum("ijab,jb->ia", fc, u),
            refs, test, spec, n_prim,
        ))
        prior40 = extractor_model(pos, cell, masses, basis, spec["cutoff"],
                                  u_train, f_train, 1.0e-3, True)
        rows.append(evaluate_model(
            "canonical_40_published_prior",
            extractor_frequencies(prior40, spec, qpath),
            extractor_frequencies(prior40, spec, qmesh),
            prior40.predict_forces, refs, test, spec, n_prim,
        ))

    for ph, label in ((fd_cell, "fd_same_cell"), (fd_small, "fd_3x3x3"),
                      (fd_large, "fd_large")):
        if ph is None:
            continue
        predict = None
        if label == "fd_same_cell":
            fc = np.asarray(ph.force_constants)
            predict = lambda u, fc=fc: -np.einsum("ijab,jb->ia", fc, u)  # noqa: E731
        rows.append(evaluate_model(
            label, phonopy_frequencies(ph, qpath), phonopy_frequencies(ph, qmesh),
            predict, refs, test, spec, n_prim,
        ))

    sotc_models = {}
    for label, mask in (("sotc_selected", sel), ("sotc_pooled", np.ones_like(sel))):
        model = extractor_model(pos, cell, masses, basis, spec["cutoff"],
                                u_all[mask], df_all[mask], 1.0e-3, True)
        sotc_models[label] = model
        rows.append(evaluate_model(
            label, extractor_frequencies(model, spec, qpath),
            extractor_frequencies(model, spec, qmesh),
            model.predict_forces, refs, test, spec, n_prim,
        ))
        phonon, diag = symfc_model(pos, cell, symbols, spec, u_all[mask],
                                   df_all[mask], spec["cutoff"])
        fc = np.asarray(phonon.force_constants)
        row = evaluate_model(
            f"{label}_symfc", phonopy_frequencies(phonon, qpath),
            phonopy_frequencies(phonon, qmesh),
            lambda u, fc=fc: -np.einsum("ijab,jb->ia", fc, u),
            refs, test, spec, n_prim,
        )
        row["design"] = diag
        rows.append(row)

    matched = {}
    if canonical is not None:
        for label in ("sotc_selected", "sotc_pooled"):
            count = report["n_sotc_selected" if label == "sotc_selected"
                            else "n_sotc_pooled"]
            count = min(count, N_TRAIN)
            series = []
            for seed in range(N_SEEDS):
                chosen = np.random.default_rng(5150 + seed).choice(
                    N_TRAIN, size=count, replace=False
                )
                model = extractor_model(pos, cell, masses, basis, spec["cutoff"],
                                        u_train[chosen], f_train[chosen],
                                        1.0e-3, True)
                row = evaluate_model(
                    f"canonical_{count}_seed{seed}",
                    extractor_frequencies(model, spec, qpath),
                    extractor_frequencies(model, spec, qmesh),
                    model.predict_forces, refs, test, spec, n_prim,
                )
                row["chosen"] = chosen.tolist()
                series.append(row)
            matched[label] = {"n_configurations": count, "canonical": series}
    report["models"] = rows
    report["matched_canonical"] = matched
    if canonical is not None:
        report["count_series"] = count_series(
            material, spec, pos, cell, symbols, masses, basis, qpath, qmesh,
            refs, test, u_train, f_train, n_prim,
        )

    # Displacement parameters at 300 K and T_des.
    temps = [300.0, spec["T"]]
    uiso = {}
    for ph, label in ((fd_large, "fd_large"), (fd_cell, "fd_same_cell")):
        if ph is not None:
            uiso[label] = phonopy_uiso(ph, temps).tolist()
    for label, model in sotc_models.items():
        uiso[label] = extractor_uiso(model, spec, temps).tolist()
    for variant in ("room_large", "room"):
        ph = load(vr / "fd_references" / variant / "phonopy_harmonic.yaml")
        if ph is not None:
            uiso[f"fd_{variant}"] = phonopy_uiso(ph, temps).tolist()
    report["u_iso_ang2"] = {"temperatures_k": temps, "values_by_atom": uiso}

    # Volume sensitivity: harmonic 3x3x3 FD at PBE versus production lattice.
    pbe = load(vr / "fd_references" / "pbe" / "phonopy_harmonic.yaml")
    summary_path = vr / "fd_references" / "fd_reference_summary.json"
    if pbe is not None and fd_small is not None:
        report["volume_sensitivity_3x3x3"] = {
            "mesh_pbe_vs_production": frequency_metrics(
                phonopy_frequencies(fd_small, qmesh), phonopy_frequencies(pbe, qmesh)
            ),
            "thermo_production": thermo(phonopy_frequencies(fd_small, qmesh),
                                        spec["T"], n_prim),
            "thermo_pbe": thermo(phonopy_frequencies(pbe, qmesh), spec["T"], n_prim),
        }
        if summary_path.exists():
            eos = json.loads(summary_path.read_text()).get("pbe", {}).get("eos")
            if eos:
                report["volume_sensitivity_3x3x3"]["eos"] = {
                    k: eos[k] for k in ("a_pbe_ang", "a_production_ang",
                                        "relative_difference", "fit")
                }

    out = vr / "robustness"
    out.mkdir(parents=True, exist_ok=True)
    (out / "robustness_summary.json").write_text(
        json.dumps(report, indent=2, default=float) + "\n"
    )
    return report


def main():
    for material in sys.argv[1:] or list(PRIMARY):
        analyze(material)
        print(f"{material}: robustness analysis written", flush=True)


if __name__ == "__main__":
    main()
