#!/usr/bin/env python3
"""Iterated sTDEP and self-consistent ZG with DFT forces in the SOTC cell.

Both start from the same-cell harmonic FD trial used by every protocol and
use the regression of ``dft_sotcd_sc.py`` (supercell space-group basis,
minimum-norm correction of the harmonic coefficients while unidentified):

* ``stdep``: sTDEP iterated to self-consistency (Shulumba et al.). Each
  iteration draws ``--per-iter`` canonical configurations (+/- pairs) from
  the current model; the FC2 is refitted to that iteration alone ("last",
  the original protocol) and to all iterations ("pooled").
* ``zgsc``: self-consistent special displacements. Each iteration adds one
  Zacharias--Giustino +/- pair designed from the current model, and all
  pairs are refitted, exactly as in SOTC-D-SC. It differs from SOTC-D-SC
  only in the sign design (ZG Walsh alternation instead of D-optimal signs),
  in the spirit of the anharmonic special-displacement method.

The model after every iteration is stored.

Writes validation_results/<m>/primary_cell/<protocol>_dft/s<seed>/history.npz.

Usage: dft_sc_protocols.py <material> <stdep|zgsc> <seed> [--budget 32] [--per-iter 4]
       [--name DIR]
"""

from __future__ import annotations

import argparse
import concurrent.futures
from pathlib import Path
import sys
from types import SimpleNamespace

import numpy as np

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "codes"))
sys.path.insert(0, str(Path(__file__).resolve().parent))

from analyze_primary_archives import PRIMARY, symfc_fc2_basis  # noqa: E402
from run_primary_cell_controls import overrides, primary_spec  # noqa: E402
from run_real_anharmonic_ifcs import run_one_state  # noqa: E402
from sotc.optimal_design import configurations, walsh_signs, zg_modes  # noqa: E402
from surrogate_design_study import gaussian_configs, paired  # noqa: E402
from surrogate_self_consistent import fit_coefficients, modes, to_fc  # noqa: E402

SEED0 = {"stdep": 83_000, "zgsc": 85_000}


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("material")
    parser.add_argument("protocol", choices=sorted(SEED0))
    parser.add_argument("seed", type=int)
    parser.add_argument("--budget", type=int, default=32)
    parser.add_argument("--per-iter", type=int, default=4)
    parser.add_argument("--n-parallel", type=int, default=4)
    parser.add_argument("--name", help="output directory (default <protocol>_dft); the "
                        "converged reference uses stdep --per-iter 128 --name stdep_reference")
    args = parser.parse_args()

    import phonopy
    from phonopy.harmonic.force_constants import compact_fc_to_full_fc

    material = args.material
    base = ROOT / "validation_results" / material / "primary_cell"
    out = base / (args.name or f"{args.protocol}_dft") / f"s{args.seed}"
    out.mkdir(parents=True, exist_ok=True)
    phonon = phonopy.load(base / "fd" / "phonopy_harmonic.yaml")
    with np.load(base / "fd" / "harmonic_reference.npz") as h:
        f0 = np.asarray(h["reference_forces_ev_ang"])
    harmonic = np.asarray(phonon.force_constants)
    if harmonic.shape[0] != harmonic.shape[1]:
        harmonic = compact_fc_to_full_fc(phonon.primitive, harmonic)
    sup = phonon.supercell
    eq, cell = np.asarray(sup.positions), np.asarray(sup.cell)
    symbols, masses = list(sup.symbols), np.asarray(sup.masses)
    basis = symfc_fc2_basis(eq, cell, symbols, PRIMARY[material]["cutoff"])
    c_harm, *_ = np.linalg.lstsq(basis.reshape(len(basis), -1).T, harmonic.ravel(), rcond=None)
    temperature = PRIMARY[material]["T"]
    spec = primary_spec(material)
    run_args = SimpleNamespace(material=material, n_parallel=args.n_parallel, tasks_per_qe=32,
                               fit_only=False, force_rerun=False)
    rng = np.random.default_rng(SEED0[args.protocol] + args.seed)

    def forces(u, start):
        with concurrent.futures.ThreadPoolExecutor(max_workers=args.n_parallel) as pool:
            res = list(pool.map(lambda j: run_one_state(start + j, eq + u[j], cell, symbols, spec,
                                                        run_args, out, overrides(spec)),
                                range(len(u))))
        df = np.asarray([np.asarray(r["forces_ev_ang"]) - f0 for r in res])
        return df - df.mean(axis=1, keepdims=True)

    fc = harmonic
    us, fs, calls, history = [], [], [], {"last": [], "pooled": []}
    per_iter = 2 if args.protocol == "zgsc" else args.per_iter
    while sum(len(u) for u in us) + per_iter <= args.budget:
        if args.protocol == "stdep":
            patterns, sigma = modes(fc, masses, temperature)
            u = paired(gaussian_configs(rng, per_iter // 2, patterns, sigma))
        else:
            phonon.force_constants = fc
            _, zp, zs = zg_modes(phonon, temperature)
            u = paired(configurations(walsh_signs(len(zs), 1), zp, zs))
        u -= u.mean(axis=1, keepdims=True)
        f = forces(u, sum(len(x) for x in us))
        us.append(u)
        fs.append(f)
        pooled = to_fc(basis, fit_coefficients(basis, np.concatenate(us), np.concatenate(fs),
                                               c_harm))
        last = to_fc(basis, fit_coefficients(basis, u, f, c_harm))
        history["pooled"].append(pooled)
        history["last"].append(last)
        # sTDEP iterates on the fit of the latest iteration; ZG-SC on the pooled fit.
        fc = last if args.protocol == "stdep" else pooled
        calls.append(sum(len(x) for x in us))
        np.savez_compressed(out / "history.npz", force_constants_pooled=np.asarray(
            history["pooled"]), force_constants_last=np.asarray(history["last"]),
            force_calls=np.asarray(calls), displacements_ang=np.concatenate(us))
        print(f"{material} {args.protocol} seed {args.seed}: {calls[-1]} calls", flush=True)


if __name__ == "__main__":
    main()
