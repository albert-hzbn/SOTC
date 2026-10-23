#!/usr/bin/env python3
"""SSCHA with DFT (Quantum ESPRESSO) forces in the SOTC cell.

python-sscha minimizes the free energy with populations of N configurations
drawn from the current auxiliary FC2, starting from the same-cell harmonic
FD trial used by every other protocol.  The whole SOTC supercell is the
CellConstructor unit cell (Gamma point only), so the result is the
self-consistent effective FC2 of exactly the cell used by the designs.
Forces of each population are computed in parallel with the cached QE
driver of ``run_primary_cell_controls.py`` and handed to SSCHA through a
lookup calculator; F(0) and E(0) of the undisplaced cell are subtracted.

Writes validation_results/<m>/primary_cell/sscha_dft/N<N>_s<seed>/history.npz
with the auxiliary FC2 after every population and the cumulative number of
force calls.

Usage: dft_sscha.py <material> <N> <seed> [--max-pop 12]
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

import surrogate_sscha  # noqa: E402
from analyze_primary_archives import PRIMARY  # noqa: E402
from run_primary_cell_controls import overrides, primary_spec  # noqa: E402
from run_real_anharmonic_ifcs import run_one_state  # noqa: E402
from surrogate_sscha import build_dynamics, dynamics_to_fc  # noqa: E402


class LookupCalculator:
    """ASE calculator returning precomputed DFT results (eV, eV/A) for
    configurations given in SSCHA atom order."""

    def __new__(cls, *args, **kwargs):
        from ase.calculators.calculator import Calculator, all_changes

        class _Calc(Calculator):
            implemented_properties = ["energy", "forces"]

            def __init__(self, positions, energies, forces, order):
                super().__init__()
                self.table = list(zip(positions, energies, forces))
                self.order = order

            def calculate(self, atoms=None, properties=("energy",),
                          system_changes=all_changes):
                super().calculate(atoms, properties, system_changes)
                pos = np.asarray(self.atoms.get_positions())
                for p, e, f in self.table:
                    if np.allclose(p, pos, atol=1.0e-6):
                        self.results = {"energy": float(e), "forces": f[self.order]}
                        return
                raise KeyError("configuration not in the precomputed population")

        return _Calc(*args, **kwargs)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("material")
    parser.add_argument("n_configs", type=int)
    parser.add_argument("seed", type=int)
    parser.add_argument("--max-pop", type=int, default=12)
    parser.add_argument("--n-parallel", type=int, default=4)
    args = parser.parse_args()

    import phonopy
    import sscha.Ensemble
    import sscha.SchaMinimizer

    material = args.material
    base = ROOT / "validation_results" / material / "primary_cell"
    out = base / "sscha_dft" / f"N{args.n_configs}_s{args.seed}"
    out.mkdir(parents=True, exist_ok=True)
    phonon = phonopy.load(base / "fd" / "phonopy_harmonic.yaml")
    with np.load(base / "fd" / "harmonic_reference.npz") as h:
        f0 = np.asarray(h["reference_forces_ev_ang"])
        e0 = float(h["reference_energy_ev"])
    surrogate_sscha.CONVENTIONAL_SIZE[material] = (1, 1, 1)
    dyn, size, order, _ = build_dynamics(phonon, phonon.force_constants, material)
    temperature = PRIMARY[material]["T"]
    spec = primary_spec(material)
    run_args = SimpleNamespace(material=material, n_parallel=args.n_parallel,
                               tasks_per_qe=32, fit_only=False, force_rerun=False)
    eq = np.asarray(phonon.supercell.positions)
    cell = np.asarray(phonon.supercell.cell)
    symbols = list(phonon.supercell.symbols)
    inv = np.linalg.inv(cell)

    np.random.seed(args.seed)
    fcs, calls, converged = [], [], []
    total = 0
    for pop in range(args.max_pop):
        ensemble = sscha.Ensemble.Ensemble(dyn, temperature, size)
        ensemble.generate(args.n_configs)
        sscha_pos, ph_pos = [], []
        for s in ensemble.structures:
            c = np.asarray(s.coords)
            d = c - eq[order]
            frac = d @ inv
            d = (frac - np.rint(frac)) @ cell
            p = eq.copy()
            p[order] = eq[order] + d
            sscha_pos.append(c)
            ph_pos.append(p)
        stage = out / f"pop{pop:02d}"
        stage.mkdir(exist_ok=True)
        with concurrent.futures.ThreadPoolExecutor(max_workers=args.n_parallel) as pool:
            results = list(pool.map(
                lambda i: run_one_state(i, ph_pos[i], cell, symbols, spec, run_args,
                                        stage, overrides(spec)),
                range(len(ph_pos))))
        energies = [float(r["energy_ev"]) - e0 for r in results]
        forces = [np.asarray(r["forces_ev_ang"]) - f0 for r in results]
        forces = [f - f.mean(axis=0) for f in forces]
        calc = LookupCalculator(sscha_pos, energies, forces, np.asarray(order))
        ensemble.compute_ensemble(calc, compute_stress=False, verbose=False)
        total += args.n_configs
        minim = sscha.SchaMinimizer.SSCHA_Minimizer(ensemble)
        minim.set_minimization_step(0.1)
        minim.meaningful_factor = 0.2
        minim.kong_liu_ratio = 0.5
        minim.minim_struct = False
        minim.init()
        minim.run()
        dyn = minim.dyn
        fcs.append(dynamics_to_fc(dyn, size, order, len(order)))
        calls.append(total)
        converged.append(bool(minim.is_converged()))
        np.savez_compressed(out / "history.npz", force_constants=np.asarray(fcs),
                            force_calls=np.asarray(calls), converged=np.asarray(converged))
        print(f"{material} N={args.n_configs} seed={args.seed} population {pop}: "
              f"{total} calls, converged={converged[-1]}", flush=True)
        if converged[-1]:
            break


if __name__ == "__main__":
    main()
