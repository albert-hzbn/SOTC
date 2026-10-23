#!/usr/bin/env python3
"""SSCHA on the DFT-fitted FC2+FC3+FC4 surrogates (Al, W, NaCl).

Runs the stochastic self-consistent harmonic approximation of python-sscha
(CellConstructor 1.6) with the surrogate as an exact, noise-free ASE
calculator, starting from the harmonic FD trial of the 3x3x3 cell.  The
auxiliary FC2 at convergence is the SCHA fixed point, which for a crystal
without free internal coordinates coincides with the self-consistent
canonical (sTDEP) effective FC2 (Stein's lemma), so it is scored in
``surrogate_self_consistent.py`` against the same ground truth as the other
self-consistent protocols.  Every population's forces are counted as force
calls.

This script needs python-sscha, which is built in a separate Python 3.13
environment (see README); it writes
validation_results/<m>/surrogate_design/sscha_runs.npz.
"""

from __future__ import annotations

import json
from pathlib import Path
import sys

import numpy as np

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "codes"))

from sotc.symfc_fit import forces_from_compact_fcs  # noqa: E402

SURROGATES = {"al": 300.0, "w": 1500.0, "nacl": 700.0, "si": 600.0, "mg": 600.0}
# Supercells that are not diagonal in the primitive cell are described by a
# conventional unit cell (supercell lattice / size) for CellConstructor.
CONVENTIONAL_SIZE = {"si": (2, 2, 2)}
RY_EV = 13.605693122994
BOHR_ANG = 0.529177210903
AMU_RY = 911.44424310865645  # electron-mass/2 units per amu
POPULATION_SIZES = (8, 16, 32, 64, 128)
REPEATS = 5
MAX_POP = 30


def load_surrogate(material):
    import h5py
    import phonopy

    base = ROOT / "validation_results" / material / "real_ifc34"
    phonon = phonopy.load(base / "phonopy_harmonic.yaml")
    ff = base / "final_force_constants"
    with h5py.File(ff / "fc234_compact.hdf5") as handle:
        fcs = {k: np.asarray(handle[f"fc{k}"]) for k in (2, 3, 4)}
    return (phonon, fcs, np.load(ff / "translation_permutations.npy"),
            np.load(ff / "p2s_map.npy"))


class SurrogateCalculator:
    """Minimal ASE calculator: E and F of the compact FC2+FC3+FC4 model."""

    def __new__(cls, *args, **kwargs):
        from ase.calculators.calculator import Calculator, all_changes

        class _Calc(Calculator):
            implemented_properties = ["energy", "forces"]

            def __init__(self, fcs, perms, p2s, reference, cell, order):
                super().__init__()
                self.fcs, self.perms, self.p2s = fcs, perms, p2s
                self.reference, self.cell_ = reference, cell
                self.order = order  # SSCHA atom k -> surrogate atom order[k]
                self.calls = 0

            def calculate(self, atoms=None, properties=("energy",),
                          system_changes=all_changes):
                super().calculate(atoms, properties, system_changes)
                pos = np.asarray(self.atoms.get_positions())
                u = np.zeros_like(self.reference)
                d = pos - self.reference[self.order]
                frac = d @ np.linalg.inv(self.cell_)
                d = (frac - np.rint(frac)) @ self.cell_
                u[self.order] = d
                energy, forces = 0.0, np.zeros_like(u)
                for k in (2, 3, 4):
                    f = forces_from_compact_fcs(u, {k: self.fcs[k]}, self.perms, self.p2s)
                    forces += f
                    energy -= float(np.sum(u * f)) / k
                self.calls += 1
                self.results = {"energy": energy, "forces": forces[self.order]}

        return _Calc(*args, **kwargs)


def full_fc(phonon, fc):
    from phonopy.harmonic.force_constants import compact_fc_to_full_fc

    fc = np.asarray(fc)
    return compact_fc_to_full_fc(phonon.primitive, fc) if fc.shape[0] != fc.shape[1] else fc


def build_dynamics(phonon, fc_ev_ang2, material=None):
    """CellConstructor Phonons of the supercell FC2 with SSCHA atom order."""
    fc_ev_ang2 = full_fc(phonon, fc_ev_ang2)
    import cellconstructor as CC
    import cellconstructor.Phonons
    import cellconstructor.Structure
    import cellconstructor.symmetries
    from ase import Atoms

    prim = phonon.primitive
    sup = phonon.supercell
    if material in CONVENTIONAL_SIZE:
        size = CONVENTIONAL_SIZE[material]
        ucell = np.asarray(sup.cell) / np.asarray(size, float)[:, None]
        # Wrap into the supercell first so that atoms on its boundary are
        # not lost to rounding.
        sfrac = np.asarray(sup.positions) @ np.linalg.inv(np.asarray(sup.cell))
        sfrac = np.mod(np.round(sfrac, 8), 1.0)
        frac = sfrac * np.asarray(size, float)
        keep = np.all(frac < 1 - 1e-6, axis=1)
        assert keep.sum() * np.prod(size) == len(sup.positions)
        usym = [s for s, k in zip(sup.symbols, keep) if k]
        upos, umass = np.asarray(sup.positions)[keep], np.asarray(sup.masses)[keep]
    else:
        ucell, usym = np.asarray(prim.cell), list(prim.symbols)
        upos, umass = np.asarray(prim.positions), np.asarray(prim.masses)
        size = tuple(int(x) for x in np.rint(np.diag(
            np.asarray(sup.cell) @ np.linalg.inv(ucell))))
    unit = CC.Structure.Structure()
    unit.generate_from_ase_atoms(Atoms(symbols=usym, positions=upos, cell=ucell, pbc=True))
    unit.masses = {s: float(m) * AMU_RY for s, m in zip(usym, umass)}
    superstruct = unit.generate_supercell(size)
    # Map SSCHA supercell atoms onto the phonopy supercell atoms.
    cell = np.asarray(phonon.supercell.cell)
    ours = np.asarray(phonon.supercell.positions)
    order = []
    for x in superstruct.coords:
        frac = (ours - x) @ np.linalg.inv(cell)
        dist = np.linalg.norm((frac - np.rint(frac)) @ cell, axis=1)
        order.append(int(np.argmin(dist)))
        assert dist.min() < 1.0e-5
    assert sorted(order) == list(range(len(ours))), "supercell mapping is not a permutation"
    order = np.asarray(order)
    n = len(order)
    fc = np.asarray(fc_ev_ang2)[np.ix_(order, order)].transpose(0, 2, 1, 3)
    fc = fc.reshape(3 * n, 3 * n) * BOHR_ANG ** 2 / RY_EV
    qgrid = CC.symmetries.GetQGrid(unit.unit_cell, size)
    dynq = CC.Phonons.GetDynQFromFCSupercell(fc, np.asarray(qgrid), unit, superstruct)
    dyn = CC.Phonons.Phonons(unit, len(qgrid))
    for i, q in enumerate(qgrid):
        dyn.dynmats[i] = dynq[i]
        dyn.q_tot[i] = q
    dyn.AdjustQStar()
    dyn.Symmetrize()
    dyn.ForcePositiveDefinite()
    return dyn, size, order, superstruct


def dynamics_to_fc(dyn, size, order, n):
    fc = np.asarray(dyn.GetRealSpaceFC(size)).real * RY_EV / BOHR_ANG ** 2
    fc = fc.reshape(n, 3, n, 3).transpose(0, 2, 1, 3)
    out = np.zeros_like(fc)
    out[np.ix_(order, order)] = fc
    return out


def run(material, n_configs, seed):
    import sscha.Ensemble
    import sscha.Relax
    import sscha.SchaMinimizer

    temperature = SURROGATES[material]
    phonon, fcs, perms, p2s = load_surrogate(material)
    dyn, size, order, superstruct = build_dynamics(phonon, phonon.force_constants, material)
    calc = SurrogateCalculator(fcs, perms, p2s, np.asarray(phonon.supercell.positions),
                               np.asarray(phonon.supercell.cell), order)
    np.random.seed(seed)
    ensemble = sscha.Ensemble.Ensemble(dyn, temperature, size)
    minim = sscha.SchaMinimizer.SSCHA_Minimizer(ensemble)
    minim.set_minimization_step(0.1)
    minim.meaningful_factor = 0.2
    minim.kong_liu_ratio = 0.5
    minim.minim_struct = False  # high-symmetry sites: no internal relaxation
    relax = sscha.Relax.SSCHA(minim, ase_calculator=calc, N_configs=n_configs,
                              max_pop=MAX_POP)
    history = []

    def record(m):
        history.append(dynamics_to_fc(m.dyn, size, order, len(order)))

    relax.setup_custom_functions(custom_function_post=record)
    relax.relax(get_stress=False)
    final = dynamics_to_fc(relax.minim.dyn, size, order, len(order))
    return final, calc.calls, relax.minim.is_converged()


def main():
    materials = sys.argv[1:] or list(SURROGATES)
    for material in materials:
        out = ROOT / "validation_results" / material / "surrogate_design"
        out.mkdir(parents=True, exist_ok=True)
        fcs, meta = [], []
        for n_configs in POPULATION_SIZES:
            for rep in range(REPEATS):
                fc, calls, converged = run(material, n_configs, 1000 * n_configs + rep)
                fcs.append(fc)
                meta.append({"n_configs": n_configs, "repeat": rep, "force_calls": calls,
                             "converged": bool(converged)})
                print(material, meta[-1], flush=True)
        np.savez_compressed(out / "sscha_runs.npz", force_constants=np.asarray(fcs),
                            meta=json.dumps(meta))


if __name__ == "__main__":
    main()
