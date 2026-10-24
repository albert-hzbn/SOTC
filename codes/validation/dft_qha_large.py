#!/usr/bin/env python3
"""Converged-cell harmonic FD FC2 on the thermal-expansion volume grid.

Same lattice parameters as ``dft_qha.py``, in the 5x5x5 primitive (125-atom)
cell with the large-cell settings of ``run_fd_references.py`` (3x3x3 k mesh),
as the converged harmonic QHA baseline.

Writes validation_results/<m>/fd_references/qha_large/a<a>/harmonic_reference.npz
and phonopy_harmonic.yaml.

Usage: dft_qha_large.py <material> <a_ang> [--n-parallel 3]
"""

from __future__ import annotations

import argparse
from pathlib import Path
import sys
from types import SimpleNamespace

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "codes"))
sys.path.insert(0, str(Path(__file__).resolve().parent))

from run_fd_references import MATERIALS, large_settings, make_phonon  # noqa: E402
from run_real_anharmonic_ifcs import harmonic_reference  # noqa: E402


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("material")
    parser.add_argument("a", type=float)
    parser.add_argument("--n-parallel", type=int, default=3)
    args = parser.parse_args()
    spec = MATERIALS[args.material]
    size, kpts = large_settings(spec)
    local = dict(spec, a=args.a, kpts=kpts)
    out = (ROOT / "validation_results" / args.material / "fd_references" / "qha_large"
           / f"a{args.a:.4f}")
    out.mkdir(parents=True, exist_ok=True)
    run_args = SimpleNamespace(material=args.material, n_parallel=args.n_parallel,
                               tasks_per_qe=32, fit_only=False, force_rerun=False)
    result = harmonic_reference(make_phonon(spec, args.a, size), local, run_args, out)
    print(f"{args.material} a={args.a:.4f}: min freq {result['min_mesh_frequency_thz']:.3f} THz",
          flush=True)


if __name__ == "__main__":
    main()
