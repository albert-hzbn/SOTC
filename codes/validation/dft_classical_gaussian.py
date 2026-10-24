#!/usr/bin/env python3
"""Classical-amplitude Gaussian ensemble in the SOTC cell (Al, 300 K).

Separates the two differences between the AIMD-TDEP FC2 and the quantum
effective FC2: canonical +/- pairs are drawn from the same-cell harmonic FD
trial as the iteration-0 canonical pool, but with classical mode amplitudes
sigma = sqrt(k_B T)/omega. The effective FC2 fitted to them differs from
the quantum iteration-0 FC2 only by the amplitudes, and from the AIMD-TDEP
FC2 only by the Gaussian displacement statistics (and the trial).

Writes validation_results/<m>/primary_cell/classical_gaussian/configs.npz.

Usage: dft_classical_gaussian.py <material> [--n 40]
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

from analyze_primary_archives import PRIMARY  # noqa: E402
from run_primary_cell_controls import overrides, primary_spec  # noqa: E402
from run_real_anharmonic_ifcs import run_one_state  # noqa: E402
from sotc.optimal_design import thermal_modes  # noqa: E402
from surrogate_design_study import gaussian_configs, paired  # noqa: E402

KB_EV = 8.617333262e-5
HBAR_EVS = 6.582119569e-16


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("material")
    parser.add_argument("--n", type=int, default=40)
    args = parser.parse_args()

    import phonopy

    material = args.material
    base = ROOT / "validation_results" / material / "primary_cell"
    out = base / "classical_gaussian"
    out.mkdir(parents=True, exist_ok=True)
    phonon = phonopy.load(base / "fd" / "phonopy_harmonic.yaml")
    with np.load(base / "fd" / "harmonic_reference.npz") as h:
        f0 = np.asarray(h["reference_forces_ev_ang"])
    sup = phonon.supercell
    eq, cell, symbols = np.asarray(sup.positions), np.asarray(sup.cell), list(sup.symbols)
    temperature = PRIMARY[material]["T"]
    freq, patterns, sigma_q = thermal_modes(np.asarray(phonon.force_constants),
                                            np.asarray(sup.masses), temperature)
    # Classical limit of sigma^2 = hbar/(2 omega) coth(hbar omega / 2 k T).
    x = HBAR_EVS * 2 * np.pi * freq * 1e12 / (2 * KB_EV * temperature)
    sigma_c = sigma_q / np.sqrt(x / np.tanh(x))
    rng = np.random.default_rng(4242)
    u = paired(gaussian_configs(rng, args.n // 2, patterns, sigma_c))
    u -= u.mean(axis=1, keepdims=True)
    spec = primary_spec(material)
    run_args = SimpleNamespace(material=material, n_parallel=4, tasks_per_qe=32,
                               fit_only=False, force_rerun=False)
    with concurrent.futures.ThreadPoolExecutor(max_workers=4) as pool:
        res = list(pool.map(lambda i: run_one_state(i, eq + u[i], cell, symbols, spec, run_args,
                                                    out / "qe", overrides(spec)),
                            range(len(u))))
    df = np.asarray([np.asarray(r["forces_ev_ang"]) - f0 for r in res])
    np.savez_compressed(out / "configs.npz", displacements_ang=u,
                        delta_forces_ev_ang=df - df.mean(axis=1, keepdims=True),
                        u_rms_classical=float(np.sqrt((u ** 2).sum(-1).mean())))
    print(f"{material}: {len(u)} classical Gaussian configurations, "
          f"u_rms {np.sqrt((u ** 2).sum(-1).mean()):.3f} A", flush=True)


if __name__ == "__main__":
    main()
