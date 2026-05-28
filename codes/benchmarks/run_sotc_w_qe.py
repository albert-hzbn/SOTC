#!/usr/bin/env python3
"""
SQTC + Quantum ESPRESSO benchmark: bcc W at 1500 K  (mildly anharmonic)
========================================================================
W (tungsten) is the highest-melting elemental metal (T_melt = 3695 K).
At T_design = 1500 K, T/T_melt = 0.41 — similar to Mo at 1500 K — making it
a clean harmonic reference for SQTC in a heavy refractory BCC metal.

Well-measured phonon dispersions exist from neutron scattering (Larose &
Brockhouse 1976) and many first-principles studies for direct comparison.

QE settings  (SSSP efficiency v1.3 pseudopotential)
-----------------------------------------------
  Pseudopotential : W_pbe_v1.2.uspp.F.UPF  (USPP GBRV-1.2)
  ecutwfc         : 30 Ry  (≈ 408 eV)  [SSSP recommended]
  ecutrho         : 240 Ry  (8 × ecutwfc for USPP)
  k-mesh (SC)     : 2×2×2  (appropriate for 64-atom bcc supercell)
  Smearing        : Marzari-Vanderbilt cold smearing, σ = 0.02 Ry
  SCF convergence : ΔE_tot < 1×10⁻⁸ Ry

Experimental references:
  T_D (calorimetric, low-T)  ≈ 310 K   (Kittel)
  ω_max ≈ 7.6 THz                       (Larose & Brockhouse 1976)
  Cv(300 K) ≈ 24.3 J/mol/K             (NIST Webbook)
  T_melt = 3695 K

Usage
-----
    QE_PW_CMD="mpirun -np 128 pw.x" python3 codes/benchmarks/run_sotc_w_qe.py
"""

import os
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).parent / "../"))

from sotc import SOTCRunner
from sotc.qe_io import QEForceCalculator

# ── Physical parameters: bcc W ───────────────────────────────────────────────

a_w      = 3.18           # Å  (experimental bcc W at ~1500 K; DFT-PBE ≈ 3.17 Å)
M_W      = 183.84         # amu
T_D_W    = 300.0          # K  (seed Debye temperature)
T_design = 1500.0         # K  (T/T_melt = 0.41 — similar to Mo)

N_SC = 64                 # 64-atom bcc supercell

# BCC primitive cell (1 atom per cell, body-centred)
prim_cell_w = 0.5 * a_w * np.array([
    [-1.0,  1.0,  1.0],
    [ 1.0, -1.0,  1.0],
    [ 1.0,  1.0, -1.0],
])

prim_pos_w = np.array([[0.0, 0.0, 0.0]])

WORK_DIR = Path("examples/sotc_w_qe_run")

# ── QE settings ───────────────────────────────────────────────────────────────

SSP_DIR = Path(
    os.environ.get(
        "SSSP_DIR",
        str(Path(__file__).parent.parent.parent / "pseudopotentials"),
    )
)
W_UPF = "W_pbe_v1.2.uspp.F.UPF"   # USPP GBRV-1.2

PW_CMD = os.environ.get(
    "QE_PW_CMD",
    "mpiexec -np 1 " + os.path.expanduser("~/Softwares/qe-7.5/bin/pw.x"),
)

# ── QE force calculator ───────────────────────────────────────────────────────

qe_calc = QEForceCalculator(
    species=["W"],
    pseudopotentials={"W": W_UPF},
    pseudo_dir=SSP_DIR,
    ecutwfc=30.0,                               # Ry  (SSSP efficiency)
    ecutrho=240.0,                              # Ry  (8 × ecutwfc for USPP)
    kpts=(2, 2, 2),                             # 2×2×2 for 64-atom bcc supercell
    smearing="marzari-vanderbilt",
    degauss=0.02,                               # Ry
    conv_thr=1.0e-8,                            # Ry
    pw_cmd=PW_CMD,
    workdir=WORK_DIR / "qe_scratch",
    prefix="w",
    n_parallel=1,
    extra_input_data={"electrons": {"diagonalization": "david", "diago_david_ndim": 4}},
)

# ── SQTC runner ───────────────────────────────────────────────────────────────

def main() -> None:
    print("\n" + "=" * 68)
    print(" SQTC + Quantum ESPRESSO: bcc W at 1500 K")
    print("=" * 68)
    print(f"  Supercell size   : {N_SC} atoms")
    print(f"  r_cutoff         : 5.0 Å  (1st + 2nd + 3rd NN for bcc W)")
    print(f"  T_design         : {T_design} K  (T/T_melt = {T_design/3695:.2f})")
    print(f"  T_D (input)      : {T_D_W} K")
    print(f"  ecutwfc          : {qe_calc.ecutwfc:.0f} Ry")
    print(f"  ecutrho          : {qe_calc.ecutrho:.0f} Ry")
    print(f"  k-mesh (SC)      : {qe_calc.kpts}")
    print(f"  pw_cmd           : {qe_calc.pw_cmd}")
    print()

    runner = SOTCRunner(
        element="W",
        mass_amu=M_W,
        prim_cell=prim_cell_w,
        prim_positions=prim_pos_w,
        T=T_design,
        T_D=T_D_W,
        n_atoms_sc=N_SC,
        force_calculator=qe_calc,
        r_cutoff=5.0,
        r_max_corr=10.0,
        n_ensemble=3,       # T/T_melt = 0.41; relatively harmonic like Mo
        work_dir=WORK_DIR,
        verbosity=1,
        ridge_alpha=1e-3,
        symmetrize_bonds=True,
    )

    T_values = np.array([100., 200., 300., 500., 700., 1000., 1500.])

    res = runner.run(
        n_sc_iterations=10,
        epsilon_conv=0.003,
        min_iterations=4,
        mixing=0.30,
        T_values=T_values,
        q_mesh_cv=(20, 20, 20),
    )

    print("\n── Final properties ──")
    cv300 = None
    for T, cv in zip(res["T_values"], res["C_V_scan"]):
        print(f"  Cv({T:.0f} K) = {cv:.4f} J/(mol·K)")
        if abs(T - 300.0) < 1:
            cv300 = cv

    print(f"\n  T_D (spectral-moment) = {res['T_D_effective']:.2f} K  (expt low-T: ~310 K)")
    if cv300 is not None:
        print(f"  Cv(300 K)             = {cv300:.4f} J/(mol·K)  (expt: 24.3)")
    print(f"  ZPE                   = {res.get('ZPE_meV', float('nan')):.2f} meV/atom")
    print(f"  Converged             : {res['converged']}")
    print(f"  Iterations            : {res['n_iterations']}")


if __name__ == "__main__":
    main()
