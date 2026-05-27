#!/usr/bin/env python3
"""
SQTC + Quantum ESPRESSO benchmark: fcc Ni at 1200 K  (moderately anharmonic)
=============================================================================
Ni at T/T_melt = 0.69, above the Curie temperature (631 K), behaves as a
paramagnetic metal.  For phonon benchmarking, non-spin-polarized DFT is used
(standard approximation in phonon databases).  SQTC tests the renormalised IFC
at high T in a d-band transition metal.

QE settings  (SSSP efficiency v1.3 pseudopotential)
-----------------------------------------------
  Pseudopotential : ni_pbe_v1.4.uspp.F.UPF  (USPP GBRV-1.4)
  ecutwfc         : 45 Ry  (≈ 612 eV)  [SSSP recommended]
  ecutrho         : 360 Ry  (8 × ecutwfc for USPP)
  k-mesh (SC)     : 2×2×2  (scales from 8×8×8 on the primitive cell)
  Smearing        : Marzari-Vanderbilt cold smearing, σ = 0.02 Ry
  SCF convergence : ΔE_tot < 1×10⁻⁸ Ry
  Note: nspin = 1 (non-spin-polarized — valid approximation above Curie T)

Experimental references:
  T_D (calorimetric, low-T)  ≈ 450 K   (Kittel; NIST)
  T_D (effective, room-T)    ≈ 390 K   (Grimvall 1981)
  Cv(300 K) ≈ 26.1 J/mol/K             (NIST Webbook; includes magnetic)
  T_melt = 1728 K

Usage
-----
    QE_PW_CMD="mpirun -np 64 pw.x" python3 codes/benchmarks/run_sotc_ni_qe.py
"""

import os
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).parent / "../"))

from sotc import SOTCRunner
from sotc.qe_io import QEForceCalculator

# ── Physical parameters: fcc Ni ──────────────────────────────────────────────

a_ni     = 3.56           # Å  (experimental fcc Ni at ~1200 K; DFT-PBE ≈ 3.52 Å)
M_Ni     = 58.6934        # amu
T_D_Ni   = 380.0          # K  (seed Debye temperature)
T_design = 1200.0         # K  (T/T_melt = 0.69)

N_SC = 32                 # 32-atom 2×2×2 fcc supercell

prim_cell_ni = 0.5 * a_ni * np.array([
    [0.0, 1.0, 1.0],
    [1.0, 0.0, 1.0],
    [1.0, 1.0, 0.0],
])

prim_pos_ni = np.array([[0.0, 0.0, 0.0]])

WORK_DIR = Path("examples/sotc_ni_qe_run")

# ── QE settings ───────────────────────────────────────────────────────────────

SSP_DIR = Path(
    os.environ.get(
        "SSSP_DIR",
        str(Path(__file__).parent.parent.parent / "pseudopotentials"),
    )
)
NI_UPF = "ni_pbe_v1.4.uspp.F.UPF"   # USPP GBRV-1.4

PW_CMD = os.environ.get(
    "QE_PW_CMD",
    "mpiexec -np 1 " + os.path.expanduser("~/Softwares/qe-7.5/bin/pw.x"),
)

# ── QE force calculator ───────────────────────────────────────────────────────

qe_calc = QEForceCalculator(
    species=["Ni"],
    pseudopotentials={"Ni": NI_UPF},
    pseudo_dir=SSP_DIR,
    ecutwfc=45.0,                               # Ry  (SSSP efficiency)
    ecutrho=360.0,                              # Ry  (8 × ecutwfc for USPP)
    kpts=(2, 2, 2),                             # 2×2×2 for 32-atom fcc supercell
    smearing="marzari-vanderbilt",
    degauss=0.02,                               # Ry
    conv_thr=1.0e-7,                            # Ry  (relaxed; d-band metals often need this)
    pw_cmd=PW_CMD,
    workdir=WORK_DIR / "qe_scratch",
    prefix="ni",
    n_parallel=1,
    extra_input_data={
        "system": {"nosym": True},       # prevent ASE eigenvalue parse error with USPP k-mesh
        "electrons": {
            "mixing_beta": 0.1,           # small beta for d-band convergence
            "mixing_mode": "local-TF",    # TF-screened mixing; better for metals with d-bands
            "electron_maxstep": 500,      # allow more iterations for thermally displaced cells
        },
    },
)

# ── SQTC runner ───────────────────────────────────────────────────────────────

def main() -> None:
    print("\n" + "=" * 68)
    print(" SQTC + Quantum ESPRESSO: fcc Ni at 1200 K")
    print("=" * 68)
    print(f"  Supercell size   : {N_SC} atoms")
    print(f"  r_cutoff         : 4.5 Å  (1st + 2nd NN for fcc Ni)")
    print(f"  T_design         : {T_design} K  (T/T_melt = {T_design/1728:.2f})")
    print(f"  T_D (input)      : {T_D_Ni} K")
    print(f"  Note             : non-spin-polarized (above Curie T = 631 K)")
    print(f"  ecutwfc          : {qe_calc.ecutwfc:.0f} Ry")
    print(f"  ecutrho          : {qe_calc.ecutrho:.0f} Ry")
    print(f"  k-mesh (SC)      : {qe_calc.kpts}")
    print(f"  pw_cmd           : {qe_calc.pw_cmd}")
    print()

    runner = SOTCRunner(
        element="Ni",
        mass_amu=M_Ni,
        prim_cell=prim_cell_ni,
        prim_positions=prim_pos_ni,
        T=T_design,
        T_D=T_D_Ni,
        n_atoms_sc=N_SC,
        force_calculator=qe_calc,
        r_cutoff=4.5,
        r_max_corr=9.0,
        n_ensemble=5,       # T/T_melt = 0.69; 5 snaps × 189 data points
        work_dir=WORK_DIR,
        verbosity=1,
        ridge_alpha=1e-3,
        symmetrize_bonds=True,
    )

    T_values = np.array([100., 200., 300., 400., 500., 700., 900., 1200.])

    res = runner.run(
        n_sc_iterations=12,
        epsilon_conv=0.003,
        min_iterations=4,
        mixing=0.25,
        T_values=T_values,
        q_mesh_cv=(20, 20, 20),
    )

    print("\n── Final properties ──")
    cv300 = None
    for T, cv in zip(res["T_values"], res["C_V_scan"]):
        print(f"  Cv({T:.0f} K) = {cv:.4f} J/(mol·K)")
        if abs(T - 300.0) < 1:
            cv300 = cv

    print(f"\n  T_D (spectral-moment) = {res['T_D']:.2f} K  (expt low-T: ~450 K)")
    if cv300 is not None:
        print(f"  Cv(300 K)             = {cv300:.4f} J/(mol·K)  (expt: ~26.1)")
    print(f"  ZPE                   = {res.get('ZPE_meV', float('nan')):.2f} meV/atom")
    print(f"  Converged             : {res['converged']}")
    print(f"  Iterations            : {res['n_iterations']}")


if __name__ == "__main__":
    main()
