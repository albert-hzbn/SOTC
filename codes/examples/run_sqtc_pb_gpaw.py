#!/usr/bin/env python3
"""
SQTC + GPAW: fcc Pb — harmonic vs anharmonic decomposition
===========================================================
Pb (lead) is one of the most anharmonic elemental metals:
  T_D  ≈  88 K  (very soft)
  T_melt = 600.6 K
  Grüneisen γ ≈ 2.73  (compare: Al ≈ 2.17, Cu ≈ 1.96)

At T = 500 K (T/T_melt ≈ 0.83) the phonon frequencies are measurably
softened relative to the harmonic (0 K) values.  This temperature
renormalization — captured by SQTC self-consistency — is the dominant
anharmonic signature.

Strategy
--------
Two independent SQTC runs at the same geometry but different temperatures:

  Run A — "harmonic reference" (T = 100 K)
    SQTC is run at T = 100 K, where anharmonicity is small.
    The converged IFCs represent the nearly-harmonic force constants.
    These IFCs are then used to compute C_V^harm(T) for all T.

  Run B — "anharmonic" (T = 500 K)
    SQTC is run at T = 500 K, where thermal fluctuations are large and
    the self-consistency loop renormalizes the IFCs (phonon softening).
    The converged IFCs give C_V^anh(T) for all T.

The anharmonic correction at temperature T is:
    ΔC_V^anh(T) = C_V^anh(T) - C_V^harm(T)

When both runs have converged, compare_al_pb_gpaw.py plots the
decomposition C_V^total = C_V^harm + ΔC_V^anh.

GPAW settings
-------------
  Plane-wave cutoff : 350 eV  (Pb PAW — relatively light cutoff; the heavy
                               core is absorbed by the PAW transformation)
  XC functional     : PBE
  k-mesh (supercell): 2×2×2  (32-atom cell, ≈14×14×14 Å box)
  Smearing          : Fermi-Dirac, σ = 0.10 eV
  SCF convergence   : ΔE < 1×10⁻⁷ eV/electron

Output
------
    sqtc_pb_gpaw_run/harmonic/sqtc_results.json    (Run A)
    sqtc_pb_gpaw_run/anharmonic/sqtc_results.json  (Run B)
    sqtc_pb_gpaw_run/pb_harm_vs_anh.json           (merged comparison data)

Usage
-----
    python codes/examples/run_sqtc_pb_gpaw.py

    # To run only one temperature:
    python codes/examples/run_sqtc_pb_gpaw.py --only harmonic
    python codes/examples/run_sqtc_pb_gpaw.py --only anharmonic
"""

import json
import sys
import argparse
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).parent / "../"))

from sqtc import SQTCRunner, PhononCalculator
from sqtc.gpaw_io import GPAWForceCalculator

# ── Physical parameters: fcc Pb ──────────────────────────────────────────────

a_pb   = 4.9508        # Å  (DFT-PBE equilibrium lattice constant)
M_Pb   = 207.2         # amu
T_D_Pb = 88.0          # K   (Debye temperature from low-T specific heat)
T_melt = 600.6         # K   (Pb melting point)

N_SC   = 32            # supercell size

prim_cell_pb = 0.5 * a_pb * np.array([
    [0.0, 1.0, 1.0],
    [1.0, 0.0, 1.0],
    [1.0, 1.0, 0.0],
])                     # fcc primitive cell [Å]

prim_pos_pb = np.array([[0.0, 0.0, 0.0]])

# Run temperatures
T_HARM = 100.0         # K  — harmonic reference (T ≪ T_melt, mild anharmonicity)
T_ANH  = 500.0         # K  — strongly anharmonic (T/T_melt = 0.83)

ROOT_DIR = Path("sqtc_pb_gpaw_run")

# Shared temperature grid for the C_V scan (both runs output C_V at these T)
T_SCAN = np.array([50.0, 88.0, 100.0, 150.0, 200.0, 250.0, 300.0,
                   350.0, 400.0, 450.0, 500.0, 550.0, 600.0])

# ── Argument parsing ──────────────────────────────────────────────────────────

parser = argparse.ArgumentParser(description="SQTC+GPAW: Pb harmonic vs anharmonic")
parser.add_argument(
    "--only",
    choices=["harmonic", "anharmonic"],
    default=None,
    help="Run only one temperature (default: run both sequentially).",
)
args = parser.parse_args()

RUN_HARM = (args.only is None) or (args.only == "harmonic")
RUN_ANH  = (args.only is None) or (args.only == "anharmonic")


# ── Helper: build GPAW calculator for a given working directory ───────────────

def make_gpaw_calc(subdir: Path) -> GPAWForceCalculator:
    """Return a fresh GPAWForceCalculator writing output to *subdir*."""
    subdir.mkdir(parents=True, exist_ok=True)
    return GPAWForceCalculator(
        species=["Pb"],  # tiled automatically to match supercell
        cutoff=350.0,
        xc="PBE",
        kpts=(2, 2, 2),
        smearing_width=0.10,
        convergence_energy=1e-7,
        eigensolver="rmm-diis",
        txt=str(subdir / "gpaw_pb.out"),
        workdir=subdir / "gpaw_restarts",
    )


# ── Helper: run one SQTC temperature ─────────────────────────────────────────

def run_sqtc_pb(T: float, label: str) -> dict:
    """
    Run SQTC for Pb at temperature *T* and return the results dict.

    The results JSON is saved to ``ROOT_DIR / label / sqtc_results.json``.
    If that file already exists the run is skipped and the file is loaded
    directly (restart behaviour — useful if the cluster job was interrupted).
    """
    work_dir = ROOT_DIR / label
    result_json = work_dir / "sqtc_results.json"

    if result_json.exists():
        print(f"\n  [{label}] Found existing results at {result_json} — skipping run.")
        with open(result_json) as fh:
            return json.load(fh)

    print(f"\n{'='*68}")
    print(f" SQTC + GPAW: fcc Pb   [{label}]   T = {T:.0f} K")
    print("=" * 68)
    print(f"  Supercell size  : {N_SC} atoms")
    print(f"  r_cutoff        : 5.5 Å  (1st + 2nd NN for fcc Pb)")
    print(f"  T_D (input)     : {T_D_Pb:.0f} K")
    print(f"  T_melt          : {T_melt:.1f} K   (T/T_melt = {T/T_melt:.2f})")
    print(f"  Grüneisen γ     : ~2.73  (strongly anharmonic)")
    print()

    calc = make_gpaw_calc(work_dir)

    runner = SQTCRunner(
        element="Pb",
        mass_amu=M_Pb,
        prim_cell=prim_cell_pb,
        prim_positions=prim_pos_pb,
        T=T,
        T_D=T_D_Pb,
        n_atoms_sc=N_SC,
        force_calculator=calc,
        r_cutoff=5.5,       # Å  — 1st NN: 3.50 Å, 2nd NN: 4.95 Å
        r_max_corr=9.0,
        n_ensemble=5,
        work_dir=work_dir,
        verbosity=1,
        ridge_alpha=5e-3,   # slightly stronger regularisation: Pb is very soft
        symmetrize_bonds=True,
    )

    res = runner.run(
        n_sc_iterations=4,
        epsilon_conv=0.003,
        mixing=0.40,
        T_values=T_SCAN,
        q_mesh_cv=(8, 8, 8),
    )

    print(f"\n  [{label}]  T_D_eff = {res['T_D_effective']:.1f} K  "
          f"  C_V({T:.0f} K) = {res['C_V_jmolk']:.3f} J/(mol·K)")

    return res


# ── Run A: harmonic reference ─────────────────────────────────────────────────

res_harm = {}
if RUN_HARM:
    res_harm = run_sqtc_pb(T_HARM, label="harmonic")

# ── Run B: anharmonic ─────────────────────────────────────────────────────────

res_anh = {}
if RUN_ANH:
    res_anh = run_sqtc_pb(T_ANH, label="anharmonic")

# ── Merge results: compute ΔC_V^anh = C_V^anh - C_V^harm ────────────────────

if RUN_HARM and RUN_ANH:
    cv_harm = np.array(res_harm["C_V_scan"])   # shape (n_T,)  harmonic IFCs
    cv_anh  = np.array(res_anh["C_V_scan"])    # shape (n_T,)  anharmonic IFCs
    delta_cv = cv_anh - cv_harm                 # anharmonic correction

    THREE_R = 3.0 * 8.314462618               # J/(mol·K) Dulong-Petit limit

    print("\n" + "=" * 68)
    print(" Pb: harmonic vs anharmonic decomposition")
    print("=" * 68)
    print(f"  T_D^harm  = {res_harm['T_D_effective']:.1f} K")
    print(f"  T_D^anh   = {res_anh['T_D_effective']:.1f} K")
    print(f"  Phonon softening: ΔT_D = "
          f"{res_harm['T_D_effective'] - res_anh['T_D_effective']:+.1f} K  "
          f"(negative = softer at high T)")
    print()
    print(f"  {'T [K]':>8}  {'C_V^harm':>12}  {'C_V^anh':>12}  "
          f"{'ΔC_V^anh':>12}  {'ΔC_V/3R':>10}")
    print("  " + "-" * 62)
    for T_v, cv_h, cv_a, dcv in zip(T_SCAN, cv_harm, cv_anh, delta_cv):
        print(f"  {T_v:8.1f}  {cv_h:12.4f}  {cv_a:12.4f}  "
              f"  {dcv:+11.4f}  {dcv/THREE_R:+9.4f}")

    # Experimental C_V for Pb (NIST / Hultgren 1973) [J/(mol·K)]
    # Selected data points from experiment
    exp_pb_T  = np.array([50.0,  100.0, 200.0, 300.0, 400.0, 500.0, 600.0])
    exp_pb_cv = np.array([18.2,   24.7,  26.4,  27.4,  28.7,  30.3,  32.1])

    comparison = {
        "system": "fcc Pb",
        "a_ang": a_pb,
        "M_amu": M_Pb,
        "T_D_input": T_D_Pb,
        "T_melt": T_melt,
        "gruneisen": 2.73,
        "T_harmonic_run": T_HARM,
        "T_anharmonic_run": T_ANH,
        "T_D_harm": res_harm["T_D_effective"],
        "T_D_anh": res_anh["T_D_effective"],
        "T_scan": T_SCAN.tolist(),
        "cv_harm_jmolk": cv_harm.tolist(),
        "cv_anh_jmolk": cv_anh.tolist(),
        "delta_cv_jmolk": delta_cv.tolist(),
        "exp_T": exp_pb_T.tolist(),
        "exp_cv_jmolk": exp_pb_cv.tolist(),
        "converged_harm": res_harm.get("converged", False),
        "converged_anh": res_anh.get("converged", False),
    }

    ROOT_DIR.mkdir(parents=True, exist_ok=True)
    out_path = ROOT_DIR / "pb_harm_vs_anh.json"
    out_path.write_text(json.dumps(comparison, indent=2))
    print(f"\n  Merged comparison data → {out_path}")
    print(f"  Pass this to compare_al_pb_gpaw.py for the full comparison plot.")

elif RUN_HARM:
    print(f"\n  Harmonic run done.  Run with --only anharmonic next.")

elif RUN_ANH:
    print(f"\n  Anharmonic run done.  Run with --only harmonic next.")
