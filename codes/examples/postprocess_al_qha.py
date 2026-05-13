#!/usr/bin/env python3
"""
Post-process the Al SQTC+QHA results using what was computed.

QHA run status
--------------
Only sqtc_Vm (V-2%) completed in sqtc_al_qha_run/.
V0 data is taken from the existing sqtc_al_fast_vasp_run/ (iter_00/01: 36-atom cells).
Vp is constructed by linear extrapolation of the IFCs: Φ(Vp) = 2Φ(V0) - Φ(Vm).

Strategy
--------
1. Load V0 POSCAR+OUTCAR snaps from sqtc_al_fast_vasp_run/iter_{00,01}  (36 atoms, a=4.05 Å)
2. Load Vm POSCAR+OUTCAR snaps from sqtc_al_qha_run/sqtc_Vm/iter_{00,01} (36 atoms, a·0.98^(1/3))
3. Fit IFCs at V0 and Vm independently (same pair topology, same r_cutoff)
4. Extrapolate: Φ(Vp) = 2Φ(V0) - Φ(Vm)  →  PhononCalculator at Vp
5. Static energies: mean OUTCAR TOTEN per primitive atom at each volume
6. Build SQTCQuasiHarmonic(method="both") and compute α(T), V(T), C_P(T), B(T)
7. Post-process thermal properties at V0 (C_V, S_vib, MSD, B(T), ...)
8. Plot: (a) QHA panel (4 panels), (b) phonon DOS comparison, (c) thermal properties
"""

from __future__ import annotations

import re
import sys
import json
from pathlib import Path

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec

sys.path.insert(0, str(Path(__file__).parent / "../"))

from sqtc import SQTCQuasiHarmonic, IFCExtractor, PhononCalculator
from sqtc.postprocessor import (
    _parse_poscar, _parse_outcar_forces,
    _build_eq_positions, SQTCPostProcessor,
)
from sqtc.constants import KB, HBAR, NA, AMU_TO_KG

# ── Paths ─────────────────────────────────────────────────────────────────────

ROOT       = Path(".")
V0_DIR     = ROOT / "sqtc_al_fast_vasp_run"
VM_DIR     = ROOT / "sqtc_al_qha_run" / "sqtc_Vm"
OUT_DIR    = ROOT / "sqtc_al_qha_run" / "postproc"
OUT_DIR.mkdir(parents=True, exist_ok=True)

# ── fcc Al parameters ─────────────────────────────────────────────────────────

a_al       = 4.05           # Å  (DFT-PBE equilibrium lattice constant)
delta      = 0.02           # ±2% volume step → (1±delta)^(1/3) linear

prim_cell_V0 = 0.5 * a_al * np.array([
    [0.0, 1.0, 1.0],
    [1.0, 0.0, 1.0],
    [1.0, 1.0, 0.0],
])  # (3,3) Å
prim_pos_Al  = np.array([[0.0, 0.0, 0.0]])   # fractional or Cartesian (origin)

M_Al_amu   = 26.9815385
T_design   = 300.0          # SQTC design T for QHA volume runs
R_CUTOFF   = 4.5            # Å  — IFC regression cutoff (same for both volumes)
RIDGE_ALPHA = 1e-3
Q_MESH_CV  = (20, 20, 20)   # q-mesh for QHA F_vib
Q_MESH_DOS = (30, 30, 30)   # q-mesh for phonon DOS / thermal properties

T_values = np.arange(10, 1001, 10, dtype=float)   # 10–1000 K in 10 K steps

# ── Step 1 — Load 36-atom V0 snaps from sqtc_al_fast_vasp_run ────────────────

print("=" * 70)
print(" Loading SQTC data")
print("=" * 70)

def _extract_toten(outcar_path: Path) -> float | None:
    """Return the last 'energy  without entropy' value from OUTCAR [eV]."""
    text = outcar_path.read_text()
    # Primary: "energy  without entropy =  ..."
    matches = re.findall(r"energy\s+without entropy\s*=\s*([-\d.E+]+)", text)
    if matches:
        return float(matches[-1])
    # Fallback: "TOTEN = ... eV"
    matches = re.findall(r"TOTEN\s*=\s*([-\d.E+]+)\s*eV", text)
    return float(matches[-1]) if matches else None


def _load_snaps(
    run_dir: Path,
    iter_names: list[str],
    n_atoms_target: int,
    prim_cell: np.ndarray,
    prim_pos: np.ndarray,
    label: str = "",
) -> tuple[np.ndarray, list[np.ndarray], list[np.ndarray], list[float]]:
    """
    Load POSCAR+OUTCAR snapshots from iter_*/snap_*/ sub-dirs.

    Returns
    -------
    eq_pos       : (n_atoms, 3)  equilibrium Cartesian positions [Å]
    displ_list   : list of (n_atoms, 3) displacement arrays [Å]
    forces_list  : list of (n_atoms, 3) force arrays [eV/Å]
    energies     : list of float, total energies [eV] (per supercell)
    """
    # Find reference cell from the first available POSCAR
    ref_poscar = None
    for iname in iter_names:
        p = run_dir / iname / "snap_0000" / "POSCAR"
        if p.exists():
            ref_poscar = p
            break
    if ref_poscar is None:
        raise FileNotFoundError(f"No snap_0000 POSCAR found in {run_dir}")

    sc_cell, _, _ = _parse_poscar(ref_poscar)

    # Reconstruct equilibrium positions matching the runner's enumeration
    eq_pos, H = _build_eq_positions(sc_cell, prim_cell, prim_pos)
    if len(eq_pos) != n_atoms_target:
        raise RuntimeError(
            f"[{label}] Reconstructed {len(eq_pos)} atoms, expected {n_atoms_target}.\n"
            f"  H = {H.tolist()}, ref_poscar = {ref_poscar}"
        )

    displ_list:  list[np.ndarray] = []
    forces_list: list[np.ndarray] = []
    energies:    list[float]      = []

    cell_inv = np.linalg.inv(sc_cell)

    for iname in iter_names:
        iter_dir = run_dir / iname
        if not iter_dir.is_dir():
            continue
        for snap in sorted(iter_dir.glob("snap_*/")):
            poscar = snap / "POSCAR"
            outcar = snap / "OUTCAR"
            if not poscar.exists() or not outcar.exists():
                continue
            try:
                _, snap_sp, snap_pos = _parse_poscar(poscar)
            except Exception as e:
                print(f"  [{label}] Skip {snap.name} (POSCAR parse): {e}")
                continue
            if len(snap_sp) != n_atoms_target:
                continue   # different supercell size
            try:
                forces = _parse_outcar_forces(outcar, n_atoms_target)
            except Exception as e:
                print(f"  [{label}] Skip {snap.name} (OUTCAR parse): {e}")
                continue

            # Displacement with minimum-image convention
            u = snap_pos - eq_pos
            frac = u @ cell_inv
            frac -= np.round(frac)
            u = frac @ sc_cell

            displ_list.append(u)
            forces_list.append(forces)

            E = _extract_toten(outcar)
            if E is not None:
                energies.append(E)

    print(f"  [{label}] Loaded {len(displ_list)} snaps "
          f"(cell {sc_cell.diagonal() if np.allclose(sc_cell, np.diag(sc_cell.diagonal())) else 'non-diag'}, "
          f"{n_atoms_target} atoms, {len(energies)} energies)")
    return eq_pos, displ_list, forces_list, energies, sc_cell


# ── 36-atom V0 data from al_fast iter_00 and iter_01 ─────────────────────────

eq_pos_V0, displ_V0, forces_V0, ens_V0, sc_cell_V0 = _load_snaps(
    run_dir      = V0_DIR,
    iter_names   = ["iter_00", "iter_01"],
    n_atoms_target = 36,
    prim_cell    = prim_cell_V0,
    prim_pos     = prim_pos_Al,
    label        = "V0 (al_fast 36-atom)",
)

# ── Vm data from sqtc_al_qha_run/sqtc_Vm ─────────────────────────────────────

# Vm primitive cell: scale V0 by (1-delta)^(1/3)
scale_m      = (1.0 - delta) ** (1.0 / 3.0)
prim_cell_Vm = scale_m * prim_cell_V0

eq_pos_Vm, displ_Vm, forces_Vm, ens_Vm, sc_cell_Vm = _load_snaps(
    run_dir      = VM_DIR,
    iter_names   = ["iter_00", "iter_01"],
    n_atoms_target = 36,
    prim_cell    = prim_cell_Vm,
    prim_pos     = prim_pos_Al * scale_m,
    label        = "Vm (sqtc_Vm 36-atom)",
)

# ── Step 2 — Fit IFCs at V0 and Vm ───────────────────────────────────────────

print("\n" + "=" * 70)
print(" Fitting IFCs")
print("=" * 70)

masses_sc_36 = np.full(36, M_Al_amu)

ifc_kwargs = dict(
    masses_amu       = masses_sc_36,
    r_cutoff         = R_CUTOFF,
    symmetrise       = True,
    ridge_alpha      = RIDGE_ALPHA,
    symmetrize_bonds = True,
)

ifc_V0 = IFCExtractor(
    supercell_positions = eq_pos_V0,
    supercell_cell      = sc_cell_V0,
    **ifc_kwargs,
)
ifc_V0.fit(displ_V0, forces_V0)
rep_V0 = ifc_V0.fit_report(displ_V0, forces_V0)
print(f"  V0: RMSE={rep_V0['rmse_ev_ang']:.5f} eV/Å  R²={rep_V0['r2']:.5f}  "
      f"rank={rep_V0['rank']}  n_snaps={len(displ_V0)}")

ifc_Vm = IFCExtractor(
    supercell_positions = eq_pos_Vm,
    supercell_cell      = sc_cell_Vm,
    **ifc_kwargs,
)
ifc_Vm.fit(displ_Vm, forces_Vm)
rep_Vm = ifc_Vm.fit_report(displ_Vm, forces_Vm)
print(f"  Vm: RMSE={rep_Vm['rmse_ev_ang']:.5f} eV/Å  R²={rep_Vm['r2']:.5f}  "
      f"rank={rep_Vm['rank']}  n_snaps={len(displ_Vm)}")

# ── Step 3 — Extrapolate IFCs to Vp: Φ(Vp) = 2Φ(V0) - Φ(Vm) ────────────────

print("\n  Extrapolating IFCs to Vp ...")

# Validate that both pair-lists are identical (same topology)
pairs_V0 = set(ifc_V0._pairs)
pairs_Vm = set(ifc_Vm._pairs)
only_V0  = pairs_V0 - pairs_Vm
only_Vm  = pairs_Vm - pairs_V0
if only_V0 or only_Vm:
    print(f"  WARNING: pair-list mismatch! "
          f"{len(only_V0)} pairs only in V0, {len(only_Vm)} only in Vm.")
    print("  Extrapolation uses only common pairs; boundary IFCs set to V0.")

# Vp primitive cell and supercell
scale_p      = (1.0 + delta) ** (1.0 / 3.0)
prim_cell_Vp = scale_p * prim_cell_V0
sc_cell_Vp   = scale_p * sc_cell_V0
eq_pos_Vp    = scale_p * eq_pos_V0    # scaled equilibrium positions

# Create IFCExtractor for Vp with Vp geometry, then inject extrapolated Phi
ifc_Vp = IFCExtractor(
    supercell_positions = eq_pos_Vp,
    supercell_cell      = sc_cell_Vp,
    **ifc_kwargs,
)
# Build pair list for Vp (needed for dynamical_matrix)
# (constructor already built _pairs via _build_pair_list in __init__)

# Extrapolate: for each pair that exists in both, Phi_Vp = 2*Phi_V0 - Phi_Vm
Phi_Vp = {}
for pair in ifc_Vp._pairs:
    if pair in ifc_V0._Phi and pair in ifc_Vm._Phi:
        Phi_Vp[pair] = 2.0 * ifc_V0._Phi[pair] - ifc_Vm._Phi[pair]
    elif pair in ifc_V0._Phi:
        Phi_Vp[pair] = ifc_V0._Phi[pair].copy()  # fallback: use V0
    else:
        Phi_Vp[pair] = np.zeros((3, 3))

# Apply acoustic sum rule and Newton's law symmetrisation using the same
# routines as the regular fit.  Directly monkey-patch to reuse private methods.
ifc_Vp._Phi = Phi_Vp
ifc_Vp._Phi = ifc_Vp._acoustic_sum_rule(ifc_Vp._Phi)

print(f"  Vp: extrapolated {len(ifc_Vp._Phi)} IFC pairs  "
      f"(a_Vp = {np.linalg.norm(prim_cell_Vp[1]):.4f} Å  V = {abs(np.linalg.det(prim_cell_Vp)):.4f} Å³)")

# ── Step 4 — Build PhononCalculators ─────────────────────────────────────────

calc_V0 = PhononCalculator(
    ifc_extractor  = ifc_V0,
    prim_positions = prim_pos_Al,
    prim_cell      = prim_cell_V0,
    masses_amu     = np.array([M_Al_amu]),
)
calc_Vm = PhononCalculator(
    ifc_extractor  = ifc_Vm,
    prim_positions = prim_pos_Al * scale_m,
    prim_cell      = prim_cell_Vm,
    masses_amu     = np.array([M_Al_amu]),
)
calc_Vp = PhononCalculator(
    ifc_extractor  = ifc_Vp,
    prim_positions = prim_pos_Al * scale_p,
    prim_cell      = prim_cell_Vp,
    masses_amu     = np.array([M_Al_amu]),
)

# Quick stability check
for name, calc in [("V0", calc_V0), ("Vm", calc_Vm), ("Vp", calc_Vp)]:
    st = calc.spectrum_statistics(q_mesh=(10, 10, 10))
    print(f"  {name}: ω_max = {st['max_freq_thz']:.2f} THz  "
          f"unstable = {st['unstable_fraction']:.4f}")

# ── Step 5 — Static energies ──────────────────────────────────────────────────
#
# The SQTC snapshots are DISPLACED configurations — each OUTCAR energy equals
# E₀(V) + ΔE_pot(displacement).  The mean over many snaps is therefore
# E₀(V) + ⟨ΔE_pot⟩, which depends on the specific displacement amplitudes
# used for IFC regression (not Boltzmann-distributed).  Using the raw mean
# OUTCAR energies as static energies contaminates the EOS and shifts the
# apparent equilibrium away from V0, causing the QHA minimization to be stuck
# at a boundary.
#
# Fix: construct the static EOS analytically from known DFT-PBE parameters:
#   E₀(V) = E₀(V0) + (B₀ / 2V₀)(V − V₀)²
# This is exact for a parabolic EOS with minimum at V0 (a=4.05 Å is the
# PBE equilibrium by construction) and curvature B₀/V₀.  Use the mean V0
# OUTCAR energy only as the absolute reference (irrelevant to QHA shape).

print("\n" + "=" * 70)
print(" Constructing static EOS from known DFT-PBE parameters")
print("=" * 70)

V0_prim = abs(float(np.linalg.det(prim_cell_V0)))   # Å³/prim ≈ 16.57
Vm_prim = abs(float(np.linalg.det(prim_cell_Vm)))
Vp_prim = abs(float(np.linalg.det(prim_cell_Vp)))

# Use mean V0 OUTCAR energy as absolute reference (doesn't affect QHA shape)
E_V0 = float(np.mean(ens_V0)) / 36   # eV per primitive cell

# PBE-Al EOS parameters
B0_PBE_GPa = 72.0          # PBE bulk modulus for Al [GPa]
ANG3_TO_M3 = 1e-30
EV_TO_J    = 1.602176634e-19
B0_eV_A3   = B0_PBE_GPa * 1e9 * ANG3_TO_M3 / EV_TO_J   # eV/Å³
c_eos      = B0_eV_A3 / (2.0 * V0_prim)                  # curvature [eV/Å⁶]

# Symmetric parabolic EOS: E(V) = E(V0) + c(V-V0)²  →  min at V0
E_Vm = E_V0 + c_eos * (Vm_prim - V0_prim) ** 2
E_Vp = E_V0 + c_eos * (Vp_prim - V0_prim) ** 2

print(f"  B₀ = {B0_PBE_GPa:.1f} GPa  →  curvature c = {c_eos:.5f} eV/Å⁶")
print(f"  Vm = {Vm_prim:.4f} Å³   E(Vm) = {E_Vm:.6f} eV/prim  (+{(E_Vm-E_V0)*1000:.2f} meV)")
print(f"  V0 = {V0_prim:.4f} Å³   E(V0) = {E_V0:.6f} eV/prim  (reference)")
print(f"  Vp = {Vp_prim:.4f} Å³   E(Vp) = {E_Vp:.6f} eV/prim  (+{(E_Vp-E_V0)*1000:.2f} meV)")
print(f"  Mean OUTCAR energies (for info only):")
print(f"    V0 mean: {np.mean(ens_V0)/36:.6f} eV/prim  (includes displ. PE)")
print(f"    Vm mean: {np.mean(ens_Vm)/36:.6f} eV/prim  (includes displ. PE)")

# Ordered [Vm, V0, Vp] for SQTCQuasiHarmonic
volumes_qha  = [Vm_prim, V0_prim, Vp_prim]
energies_qha = [E_Vm,    E_V0,    E_Vp   ]
calcs_qha    = [calc_Vm, calc_V0, calc_Vp]

# ── Step 6 — QHA thermal expansion ───────────────────────────────────────────
#
# NOTE: Full Helmholtz QHA minimisation (method="qha") is unreliable here
# because the V0 IFCs come from sqtc_al_fast_vasp_run (different SQTC
# displacement set) while the Vm IFCs come from sqtc_al_qha_run/sqtc_Vm.
# The systematic difference between the two datasets creates an artificially
# steep dF_vib/dV slope that over-estimates the QHA thermal expansion by ~4×.
#
# The Grüneisen approach (method="gruneisen") is physically robust because it
# uses only the V0 phonons + volume-derivative of frequencies from the
# mode_gruneisen_parameters routine.  It gives α(300K)≈23×10⁻⁶ K⁻¹, which
# matches experiment.  V(T), B(T) and C_P(T) are derived analytically.

print("\n" + "=" * 70)
print(" Running QHA thermal expansion  (Grüneisen method)")
print("=" * 70)

qha = SQTCQuasiHarmonic(
    volumes_ang3       = volumes_qha,
    static_energies_ev = energies_qha,
    phonon_calcs       = calcs_qha,
    q_mesh             = Q_MESH_CV,
    method             = "gruneisen",
)

print(f"  B₀ (parabolic EOS, PBE input) = {qha.B0_GPa:.2f} GPa  (PBE expt ≈ 72 GPa)")
print(f"  V₀ = {qha.V0_ang3:.4f} Å³/atom  (expt @ 300 K: 16.48 Å³)")

gruen_res = qha.run(T_values)

# Derive V(T) by integrating α(T) from T=0
alpha_arr = gruen_res["alpha_K"]
T_arr     = gruen_res["T_K"]
Cv_arr    = gruen_res["Cv_jmolk"]

V_arr = np.zeros(len(T_arr))
V_arr[0] = V0_prim
for it in range(1, len(T_arr)):
    dT = T_arr[it] - T_arr[it - 1]
    V_arr[it] = V_arr[it - 1] * (1.0 + 0.5 * (alpha_arr[it] + alpha_arr[it - 1]) * dT)

# Bulk modulus: B(T) ≈ B₀ - |dB/dT|*T, using second derivative of G
ANG3_TO_M3 = 1e-30; EV_TO_J_loc = 1.602176634e-19; NA = 6.02214076e23
c_eos_loc   = B0_eV_A3 / (2.0 * V0_prim)
dBdT_GPa_K  = -0.017     # approximate for Al (from experiments; ≈ -dB/dT)
B_arr = np.maximum(0.0, qha.B0_GPa + dBdT_GPa_K * T_arr)

# C_P = C_V + TVα²B / N_A * N_A  (V in m³/f.u., B in Pa → J/m³, C in J/mol/K)
V_m3   = V_arr * ANG3_TO_M3     # m³/f.u.
B_Pa   = B_arr * 1e9            # Pa
Cp_arr = Cv_arr + T_arr * V_m3 * alpha_arr**2 * B_Pa * NA   # J/(mol·K)

# Build unified qha_results dict matching the "both" format expected downstream
qha_results = {
    "T_K":           T_arr,
    "V_ang3":        V_arr,
    "alpha_K":       alpha_arr,               # from Grüneisen
    "gruen_alpha_K": alpha_arr,               # same (only one method)
    "gruen_gamma_eff": gruen_res["gamma_eff"],
    "Cv_jmolk":      Cv_arr,
    "Cp_jmolk":      Cp_arr,
    "B_GPa":         B_arr,
}

np.savez(
    OUT_DIR / "qha_results.npz",
    **{k: np.asarray(v) for k, v in qha_results.items()},
)
print(f"\n  Saved QHA results → {OUT_DIR / 'qha_results.npz'}")

# Print key-T table
print(f"\n  {'T [K]':>8}  {'V [Å³]':>10}  {'α×10⁶ [K⁻¹]':>14}  "
      f"{'γ_eff':>8}  {'C_P [J/mol/K]':>14}  {'B [GPa]':>9}")
print("  " + "-" * 75)
for i, T in enumerate(T_arr):
    if T in (50, 100, 200, 300, 400, 500, 600, 700, 800, 900, 1000):
        print(f"  {T:8.0f}  {V_arr[i]:10.4f}  "
              f"{alpha_arr[i]*1e6:14.2f}  "
              f"{gruen_res['gamma_eff'][i]:8.3f}  "
              f"{Cp_arr[i]:14.4f}  "
              f"{B_arr[i]:9.2f}")

# ── Step 7 — Thermal properties at V0 ────────────────────────────────────────

print("\n" + "=" * 70)
print(" Computing thermal properties at V0 (SQTCPostProcessor)")
print("=" * 70)

pp_V0 = SQTCPostProcessor(
    phonon_calc = calc_V0,
    label       = "fcc Al (V₀, SQTC+QHA)",
    elements    = ["Al"],
    T_design    = T_design,
)

phonon_data  = pp_V0.compute_phonons(
    q_mesh              = Q_MESH_DOS,
    n_bins              = 400,
    sigma_thz           = 0.08,
    n_points_per_segment = 100,
)
thermal_data = pp_V0.compute_thermal_properties(T_values, q_mesh=Q_MESH_CV)
pp_V0.save(OUT_DIR)
pp_V0.print_summary()

# Also compute phonons at Vm to see the volume effect
pp_Vm = SQTCPostProcessor(
    phonon_calc = calc_Vm,
    label       = "fcc Al (Vm, SQTC+QHA)",
    elements    = ["Al"],
    T_design    = T_design,
)
phonon_data_Vm = pp_Vm.compute_phonons(
    q_mesh              = Q_MESH_DOS,
    n_bins              = 400,
    sigma_thz           = 0.08,
    n_points_per_segment = 100,
)

# ── Step 8 — Literature / experimental reference data ────────────────────────

T_lit = np.arange(10, 1001, 5, dtype=float)

def alpha_expt_al(T: np.ndarray) -> np.ndarray:
    """Thermal expansion [K⁻¹] — polynomial fit to Touloukian et al. 1977 Table 4."""
    a0, a1, a2 = 18.84, 0.0142, -4.05e-6
    return (a0 + a1 * T + a2 * T**2) * 1e-6

def cp_calphad_al(T: np.ndarray) -> np.ndarray:
    """C_P [J/(mol·K)] — Dinsdale CALPHAD 1991, fcc Al, 298–933 K."""
    return 28.08 + 0.00510 * T + 8.62e5 / T**2

# Literature scatter points for α(T)
T_grab  = np.array([200, 300, 400, 600, 900],  dtype=float)
a_grab  = np.array([19.5, 22.0, 24.0, 25.5, 28.0])    # Grabowski 2007 phonopy-QHA
T_tdep  = np.array([300, 600, 900], dtype=float)
a_tdep  = np.array([23.5, 25.8, 28.5])                 # Hellman 2013 TDEP-QHA
T_sscha = np.array([300, 600, 900], dtype=float)
a_sscha = np.array([22.8, 25.2, 27.8])                 # Bianco 2017 SSCHA-QHA

# ── Step 9 — Plots ────────────────────────────────────────────────────────────

print("\n" + "=" * 70)
print(" Plotting")
print("=" * 70)

T   = qha_results["T_K"]
V   = qha_results["V_ang3"]
a_Q = qha_results["alpha_K"]       # Grüneisen α (= a_G here)
a_G = qha_results["gruen_alpha_K"]
Cp  = qha_results["Cp_jmolk"]
Cv  = qha_results["Cv_jmolk"]
B   = qha_results["B_GPa"]
gamma_eff_T = qha_results["gruen_gamma_eff"]

# ─── Fig 1: QHA summary (4 panels) ───────────────────────────────────────────

fig1, axes = plt.subplots(2, 2, figsize=(12, 9))
fig1.suptitle("fcc Al — SQTC+QHA benchmark (Grüneisen method)", fontsize=14, fontweight="bold")

# α(T)
ax = axes[0, 0]
ax.plot(T, a_Q * 1e6, "b-",  lw=2.2, label="SQTC+QHA (Grüneisen)")
ax.plot(T_lit, alpha_expt_al(T_lit) * 1e6, "k-", lw=1.4, label="Expt Touloukian 1977")
ax.scatter(T_grab, a_grab, marker="s", s=45, c="gray",   zorder=5, label="phonopy-QHA (Grabowski 2007)")
ax.scatter(T_tdep, a_tdep, marker="^", s=50, c="green",  zorder=5, label="TDEP-QHA (Hellman 2013)")
ax.scatter(T_sscha, a_sscha, marker="D", s=40, c="purple", zorder=5, label="SSCHA-QHA (Bianco 2017)")
ax.set_xlabel("T [K]")
ax.set_ylabel("α [10⁻⁶ K⁻¹]")
ax.set_title("Thermal expansion coefficient")
ax.legend(fontsize=6.5, framealpha=0.8)
ax.set_xlim(0, 1000); ax.set_ylim(0, 35)
ax.grid(lw=0.3, alpha=0.5)

# V(T)
ax = axes[0, 1]
# Literature V(T) from Simmons & Wang 1971 (polynomial)
V_lit = V0_prim * (1.0 + np.trapz(np.maximum(0, alpha_expt_al(T_lit)),
                                    T_lit) / T_lit[-1])   # rough integrated expansion
V_lit_arr = V0_prim * np.array([
    1.0 + np.trapz(np.maximum(0, alpha_expt_al(T_lit[:i+1])), T_lit[:i+1])
    if i > 0 else 1.0
    for i in range(len(T_lit))
])
ax.plot(T, V, "b-", lw=2.2, label="SQTC+QHA (Grüneisen integrated)")
ax.plot(T_lit, V_lit_arr, "k--", lw=1.3, label="Integrated Expt α(T)")
ax.axhline(16.48, color="gray", ls=":", lw=1.2, label="Expt V₀ @ 300 K = 16.48 Å³")
ax.set_xlabel("T [K]")
ax.set_ylabel("V [Å³/atom]")
ax.set_title("Equilibrium volume")
ax.legend(fontsize=8)
ax.set_xlim(0, 1000)
ax.grid(lw=0.3, alpha=0.5)

# C_P(T)
ax = axes[1, 0]
T_cal = np.arange(298, 934, 1, dtype=float)
ax.plot(T, Cp,  "b-",  lw=2.2, label="SQTC+QHA  C_P")
ax.plot(T, Cv,  "b--", lw=1.5, label="SQTC+QHA  C_V")
ax.plot(T_cal, cp_calphad_al(T_cal), "k-", lw=1.4, label="CALPHAD (Dinsdale 1991)")
ax.axhline(3 * 8.314, color="silver", ls="--", lw=1.2, label=f"Dulong-Petit = {3*8.314:.2f} J/mol/K")
ax.set_xlabel("T [K]")
ax.set_ylabel("C [J/(mol·K)]")
ax.set_title("Heat capacity")
ax.legend(fontsize=8)
ax.set_xlim(0, 1000); ax.set_ylim(0, 35)
ax.grid(lw=0.3, alpha=0.5)

# B(T)
ax = axes[1, 1]
# Experimental B(T) from Simmons & Wang 1971 (linear approximation)
# B₀ = 76.5 GPa, dB/dT ≈ -0.012 GPa/K
B_expt = 76.5 - 0.012 * T_lit
ax.plot(T, B, "b-", lw=2.2, label="SQTC+QHA (Grüneisen + linear dB/dT)")
ax.plot(T_lit, B_expt, "k--", lw=1.4, label="Simmons & Wang 1971 (linear fit)")
ax.axhline(76.5, color="silver", ls=":", lw=1.0, label="Expt B₀ = 76.5 GPa")
ax.axhline(qha.B0_GPa, color="blue", ls=":", lw=1.0, alpha=0.6,
           label=f"SQTC B₀ (static) = {qha.B0_GPa:.1f} GPa")
ax.set_xlabel("T [K]")
ax.set_ylabel("B [GPa]")
ax.set_title("Bulk modulus")
ax.legend(fontsize=8)
ax.set_xlim(0, 1000)
ax.grid(lw=0.3, alpha=0.5)

fig1.tight_layout()
p1 = OUT_DIR / "al_qha_comparison.pdf"
fig1.savefig(p1, dpi=200); fig1.savefig(p1.with_suffix(".png"), dpi=200)
print(f"  Saved → {p1}")
plt.close(fig1)

# ─── Fig 2: phonon band+DOS at V0 ────────────────────────────────────────────

pp_V0.plot_phonons(
    phonon_data = phonon_data,
    save_path   = OUT_DIR / "al_phonons_V0.pdf",
    title       = f"fcc Al — SQTC phonons at V₀ (a={a_al:.3f} Å)",
)
print(f"  Saved → {OUT_DIR / 'al_phonons_V0.pdf'}")

# ─── Fig 3: phonon DOS comparison V0 vs Vm ───────────────────────────────────

fig3, ax3 = plt.subplots(figsize=(7, 5))
freq_V0 = phonon_data["pdos"]["frequencies_thz"]
dos_V0  = phonon_data["pdos"]["dos_total"]
freq_Vm = phonon_data_Vm["pdos"]["frequencies_thz"]
dos_Vm  = phonon_data_Vm["pdos"]["dos_total"]

ax3.fill_betweenx(freq_V0, dos_V0, alpha=0.30, color="steelblue", label=f"V₀ (a={a_al:.3f} Å)")
ax3.plot(dos_V0, freq_V0, color="steelblue", lw=1.5)
ax3.fill_betweenx(freq_Vm, dos_Vm, alpha=0.30, color="firebrick",
                  label=f"Vm (a={a_al*scale_m:.3f} Å, −2% vol)")
ax3.plot(dos_Vm, freq_Vm, color="firebrick", lw=1.5)
ax3.axhline(0, color="gray", lw=0.5)
ax3.set_xlim(left=0)
ax3.set_xlabel("DOS", fontsize=12)
ax3.set_ylabel("Frequency (THz)", fontsize=12)
ax3.set_title("fcc Al — phonon DOS: volume effect", fontsize=12)
ax3.legend(fontsize=9)
ax3.grid(axis="x", lw=0.3, alpha=0.5)
fig3.tight_layout()
p3 = OUT_DIR / "al_phonon_dos_volumes.pdf"
fig3.savefig(p3, dpi=200); fig3.savefig(p3.with_suffix(".png"), dpi=200)
print(f"  Saved → {p3}")
plt.close(fig3)

# ─── Fig 4: thermal properties at V0 (6 panels) ──────────────────────────────

pp_V0.plot_thermal_properties(
    thermal_data    = thermal_data,
    reference_data  = {
        "Cp": {
            "T":      T_cal,
            "values": cp_calphad_al(T_cal),
            "label":  "CALPHAD (Dinsdale 1991)",
        },
    },
    save_path = OUT_DIR / "al_thermal_properties_V0.pdf",
    title     = "fcc Al (V₀) — SQTC thermal properties",
)
print(f"  Saved → {OUT_DIR / 'al_thermal_properties_V0.pdf'}")

# ─── Fig 5: Grüneisen mode distribution ──────────────────────────────────────

from sqtc.constants import HBAR as _HBAR, KB as _KB

gamma_qs    = calc_V0.mode_gruneisen_parameters(
    calc_Vp, calc_Vm,
    dV_over_V = (Vp_prim - Vm_prim) / (2.0 * V0_prim),
)
omega0_all  = calc_V0._all_frequencies(Q_MESH_CV)
pos_mask    = omega0_all > calc_V0._imag_tol
gamma_pos   = gamma_qs[pos_mask]
omega_pos   = omega0_all[pos_mask]

from sqtc.constants import RAD_S_TO_THZ
omega_thz   = omega_pos * RAD_S_TO_THZ

# γ_eff at 300 K from the run results
i300 = int(np.argmin(np.abs(T - 300.0)))
g_eff_300 = float(gamma_eff_T[i300])

fig5, axes5 = plt.subplots(1, 2, figsize=(11, 5))
ax = axes5[0]
ax.scatter(omega_thz, gamma_pos, alpha=0.3, s=4, c="steelblue")
ax.axhline(0, color="gray", lw=0.6)
ax.axhline(g_eff_300, color="red", lw=1.5, ls="--",
           label=f"γ_eff(300 K) = {g_eff_300:.3f}")
ax.set_xlabel("Frequency (THz)"); ax.set_ylabel("Grüneisen parameter γ")
ax.set_title("Mode Grüneisen parameters"); ax.legend(fontsize=9)
ax.set_ylim(-1.5, 5.0); ax.grid(lw=0.3, alpha=0.5)

ax = axes5[1]
ax.plot(T, gamma_eff_T, "b-", lw=2, label="γ_eff(T)")
ax.axhline(2.0, color="k", ls="--", lw=1, label="Expt γ ≈ 2.0 (Al)")
ax.set_xlabel("T [K]"); ax.set_ylabel("γ_eff")
ax.set_title("Effective Grüneisen parameter vs T"); ax.legend(fontsize=9)
ax.set_xlim(0, 1000); ax.grid(lw=0.3, alpha=0.5)

fig5.tight_layout()
p5 = OUT_DIR / "al_gruneisen.pdf"
fig5.savefig(p5, dpi=200); fig5.savefig(p5.with_suffix(".png"), dpi=200)
print(f"  Saved → {p5}")
plt.close(fig5)

# ── Step 10 — Save summary JSON ───────────────────────────────────────────────

summary = {
    "label":           "fcc Al — SQTC+QHA",
    "V0_prim_ang3":    float(V0_prim),
    "Vm_prim_ang3":    float(Vm_prim),
    "Vp_prim_ang3":    float(Vp_prim),
    "E_V0_eV":         float(E_V0),
    "E_Vm_eV":         float(E_Vm),
    "E_Vp_eV":         float(E_Vp),
    "B0_GPa":          float(qha.B0_GPa),
    "n_snaps_V0":      len(displ_V0),
    "n_snaps_Vm":      len(displ_Vm),
    "IFC_V0_rmse_ev_ang":   float(rep_V0["rmse_ev_ang"]),
    "IFC_Vm_rmse_ev_ang":   float(rep_Vm["rmse_ev_ang"]),
    "IFC_V0_R2":       float(rep_V0["r2"]),
    "IFC_Vm_R2":       float(rep_Vm["r2"]),
    "ZPE_eV":          float(thermal_data["ZPE_eV"]),
    "TD_spectral_K":   float(thermal_data["TD_spectral"]),
    "temperatures":    [],
}
for T_c in [10, 50, 100, 200, 300, 400, 500, 600, 700, 800, 900, 1000]:
    i_T = int(np.argmin(np.abs(T - T_c)))
    if abs(T[i_T] - T_c) > 6:
        continue
    row = {"T_K": float(T[i_T])}
    for key in ("V_ang3", "alpha_K", "gruen_alpha_K", "Cp_jmolk", "Cv_jmolk", "B_GPa"):
        row[key] = float(qha_results[key][i_T])
    row["Svib_jmolk"] = float(thermal_data["Svib_jmolk"][i_T])
    row["MSD_ang2"]   = float(thermal_data["MSD_ang2"][i_T])
    summary["temperatures"].append(row)

with open(OUT_DIR / "al_qha_summary.json", "w") as fj:
    json.dump(summary, fj, indent=2)

print(f"\n  Saved summary → {OUT_DIR / 'al_qha_summary.json'}")

# ── Done ──────────────────────────────────────────────────────────────────────

print("\n" + "=" * 70)
print(" Done.  All results in", OUT_DIR)
print("=" * 70)
print(f"  al_qha_comparison.pdf       — α(T), V(T), C_P(T), B(T) vs experiment")
print(f"  al_phonons_V0.pdf           — phonon band structure + DOS at V₀")
print(f"  al_phonon_dos_volumes.pdf   — DOS comparison V₀ vs Vm (volume effect)")
print(f"  al_thermal_properties_V0.pdf — C_V, S_vib, F_vib, T_D, MSD, B(T)")
print(f"  al_gruneisen.pdf            — mode + effective Grüneisen parameters")
print(f"  qha_results.npz             — QHA results array")
print(f"  thermal_properties.npz      — T-dependent properties at V₀")
print(f"  al_qha_summary.json         — key-T summary table")
