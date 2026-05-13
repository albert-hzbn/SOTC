#!/usr/bin/env python3
"""
Post-process Al SQTC run: compute all temperature-dependent properties.

Reads force/displacement data from sqtc_al_fast_vasp_run (all iterations),
re-fits the IFCs, and computes:
  - C_V(T), C_P(T)  [J/(mol·K)]
  - S_vib(T)        [J/(mol·K)]
  - F_vib(T)        [eV/f.u.]
  - ZPE             [eV/f.u.]
  - T_D_eff(T)      effective Debye temperature from C_V inversion [K]
  - T_D_spectral     spectral-moment Debye temperature [K]
  - MSD(T)          mean-squared displacement [Å²]
  - Debye-Waller B  [Å²]
  - Phonon DOS at T=0 and T=300 K
  - Phonon band structure

All results saved to sqtc_al_fast_vasp_run/postproc/ as .npz + .json + plots.
"""

import json
import os
import re
import sys
from pathlib import Path

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "../"))

from sqtc.ifc_extractor import IFCExtractor
from sqtc.phonons import PhononCalculator
from sqtc.constants import KB, HBAR, NA, AMU_TO_KG, EV_TO_J

# ── fcc Al geometry (must match run_sqtc_al_fast_vasp.py) ────────────────────

a_al         = 4.05
prim_cell_al = 0.5 * a_al * np.array([[0,1,1],[1,0,1],[1,1,0]], dtype=float)
prim_pos_al  = np.array([[0.0, 0.0, 0.0]])
M_Al_amu     = 26.9815385
r_cutoff     = 4.5    # same as runner
ridge_alpha  = 1e-3
symmetrize_bonds = True

RUN_DIR  = Path("sqtc_al_fast_vasp_run")
OUT_DIR  = RUN_DIR / "postproc"
OUT_DIR.mkdir(exist_ok=True)

Q_MESH   = (20, 20, 20)
T_MIN, T_MAX, T_STEP = 10.0, 1000.0, 10.0
T_values = np.arange(T_MIN, T_MAX + T_STEP, T_STEP)

# ─────────────────────────────────────────────────────────────────────────────
# Step 1: parse POSCAR → equilibrium supercell positions + cell
# ─────────────────────────────────────────────────────────────────────────────

def parse_poscar(path: Path):
    """Return (cell [Å], species list, positions [Å]).
    Handles both VASP4 (no species-name line) and VASP5 (with species-name line).
    """
    lines = Path(path).read_text().splitlines()
    scale = float(lines[1])
    cell  = scale * np.array([
        [float(x) for x in lines[2].split()],
        [float(x) for x in lines[3].split()],
        [float(x) for x in lines[4].split()],
    ])
    # Detect VASP5 (line 5 = species names, line 6 = counts)
    # vs VASP4 (line 5 = counts directly)
    try:
        counts = [int(x) for x in lines[5].split()]
        species_names_line = None
        counts_line_idx = 5
    except ValueError:
        # VASP5: line 5 is species names
        species_names_line = lines[5].split()
        counts = [int(x) for x in lines[6].split()]
        counts_line_idx = 6

    n_atoms = sum(counts)
    if species_names_line is not None:
        species_names = []
        for s, n in zip(species_names_line, counts):
            species_names += [s] * n
    else:
        species_names = ["X"] * n_atoms

    # Next line after counts: optional "Selective dynamics", then coord type
    coord_idx = counts_line_idx + 1
    if lines[coord_idx].strip().lower()[0] == "s":   # Selective dynamics
        coord_idx += 1
    coord_type = lines[coord_idx].strip().lower()[0]  # 'D' or 'C'
    pos_start  = coord_idx + 1
    pos_lines  = lines[pos_start: pos_start + n_atoms]
    positions  = np.array([[float(x) for x in l.split()[:3]]
                            for l in pos_lines if l.strip()])
    if coord_type == "d":
        positions = positions @ cell
    return cell, species_names, positions


def parse_outcar_forces(path: Path, n_atoms: int) -> np.ndarray:
    """Extract the last set of forces from OUTCAR → (n_atoms, 3) eV/Å."""
    text = Path(path).read_text()
    pattern = (
        r"TOTAL-FORCE \(eV/Angst\)\s*[-\s]+\n"
        r"((?:\s+[-\d.]+\s+[-\d.]+\s+[-\d.]+\s+[-\d.]+\s+[-\d.]+\s+[-\d.]+\n)+)"
    )
    blocks = re.findall(pattern, text)
    if not blocks:
        raise ValueError(f"No forces found in {path}")
    rows = blocks[-1].strip().split("\n")
    forces = np.array([
        [float(x) for x in row.split()[3:6]] for row in rows if row.strip()
    ])
    return forces[:n_atoms]


# ── Build equilibrium supercell from known prim_cell ─────────────────────────

def build_sc_positions(prim_cell, prim_pos, sc_cell):
    """
    Enumerate all prim-cell lattice points inside sc_cell and return
    equilibrium supercell positions.  Matches the runner's _choose_supercell().
    """
    H = np.round(sc_cell @ np.linalg.inv(prim_cell)).astype(int)
    H_inv = np.linalg.inv(H.astype(float))
    max_idx = max(
        H[0, 0],
        abs(H[0, 1]) + H[1, 1],
        abs(H[0, 2]) + abs(H[1, 2]) + H[2, 2],
    ) + 1
    sc_pos = []
    for n1 in range(0, max_idx):
        for n2 in range(0, max_idx):
            for n3 in range(0, max_idx):
                m    = np.array([n1, n2, n3], dtype=float)
                frac = m @ H_inv
                if np.all(frac >= -1e-9) and np.all(frac < 1.0 - 1e-9):
                    cart_shift = m @ prim_cell
                    for p in prim_pos:
                        sc_pos.append(p + cart_shift)
    return np.array(sc_pos)


# Read cell from the LAST iteration's snap_0000 (most converged supercell choice)
last_iter  = sorted(RUN_DIR.glob("iter_*/snap_0000/POSCAR"))[-1]
sc_cell, species, _ = parse_poscar(last_iter)
n_atoms              = len(species)
eq_positions         = build_sc_positions(prim_cell_al, prim_pos_al, sc_cell)
masses_amu           = np.full(n_atoms, M_Al_amu)

print(f"Supercell: {n_atoms} atoms  (from {last_iter.parent.parent.name})")
print(f"Cell:\n{sc_cell}")
print(f"Equilibrium pos built: {len(eq_positions)} atoms")

# ─────────────────────────────────────────────────────────────────────────────
# Step 2: collect all (displacement, force) pairs across all iterations
# Only use snapshots whose POSCAR has the same atom count as the chosen cell.
# ─────────────────────────────────────────────────────────────────────────────

displacements_list = []
forces_list        = []

iter_dirs = sorted(RUN_DIR.glob("iter_*/"))
for iter_dir in iter_dirs:
    snaps = sorted(iter_dir.glob("snap_*/"))
    for snap in snaps:
        poscar_path = snap / "POSCAR"
        outcar_path = snap / "OUTCAR"
        if not poscar_path.exists() or not outcar_path.exists():
            continue
        try:
            _, snap_species, disp_pos = parse_poscar(poscar_path)
        except Exception as e:
            print(f"  Skipping {snap} (POSCAR parse): {e}")
            continue
        # Skip snapshots from a different supercell (different atom count)
        if len(snap_species) != n_atoms:
            continue
        try:
            forces = parse_outcar_forces(outcar_path, n_atoms)
        except Exception as e:
            print(f"  Skipping {snap} (OUTCAR parse): {e}")
            continue
        # Displacement = displaced - equilibrium (minimum image)
        u = disp_pos - eq_positions
        # Apply minimum image convention
        for i in range(n_atoms):
            frac = u[i] @ np.linalg.inv(sc_cell)
            frac -= np.round(frac)
            u[i] = frac @ sc_cell
        displacements_list.append(u)
        forces_list.append(forces)

print(f"\nLoaded {len(displacements_list)} (displacement, force) snapshots "
      f"from {len(iter_dirs)} iterations")

# ─────────────────────────────────────────────────────────────────────────────
# Step 3: fit IFCs
# ─────────────────────────────────────────────────────────────────────────────

print("\nFitting IFCs ...")
ifc = IFCExtractor(
    supercell_positions=eq_positions,
    supercell_cell=sc_cell,
    masses_amu=masses_amu,
    r_cutoff=r_cutoff,
    symmetrise=True,
    ridge_alpha=ridge_alpha,
    symmetrize_bonds=symmetrize_bonds,
)
ifc.fit(displacements_list, forces_list)
report = ifc.fit_report(displacements_list, forces_list)
print(f"  RMSE = {report['rmse_ev_ang']:.5f} eV/Å   R² = {report['r2']:.5f}   rank = {report['rank']}")

# ─────────────────────────────────────────────────────────────────────────────
# Step 4: build PhononCalculator
# ─────────────────────────────────────────────────────────────────────────────

calc = PhononCalculator(
    ifc_extractor=ifc,
    prim_positions=prim_pos_al,
    prim_cell=prim_cell_al,
    masses_amu=np.array([M_Al_amu]),
)

# Quick sanity check
stats = calc.spectrum_statistics(q_mesh=(10, 10, 10))
print(f"\nSpectrum: ω_max = {stats['max_freq_thz']:.2f} THz   "
      f"unstable fraction = {stats['unstable_fraction']:.4f}")

# ─────────────────────────────────────────────────────────────────────────────
# Step 5: compute all T-dependent properties
# ─────────────────────────────────────────────────────────────────────────────

print(f"\nComputing T-dependent properties for {len(T_values)} temperatures "
      f"({T_MIN:.0f}–{T_MAX:.0f} K) ...")

n_T = len(T_values)

Cv_arr   = np.zeros(n_T)   # C_V  [J/(mol·K)]
S_arr    = np.zeros(n_T)   # S_vib [J/(mol·K)]
F_arr    = np.zeros(n_T)   # F_vib [eV/f.u.]
Td_arr   = np.zeros(n_T)   # T_D calorimetric [K]
MSD_arr  = np.zeros(n_T)   # mean-squared displacement [Å²]
DW_arr   = np.zeros(n_T)   # Debye-Waller B factor [Å²]

# ZPE is T-independent
ZPE_eV = calc.zero_point_energy(q_mesh=Q_MESH)
T_D_spectral = calc.debye_temperature_from_dos(q_mesh=Q_MESH)
print(f"  ZPE            = {ZPE_eV:.5f} eV/f.u.")
print(f"  T_D (spectral) = {T_D_spectral:.2f} K")

# Vectorised C_V scan (fast path)
Cv_arr = calc.heat_capacity_scan(T_values, q_mesh=Q_MESH)

# Loop for per-T properties that don't have a vectorised path
all_omegas = calc._all_frequencies(Q_MESH)  # (n_q, n_branches) rad/s
n_q_pts    = all_omegas.shape[0]
omegas_flat = all_omegas.ravel()
valid_mask, pos_mask, omega_use = calc._classify_omegas(omegas_flat)

for it, T in enumerate(T_values):
    # F_vib [eV/f.u.]
    F_arr[it] = calc.vibrational_free_energy(T, q_mesh=Q_MESH)

    # S_vib [J/(mol·K)]
    S_arr[it] = calc.vibrational_entropy(T, q_mesh=Q_MESH) * NA

    # Calorimetric T_D  — use already-computed Cv (avoids re-running _all_frequencies)
    Td_arr[it] = calc._calorimetric_debye_temperature(Cv_arr[it], T)

    # Mean-squared displacement <u²> = (ħ/N) Σ_qs coth(ħω/2kT)/(2ω M)
    # Units: Å²  (Å² = 1e-20 m²)
    if T < 1e-9:
        coth_vals = np.ones_like(omega_use)
    else:
        x = np.clip(HBAR * omega_use / (2.0 * KB * T), 0, 350.0)
        coth_vals = np.where(x > 1e-10, 1.0 / np.tanh(x), 1.0 / x)

    # MSD per atom from quantum fluctuations
    # <u²> = ħ/(2M) × (1/N_q) Σ_{qs} coth(ħω/2kT)/ω
    # Exclude acoustic (ω~0) and imaginary modes
    omega_safe = np.where(pos_mask, omega_use, np.inf)
    msd_contrib = np.where(pos_mask, coth_vals / omega_safe, 0.0)
    M_SI = M_Al_amu * AMU_TO_KG
    msd_SI = (HBAR / (2.0 * M_SI)) * np.sum(msd_contrib) / n_q_pts  # m²
    MSD_arr[it] = msd_SI * 1e20  # → Å²
    # Debye-Waller B = 8π²<u²>/3 (isotropic, powder)
    DW_arr[it]  = 8.0 * np.pi**2 * MSD_arr[it] / 3.0

print("  Done.")

# ─────────────────────────────────────────────────────────────────────────────
# Step 6: phonon DOS
# ─────────────────────────────────────────────────────────────────────────────

print("\nComputing phonon DOS ...")
dos_result = calc.compute_dos(q_mesh=(30, 30, 30), n_bins=300)
freq_dos = dos_result["frequencies_thz"]
dos_vals = dos_result["dos"]

# ─────────────────────────────────────────────────────────────────────────────
# Step 7: phonon band structure (FCC high-symmetry path Γ-X-W-L-Γ-K)
# ─────────────────────────────────────────────────────────────────────────────

print("Computing phonon band structure ...")
bs_result = calc.compute_band_structure(n_points_per_segment=100)

# ─────────────────────────────────────────────────────────────────────────────
# Step 8: save all results
# ─────────────────────────────────────────────────────────────────────────────

np.savez(
    OUT_DIR / "thermal_properties.npz",
    T_K         = T_values,
    Cv_jmolk    = Cv_arr,
    Svib_jmolk  = S_arr,
    Fvib_ev     = F_arr,
    TD_caloric  = Td_arr,
    MSD_ang2    = MSD_arr,
    DW_B_ang2   = DW_arr,
    ZPE_eV      = np.array([ZPE_eV]),
    TD_spectral = np.array([T_D_spectral]),
)

np.savez(
    OUT_DIR / "phonon_dos.npz",
    frequencies_thz = freq_dos,
    dos             = dos_vals,
)

np.savez(
    OUT_DIR / "phonon_bs.npz",
    distances          = bs_result["distances"],
    frequencies_thz    = bs_result["frequencies_thz"],
    q_points           = bs_result["q_points"],
    labels             = np.array(bs_result["labels"]),
    label_positions    = np.array(bs_result["label_positions"]),
)

# JSON summary at key T
summary_json = {
    "ZPE_eV":          float(ZPE_eV),
    "T_D_spectral_K":  float(T_D_spectral),
    "IFC_RMSE_eV_ang": float(report["rmse_ev_ang"]),
    "IFC_R2":          float(report["r2"]),
    "n_snapshots":     len(displacements_list),
    "temperatures": [],
}
for T_c in [10, 50, 100, 200, 300, 400, 500, 600, 700, 800, 900, 1000]:
    i = int(np.argmin(np.abs(T_values - T_c)))
    summary_json["temperatures"].append({
        "T_K":          float(T_values[i]),
        "Cv_jmolk":     float(Cv_arr[i]),
        "Svib_jmolk":   float(S_arr[i]),
        "Fvib_ev":      float(F_arr[i]),
        "TD_caloric_K": float(Td_arr[i]),
        "MSD_ang2":     float(MSD_arr[i]),
        "DW_B_ang2":    float(DW_arr[i]),
    })

with open(OUT_DIR / "thermal_summary.json", "w") as f:
    json.dump(summary_json, f, indent=2)

print(f"\nSaved results → {OUT_DIR}/")

# ─────────────────────────────────────────────────────────────────────────────
# Step 9: plots
# ─────────────────────────────────────────────────────────────────────────────

# Experimental / CALPHAD reference data for Al
def cv_dulong_petit():
    return 3.0 * 8.314  # J/(mol·K)

def cp_calphad(T):
    """Dinsdale 1991 CALPHAD C_P [J/(mol·K)], valid 298–933 K."""
    return 28.08 + 0.00510 * T + 8.62e5 / T**2

def s_calphad(T):
    """Entropy from Dinsdale 1991 CALPHAD, 298–933 K."""
    return 28.08 * np.log(T / 298) + 0.00510 * (T - 298) - 0.5 * 8.62e5 * (1/T**2 - 1/298**2) + 28.35

T_lit = np.linspace(100, 900, 200)

fig, axes = plt.subplots(3, 2, figsize=(12, 13))
fig.suptitle("fcc Al — Temperature-dependent properties from SQTC", fontsize=13)

# ── C_V(T) ──────────────────────────────────────────────────────────────────
ax = axes[0, 0]
ax.plot(T_values, Cv_arr, "b-", lw=2, label="SQTC C_V")
ax.plot(T_lit, cp_calphad(T_lit), "k-", lw=1.2, label="CALPHAD C_P (Dinsdale 1991)")
ax.axhline(cv_dulong_petit(), color="gray", ls="--", lw=1, label="Dulong-Petit (3R)")
# Experimental C_V from NIST (Desai 1987)
T_exp_cv = np.array([100, 200, 298, 400, 500, 600, 700, 800, 900])
cv_exp   = np.array([12.99, 21.36, 24.35, 25.91, 26.62, 27.22, 27.80, 28.51, 29.22])
ax.scatter(T_exp_cv, cv_exp, marker="o", c="red", s=40, zorder=5, label="Expt C_P (Desai 1987)")
ax.set_xlabel("T [K]")
ax.set_ylabel("C_V, C_P  [J/(mol·K)]")
ax.set_title("Heat capacity")
ax.legend(fontsize=7.5)
ax.set_xlim(0, 1000)
ax.set_ylim(0, 32)

# ── S_vib(T) ─────────────────────────────────────────────────────────────────
ax = axes[0, 1]
ax.plot(T_values, S_arr, "b-", lw=2, label="SQTC S_vib")
ax.plot(T_lit, s_calphad(T_lit), "k-", lw=1.2, label="CALPHAD S (Dinsdale 1991)")
# Experimental S (Barin 1995)
T_exp_s = np.array([200, 298, 400, 600, 800])
s_exp   = np.array([18.51, 28.35, 36.60, 49.81, 60.16])
ax.scatter(T_exp_s, s_exp, marker="s", c="red", s=40, zorder=5, label="Expt S (Barin 1995)")
ax.set_xlabel("T [K]")
ax.set_ylabel("S_vib  [J/(mol·K)]")
ax.set_title("Vibrational entropy")
ax.legend(fontsize=8)
ax.set_xlim(0, 1000)

# ── F_vib(T) ─────────────────────────────────────────────────────────────────
ax = axes[1, 0]
ax.plot(T_values, F_arr * 1000, "b-", lw=2, label="SQTC F_vib")
ax.axhline(ZPE_eV * 1000, color="gray", ls="--", lw=1,
           label=f"ZPE = {ZPE_eV*1000:.2f} meV")
ax.set_xlabel("T [K]")
ax.set_ylabel("F_vib  [meV/f.u.]")
ax.set_title("Vibrational free energy")
ax.legend(fontsize=8)
ax.set_xlim(0, 1000)

# ── T_D(T) ───────────────────────────────────────────────────────────────────
ax = axes[1, 1]
ax.plot(T_values, Td_arr, "b-", lw=2, label="T_D calorimetric (SQTC)")
ax.axhline(T_D_spectral, color="orange", ls="--", lw=1.5,
           label=f"T_D spectral = {T_D_spectral:.0f} K")
ax.axhline(390.0, color="gray", ls=":", lw=1.2, label="T_D input (Debye seed)")
# Experimental T_D (Grimvall, Thermophysical Props)
T_exp_td = np.array([100, 200, 300, 400, 600, 800])
td_exp   = np.array([429, 422, 418, 413, 405, 395])
ax.scatter(T_exp_td, td_exp, marker="^", c="red", s=40, zorder=5,
           label="Expt T_D calorimetric (Grimvall)")
ax.set_xlabel("T [K]")
ax.set_ylabel("T_D  [K]")
ax.set_title("Effective Debye temperature")
ax.legend(fontsize=7.5)
ax.set_xlim(0, 1000)
ax.set_ylim(300, 500)

# ── MSD(T) ───────────────────────────────────────────────────────────────────
ax = axes[2, 0]
ax.plot(T_values, MSD_arr, "b-", lw=2, label="SQTC <u²>")
# Classical limit: <u²>_cl = 3k_BT / (M ω_D²)
# Debye result: <u²>_D = 3ħ²/(Mk_BT_D) × [1/4 + (T/T_D)²·(1/3)...]
# Approximate: linear for T>>T_D
T_exp_msd = np.array([100, 200, 300, 400, 500])
# Expt B-factors from X-ray diffraction (Walker & Chipman 1953, Al)
B_expt = np.array([0.48, 0.73, 0.97, 1.21, 1.45])   # Å²
msd_expt = B_expt / (8 * np.pi**2) * 3.0             # <u²> ≈ B/8π² × 3
ax.scatter(T_exp_msd, msd_expt, marker="D", c="red", s=40, zorder=5,
           label="Expt (Walker & Chipman 1953)")
ax.set_xlabel("T [K]")
ax.set_ylabel("<u²>  [Å²]")
ax.set_title("Mean-squared displacement")
ax.legend(fontsize=8)
ax.set_xlim(0, 1000)

# ── Debye-Waller B(T) ─────────────────────────────────────────────────────────
ax = axes[2, 1]
ax.plot(T_values, DW_arr, "b-", lw=2, label="SQTC B(T)")
ax.scatter(T_exp_msd, B_expt, marker="D", c="red", s=40, zorder=5,
           label="Expt (Walker & Chipman 1953)")
ax.set_xlabel("T [K]")
ax.set_ylabel("B = 8π²<u²>/3  [Å²]")
ax.set_title("Debye-Waller B factor")
ax.legend(fontsize=8)
ax.set_xlim(0, 1000)

plt.tight_layout()
plt.savefig(OUT_DIR / "al_thermal_properties.png", dpi=150)
plt.savefig(OUT_DIR / "al_thermal_properties.pdf")
print(f"Saved plot → {OUT_DIR / 'al_thermal_properties.png'}")

# ── Phonon DOS plot ───────────────────────────────────────────────────────────
fig2, ax2 = plt.subplots(figsize=(6, 4))
ax2.fill_between(freq_dos, dos_vals, alpha=0.3, color="steelblue")
ax2.plot(freq_dos, dos_vals, "b-", lw=1.5, label="SQTC phonon DOS")
# Literature: INS measurement of Al (Stedman & Nilsson, PR 1966)
freq_ins = np.array([0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10])
# Approximate shape only — normalised to 1
dos_ins  = np.array([0, 0.005, 0.030, 0.090, 0.160, 0.200, 0.230, 0.200, 0.060, 0.020, 0])
dos_ins /= np.trapz(dos_ins, freq_ins)
ax2.plot(freq_ins, dos_ins, "k--", lw=1.2, label="INS (Stedman & Nilsson 1966, schematic)")
ax2.set_xlabel("Frequency [THz]")
ax2.set_ylabel("DOS [1/THz]")
ax2.set_title("fcc Al — Phonon DOS")
ax2.legend(fontsize=9)
ax2.set_xlim(0, None)
ax2.set_ylim(bottom=0)
plt.tight_layout()
plt.savefig(OUT_DIR / "al_phonon_dos.png", dpi=150)
plt.savefig(OUT_DIR / "al_phonon_dos.pdf")

# ── Phonon band structure plot ────────────────────────────────────────────────
fig3, ax3 = plt.subplots(figsize=(7, 5))
dists = bs_result["distances"]
freqs = bs_result["frequencies_thz"]   # (n_points, n_branches)
for branch in range(freqs.shape[1]):
    ax3.plot(dists, freqs[:, branch], "b-", lw=1.2, alpha=0.85)
for lp in bs_result["label_positions"]:
    ax3.axvline(lp, color="k", lw=0.6, ls="--", alpha=0.5)
ax3.set_xticks(bs_result["label_positions"])
ax3.set_xticklabels(bs_result["labels"])
ax3.set_xlabel("")
ax3.set_ylabel("Frequency [THz]")
ax3.set_title("fcc Al — Phonon dispersion (SQTC)")
ax3.set_ylim(bottom=0)
plt.tight_layout()
plt.savefig(OUT_DIR / "al_phonon_bs.png", dpi=150)
plt.savefig(OUT_DIR / "al_phonon_bs.pdf")
print(f"Saved phonon plots → {OUT_DIR}/")

# ─────────────────────────────────────────────────────────────────────────────
# Step 10: print comparison table
# ─────────────────────────────────────────────────────────────────────────────

print("\n" + "="*80)
print(f" fcc Al — Temperature-dependent properties summary")
print(f" IFC fit: RMSE={report['rmse_ev_ang']:.5f} eV/Å  R²={report['r2']:.5f}  "
      f"{len(displacements_list)} snapshots")
print(f" ZPE = {ZPE_eV*1000:.3f} meV/f.u.     T_D(spectral) = {T_D_spectral:.1f} K")
print("="*80)
print(f"  {'T [K]':>7}  {'C_V [J/mol/K]':>14}  {'S [J/mol/K]':>12}  "
      f"{'F_vib [meV]':>12}  {'T_D [K]':>8}  {'MSD [Å²]':>10}  {'B [Å²]':>8}")
print("  " + "-"*82)
for row in summary_json["temperatures"]:
    T_c  = row["T_K"]
    cv   = row["Cv_jmolk"]
    s    = row["Svib_jmolk"]
    fv   = row["Fvib_ev"] * 1000
    td   = row["TD_caloric_K"]
    msd  = row["MSD_ang2"]
    b    = row["DW_B_ang2"]
    print(f"  {T_c:7.0f}  {cv:14.4f}  {s:12.4f}  {fv:12.3f}  {td:8.1f}  {msd:10.5f}  {b:8.5f}")

print("\nDone.  All results in", OUT_DIR)
