#!/usr/bin/env python3
"""
Plot phonon band structure + partial/total DOS for all completed SQTC runs.

For each compound, saves:
  <work_dir>/phonon_bandstructure.npz
  <work_dir>/phonon_dos.npz
  <work_dir>/phonon_combined.pdf / .png

Also saves a summary grid of all DOS plots:
  phonon_dos_summary.pdf / .png

Usage:
  cd /u/alli/calculations/sqtc
  .venv/bin/python codes/examples/plot_all_phonons.py
"""

import json
import sys
from pathlib import Path

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec

try:
    from ase.io import read as ase_read
except ImportError:
    raise ImportError("ASE required: pip install ase")

ROOT = Path(__file__).resolve().parent.parent.parent  # /u/alli/calculations/sqtc
sys.path.insert(0, str(ROOT / "codes"))

from sqtc.ifc_extractor import IFCExtractor
from sqtc.phonons import PhononCalculator

# ── Run definitions ─────────────────────────────────────────────────────────────
# Each entry specifies structural and fitting parameters to reconstruct the
# IFCExtractor exactly as in the original SQTC run.

def _fcc(a):
    return 0.5 * a * np.array([[0.,1,1],[1,0,1],[1,1,0]])

def _bcc(a):
    return 0.5 * a * np.array([[-1.,1,1],[1,-1,1],[1,1,-1]])

def _rocksalt_prim(a):
    return 0.5 * a * np.array([[0.,1,1],[1,0,1],[1,1,0]])

a_al    = 4.05;  a_ag = 4.085; a_au = 4.13;  a_cu = 3.67;  a_ni = 3.57
a_ti    = 3.27;  a_mo = 3.16;  a_w  = 3.19
a_nacl  = 5.73;  a_mgo = 4.26; a_lif = 4.10
a_mg    = 3.24;  c_mg = 5.26

RUNS = [
    # ── FCC monatomic metals ──────────────────────────────────────────────────
    {
        "label": "fcc Al",     "group": "FCC metals",
        "work_dir": "sqtc_al_fast_vasp_run",
        "elements_per_atom": ["Al"],
        "masses_amu": [26.9815385],
        "prim_cell": _fcc(a_al),
        "prim_pos": np.array([[0., 0, 0]]),
        "r_cutoff": 4.5,  "ridge_alpha": 1e-3,  "symmetrize_bonds": True,
    },
    {
        "label": "fcc Ag",     "group": "FCC metals",
        "work_dir": "sqtc_ag_vasp_run",
        "elements_per_atom": ["Ag"],
        "masses_amu": [107.8682],
        "prim_cell": _fcc(a_ag),
        "prim_pos": np.array([[0., 0, 0]]),
        "r_cutoff": 4.5,  "ridge_alpha": 1e-3,  "symmetrize_bonds": True,
    },
    {
        "label": "fcc Au",     "group": "FCC metals",
        "work_dir": "sqtc_au_vasp_run",
        "elements_per_atom": ["Au"],
        "masses_amu": [196.9665],
        "prim_cell": _fcc(a_au),
        "prim_pos": np.array([[0., 0, 0]]),
        "r_cutoff": 4.5,  "ridge_alpha": 1e-3,  "symmetrize_bonds": True,
    },
    {
        "label": "fcc Cu",     "group": "FCC metals",
        "work_dir": "sqtc_cu_vasp_run",
        "elements_per_atom": ["Cu"],
        "masses_amu": [63.546],
        "prim_cell": _fcc(a_cu),
        "prim_pos": np.array([[0., 0, 0]]),
        "r_cutoff": 4.5,  "ridge_alpha": 1e-3,  "symmetrize_bonds": True,
    },
    {
        "label": "fcc Ni",     "group": "FCC metals",
        "work_dir": "sqtc_fccni_vasp_run",
        "elements_per_atom": ["Ni"],
        "masses_amu": [58.6934],
        "prim_cell": _fcc(a_ni),
        "prim_pos": np.array([[0., 0, 0]]),
        "r_cutoff": 4.5,  "ridge_alpha": 1e-3,  "symmetrize_bonds": True,
    },
    # ── BCC monatomic metals ──────────────────────────────────────────────────
    {
        "label": "bcc Ti",     "group": "BCC metals",
        "work_dir": "sqtc_bccti_vasp_run",
        "elements_per_atom": ["Ti"],
        "masses_amu": [47.867],
        "prim_cell": _bcc(a_ti),
        "prim_pos": np.array([[0., 0, 0]]),
        "r_cutoff": 4.0,  "ridge_alpha": 1e-2,  "symmetrize_bonds": False,
    },
    {
        "label": "bcc Mo",     "group": "BCC metals",
        "work_dir": "sqtc_bccmo_vasp_run",
        "elements_per_atom": ["Mo"],
        "masses_amu": [95.96],
        "prim_cell": _bcc(a_mo),
        "prim_pos": np.array([[0., 0, 0]]),
        "r_cutoff": 4.0,  "ridge_alpha": 1e-2,  "symmetrize_bonds": False,
    },
    {
        "label": "bcc W",      "group": "BCC metals",
        "work_dir": "sqtc_bccw_vasp_run",
        "elements_per_atom": ["W"],
        "masses_amu": [183.84],
        "prim_cell": _bcc(a_w),
        "prim_pos": np.array([[0., 0, 0]]),
        "r_cutoff": 4.0,  "ridge_alpha": 1e-2,  "symmetrize_bonds": False,
    },
    # ── HCP ───────────────────────────────────────────────────────────────────
    {
        "label": "hcp Mg",     "group": "HCP",
        "work_dir": "sqtc_hcpmg_vasp_run",
        "elements_per_atom": ["Mg", "Mg"],
        "masses_amu": [24.305, 24.305],
        "prim_cell": np.array([
            [a_mg,           0.,                   0.],
            [-a_mg / 2., a_mg * np.sqrt(3) / 2.,   0.],
            [0.,             0.,                   c_mg],
        ]),
        "prim_pos": np.array([
            [0.,  0.,                   0.],
            [0.,  a_mg / np.sqrt(3),    c_mg / 2.],
        ]),
        "r_cutoff": 4.5,  "ridge_alpha": 1e-3,  "symmetrize_bonds": False,
    },
    # ── Rocksalt compounds ────────────────────────────────────────────────────
    {
        "label": "NaCl",       "group": "Rocksalt",
        "work_dir": "sqtc_nacl_vasp_run",
        "elements_per_atom": ["Na", "Cl"],       # Na at index 0, Cl at index 1
        "masses_amu": [22.9898, 35.453],
        "prim_cell": _rocksalt_prim(a_nacl),
        "prim_pos": np.array([[0., 0, 0], [a_nacl / 2., 0, 0]]),
        "r_cutoff": 4.3,  "ridge_alpha": 1e-3,  "symmetrize_bonds": True,
    },
    {
        "label": "MgO",        "group": "Rocksalt",
        "work_dir": "sqtc_mgo_vasp_run",
        "elements_per_atom": ["Mg", "O"],
        "masses_amu": [24.305, 15.999],
        "prim_cell": _rocksalt_prim(a_mgo),
        "prim_pos": np.array([[0., 0, 0], [a_mgo / 2., 0, 0]]),
        "r_cutoff": 3.2,  "ridge_alpha": 1e-3,  "symmetrize_bonds": True,
    },
    {
        "label": "LiF",        "group": "Rocksalt",
        "work_dir": "sqtc_lif_vasp_run",
        "elements_per_atom": ["Li", "F"],        # Li at index 0, F at index 1
        "masses_amu": [6.941, 18.998],
        "prim_cell": _rocksalt_prim(a_lif),
        "prim_pos": np.array([[0., 0, 0], [a_lif / 2., 0, 0]]),
        "r_cutoff": 3.5,  "ridge_alpha": 1e-3,  "symmetrize_bonds": True,
    },
]

# ── Element colour map for partial DOS ────────────────────────────────────────
ELEMENT_COLORS = {
    "Al": "#1f77b4", "Ag": "#aec7e8", "Au": "#ffbb78", "Cu": "#d62728",
    "Ni": "#2ca02c", "Ti": "#9467bd", "Mo": "#8c564b", "W":  "#7f7f7f",
    "Mg": "#bcbd22", "Na": "#17becf", "Cl": "#ff7f0e", "O":  "#e377c2",
    "Li": "#e7969c", "F":  "#98df8a",
}

# ── Helper: unsort atoms from VASP-sorted to original primitive-cell order ────
def _unsort_atoms(arrays, elements_per_atom, n_total):
    """
    Undo the alphabetical-by-species sorting that VASPRunner applies.

    Parameters
    ----------
    arrays : list of (n_total, 3) ndarray   — arrays in sorted order (from POSCAR/vasprun)
    elements_per_atom : list of str          — species in original prim-cell order
    n_total : int                            — total number of atoms in supercell

    Returns
    -------
    list of (n_total, 3) ndarray in original (primitive-cell-repeating) order.
    """
    n_b = len(elements_per_atom)
    n_fu = n_total // n_b
    # Build species list in original order
    species = elements_per_atom * n_fu     # ['Na','Cl','Na','Cl',...]
    sort_order = sorted(range(n_total), key=lambda i: species[i])
    # sort_order[new_i] = old_i
    # Invert: inv_order[old_i] = new_i
    inv_order = np.empty(n_total, dtype=int)
    for new_i, old_i in enumerate(sort_order):
        inv_order[old_i] = new_i
    # original = sorted[inv_order]   (same formula as in vasp_io.py for forces)
    return [arr[inv_order] for arr in arrays]

def _build_eq_positions(sc_cell, prim_cell, prim_pos):
    """
    Derive exact equilibrium positions and the HNF matrix H from the actual
    supercell cell matrix (read from POSCAR).

    This replicates runner._choose_supercell()'s enumeration loop verbatim
    so the atom ordering is identical to the POSCAR written during the run.

    Parameters
    ----------
    sc_cell  : (3,3) float  — supercell lattice vectors (rows), from POSCAR
    prim_cell: (3,3) float  — primitive cell lattice vectors (rows)
    prim_pos : (n_b, 3) float — primitive-cell basis positions [Å]

    Returns
    -------
    eq_positions : (n_total, 3) float
    H            : (3,3) int — HNF transformation matrix
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
                m = np.array([n1, n2, n3], dtype=float)
                frac = m @ H_inv
                if np.all(frac >= -1e-9) and np.all(frac < 1.0 - 1e-9):
                    cart_shift = m @ prim_cell
                    for pos in prim_pos:
                        sc_pos.append(pos + cart_shift)
    return np.array(sc_pos), H

def _need_unsort(elements_per_atom):
    """True if alphabetical sort would change the species order."""
    n_b = len(elements_per_atom)
    if n_b == 1:
        return False
    if len(set(elements_per_atom)) == 1:
        return False  # monatomic (e.g., hcp Mg with 2 identical atoms)
    return True

# ── Main processing function ───────────────────────────────────────────────────
def process_run(run_cfg, q_mesh=(30, 30, 30)):
    label        = run_cfg["label"]
    work_dir     = ROOT / run_cfg["work_dir"]
    elems        = run_cfg["elements_per_atom"]
    masses_amu   = run_cfg["masses_amu"]
    prim_cell    = run_cfg["prim_cell"]
    prim_pos     = run_cfg["prim_pos"]
    r_cutoff     = run_cfg["r_cutoff"]
    ridge_alpha  = run_cfg["ridge_alpha"]
    symm_bonds   = run_cfg["symmetrize_bonds"]
    n_b          = len(elems)

    print(f"\n{'='*60}")
    print(f"  {label}")
    print(f"{'='*60}")

    # Load results JSON to find selected iteration
    results_json = work_dir / "sqtc_results.json"
    if not results_json.exists():
        print(f"  SKIP: no sqtc_results.json in {work_dir}")
        return None
    saved = json.loads(results_json.read_text())
    selected_it = saved.get("selected_iteration", None)
    if selected_it is None:
        print(f"  SKIP: no selected_iteration in results")
        return None

    iter_name = f"iter_{selected_it - 1:02d}"
    iter_dir = work_dir / iter_name
    if not iter_dir.exists():
        print(f"  SKIP: {iter_dir} does not exist")
        return None

    snap_dirs = sorted(iter_dir.glob("snap_*"))
    if not snap_dirs:
        print(f"  SKIP: no snaps in {iter_dir}")
        return None

    print(f"  Using {iter_name}  ({len(snap_dirs)} snaps)")

    # Read POSCAR + vasprun.xml for each snap
    all_positions = []
    all_forces    = []
    sc_cell_poscar = None

    for sd in snap_dirs:
        poscar   = sd / "POSCAR"
        vasprun  = sd / "vasprun.xml"
        if not poscar.exists() or not vasprun.exists():
            print(f"    WARNING: skipping {sd.name}")
            continue
        atoms_pos   = ase_read(str(poscar),  format="vasp")
        atoms_vasp  = ase_read(str(vasprun), format="vasp-xml")
        if sc_cell_poscar is None:
            sc_cell_poscar = atoms_pos.get_cell().array   # (3,3) Å, rows=vectors
        all_positions.append(atoms_pos.get_positions())
        all_forces.append(atoms_vasp.get_forces())

    n_snaps = len(all_positions)
    if n_snaps == 0:
        print("  SKIP: no readable snapshots")
        return None
    n_total = all_positions[0].shape[0]
    print(f"  N_atoms = {n_total}, N_snaps = {n_snaps}")

    # Unsort from VASP alphabetical-sort to primitive-cell repeating order
    if _need_unsort(elems):
        all_positions = _unsort_atoms(all_positions, elems, n_total)
        all_forces    = _unsort_atoms(all_forces,    elems, n_total)

    # Reconstruct exact equilibrium positions from the POSCAR supercell cell.
    # Computing H = round(sc_cell @ prim_cell⁻¹) and enumerating with the
    # runner's exact loop guarantees atom ordering is identical to the POSCAR.
    eq_positions, H = _build_eq_positions(sc_cell_poscar, prim_cell, prim_pos)
    masses_sc = np.tile(masses_amu, n_total // n_b)
    if len(eq_positions) != n_total:
        raise RuntimeError(
            f"Reconstructed {len(eq_positions)} atoms from POSCAR cell, "
            f"expected {n_total}.  H={H.tolist()}"
        )
    print(f"  Supercell H={H.tolist()}, {n_total} atoms (from POSCAR cell)")

    # Displacement and delta-force (subtract mean residual)
    displacements = [pos - eq_positions for pos in all_positions]
    mean_force    = np.mean(all_forces, axis=0)
    delta_forces  = [f - mean_force for f in all_forces]

    # IFC extraction
    extractor = IFCExtractor(
        supercell_positions=eq_positions,
        supercell_cell=sc_cell_poscar,
        masses_amu=masses_sc,
        r_cutoff=r_cutoff,
        symmetrise=True,
        ridge_alpha=ridge_alpha,
        symmetrize_bonds=symm_bonds,
    )
    extractor.fit(displacements, delta_forces)
    rep = extractor.fit_report(displacements, delta_forces)
    print(f"  IFC fit: RMSE={rep['rmse_ev_ang']:.4f} eV/Å  "
          f"R²={rep['r2']:.4f}  rank={rep['rank']}")

    # PhononCalculator
    phonon_calc = PhononCalculator(
        ifc_extractor=extractor,
        prim_positions=prim_pos,
        prim_cell=prim_cell,
        masses_amu=np.array(masses_amu),
    )

    # Band structure
    print(f"  Computing band structure ...")
    bs = phonon_calc.compute_band_structure(n_points_per_segment=80)
    np.savez(
        work_dir / "phonon_bandstructure.npz",
        distances=bs["distances"],
        frequencies_thz=bs["frequencies_thz"],
        q_points=bs["q_points"],
        labels=np.array(bs["labels"], dtype=str),
        label_positions=bs["label_positions"],
    )
    print(f"    Path: {' - '.join(bs['labels'])}  "
          f"  f_max={bs['frequencies_thz'].max():.2f} THz")

    # Partial + total DOS
    print(f"  Computing partial DOS ({q_mesh[0]}³ mesh) ...")
    pdos = phonon_calc.compute_partial_dos(
        q_mesh=q_mesh, n_bins=500, sigma_thz=0.12
    )
    np.savez(
        work_dir / "phonon_dos.npz",
        frequencies_thz=pdos["frequencies_thz"],
        dos_total=pdos["dos_total"],
        dos_partial=pdos["dos_partial"],
        elements=np.array(elems, dtype=str),
    )

    return {
        "label":      label,
        "work_dir":   work_dir,
        "T":          saved.get("T"),
        "T_D_eff":    saved.get("T_D_effective"),
        "elements":   elems,
        "bs":         bs,
        "pdos":       pdos,
    }

# ── Plotting ──────────────────────────────────────────────────────────────────
def _label_tex(lbl):
    return lbl.replace("G", r"$\Gamma$")

def _element_label(elem, n_b, elems):
    """Return display label for a partial DOS trace."""
    # For hcp Mg: two 'Mg' basis atoms → merge into one 'Mg' curve
    return elem

def plot_compound(result):
    """Make band+DOS figure for one compound and save to work_dir."""
    if result is None:
        return
    bs      = result["bs"]
    pdos    = result["pdos"]
    label   = result["label"]
    elems   = result["elements"]
    T       = result["T"]
    T_D     = result["T_D_eff"]
    wdir    = result["work_dir"]

    dists      = bs["distances"]
    freqs      = bs["frequencies_thz"]
    n_branches = freqs.shape[1]

    freq_bins = pdos["frequencies_thz"]
    dos_total = pdos["dos_total"]
    dos_parts = pdos["dos_partial"]   # (n_b, n_bins)

    # Merge duplicate elements (e.g. hcp Mg has 2×Mg basis atoms) — sum their partial DOSes
    unique_elems = list(dict.fromkeys(elems))  # order-preserving unique
    dos_by_elem = {}
    for elem in unique_elems:
        mask = np.array([e == elem for e in elems])
        dos_by_elem[elem] = dos_parts[mask].sum(axis=0)

    fig = plt.figure(figsize=(11, 5))
    gs  = gridspec.GridSpec(1, 2, width_ratios=[3, 1])
    ax1 = fig.add_subplot(gs[0])
    ax2 = fig.add_subplot(gs[1], sharey=ax1)

    # — Band structure —
    for b in range(n_branches):
        ax1.plot(dists, freqs[:, b], color="steelblue", lw=1.2, alpha=0.9)
    for lpos in bs["label_positions"]:
        ax1.axvline(x=lpos, color="gray", lw=0.7, ls="--")
    ax1.axhline(y=0, color="gray", lw=0.5)
    ax1.set_xticks(bs["label_positions"])
    ax1.set_xticklabels([_label_tex(l) for l in bs["labels"]], fontsize=12)
    ax1.set_xlim(dists[0], dists[-1])
    ax1.set_ylabel("Frequency (THz)", fontsize=12)
    ax1.tick_params(axis="x", length=0)
    ax1.grid(axis="y", lw=0.3, alpha=0.5)
    ax1.set_title(f"{label} — SQTC phonons at {T:.0f} K  ($T_D$={T_D:.0f} K)", fontsize=12)

    # — DOS / partial DOS —
    if len(unique_elems) > 1:
        # Binary: overlapping fills (each from zero) + total black line
        for elem in unique_elems:
            d   = dos_by_elem[elem]
            col = ELEMENT_COLORS.get(elem, "#aaaaaa")
            ax2.fill_betweenx(freq_bins, d, alpha=0.30, color=col, label=elem)
            ax2.plot(d, freq_bins, color=col, lw=1.2, ls="--", alpha=0.8)
        ax2.plot(dos_total, freq_bins, color="k", lw=1.5, label="Total", zorder=5)
        ax2.legend(fontsize=8, loc="upper right", framealpha=0.7)
    else:
        # Monatomic: coral fill + firebrick line (matches the Al style)
        ax2.fill_betweenx(freq_bins, dos_total, alpha=0.35, color="coral")
        ax2.plot(dos_total, freq_bins, color="firebrick", lw=1.5)

    ax2.axhline(y=0, color="gray", lw=0.5)
    ax2.set_xlabel("DOS", fontsize=11)
    ax2.set_xlim(left=0)
    plt.setp(ax2.get_yticklabels(), visible=False)
    ax2.tick_params(axis="y", length=0)
    ax2.grid(axis="y", lw=0.3, alpha=0.5)

    # y-axis limits — same formula as plot_al_phonons.py
    y_max = max(freqs.max(),
                freq_bins[dos_total > 0.01 * dos_total.max()].max()
                if dos_total.max() > 0 else 1.0) * 1.05
    ax1.set_ylim(bottom=min(-0.5, freqs.min() * 1.1), top=y_max)

    fig.subplots_adjust(left=0.10, right=0.97, top=0.92, bottom=0.12, wspace=0.05)
    for ext in ("pdf", "png"):
        fig.savefig(wdir / f"phonon_combined.{ext}", dpi=200)
    print(f"  Saved: {wdir}/phonon_combined.pdf")
    plt.close(fig)

def plot_summary(results):
    """Make a grid of all DOS plots (total + partial) on one page."""
    valid = [r for r in results if r is not None]
    n = len(valid)
    ncols = 4
    nrows = (n + ncols - 1) // ncols
    fig, axes = plt.subplots(nrows, ncols, figsize=(ncols * 3.2, nrows * 3.4))
    axes = np.array(axes).ravel()

    for ax, result in zip(axes, valid):
        pdos       = result["pdos"]
        elems      = result["elements"]
        freq_bins  = pdos["frequencies_thz"]
        dos_total  = pdos["dos_total"]
        dos_parts  = pdos["dos_partial"]
        unique_elems = list(dict.fromkeys(elems))

        if len(unique_elems) > 1:
            bottom = np.zeros_like(freq_bins)
            for elem in unique_elems:
                mask = np.array([e == elem for e in elems])
                d    = dos_parts[mask].sum(axis=0)
                col  = ELEMENT_COLORS.get(elem, "#aaaaaa")
                ax.fill_betweenx(freq_bins, bottom, bottom + d,
                                 alpha=0.55, color=col, label=elem)
                ax.plot(bottom + d, freq_bins, color=col, lw=0.8)
                bottom += d
            ax.plot(dos_total, freq_bins, color="k", lw=1.2, label="Total")
            ax.legend(fontsize=6, loc="upper right")
        else:
            elem = unique_elems[0]
            col  = ELEMENT_COLORS.get(elem, "steelblue")
            ax.fill_betweenx(freq_bins, dos_total, alpha=0.45, color=col)
            ax.plot(dos_total, freq_bins, color=col, lw=1.2)

        ax.axhline(0, color="gray", lw=0.4)
        ax.set_xlim(left=0)
        ax.set_ylim(bottom=min(0., freq_bins.min()))
        ax.set_title(f"{result['label']}\nT={result['T']:.0f} K, "
                     f"T$_D$={result['T_D_eff']:.0f} K", fontsize=8)
        ax.set_xlabel("DOS", fontsize=7)
        ax.set_ylabel("Freq. (THz)", fontsize=7)
        ax.tick_params(labelsize=6)

    # Hide unused axes
    for ax in axes[len(valid):]:
        ax.set_visible(False)

    fig.suptitle("Phonon DOS — SQTC benchmark set", fontsize=12, fontweight="bold")
    fig.tight_layout(rect=[0, 0, 1, 0.96])
    for ext in ("pdf", "png"):
        fig.savefig(ROOT / f"phonon_dos_summary.{ext}", dpi=200)
    print(f"\nSaved summary: {ROOT}/phonon_dos_summary.pdf")
    plt.close(fig)

# ── Entry point ────────────────────────────────────────────────────────────────
if __name__ == "__main__":
    results = []
    for run_cfg in RUNS:
        try:
            result = process_run(run_cfg, q_mesh=(30, 30, 30))
            plot_compound(result)
            results.append(result)
        except Exception as exc:
            import traceback
            print(f"  ERROR in {run_cfg['label']}: {exc}")
            traceback.print_exc()
            results.append(None)

    plot_summary(results)

    print("\n" + "="*60)
    print("All done.")
    print(f"Individual plots: <work_dir>/phonon_combined.{{pdf,png}}")
    print(f"Summary: {ROOT}/phonon_dos_summary.{{pdf,png}}")
