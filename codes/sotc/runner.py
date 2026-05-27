"""
SOTC self-consistent runner.

Implements the full SOTC algorithm loop (Section 3 of SOTC_framework.md):

  Step 1  Compute target correlators C̄₂^target(R,T)  [Debye or phonopy]
  Step 2  Enumerate HNF supercell candidates
  Step 3  Optimise displacement fields via SA to minimise Q_SOTC
  Step 4  Evaluate forces: VASP (real) or pair-potential (mock)
  Step 5  Extract IFCs by linear regression
  Step 6  Reconstruct phonon dispersion and C_V(T)
  Step 7  Update target correlators from DFT IFCs → go to Step 3

Convergence is checked on ‖C̄₂^(n+1) − C̄₂^(n)‖₂ < ε_conv.

Usage (mock/test mode)
----------------------
    from sqtc import SOTCRunner
    from sotc.mock_forces import AzizHFDPotential

    runner = SOTCRunner(
        element="He",
        mass_amu=4.0026,
        prim_cell=he_cell,
        prim_positions=he_pos,
        T=1.6,          # K
        T_D=26.0,       # K (Debye temperature)
        n_atoms_sc=16,  # target supercell size
        force_calculator=AzizHFDPotential(),
    )
    results = runner.run(n_sc_iterations=3)
    print(results['C_V_jmolk'])
"""

from __future__ import annotations

import json
import time
import concurrent.futures
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np

from .constants import KB, HBAR, AMU_TO_KG, ANG_TO_M, NA
from .correlators import DebyeCorrelator
from .cell_design import HNFEnumerator, DisplacementOptimizer
from .ifc_extractor import IFCExtractor
from .phonons import PhononCalculator
from .mock_forces import mock_vasp_results


class SOTCRunner:
    """
    Orchestrates the full SOTC self-consistent loop.

    Parameters
    ----------
    element : str
        Chemical symbol, e.g. "He" or "H".
    mass_amu : float
        Atomic mass [amu].
    prim_cell : (3,3) ndarray [Å]
        Primitive cell lattice vectors (rows).
    prim_positions : (n_b, 3) ndarray [Å]
        Equilibrium atomic positions in the primitive cell.
    T : float
        Physical temperature [K].
    T_D : float
        Debye temperature [K]  (used for initial seed correlators).
    n_atoms_sc : int
        Target number of atoms in the SOTC supercell.
    force_calculator : callable, optional
        Object with .compute(positions, cell) → (N,3) forces [eV/Å].
        If None, a VASPRunner is used (requires vasp_cmd).
    vasp_cmd : str, optional
        VASP launch command, e.g. 'mpirun -np 8 vasp_std'.
    r_cutoff : float [Å]
        IFC regression cutoff radius.
    xi_T : float [Å]
        Thermal coherence length (overrides Debye estimate if given).
    r_max_corr : float [Å]
        Maximum shell distance included in Q_SOTC.
    n_ensemble : int
        Number of SOTC displacement configurations.
    work_dir : str or Path
        Working directory for SOTC calculations.
    verbosity : int
        0 = silent, 1 = progress, 2 = detailed.
    ridge_alpha : float [eV²/Å⁴]
        Tikhonov regularisation for the IFC regression.  Prevents near-zero
        force constants when the system is underdetermined (few snapshots
        relative to the number of IFC parameters).  Typical range: 1e-4–1e-2.
        Set to 0 to use unregularised least-squares (default, backward-compat).
    symmetrize_bonds : bool
        After IFC regression, project each bond class onto the central-force
        form Φ(R)=α·r̂⊗r̂+β·(I−r̂⊗r̂) and average α,β over symmetry-equivalent
        shells.  Strongly recommended for monatomic high-symmetry crystals
        (fcc, bcc) with small displacement ensembles: reduces the effective IFC
        parameter count from K×9 to 2 per shell, eliminating the overfitting
        that produces negative longitudinal force constants and imaginary modes.
    """

    def __init__(
        self,
        element: str,
        mass_amu: float,
        prim_cell: np.ndarray,
        prim_positions: np.ndarray,
        T: float,
        T_D: float,
        n_atoms_sc: int = 16,
        force_calculator=None,
        vasp_cmd: Optional[str] = None,
        vasp_settings: Optional[Dict] = None,
        r_cutoff: float = 6.0,
        xi_T: Optional[float] = None,
        r_max_corr: float = 8.0,
        n_ensemble: int = 5,
        work_dir: str | Path = "sqtc_run",
        verbosity: int = 1,
        ridge_alpha: float = 0.0,
        symmetrize_bonds: bool = False,
        elements: Optional[List[str]] = None,
        masses_amu: Optional[List[float]] = None,
    ):
        self.element = element
        self.mass_amu = mass_amu
        self.prim_cell = np.asarray(prim_cell, dtype=float)
        self.prim_pos = np.asarray(prim_positions, dtype=float)
        self.T = T
        self.T_D = T_D
        self.n_atoms_sc = n_atoms_sc
        self.force_calc = force_calculator
        self.vasp_cmd = vasp_cmd
        self.vasp_settings = vasp_settings or {}
        self.r_cutoff = r_cutoff
        self.r_max_corr = r_max_corr
        self.n_ensemble = n_ensemble
        self.work_dir = Path(work_dir)
        self.verbosity = verbosity
        self.ridge_alpha = ridge_alpha
        self.symmetrize_bonds = symmetrize_bonds

        self.n_b = len(prim_positions)

        # Per-basis-atom species and masses (compound support).
        # If elements/masses_amu lists are given, use them; otherwise replicate
        # the single element/mass_amu for each basis atom.
        self.elements_per_atom: List[str] = (
            list(elements) if elements is not None else [element] * self.n_b
        )
        self.masses_per_atom: np.ndarray = (
            np.asarray(masses_amu, dtype=float)
            if masses_amu is not None
            else np.full(self.n_b, mass_amu)
        )
        if len(self.elements_per_atom) != self.n_b or len(self.masses_per_atom) != self.n_b:
            raise ValueError(
                f"elements and masses_amu must each have length n_b={self.n_b}; "
                f"got {len(self.elements_per_atom)} and {len(self.masses_per_atom)}."
            )

        # Initial Debye correlator (seed for iteration 0).
        # Use mean atomic mass for the Debye seed (appropriate for compounds).
        V_prim_ang3 = abs(float(np.linalg.det(np.array(prim_cell, dtype=float)))) / self.n_b
        M_eff_amu = float(np.mean(self.masses_per_atom))
        self.debye_corr = DebyeCorrelator(T_D=T_D, M_amu=M_eff_amu, V_per_atom_ang3=V_prim_ang3)

        # Thermal coherence length
        if xi_T is not None:
            self.xi_T = xi_T
        else:
            xi_est = self.debye_corr.coherence_length(T, gruneisen=2.0)
            # ξ_T = ℏv_D/(γk_BT) gives the phonon thermal de Broglie length, which
            # for most metals at 300K is <1 Å — too small to usefully weight the
            # displacement quality functional beyond the on-site term.  Clamp to at
            # least 2 Å (≈ nearest-neighbor distance) so the optimizer penalises
            # mismatch at all shells within r_cutoff.
            self.xi_T = max(xi_est, 2.0)
            if verbosity >= 1:
                print(f"  ξ_T = {self.xi_T:.1f} Å  (Debye estimate at T={T} K, floor 2 Å)")

        self._results: Dict = {}
        self._iteration_log: List[Dict] = []

    # ── Step 1: target correlators ────────────────────────────────────────────

    def _target_correlators(
        self,
        R_shells: np.ndarray,
        current_corr=None,
    ) -> np.ndarray:
        """
        Compute C̄₂^target(R,T) [Å²] for each shell distance.

        Uses current_corr if provided (after first SC iteration),
        otherwise falls back to Debye model.
        """
        if current_corr is None:
            current_corr = self.debye_corr
        return current_corr.correlator_table(R_shells, self.T)

    # ── Step 2: choose supercell ──────────────────────────────────────────────

    def _choose_supercell(self) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
        """
        Choose the best HNF supercell of size ≈ n_atoms_sc.

        Returns
        -------
        H : (3,3) int — HNF transformation matrix
        sc_cell : (3,3) float [Å] — supercell lattice
        sc_pos : (N,3) float [Å] — equilibrium atomic positions in supercell
        masses : (N,) float [amu]
        """
        n_prim_cells = self.n_atoms_sc // self.n_b
        enumerator = HNFEnumerator(
            self.prim_cell, self.n_b,
            n_min=max(2, n_prim_cells - 2),
            n_max=n_prim_cells + 4,
        )

        # Generate more candidates so that NN-complete supercells are found.
        candidates = enumerator.generate(max_per_n=20)
        if not candidates:
            # Fallback: simple cubic repetition
            n = round(n_prim_cells ** (1.0/3.0))
            n = max(n, 2)
            H = np.diag([n, n, n])
            sc_cell = H @ self.prim_cell
            candidates = [(H, sc_cell)]

        # Count NN of atom 0 in each candidate supercell.
        # For correct IFCs the supercell must contain all symmetrically
        # distinct nearest-neighbour shells within r_cutoff.  A supercell
        # that is missing even one NN direction produces an asymmetric
        # dynamical matrix with spurious imaginary branches.
        # Score: (missing_NN_fraction, condition_number) — smaller is better.
        import itertools as _it
        _images = list(_it.product(range(-2, 3), repeat=3))

        def _count_nn(H_cand, sc_cell_cand):
            """Count how many NN (d < r_cutoff) atom 0 sees in the supercell."""
            H_inv = np.linalg.inv(H_cand.astype(float))
            max_idx = int(np.max(np.abs(H_cand))) + 2
            sc_pos_c = []
            for n1 in range(max_idx + 1):
                for n2 in range(max_idx + 1):
                    for n3 in range(max_idx + 1):
                        m = np.array([n1, n2, n3], dtype=float)
                        frac = m @ H_inv
                        if np.all(frac >= -1e-9) and np.all(frac < 1.0 - 1e-9):
                            sc_pos_c.append(m @ self.prim_cell)
            sc_pos_c = np.array(sc_pos_c)
            nn = 0
            for j in range(1, len(sc_pos_c)):
                dR = sc_pos_c[j] - sc_pos_c[0]
                d_min = min(
                    np.linalg.norm(dR + np.array(img) @ sc_cell_cand)
                    for img in _images
                )
                if d_min <= self.r_cutoff:
                    nn += 1
            return nn

        best_score = (np.inf, np.inf)
        best = candidates[0]

        # Estimate expected NN count in the infinite crystal (no PBC).
        _nn_inf = sum(
            1 for n1 in range(-3, 4) for n2 in range(-3, 4) for n3 in range(-3, 4)
            if 1e-3 < np.linalg.norm(np.array([n1, n2, n3]) @ self.prim_cell) <= self.r_cutoff
        )

        for H, sc_cell in candidates:
            nn = _count_nn(H, sc_cell)
            missing = max(0, _nn_inf - nn)
            c = np.linalg.cond(sc_cell)
            score = (missing, c)
            if score < best_score:
                best_score = score
                best = (H, sc_cell)

        H, sc_cell = best
        n_sc = int(round(abs(np.linalg.det(H))))
        if self.verbosity >= 1:
            _nn_sel = _count_nn(H, sc_cell)
            print(f"  Selected H = {H.tolist()}, NN count = {_nn_sel}/{_nn_inf}, cond = {np.linalg.cond(sc_cell):.2f}")

        # Build supercell positions by enumerating integer coefficient vectors
        # n = (n1, n2, n3) such that n @ prim_cell stays inside the supercell.
        # For a general HNF matrix H (upper triangular, possibly off-diagonal),
        # simply iterating range(H[0,0]) × range(H[1,1]) × range(H[2,2]) is
        # WRONG for non-diagonal H — it misses shifted basis positions.
        # The correct set: enumerate all integer m and keep those whose
        # fractional supercell coordinates are in [0,1).
        # With rows-as-vectors convention (pos = m @ prim_cell, r = frac @ sc_cell):
        #   frac = m @ H_inv   (NOT H_inv @ m, which is for column-vector convention)
        # For diagonal H both give the same result, but they differ for non-diagonal H.
        H_inv = np.linalg.inv(H.astype(float))
        sc_pos = []
        # With the row-vector convention frac = m @ H_inv, the maximum value
        # that coordinate k can reach is bounded by the sum of column k of H
        # (all elements H[i,k] for i ≤ k in the upper-triangular HNF).
        # Using only max(diagonal) + max(off-diagonal) is too tight when
        # multiple off-diagonal elements compound along the same axis, causing
        # valid lattice points to be silently skipped and the atom count to be
        # wrong (e.g. H=[[4,1,2],[0,4,1],[0,0,4]] needs n3 up to 6, not 5).
        max_idx = max(
            H[0, 0],
            abs(H[0, 1]) + H[1, 1],
            abs(H[0, 2]) + abs(H[1, 2]) + H[2, 2],
        ) + 1   # +1: safe margin; the inside-check filters extras
        for n1 in range(0, max_idx):
            for n2 in range(0, max_idx):
                for n3 in range(0, max_idx):
                    m = np.array([n1, n2, n3], dtype=float)
                    frac = m @ H_inv          # fractional coords in supercell
                    if np.all(frac >= -1e-9) and np.all(frac < 1.0 - 1e-9):
                        cart_shift = m @ self.prim_cell
                        for pos in self.prim_pos:
                            sc_pos.append(pos + cart_shift)

        sc_pos = np.array(sc_pos)
        if len(sc_pos) != n_sc * self.n_b:
            raise RuntimeError(
                f"Supercell tiling produced {len(sc_pos)} atoms, "
                f"expected {n_sc * self.n_b}.  H = {H.tolist()}"
            )
        masses = np.tile(self.masses_per_atom, len(sc_pos) // self.n_b)

        if self.verbosity >= 1:
            print(f"  Supercell: {n_sc} prim. cells, {len(sc_pos)} atoms")
            print(f"  H = {H.tolist()}")

        return H, sc_cell, sc_pos, masses

    # ── Step 3: displacement optimisation ─────────────────────────────────────

    def _optimise_ensemble(
        self,
        sc_cell: np.ndarray,
        sc_pos: np.ndarray,
        masses: np.ndarray,
        target_corr=None,
        iteration: int = 0,
    ) -> List[np.ndarray]:
        """Run SA to produce n_ensemble SOTC displacement configurations."""
        optimizer = DisplacementOptimizer(
            target_correlator=target_corr or self.debye_corr,
            T=self.T,
            xi_T=self.xi_T,
            r_max=self.r_max_corr,
        )

        if self.verbosity >= 1:
            print(f"  Optimising {self.n_ensemble} displacement configurations ...")

        ensemble = optimizer.generate_ensemble(
            positions=sc_pos,
            cell=sc_cell,
            masses_amu=masses,
            n_members=self.n_ensemble,
            seed=42 + iteration * 100,
            verbose=(self.verbosity >= 2),
        )
        return ensemble

    # ── Step 4: force evaluation ──────────────────────────────────────────────

    def _equilibrium_forces(
        self,
        sc_cell: np.ndarray,
        sc_pos: np.ndarray,
    ) -> np.ndarray:
        """
        Forces [eV/Å] at the undisplaced (equilibrium) positions.

        These are subtracted from all displaced-configuration forces so that
        the IFC regression sees only ΔF = F(u) − F(0), which is linear in u
        even when the lattice is not at the potential-energy minimum (e.g.
        when using an external pressure or a slightly mis-matched lattice
        constant).
        """
        if self.force_calc is not None:
            res0 = mock_vasp_results(sc_pos, sc_cell, self.force_calc)
            F0 = res0["forces_ev_ang"]
        else:
            # For real VASP, callers should supply the relaxed structure;
            # returning zero here is the safe fallback.
            F0 = np.zeros((len(sc_pos), 3))
        if self.verbosity >= 1:
            print(
                f"  Equilibrium forces: |F0|_max = {np.abs(F0).max():.4f} eV/Å  "
                f"  |F0|_rms = {np.sqrt(np.mean(F0**2)):.4f} eV/Å"
            )
        return F0

    def _evaluate_forces(
        self,
        sc_cell: np.ndarray,
        sc_pos: np.ndarray,
        ensemble: List[np.ndarray],
        iteration: int = 0,
    ) -> Tuple[List[np.ndarray], List[np.ndarray]]:
        """
        Evaluate DFT (or mock) forces for all SOTC configurations in parallel.

        Returns
        -------
        displacements_list : list of (N,3) arrays [Å]
        forces_list        : list of (N,3) arrays [eV/Å]
            Force *differences* ΔF = F(u) − F(0).
        """
        F_eq = self._equilibrium_forces(sc_cell, sc_pos)

        def _compute_snapshot(m_u):
            m, u = m_u
            pos_displaced = sc_pos + u
            if self.force_calc is not None:
                res = mock_vasp_results(pos_displaced, sc_cell, self.force_calc)
            elif self.vasp_cmd is not None:
                from .vasp_io import VASPRunner
                runner = VASPRunner(
                    vasp_cmd=self.vasp_cmd,
                    base_dir=self.work_dir / f"iter_{iteration:02d}",
                )
                species = list(self.elements_per_atom) * (len(sc_pos) // self.n_b)
                vs = self.vasp_settings
                res = runner.run_snapshot(
                    m, sc_cell, species, sc_pos, u,
                    kgrid=vs.get("kgrid", (4, 4, 4)),
                    encut=vs.get("encut", 500.0),
                    functional=vs.get("functional", "PBE"),
                    ncore=vs.get("ncore", 4),
                    pp_base_dir=vs.get("pp_base_dir"),
                    pp_set=vs.get("pp_set", "PAW_PBE"),
                    extra_incar=vs.get("extra_incar"),
                    timeout=vs.get("timeout", 7200),
                )
            else:
                raise RuntimeError(
                    "Provide either force_calculator (mock) or vasp_cmd (DFT)."
                )
            delta_F = res["forces_ev_ang"] - F_eq
            return m, u, delta_F, res

        # ThreadPoolExecutor: vectorized numpy/scipy releases GIL → real parallelism
        with concurrent.futures.ThreadPoolExecutor() as executor:
            snapshot_results = list(executor.map(_compute_snapshot, enumerate(ensemble)))

        displacements_list = []
        forces_list = []
        for m, u, delta_F, res in snapshot_results:
            if not res["converged"] and self.verbosity >= 1:
                print(f"  WARNING: snapshot {m} did not converge!")
            displacements_list.append(u)
            forces_list.append(delta_F)
            if self.verbosity >= 1:
                print(
                    f"  Snapshot {m+1}/{len(ensemble)}: "
                    f"E = {res['energy_ev']:.4f} eV  "
                    f"|ΔF|_max = {np.abs(delta_F).max():.4f} eV/Å"
                )

        return displacements_list, forces_list

    # ── Step 5: IFC extraction ─────────────────────────────────────────────────

    def _extract_ifcs(
        self,
        sc_cell: np.ndarray,
        sc_pos: np.ndarray,
        masses: np.ndarray,
        displacements_list: List[np.ndarray],
        forces_list: List[np.ndarray],
    ) -> IFCExtractor:
        """Linear regression for IFCs from force-displacement data."""
        extractor = IFCExtractor(
            supercell_positions=sc_pos,
            supercell_cell=sc_cell,
            masses_amu=masses,
            r_cutoff=self.r_cutoff,
            symmetrise=True,
            ridge_alpha=self.ridge_alpha,
            symmetrize_bonds=self.symmetrize_bonds,
        )
        extractor.fit(displacements_list, forces_list)

        if self.verbosity >= 1:
            report = extractor.fit_report(displacements_list, forces_list)
            print(
                f"  IFC fit: RMSE = {report['rmse_ev_ang']:.4f} eV/Å  "
                f"R² = {report['r2']:.4f}  rank = {report['rank']}"
            )

        return extractor

    # ── Step 6: phonon reconstruction ─────────────────────────────────────────

    def _compute_phonons(
        self,
        extractor: IFCExtractor,
        sc_pos: np.ndarray,
    ) -> PhononCalculator:
        """Build PhononCalculator and compute C_V."""
        calc = PhononCalculator(
            ifc_extractor=extractor,
            prim_positions=self.prim_pos,
            prim_cell=self.prim_cell,
            masses_amu=self.masses_per_atom,
        )
        return calc

    # ── Main loop ─────────────────────────────────────────────────────────────

    def run(
        self,
        n_sc_iterations: int = 15,
        epsilon_conv: float = 0.002,
        T_values: Optional[np.ndarray] = None,
        q_mesh_cv: Tuple[int, int, int] = (8, 8, 8),
        mixing: float = 0.5,
        min_iterations: int = 2,
    ) -> Dict:
        """
        Run the full SOTC self-consistency loop.

        Parameters
        ----------
        n_sc_iterations : int
            Maximum number of self-consistency iterations.
        epsilon_conv : float [Å²]
            Convergence threshold on ‖ΔC̄₂‖₂.
        min_iterations : int
            Minimum number of iterations to run before checking convergence.
            Prevents early stopping when the correlator floor is hit twice
            with the same value (delta=0) in only 2 iterations.
        T_values : ndarray, optional
            Temperatures [K] at which to compute C_V in the final step.
            Defaults to 10 points from 0.1·T to 3·T.
        q_mesh_cv : tuple of int
            q-mesh for final C_V integration.

        Parameters (additional)
        --------------------------
        mixing : float  (0 < mixing ≤ 1)
            Linear-mixing coefficient for the C̄₂ self-consistency update:
                C̄₂^{n+1} = mixing * C̄₂^{new} + (1−mixing) * C̄₂^{n}
            mixing=1 is pure substitution (fast but may oscillate);
            mixing=0.5 damps oscillations while staying reasonably fast.

        Returns
        -------
        results : dict with keys:
            'T'            : float          temperature [K]
            'C_V_jmolk'   : float          C_V at T [J/(mol·K)]
            'C_V_scan'    : (n_T,)         C_V over T_values [J/(mol·K)]
            'T_values'    : (n_T,)         temperature array [K]
            'omega_D_K'   : float          effective Debye temperature from IFCs [K]
            'ZPE_eV'      : float          zero-point energy per atom [eV]
            'MSD_ang2'    : float          mean-square displacement [Å²]
            'converged'   : bool
            'n_iterations' : int
            'iteration_log': list of dicts per iteration
        """
        self.work_dir.mkdir(parents=True, exist_ok=True)

        if T_values is None:
            T_values = np.linspace(0.1 * self.T, 3.0 * self.T, 20)

        t_start = time.time()

        # ── Choose supercell (fixed across iterations) ────────────────────────
        if self.verbosity >= 1:
            formula = "".join(self.elements_per_atom) if self.n_b > 1 else self.element
            print(f"\n{'='*60}")
            print(f"SOTC run: {formula}, T={self.T} K, T_D={self.T_D} K")
            print(f"{'='*60}")

        H, sc_cell, sc_pos, masses = self._choose_supercell()

        current_corr = None     # None → use Debye seed in first iteration
        prev_C2 = None
        converged = False
        phonon_calc = None
        extractor = None

        # Track best iterations so final reporting is robust against
        # late-iteration numerical blowups.
        best_physical_state = None
        best_physical_score = (np.inf, np.inf)
        best_any_state = None
        best_any_score = (np.inf, np.inf)
        prev_delta = None
        divergence_detected = False

        # Shell distances for convergence check
        R_shells = np.linspace(np.linalg.norm(self.prim_cell[0]) * 0.5, self.r_max_corr, 20)

        for it in range(n_sc_iterations):
            t_iter = time.time()
            if self.verbosity >= 1:
                print(f"\n── Iteration {it+1}/{n_sc_iterations} ──")

            # Step 3: displacement optimisation
            ensemble = self._optimise_ensemble(sc_cell, sc_pos, masses, current_corr, it)

            # Step 4: forces
            disps, forces = self._evaluate_forces(sc_cell, sc_pos, ensemble, it)

            # Step 5: IFC extraction
            extractor = self._extract_ifcs(sc_cell, sc_pos, masses, disps, forces)

            # Step 6: phonon reconstruction
            phonon_calc = self._compute_phonons(extractor, sc_pos)

            # Compute key thermodynamic and spectral-stability metrics in one pass
            thermo = phonon_calc.thermodynamic_summary(self.T, q_mesh=q_mesh_cv)
            cv_current = thermo["cv_jmolk"]
            T_D_eff = thermo["t_debye_k"]            # spectral-moment (robust)
            T_D_caloric = thermo["t_debye_caloric_k"]  # calorimetric (T ≈ T_D only)
            ZPE = thermo["zpe_ev"]
            unstable_fraction = thermo["unstable_fraction"]

            # Updated correlators from DFT IFCs
            C2_new = phonon_calc.updated_correlators(self.T, R_shells, q_mesh=q_mesh_cv)

            # ── Correlator stabilisation ──────────────────────────────────────
            # updated_correlators skips imaginary modes, so C2_new represents
            # only the (1 - unstable_fraction) stable fraction of the spectral
            # weight.  Compensate by rescaling:
            #
            #   C2_total ≈ C2_new / (1 - f)
            #
            # This preserves the *shape* of the correlator (including its
            # oscillations at longer R), unlike the previous additive Debye
            # blending which distorted the profile and overcorrected T_D.
            # Cap the scale at 2.0 to avoid over-amplification when f > 0.5.
            if unstable_fraction > 0.05:
                scale = min(1.0 / max(1.0 - unstable_fraction, 0.15), 2.0)
                C2_new = C2_new * scale
                if self.verbosity >= 1:
                    print(
                        f"  Correlator rescaled \u00d7{scale:.3f} "
                        f"({unstable_fraction*100:.1f}% imaginary modes compensated)."
                    )

            # Guard against NaN/Inf from a catastrophic IFC fit (e.g. all modes imaginary).
            if not np.all(np.isfinite(C2_new)):
                if self.verbosity >= 1:
                    print("  WARNING: non-finite correlators; falling back to Debye reference.")
                C2_new = self.debye_corr.correlator_table(R_shells, self.T)

            # Physical lower bound: never let C2 collapse below 20% of the Debye
            # reference.  If the IFC fit fails (e.g. R² << 0), the phonon modes
            # are unphysically stiff (large ω), producing near-zero correlators.
            # This guard prevents the optimizer from generating vanishingly small
            # displacements in the next iteration, which would destroy force SNR.
            C2_debye_ref = self.debye_corr.correlator_table(R_shells, self.T)
            C2_floor = 0.20 * C2_debye_ref
            if np.any(C2_new < C2_floor):
                if self.verbosity >= 1:
                    print(
                        f"  Correlator floor applied: C2[0]={C2_new[0]:.4f} -> "
                        f">= {C2_floor[0]:.4f} Å² (Debye 20% floor)."
                    )
                C2_new = np.maximum(C2_new, C2_floor)

            iter_dict = {
                "iteration": it + 1,
                "C_V_jmolk": cv_current,
                "T_D_eff_K": T_D_eff,
                "ZPE_eV": ZPE,
                "unstable_fraction": unstable_fraction,
                "C2_shells": C2_new.tolist(),
                "wall_time_s": time.time() - t_iter,
            }

            # Convergence check
            delta_C2 = None
            if prev_C2 is not None:
                delta_C2 = np.linalg.norm(C2_new - prev_C2)
                iter_dict["delta_C2"] = delta_C2
                if self.verbosity >= 1:
                    print(f"  ‖ΔC̄₂‖₂ = {delta_C2:.4e} Å²  (threshold: {epsilon_conv:.4e})")
                if delta_C2 < epsilon_conv and it + 1 >= min_iterations:
                    converged = True
                    if self.verbosity >= 1:
                        print(f"  ✓ Converged after {it+1} iterations!")
            else:
                iter_dict["delta_C2"] = None

            # Save the best iteration for robust final reporting.
            score = (
                unstable_fraction,
                delta_C2 if delta_C2 is not None else np.inf,
            )
            if best_any_state is None or score < best_any_score:
                best_any_score = score
                best_any_state = {
                    "iteration": it + 1,
                    "phonon_calc": phonon_calc,
                    "extractor": extractor,
                    "thermo": thermo,
                    "physical": False,
                }

            is_physical = (
                np.all(np.isfinite(C2_new))
                and np.isfinite(T_D_eff)
                and np.isfinite(ZPE)
                and (T_D_eff > 1.0)
                and (ZPE > 1e-4)
                and (unstable_fraction <= 0.15)
            )
            if is_physical:
                if best_physical_state is None or score < best_physical_score:
                    best_physical_score = score
                    best_physical_state = {
                        "iteration": it + 1,
                        "phonon_calc": phonon_calc,
                        "extractor": extractor,
                        "thermo": thermo,
                        "physical": True,
                    }

            # Detect sudden SC blowups and stop before accepting an unstable tail.
            if delta_C2 is not None and prev_delta is not None:
                blowup_factor = 8.0
                absolute_guard = 20.0 * epsilon_conv
                if delta_C2 > max(blowup_factor * prev_delta, absolute_guard):
                    divergence_detected = True
                    iter_dict["unstable_iteration"] = True
                    if self.verbosity >= 1:
                        print(
                            "  WARNING: Detected SC divergence spike; "
                            "stopping and rolling back to best stable iteration."
                        )

            self._iteration_log.append(iter_dict)

            if delta_C2 is not None:
                prev_delta = delta_C2

            # Linear mixing to damp oscillations:
            # C̄₂^{n+1} = mixing * C̄₂^{new} + (1−mixing) * C̄₂^{prev}
            if prev_C2 is not None and mixing < 1.0:
                C2_mixed = mixing * C2_new + (1.0 - mixing) * prev_C2
            else:
                C2_mixed = C2_new.copy()

            prev_C2 = C2_mixed.copy()

            # Build updated correlator object for next iteration
            # (use a simple wrapper that returns C2_mixed for each R)
            current_corr = _InterpolatedCorrelator(
                R_shells, C2_mixed, self.debye_corr, self.T
            )

            if converged or divergence_detected:
                break

        # ── Final phonon properties ───────────────────────────────────────────
        if phonon_calc is None:
            raise RuntimeError("No phonon calculation was completed.")

        selected_iteration = len(self._iteration_log)
        selected_thermo = None
        selected_state = best_physical_state or best_any_state
        selected_state_physical = best_physical_state is not None
        if selected_state is not None:
            phonon_calc = selected_state["phonon_calc"]
            extractor = selected_state["extractor"]
            selected_iteration = selected_state["iteration"]
            selected_thermo = selected_state.get("thermo")
            if self.verbosity >= 1:
                if selected_state_physical:
                    print(
                        f"  Using best physical iteration {selected_iteration} "
                        "for final properties."
                    )
                else:
                    print(
                        f"  WARNING: no physical iteration found; using least unstable iteration {selected_iteration} for final properties."
                    )

        if self.verbosity >= 1:
            print(f"\n── Final properties ──")

        if selected_thermo is None:
            selected_thermo = phonon_calc.thermodynamic_summary(self.T, q_mesh=q_mesh_cv)

        cv_final = selected_thermo["cv_jmolk"]
        cv_scan  = phonon_calc.heat_capacity_scan(T_values, q_mesh=(6, 6, 6))
        T_D_eff      = selected_thermo["t_debye_k"]           # spectral-moment
        T_D_caloric  = selected_thermo["t_debye_caloric_k"]   # calorimetric
        ZPE      = selected_thermo["zpe_ev"]
        unstable_fraction_selected = selected_thermo["unstable_fraction"]

        # MSD from Debye model with effective T_D
        debye_eff = DebyeCorrelator(
            T_D=T_D_eff if T_D_eff > 0.1 else self.T_D,
            M_amu=self.mass_amu,
        )
        msd = debye_eff.msd_ang2(self.T)

        if self.verbosity >= 1:
            print(f"  C_V({self.T} K) = {cv_final:.4f} J/(mol·K)")
            print(f"  T_D (spectral-moment)  = {T_D_eff:.2f} K")
            if abs(self.T / max(T_D_eff, 1.0)) < 3.0:
                # Calorimetric T_D is reliable when T < 3×T_D
                print(f"  T_D (calorimetric)     = {T_D_caloric:.2f} K")
            else:
                print(f"  T_D (calorimetric)     = {T_D_caloric:.2f} K  [ill-cond. T≫T_D]")
            print(f"  ZPE             = {ZPE*1000:.2f} meV/atom")
            print(f"  MSD (Debye)     = {msd:.4f} Å²")
            print(f"  Total wall time = {time.time()-t_start:.1f} s")

        self._results = {
            "T": self.T,
            "T_D_input": self.T_D,
            "T_D_effective": T_D_eff,
            "T_D_caloric": T_D_caloric,
            "C_V_jmolk": cv_final,
            "C_V_scan": cv_scan,
            "T_values": T_values,
            "ZPE_eV": ZPE,
            "MSD_ang2": msd,
            "unstable_fraction": unstable_fraction_selected,
            "converged": converged,
            "n_iterations": len(self._iteration_log),
            "selected_iteration": selected_iteration,
            "selected_state_physical": selected_state_physical,
            "iteration_log": self._iteration_log,
            "phonon_calculator": phonon_calc,
            "ifc_extractor": extractor,
        }

        # ── Phonon band structure and DOS ─────────────────────────────────────
        if self.verbosity >= 1:
            print(f"\n── Phonon band structure & DOS ──")

        try:
            bs = phonon_calc.compute_band_structure()
            np.savez(
                self.work_dir / "phonon_bandstructure.npz",
                distances=bs["distances"],
                frequencies_thz=bs["frequencies_thz"],
                q_points=bs["q_points"],
                labels=np.array(bs["labels"], dtype=str),
                label_positions=bs["label_positions"],
            )
            self._results["phonon_bandstructure"] = bs
            if self.verbosity >= 1:
                print(f"  Band structure saved: {self.work_dir / 'phonon_bandstructure.npz'}")
                print(f"  Path: {' - '.join(bs['labels'])}")
        except Exception as _exc:
            if self.verbosity >= 1:
                print(f"  WARNING: band structure calculation failed: {_exc}")

        try:
            dos = phonon_calc.compute_dos(q_mesh=(30, 30, 30))
            np.savez(
                self.work_dir / "phonon_dos.npz",
                frequencies_thz=dos["frequencies_thz"],
                dos=dos["dos"],
            )
            self._results["phonon_dos"] = dos
            if self.verbosity >= 1:
                print(f"  DOS saved: {self.work_dir / 'phonon_dos.npz'}")
        except Exception as _exc:
            if self.verbosity >= 1:
                print(f"  WARNING: DOS calculation failed: {_exc}")

        # Save a JSON summary (excluding non-serialisable objects)
        _skip = {"phonon_calculator", "ifc_extractor",
                 "phonon_bandstructure", "phonon_dos"}
        summary = {k: v for k, v in self._results.items()
                   if k not in _skip}
        summary["C_V_scan"] = list(cv_scan)
        summary["T_values"] = list(T_values)
        summary["r_cutoff"] = self.r_cutoff
        summary["ridge_alpha"] = self.ridge_alpha
        summary["symmetrize_bonds"] = self.symmetrize_bonds
        summary["n_ensemble"] = self.n_ensemble

        def _json_default(obj):
            if isinstance(obj, np.ndarray):
                return obj.tolist()
            if isinstance(obj, (np.floating, np.integer)):
                return obj.item()
            raise TypeError(f"Object of type {type(obj)} is not JSON serializable")

        (self.work_dir / "sotc_results.json").write_text(
            json.dumps(summary, indent=2, default=_json_default)
        )

        return self._results


# ── Helper: interpolated correlator wrapper ───────────────────────────────────

class _InterpolatedCorrelator:
    """
    Wrap a precomputed (R_shells, C2_shells) array so it can be used as a
    drop-in for DebyeCorrelator in subsequent SOTC iterations.
    """

    def __init__(self, R_shells, C2_shells, fallback_corr, T):
        self._R = R_shells
        self._C2 = C2_shells
        self._fallback = fallback_corr
        self._T = T

    def pair_correlator_ang2(self, R_ang: float, T: float) -> float:
        if R_ang <= self._R[0]:
            return float(self._C2[0])
        if R_ang >= self._R[-1]:
            return float(self._C2[-1])
        return float(np.interp(R_ang, self._R, self._C2))

    def msd_ang2(self, T: float) -> float:
        return self._fallback.msd_ang2(T)

    def correlator_table(self, R_values, T):
        return np.array([self.pair_correlator_ang2(R, T) for R in R_values])

    def coherence_length(self, T, gruneisen=2.0):
        return self._fallback.coherence_length(T, gruneisen)

    @property
    def _M(self):
        return self._fallback._M

    @property
    def _omega_D(self):
        return self._fallback._omega_D

    @property
    def _v_D(self):
        return self._fallback._v_D
