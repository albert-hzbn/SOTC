"""
SQTC cell design: Hermite Normal Form supercell enumeration and
displacement field optimisation via simulated annealing.

Workflow
--------
1.  HNFEnumerator.generate()  → list of candidate ASE Atoms supercells.
2.  DisplacementOptimizer.optimise()  → optimised {u_i} for a given cell.
3.  The optimised cell is written to POSCAR for DFT evaluation.

Displacement expansion (Section 2.4 of SQTC_framework.md):

    u_i^α = Σ_{q,s} A_{qs} Re[e_i^α(q,s) exp(iq·R_i)]

where A_{qs} is the harmonic amplitude:

    A_{qs}^harm = sqrt(ℏ/(2M ω_{qs}) coth(ℏω_{qs}/2k_BT))

Simulated annealing minimises the SQTC quality functional:

    Q = Σ_R w(R) [C̄₂^SQTC(R) - C̄₂^target(R,T)]²

with w(R) = exp(−R/ξ_T).
"""

from __future__ import annotations

import concurrent.futures
import itertools
from typing import List, Optional, Tuple

import numpy as np
from scipy.optimize import minimize

from .constants import HBAR, KB, AMU_TO_KG, ANG_TO_M, coth


# ── Module-level helper for ProcessPoolExecutor (must be picklable) ──────────

def _run_optimise_member(args):
    """Top-level wrapper so ProcessPoolExecutor can pickle the call."""
    optimizer, positions, cell, masses_amu, seed, verbose = args
    u, Q = optimizer.optimise(positions, cell, masses_amu, seed=seed, verbose=verbose)
    return u, Q


# ── Hermite Normal Form enumeration ──────────────────────────────────────────

class HNFEnumerator:
    """
    Enumerate candidate SQTC supercells via Hermite Normal Form matrices.

    A supercell transformation matrix H in upper-triangular HNF satisfies:
        det(H) = n   (n = number of primitive cells in the supercell)

    and has the form:
        H = [[h11,  h12, h13],
             [0,    h22, h23],
             [0,    0,   h33]]
    with h11*h22*h33 = n and 0 ≤ h12 < h22, 0 ≤ h13,h23 < h33.

    Parameters
    ----------
    primitive_cell : np.ndarray, shape (3,3)
        Rows are primitive lattice vectors [Å].
    n_atoms_prim : int
        Number of atoms in the primitive cell.
    n_min, n_max : int
        Range of desired formula units (number of prim cells).
    """

    def __init__(
        self,
        primitive_cell: np.ndarray,
        n_atoms_prim: int,
        n_min: int = 4,
        n_max: int = 32,
    ):
        self.prim_cell = np.array(primitive_cell, dtype=float)
        self.n_atoms_prim = n_atoms_prim
        self.n_min = n_min
        self.n_max = n_max

    def _hnf_matrices(self, n: int) -> List[np.ndarray]:
        """Return all distinct upper-triangular HNF matrices with det = n."""
        mats = []
        for h11 in range(1, n + 1):
            if n % h11 != 0:
                continue
            for h22 in range(1, n // h11 + 1):
                h33 = n // (h11 * h22)
                if h11 * h22 * h33 != n:
                    continue
                for h12 in range(h22):
                    for h13 in range(h33):
                        for h23 in range(h33):
                            H = np.array(
                                [[h11, h12, h13],
                                 [0,   h22, h23],
                                 [0,   0,   h33]],
                                dtype=int,
                            )
                            mats.append(H)
        return mats

    def generate(
        self,
        n_values: Optional[List[int]] = None,
        max_per_n: int = 5,
    ) -> List[Tuple[np.ndarray, np.ndarray]]:
        """
        Generate candidate (supercell_matrix, supercell_lattice) pairs.

        Parameters
        ----------
        n_values : list of int, optional
            Number of primitive cells per supercell to consider.
            Default: all even n in [n_min, n_max].
        max_per_n : int
            Maximum number of HNF shapes to keep per n value.
            Shapes are selected to minimise cell aspect ratio.

        Returns
        -------
        candidates : list of (H, A_sc) tuples
            H : (3,3) int  — supercell transformation matrix
            A_sc : (3,3) float  — supercell lattice [Å], rows = vectors
        """
        if n_values is None:
            n_values = [n for n in range(self.n_min, self.n_max + 1, 2)]

        candidates = []
        for n in n_values:
            hnf_list = self._hnf_matrices(n)
            # Score by condition number (cubic-like cells preferred)
            scored = []
            for H in hnf_list:
                A_sc = H @ self.prim_cell
                # Low condition number → more cubic shape
                try:
                    cond = np.linalg.cond(A_sc)
                except np.linalg.LinAlgError:
                    cond = 1e9
                scored.append((cond, H, A_sc))
            scored.sort(key=lambda t: t[0])
            for _, H, A_sc in scored[:max_per_n]:
                candidates.append((H, A_sc))

        return candidates

    def generate_supercell_atoms(
        self,
        prim_atoms,          # ase.Atoms primitive cell
        n_values: Optional[List[int]] = None,
        max_per_n: int = 5,
    ) -> List:
        """
        Generate candidate ASE Atoms supercells.

        Requires ASE to be installed.
        Returns list of ase.Atoms objects.
        """
        from ase.build import make_supercell as _ase_make_sc

        candidates = []
        for H, _ in self.generate(n_values, max_per_n):
            sc = _ase_make_sc(prim_atoms, H)
            sc.info["sqtc_H"] = H.tolist()
            candidates.append(sc)
        return candidates


# ── Displacement optimiser ────────────────────────────────────────────────────

class DisplacementOptimizer:
    """
    Optimise atomic displacement field {u_i} in an SQTC candidate cell to
    minimise the quality functional Q[C̄₂^SQTC − C̄₂^target].

    The optimisation is over displacement amplitudes A_{qs} (one scalar per
    phonon mode), with the displacement of atom i given by:

        u_i = Σ_{qs} A_{qs} Re[e_i(q,s) exp(iq·R_i)]

    Starting point: harmonic amplitudes A_{qs}^harm(T).

    For systems without phonopy, the Debye model is used to generate an
    approximate orthogonal displacement basis (random normal modes).

    Parameters
    ----------
    target_correlator : DebyeCorrelator or PhononCorrelator
        Provides C̄₂^target(R,T).
    T : float
        Temperature [K].
    xi_T : float
        Thermal coherence length [Å].  Controls the distance weighting.
    r_max : float
        Maximum pair-shell distance to include in Q [Å].
    lambda_2 : float
        Weight of pair-correlator term (λ₂ in SQTC framework).
    """

    def __init__(
        self,
        target_correlator,
        T: float,
        xi_T: float = 20.0,
        r_max: float = 10.0,
        lambda_2: float = 1.0,
    ):
        self.corr = target_correlator
        self.T = T
        self.xi_T = xi_T
        self.r_max = r_max
        self.lambda_2 = lambda_2

    # ── Quality functional ────────────────────────────────────────────────────

    def quality(
        self,
        displacements: np.ndarray,
        positions: np.ndarray,
        cell: np.ndarray,
        shells: List[Tuple[float, List[Tuple[int, int]]]],
        target_C2: np.ndarray,
    ) -> float:
        """
        Evaluate Q[{u_i}].

        Parameters
        ----------
        displacements : (n_atoms, 3) float [Å]
        positions : (n_atoms, 3) float [Å]  equilibrium positions
        cell : (3, 3) float [Å]  supercell lattice (rows = vectors)
        shells : list of (R_ang, [(i,j), ...]) — pair shells
        target_C2 : (n_shells,) float [Å²]  target C̄₂ values

        Returns
        -------
        Q : float  (dimensionless, in Å⁴)
        """
        u = displacements  # (n_atoms, 3)
        Q = 0.0
        for k, (R, pairs) in enumerate(shells):
            if R > self.r_max:
                break
            w = np.exp(-R / self.xi_T)
            # empirical C̄₂^SQTC(R)
            c2_sqtc = 0.0
            for i, j in pairs:
                c2_sqtc += np.dot(u[i], u[j])
            c2_sqtc /= 3.0 * max(len(pairs), 1)
            Q += w * (c2_sqtc - target_C2[k]) ** 2
        return self.lambda_2 * Q

    # ── Pair shell finder ─────────────────────────────────────────────────────

    def find_shells(
        self,
        positions: np.ndarray,
        cell: np.ndarray,
        tol: float = 0.05,
    ) -> Tuple[List[Tuple[float, List[Tuple[int, int]]]], np.ndarray]:
        """
        Find all pair shells within r_max and compute target C̄₂ for each.

        Returns
        -------
        shells : list of (R [Å], [(i,j) pairs])
        target_C2 : (n_shells,) [Å²]
        """
        n = len(positions)
        # lattice images: ±1 along each axis
        images = list(itertools.product([-1, 0, 1], repeat=3))
        distances: dict[float, List[Tuple[int, int]]] = {}

        for i in range(n):
            for j in range(n):
                for img in images:
                    if i == j and img == (0, 0, 0):
                        continue  # skip self at origin
                    R_vec = positions[j] - positions[i] + np.array(img) @ cell
                    dist = np.linalg.norm(R_vec)
                    if dist < 1e-3 or dist > self.r_max:
                        continue
                    # Round to tolerance
                    key = round(dist / tol) * tol
                    if key not in distances:
                        distances[key] = []
                    distances[key].append((i, j))

        sorted_keys = sorted(distances.keys())
        shells = [(k, distances[k]) for k in sorted_keys]

        # Compute target C̄₂ for each shell
        target_C2 = np.array([
            self.corr.pair_correlator_ang2(R, self.T)
            for R, _ in shells
        ])

        # Also add R=0 (on-site, gives MSD/3)
        msd3 = self.corr.msd_ang2(self.T) / 3.0
        shells.insert(0, (0.0, [(i, i) for i in range(n)]))
        target_C2 = np.concatenate([[msd3], target_C2])

        return shells, target_C2

    # ── Debye displacement basis ──────────────────────────────────────────────

    def debye_displacement_basis(
        self,
        positions: np.ndarray,
        n_modes: Optional[int] = None,
        seed: int = 42,
    ) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        """
        Build an approximate normal-mode basis using the Debye model.

        For each q-vector on the reciprocal supercell mesh, generate 3 plane
        waves with random orthonormal polarisations.  This is exact in the
        isotropic Debye limit.

        Returns
        -------
        q_vectors : (n_modes, 3) float [rad/Å]
        e_vectors : (n_modes, n_atoms, 3) float  polarisation
        A_harm : (n_modes,) float [Å]  harmonic amplitude
        """
        rng = np.random.default_rng(seed)
        n_atoms = len(positions)
        if n_modes is None:
            n_modes = 3 * n_atoms

        # q-vectors: simple grid over reciprocal cell
        n_q = n_modes // 3
        q_vecs_list = []
        # Use reciprocal lattice vectors (not implemented for general cell —
        # use Γ + small q vectors scaled to match Debye sphere)
        omega_D = getattr(self.corr, '_omega_D', 1e13)
        v_D = getattr(self.corr, '_v_D', 2000.0)  # m/s
        q_D_m = omega_D / v_D  # Debye wave-vector [rad/m]
        q_D = q_D_m * 1e-10    # [rad/Å]

        # Uniform sampling of q magnitudes from 0 to q_D
        for i in range(1, n_q + 1):
            q_mag = q_D * i / n_q
            # Random direction
            phi = rng.uniform(0, 2 * np.pi)
            theta = np.arccos(rng.uniform(-1, 1))
            q_hat = np.array([
                np.sin(theta) * np.cos(phi),
                np.sin(theta) * np.sin(phi),
                np.cos(theta),
            ])
            q_vecs_list.append(q_hat * q_mag)

        q_vecs = np.array(q_vecs_list)  # (n_q, 3)

        # 3 polarisations per q (LA + 2 TA) — random orthonormal frame
        e_vecs = np.zeros((n_q, 3, n_atoms, 3))
        for iq, q in enumerate(q_vecs):
            frame = self._random_frame(q, rng)  # (3, 3) rows = polarisations
            e_vecs[iq] = frame[:, np.newaxis, :]  # broadcast over atoms

        # Flatten (n_q * 3, n_atoms, 3)
        e_flat = e_vecs.reshape(-1, n_atoms, 3)
        q_flat = np.repeat(q_vecs, 3, axis=0)  # (n_modes, 3)

        # Apply phase factors exp(iq·R_i)
        # Real displacement: u_i = A Re[e_i exp(iq·R_i)]
        phases = np.exp(1j * positions @ q_flat.T).T  # (n_modes, n_atoms)
        for m in range(len(e_flat)):
            e_flat[m] = np.real(e_flat[m] * phases[m, :, np.newaxis])

        # Harmonic amplitudes
        M_kg = getattr(self.corr, '_M', 6.646e-27)
        # q_flat is in rad/Å; v_D is in m/s.  Convert: 1 rad/Å = 1e10 rad/m
        omega_per_mode = np.linalg.norm(q_flat, axis=1) * v_D * 1e10  # rad/s
        omega_per_mode = np.where(omega_per_mode < 1e6, 1e6, omega_per_mode)
        A_harm = np.sqrt(
            HBAR / (2.0 * M_kg * omega_per_mode)
            * coth(HBAR * omega_per_mode / (2.0 * KB * self.T + 1e-30))
        ) * 1e10  # m → Å

        return q_flat, e_flat, A_harm

    def _random_frame(self, q_vec: np.ndarray, rng) -> np.ndarray:
        """Return (3,3) orthonormal frame with first axis along q_vec (or random if q=0)."""
        q_norm = np.linalg.norm(q_vec)
        if q_norm < 1e-10:
            return np.eye(3)
        e1 = q_vec / q_norm
        # Generate orthogonal vectors
        perp = rng.standard_normal(3)
        perp -= np.dot(perp, e1) * e1
        perp /= np.linalg.norm(perp)
        e2 = perp
        e3 = np.cross(e1, e2)
        return np.array([e1, e2, e3])

    # ── Simulated annealing optimiser ─────────────────────────────────────────

    def optimise(
        self,
        positions: np.ndarray,
        cell: np.ndarray,
        masses_amu: np.ndarray,
        n_sa_steps: int = 5000,
        T_sa_init: float = 1.0,
        T_sa_final: float = 1e-4,
        seed: int = 42,
        verbose: bool = False,
    ) -> Tuple[np.ndarray, float]:
        """
        Optimise displacement amplitudes via simulated annealing.

        Parameters
        ----------
        positions : (n_atoms, 3) [Å]  equilibrium positions
        cell : (3, 3) [Å]  supercell cell matrix
        masses_amu : (n_atoms,) [amu]
        n_sa_steps : int  number of SA iterations
        T_sa_init, T_sa_final : float  SA temperature schedule
        seed : int  random seed for reproducibility
        verbose : bool  print progress every 500 steps

        Returns
        -------
        displacements : (n_atoms, 3) [Å]  optimised displacement field
        Q_final : float  final quality functional value
        """
        rng = np.random.default_rng(seed)
        n_atoms = len(positions)

        # Build displacement basis
        q_vecs, e_basis, A_harm = self.debye_displacement_basis(positions)
        n_modes = len(A_harm)

        # Find pair shells and target correlators
        shells, target_C2 = self.find_shells(positions, cell)

        # Initialise amplitudes at harmonic values (random signs)
        A = A_harm.copy() * rng.choice([-1.0, 1.0], size=n_modes)

        def disps_from_A(A_vec):
            """Compute (n_atoms, 3) displacements from amplitude vector."""
            # Vectorised: e_basis is (n_modes, n_atoms, 3)
            return np.tensordot(A_vec, e_basis, axes=([0], [0]))

        u_current = disps_from_A(A)
        Q_current = self.quality(u_current, positions, cell, shells, target_C2)

        Q_best = Q_current
        A_best = A.copy()

        # SA temperature schedule (geometric cooling)
        tau = -n_sa_steps / np.log(T_sa_final / T_sa_init)

        for step in range(n_sa_steps):
            T_sa = T_sa_init * np.exp(-step / tau)

            # Propose: perturb a random mode amplitude
            m = rng.integers(n_modes)
            delta = rng.standard_normal() * A_harm[m] * 0.3
            A_new = A.copy()
            A_new[m] += delta

            u_new = disps_from_A(A_new)
            Q_new = self.quality(u_new, positions, cell, shells, target_C2)

            dQ = Q_new - Q_current
            if dQ < 0 or rng.random() < np.exp(-dQ / (T_sa + 1e-30)):
                A = A_new
                u_current = u_new
                Q_current = Q_new

            if Q_current < Q_best:
                Q_best = Q_current
                A_best = A.copy()

            if verbose and step % 500 == 0:
                print(f"  SA step {step:5d}: Q = {Q_current:.4e}  T_sa = {T_sa:.4e}")

        u_opt = disps_from_A(A_best)

        # Polish with L-BFGS-B
        def objective(a_vec):
            u = disps_from_A(a_vec)
            return self.quality(u, positions, cell, shells, target_C2)

        res = minimize(
            objective, A_best, method="L-BFGS-B",
            options={"maxiter": 500, "ftol": 1e-14, "gtol": 1e-8},
        )
        if res.fun < Q_best:
            u_opt = disps_from_A(res.x)
            Q_best = res.fun

        return u_opt, Q_best

    # ── Ensemble generation ───────────────────────────────────────────────────

    def generate_ensemble(
        self,
        positions: np.ndarray,
        cell: np.ndarray,
        masses_amu: np.ndarray,
        n_members: int = 8,
        seed: int = 0,
        verbose: bool = False,
    ) -> List[np.ndarray]:
        """
        Generate an ensemble of M SQTC displacement configurations by
        repeating the SA with different random seeds.

        The ensemble collectively samples the thermal distribution up to the
        pair-correlator level.

        Returns
        -------
        ensemble : list of (n_atoms, 3) displacement arrays [Å]
        """
        args_list = [
            (self, positions, cell, masses_amu, seed + m, verbose)
            for m in range(n_members)
        ]
        # CPU-bound SA optimisation: use ProcessPoolExecutor for true parallelism
        with concurrent.futures.ProcessPoolExecutor() as executor:
            results = list(executor.map(_run_optimise_member, args_list))

        ensemble = [u for u, Q in results]
        if verbose:
            for m, (u, Q) in enumerate(results):
                print(f"  Member {m+1}/{n_members}: Q_final = {Q:.3e}")
        return ensemble
