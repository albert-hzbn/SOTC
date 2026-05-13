"""
Interatomic Force Constant (IFC) extraction from SQTC force-displacement data.

The linear regression solves:

    F_i^α(m) = -Σ_{j,β} Φ_{ij}^{αβ} u_j^β(m),   m = 1,...,M (snapshots)

stacked into a single least-squares system:

    A · x = b

where
    b : (M · 3N,)   — observed DFT forces (all snapshots)
    x : (n_IFC,)    — independent IFC matrix elements
    A : (M·3N, n_IFC) — design matrix built from displacements

Translational symmetry (Bravais lattice):
    IFCs depend only on the displacement vector R = R_j - R_i, not on which
    specific pair (i,j) is chosen.  All pairs at the same minimum-image
    displacement share the same 3×3 tensor Φ(R).  The fit therefore has
    n_unique_displacements × 9 free parameters instead of n_pairs × 9, which
    makes the system well-overdetermined even for small ensembles (e.g. 6
    snapshots of a 16-atom supercell).

Constraints applied post-regression:
  1. Newton's third law: Φ(-R) = Φ(R)^T  (enforced by symmetrisation)
  2. Acoustic sum rule:  Φ_{ii}^{αβ} = -Σ_{j≠i} Φ_{ij}^{αβ}

Pairs beyond r_cutoff are ignored (set to zero).
"""

from __future__ import annotations

from typing import Dict, List, Optional, Tuple

import numpy as np
from scipy import linalg


class IFCExtractor:
    """
    Extract second-order IFCs from a set of (displacement, force) snapshots.

    Parameters
    ----------
    supercell_positions : (n_atoms, 3) float [Å]
        Equilibrium atomic positions in the supercell.
    supercell_cell : (3, 3) float [Å]
        Supercell lattice (rows = vectors).
    masses_amu : (n_atoms,) float [amu]
    r_cutoff : float [Å]
        Pairs beyond this distance are excluded from the IFC fit.
    symmetrise : bool
        Enforce Newton's third law Φ_{ij} = Φ_{ji}^T after regression.
    ridge_alpha : float [eV²/Å⁴]
        Tikhonov regularisation strength.  A value > 0 adds a penalty
        ``alpha * ||Φ||²`` to the least-squares cost, preventing near-zero
        IFCs when the system is underdetermined (few snapshots relative to
        the number of IFC parameters).  Typical range: 1e-4 – 1e-2.
    symmetrize_bonds : bool
        After regression, project each unique bond class onto the central-force
        (radial) form  Φ(R) = α·r̂⊗r̂ + β·(I − r̂⊗r̂)  and average α, β over
        all classes at the same shell distance.  This enforces the crystal
        point-group symmetry (all equivalent bonds have identical force
        constants) and reduces the effective parameter count from K×9 to 2 per
        shell, preventing the overfitting that produces negative longitudinal
        force constants (and hence imaginary phonon modes) when K×9 >> n_obs.
        Recommended for monatomic high-symmetry crystals (fcc, bcc, hcp, …)
        with small displacement ensembles.
    """

    def __init__(
        self,
        supercell_positions: np.ndarray,
        supercell_cell: np.ndarray,
        masses_amu: np.ndarray,
        r_cutoff: float = 6.0,
        symmetrise: bool = True,
        ridge_alpha: float = 0.0,
        symmetrize_bonds: bool = False,
    ):
        self.pos = np.asarray(supercell_positions, dtype=float)   # (N, 3)
        self.cell = np.asarray(supercell_cell, dtype=float)       # (3, 3)
        self.masses = np.asarray(masses_amu, dtype=float)         # (N,)
        self.r_cutoff = r_cutoff
        self.symmetrise = symmetrise
        self.symmetrize_bonds = symmetrize_bonds
        self.ridge_alpha = ridge_alpha
        self.N = len(self.pos)

        # Precompute neighbour list
        self._pairs, self._pair_vecs = self._build_pair_list()
        # IFC storage: dict (i, j) -> (3, 3) array
        self._Phi: Dict[Tuple[int, int], np.ndarray] = {}

    # ── Neighbour list ────────────────────────────────────────────────────────

    def _build_pair_list(self):
        """
        Find all unique pairs (i,j) with i<j and |R_ij| <= r_cutoff,
        considering periodic images.  Then group pairs by unique displacement
        vector, exploiting the translational symmetry of the Bravais lattice:
        Φ(R_i, R_j) = Φ(R_j - R_i) for a monatomic crystal.

        Returns
        -------
        pairs : list of (i, j) int tuples
        pair_vecs : dict (i,j) -> R_ij [Å] (vector from i to j)

        Side effects
        ------------
        self._unique_vecs : list of K unique displacement vectors [Å]
        self._pair_to_class : dict (i,j) -> (class index k, sign ∈ {±1})
            sign=+1: R_ij = +R_k  → IFC tensor = Φ_cls[k]
            sign=-1: R_ij = -R_k  → IFC tensor = Φ_cls[k]^T  (Newton's 3rd law)
        """
        import itertools
        pairs = []
        pair_vecs: Dict[Tuple[int, int], np.ndarray] = {}
        images = list(itertools.product([-1, 0, 1], repeat=3))

        for i in range(self.N):
            for j in range(i + 1, self.N):
                # Find minimum-image vector
                R_ij_min = None
                d_min = np.inf
                for img in images:
                    R = self.pos[j] - self.pos[i] + np.array(img) @ self.cell
                    d = np.linalg.norm(R)
                    if d < d_min:
                        d_min = d
                        R_ij_min = R
                if d_min <= self.r_cutoff:
                    pairs.append((i, j))
                    pair_vecs[(i, j)] = R_ij_min

        # ── Translational + inversion symmetry: group ±R pairs together ──────
        # Newton's 3rd law states Φ(-R) = Φ(R)^T.  By treating +R and -R as
        # the SAME displacement class (related by transposition), we:
        #   1. Halve the number of free parameters (K ≈ n_pairs/2)
        #   2. Enforce Newton's 3rd law exactly at the tensor level, not just
        #      as a post-hoc averaging step
        #
        # Convention: the canonical representative for each class is the first
        # vector seen (+R).  A pair with R_ij ≈ -R_k is assigned (cls=k, sign=-1)
        # meaning its IFC tensor is Φ_cls[k]^T.  A pair with R_ij ≈ +R_k gets
        # (cls=k, sign=+1).
        tol = 1e-3   # Å — numerical tolerance for vector equality
        unique_vecs: List[np.ndarray] = []
        pair_to_class: Dict[Tuple[int, int], Tuple[int, int]] = {}

        for (i, j) in pairs:
            R = pair_vecs[(i, j)]
            cls = -1
            sign = 0
            for k, R_k in enumerate(unique_vecs):
                if np.linalg.norm(R - R_k) < tol:
                    cls = k
                    sign = +1
                    break
                if np.linalg.norm(R + R_k) < tol:
                    cls = k
                    sign = -1
                    break
            if cls == -1:
                cls = len(unique_vecs)
                unique_vecs.append(R.copy())
                sign = +1
            pair_to_class[(i, j)] = (cls, sign)

        self._unique_vecs = unique_vecs      # K unique R vectors (canonical +R)
        self._pair_to_class = pair_to_class  # (i,j) → (class index, ±1 sign)
        return pairs, pair_vecs

    def n_pairs(self) -> int:
        return len(self._pairs)

    def n_unique_displacements(self) -> int:
        """Number of unique displacement classes (fit parameters = this × 9)."""
        return len(self._unique_vecs)

    # ── Design matrix ─────────────────────────────────────────────────────────

    def _build_design_matrix(
        self,
        displacements: np.ndarray,  # (N, 3) single snapshot
    ) -> np.ndarray:
        """
        Build one block (3N × 9·K) of the design matrix for a snapshot,
        where K = n_unique_displacements (translational symmetry applied).

        Pairs in the same displacement class k share IFC tensor Φ_cls[k].
        The sign (±1) determines orientation:

          sign=+1 → R_ij = +R_k → Φ(R_ij) = Φ_cls[k]
            F_i^α = -Φ_cls[k]^{αβ} (u_j^β - u_i^β)   col = 9k + α*3 + β
            F_j^β = -Φ_cls[k]^{αβ} (u_i^α - u_j^α)   col = 9k + α*3 + β  (NTL)

          sign=-1 → R_ij = -R_k → Φ(R_ij) = Φ_cls[k]^T
            F_i^α = -Φ_cls[k]^{βα} (u_j^β - u_i^β)   col = 9k + β*3 + α  (transposed)
            F_j^β = -Φ_cls[k]^{βα} (u_i^α - u_j^α)   col = 9k + β*3 + α  (same)

        NOTE: Relative displacements (u_j - u_i) are used instead of absolute
        u_j.  For collective (all-atoms) SQTC snapshots the acoustic sum rule
        requires F_i = -Σ_j Φ(ij)(u_j - u_i), so using absolute displacements
        omits the diagonal self-term and causes R² << 1 for acoustic-mode-rich
        configurations.  Single-atom (phonopy-style) displacements are
        unaffected because u_i = 0 for the displaced atom's neighbours.
        """
        u = displacements  # (N, 3)
        K = len(self._unique_vecs)
        A_block = np.zeros((3 * self.N, K * 9))

        for (i, j) in self._pairs:
            cls, sign = self._pair_to_class[(i, j)]
            col_base = cls * 9
            for a in range(3):
                for b in range(3):
                    # Column index: normal for sign=+1, transposed for sign=-1
                    col = col_base + (a * 3 + b if sign == +1 else b * 3 + a)
                    A_block[3 * i + a, col] -= (u[j, b] - u[i, b])
                    A_block[3 * j + b, col] -= (u[i, a] - u[j, a])

        return A_block

    # ── Regression ────────────────────────────────────────────────────────────

    def fit(
        self,
        displacements_list: List[np.ndarray],
        forces_list: List[np.ndarray],
        rcond: float = 1e-10,
    ) -> Dict[Tuple[int, int], np.ndarray]:
        """
        Fit IFCs by least-squares regression over all snapshots.

        Parameters
        ----------
        displacements_list : list of (N, 3) arrays [Å]
        forces_list        : list of (N, 3) arrays [eV/Å]

        Returns
        -------
        Phi : dict (i,j) -> (3,3) ndarray [eV/Å²]
            Second-order IFCs for all pairs within r_cutoff.
        """
        n_snap = len(displacements_list)
        n_obs = n_snap * 3 * self.N
        K = len(self._unique_vecs)          # number of unique displacement classes
        n_ifc = K * 9

        A_full = np.zeros((n_obs, n_ifc))
        b_full = np.zeros(n_obs)

        for m, (u, F) in enumerate(zip(displacements_list, forces_list)):
            row_start = m * 3 * self.N
            row_end   = row_start + 3 * self.N
            A_full[row_start:row_end, :] = self._build_design_matrix(u)
            b_full[row_start:row_end]    = F.ravel()

        # Tikhonov (ridge) regularisation: augment [A; √α I] x = [b; 0]
        # With translational symmetry the system is typically well-overdetermined
        # (n_obs >> n_ifc), so ridge_alpha mainly guards against numerical
        # ill-conditioning of nearly-degenerate shell distances.
        alpha = self.ridge_alpha
        if alpha > 0.0:
            sqrt_alpha = np.sqrt(alpha)
            A_full = np.vstack([A_full, sqrt_alpha * np.eye(n_ifc)])
            b_full = np.concatenate([b_full, np.zeros(n_ifc)])

        # Least-squares solve: A·x = b  → x = IFC elements
        x, residuals, rank, sv = linalg.lstsq(
            A_full, b_full, cond=rcond, overwrite_a=False, overwrite_b=False
        )

        self._rank = rank
        self._singular_values = sv

        # Unpack solution: one 3×3 tensor per unique class, then expand to
        # per-pair IFC dict using the sign to apply or invert transposition.
        # Newton's 3rd law Φ(-R) = Φ(R)^T is exact by construction.
        Phi_cls = [x[k * 9: (k + 1) * 9].reshape(3, 3) for k in range(K)]
        Phi: Dict[Tuple[int, int], np.ndarray] = {}
        for (i, j) in self._pairs:
            cls, sign = self._pair_to_class[(i, j)]
            if sign == +1:
                Phi[(i, j)] = Phi_cls[cls].copy()    # Φ(+R_k)
                Phi[(j, i)] = Phi_cls[cls].T.copy()  # Φ(-R_k) = Φ(R_k)^T
            else:  # sign == -1
                Phi[(i, j)] = Phi_cls[cls].T.copy()  # Φ(-R_k) = Φ(R_k)^T
                Phi[(j, i)] = Phi_cls[cls].copy()    # Φ(+R_k)

        if self.symmetrise:
            Phi = self._symmetrise(Phi)

        if self.symmetrize_bonds:
            Phi = self._symmetrize_bonds_central_force(Phi)

        # Apply acoustic sum rule to diagonal
        Phi = self._acoustic_sum_rule(Phi)

        self._Phi = Phi
        return Phi

    def _symmetrise(
        self,
        Phi: Dict[Tuple[int, int], np.ndarray],
    ) -> Dict[Tuple[int, int], np.ndarray]:
        """Enforce Φ_{ij}^{αβ} = Φ_{ji}^{βα} by averaging."""
        for i, j in self._pairs:
            Phi_ij = Phi[(i, j)]
            Phi_ji = Phi[(j, i)]
            sym = 0.5 * (Phi_ij + Phi_ji.T)
            Phi[(i, j)] = sym
            Phi[(j, i)] = sym.T
        return Phi

    def _symmetrize_bonds_central_force(
        self,
        Phi: Dict[Tuple[int, int], np.ndarray],
    ) -> Dict[Tuple[int, int], np.ndarray]:
        """
        Project IFCs onto the central-force (radial) form and average over
        symmetry-equivalent bond classes at the same shell distance.

        For a bond along unit vector r̂:
            Φ(R) = α · r̂⊗r̂  +  β · (I − r̂⊗r̂)

        Steps:
          1. For each unique class k, extract α_k = r̂_k·Φ_k·r̂_k (longitudinal)
             and β_k = (Tr(Φ_k) − α_k) / 2 (transverse average).
          2. Group classes by shell distance (within 1e-3 Å).
          3. Average α and β within each shell.
          4. Reconstruct Φ_k = ᾱ·r̂_k⊗r̂_k + β̄·(I − r̂_k⊗r̂_k) for every k.
          5. Update all per-pair entries in Phi accordingly.

        This enforces the crystal point-group symmetry for monatomic high-
        symmetry lattices (fcc, bcc, …) and reduces the effective parameter
        count from K×9 to 2 per shell.
        """
        from collections import defaultdict

        # --- Step 1 & 2: extract (alpha, beta) per class, group by |R| ---
        shell_classes: Dict = defaultdict(list)   # |R| (rounded) → list of k
        alpha_k = {}
        beta_k  = {}

        for k, R_k in enumerate(self._unique_vecs):
            r_hat = R_k / np.linalg.norm(R_k)
            # Recover Phi_cls[k] (canonical +R form) from any sign=+1 pair
            Phi_k = None
            for (i, j) in self._pairs:
                cls, sign = self._pair_to_class[(i, j)]
                if cls == k:
                    Phi_k = Phi[(i, j)].copy() if sign == +1 else Phi[(i, j)].T.copy()
                    break
            if Phi_k is None:
                continue
            # Symmetrize Phi_k first (make it symmetric in bond frame)
            Phi_k = 0.5 * (Phi_k + Phi_k.T)
            a = float(r_hat @ Phi_k @ r_hat)                     # longitudinal
            b = float((np.trace(Phi_k) - a) / 2.0)              # transverse avg
            alpha_k[k] = a
            beta_k[k]  = b
            shell_key = round(np.linalg.norm(R_k), 3)
            shell_classes[shell_key].append(k)

        # --- Step 3: average within each shell ---
        alpha_shell: Dict = {}   # k → shell-averaged alpha
        beta_shell:  Dict = {}

        for _r, ks in shell_classes.items():
            a_avg = float(np.mean([alpha_k[k] for k in ks]))
            b_avg = float(np.mean([beta_k[k]  for k in ks]))
            for k in ks:
                alpha_shell[k] = a_avg
                beta_shell[k]  = b_avg

        # --- Steps 4 & 5: reconstruct Phi_k and update all pairs ---
        for k, R_k in enumerate(self._unique_vecs):
            if k not in alpha_shell:
                continue
            r_hat = R_k / np.linalg.norm(R_k)
            a = alpha_shell[k]
            b = beta_shell[k]
            Phi_k_new = a * np.outer(r_hat, r_hat) + b * (np.eye(3) - np.outer(r_hat, r_hat))

            for (i, j) in self._pairs:
                cls, sign = self._pair_to_class[(i, j)]
                if cls != k:
                    continue
                if sign == +1:
                    Phi[(i, j)] = Phi_k_new.copy()
                    Phi[(j, i)] = Phi_k_new.T.copy()
                else:  # sign == -1  → R_{ij} = −R_k  → Φ(−R_k) = Φ(R_k)^T
                    Phi[(i, j)] = Phi_k_new.T.copy()
                    Phi[(j, i)] = Phi_k_new.copy()

        return Phi


    def _acoustic_sum_rule(
        self,
        Phi: Dict[Tuple[int, int], np.ndarray],
    ) -> Dict[Tuple[int, int], np.ndarray]:
        """
        Set on-site IFCs Φ_{ii} = -Σ_{j≠i} Φ_{ij}  (acoustic sum rule).
        """
        for i in range(self.N):
            on_site = np.zeros((3, 3))
            for j in range(self.N):
                if j == i:
                    continue
                if (i, j) in Phi:
                    on_site += Phi[(i, j)]
            Phi[(i, i)] = -on_site
        return Phi

    # ── Dynamical matrix from fitted IFCs ─────────────────────────────────────

    def dynamical_matrix(
        self,
        q_cart: np.ndarray,
        prim_positions: np.ndarray,
        prim_cell: np.ndarray,
        n_prim_atoms: int,
    ) -> np.ndarray:
        """
        Build the (3n_b × 3n_b) dynamical matrix at wavevector q [rad/Å].

        D^{αβ}_{ij}(q) = (1/√(M_i M_j)) Σ_R Φ_{ij}^{αβ}(R) exp(iq·R)

        where the sum is over all supercell images R of atom j relative to i.

        Parameters
        ----------
        q_cart : (3,) float [rad/Å]
        prim_positions : (n_b, 3) float [Å]  positions in primitive cell
        prim_cell : (3, 3) float [Å]
        n_prim_atoms : int

        Returns
        -------
        D : (3*n_b, 3*n_b) complex ndarray [amu⁻¹ · eV/Å²]
        """
        from .constants import AMU_TO_KG

        n_b = n_prim_atoms
        D = np.zeros((3 * n_b, 3 * n_b), dtype=complex)

        for (i, j), Phi_ij in self._Phi.items():
            i_prim = i % n_b
            j_prim = j % n_b
            R_ij = self._pair_vecs.get((i, j) if i < j else (j, i))
            if R_ij is None:
                R_ij = np.zeros(3)
            if i > j:
                R_ij = -R_ij

            phase = np.exp(1j * np.dot(q_cart, R_ij))
            M_i = self.masses[i] * AMU_TO_KG
            M_j = self.masses[j] * AMU_TO_KG
            prefac = 1.0 / np.sqrt(M_i * M_j)

            D[3*i_prim:3*i_prim+3, 3*j_prim:3*j_prim+3] += (
                prefac * phase * Phi_ij
            )

        return D

    # ── Fit quality metrics ────────────────────────────────────────────────────

    def fit_report(
        self,
        displacements_list: List[np.ndarray],
        forces_list: List[np.ndarray],
    ) -> Dict:
        """
        Compute force prediction errors after fitting.

        Returns dict with keys: 'rmse_ev_ang', 'max_error_ev_ang', 'r2'.
        """
        if not self._Phi:
            raise RuntimeError("Call fit() before fit_report().")

        residuals = []
        f_all = []
        for u, F in zip(displacements_list, forces_list):
            F_pred = self.predict_forces(u)
            residuals.append((F_pred - F).ravel())
            f_all.append(F.ravel())

        res = np.concatenate(residuals)
        f_ref = np.concatenate(f_all)
        rmse = np.sqrt(np.mean(res ** 2))
        max_err = np.max(np.abs(res))
        ss_tot = np.sum((f_ref - f_ref.mean()) ** 2)
        ss_res = np.sum(res ** 2)
        r2 = 1.0 - ss_res / (ss_tot + 1e-30)

        return {
            "rmse_ev_ang": rmse,
            "max_error_ev_ang": max_err,
            "r2": r2,
            "rank": self._rank,
        }

    def predict_forces(self, displacements: np.ndarray) -> np.ndarray:
        """
        Predict forces [eV/Å] for a given displacement field [Å].
        """
        u = displacements
        F = np.zeros_like(u)
        for (i, j), Phi_ij in self._Phi.items():
            if i == j:
                F[i] += -Phi_ij @ u[i]
            else:
                F[i] += -Phi_ij @ u[j]
        return F
