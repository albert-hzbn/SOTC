"""
Quantum ESPRESSO force calculator for SOTC.

Wraps the Quantum ESPRESSO pw.x code (via ASE's Espresso calculator) as an
SOTC force calculator, providing the .compute() / .energy() interface
expected by SOTCRunner and mock_vasp_results().

Quantum ESPRESSO: https://www.quantum-espresso.org
ASE Espresso:    https://wiki.fysik.dtu.dk/ase/ase/calculators/espresso.html

Installation
------------
    # QE binary (compile or download):
    https://github.com/QEF/q-e/releases

    # SSSP efficiency pseudopotentials (recommended):
    https://www.materialscloud.org/discover/sssp

    # Install ASE (already in conda env):
    pip install ase

Usage
-----
    from sotc.qe_io import QEForceCalculator

    calc = QEForceCalculator(
        species=['Al'],                 # tiled to supercell size automatically
        pseudopotentials={'Al': 'Al.pbe-n-kjpaw_psl.1.0.0.UPF'},
        pseudo_dir='/path/to/SSSP/',
        ecutwfc=40.0,                   # Ry  (≈ 544 eV for Al PAW)
        kpts=(2, 2, 2),
        workdir='sqtc_al_qe_run/qe_scratch',
    )
    runner = SOTCRunner(
        element='Al', ...,
        force_calculator=calc,
    )

Thread safety
-------------
Each snapshot calculation runs in its own uniquely-named subdirectory of
*workdir*, so multiple pw.x processes do not collide on the filesystem.
A threading.Semaphore controls how many pw.x processes run concurrently
(default: n_parallel=1, i.e. sequential, which is safe for a single-node
MPI pw.x that already uses all cores).  Set n_parallel > 1 only when pw.x
is configured to use fewer cores than available (e.g. to run several small
single-core QE jobs in parallel on many-core nodes).

A result cache ensures that mock_vasp_results() calling .compute() then
.energy() with the same positions does not run two separate pw.x jobs.

Unit conventions
----------------
ASE works in eV and Å.  QE works in Rydberg and Bohr.  The Espresso
calculator handles all conversions automatically.  Input parameters
ecutwfc, ecutrho, degauss, and conv_thr are passed in Rydberg as QE expects.
"""

from __future__ import annotations

import os
import threading
from pathlib import Path
from typing import Dict, List, Optional, Sequence, Tuple, Union

import numpy as np


class QEForceCalculator:
    """
    Subprocess-based Quantum ESPRESSO force calculator for SOTC.

    Parameters
    ----------
    species : list of str
        Chemical symbols for one formula unit (or the full supercell).
        For a monatomic system pass ``['Al']``; the list is tiled to match
        the number of atoms in each call.  For a compound use the primitive
        ordering, e.g. ``['Na', 'Cl']``.
    pseudopotentials : dict
        Mapping from element symbol to UPF filename, e.g.
        ``{'Al': 'Al.pbe-n-kjpaw_psl.1.0.0.UPF'}``.
    pseudo_dir : str or Path
        Directory that contains the UPF files.
    ecutwfc : float [Ry]
        Plane-wave kinetic energy cutoff.  Typical values (SSSP efficiency):
        Al≈40, Cu≈45, Au≈45, Fe≈90, Pb≈44.  1 Ry = 13.6057 eV.
    ecutrho : float or None [Ry]
        Charge-density cutoff.  If None, uses 8 × ecutwfc (suitable for PAW
        and USPP; for norm-conserving pseudos 4 × is usually sufficient but
        8 × is always safe).
    kpts : tuple of 3 int
        Monkhorst-Pack k-point grid for the *supercell*.  Scale down from
        the primitive-cell grid proportionally to the supercell dimensions,
        e.g. (2,2,2) for a 32-atom fcc supercell where (8,8,8) would be
        used for the 1-atom primitive cell.
    smearing : str
        Occupation smearing scheme for metallic systems.  QE options:
        ``'marzari-vanderbilt'`` (cold, recommended for metals),
        ``'methfessel-paxton'``, ``'fermi-dirac'``, ``'gaussian'``.
        For insulators set smearing=None and the input_data will omit it.
    degauss : float [Ry]
        Smearing width.  0.01–0.02 Ry is standard for metals; larger values
        speed convergence but reduce accuracy.  Ignored when smearing=None.
    conv_thr : float [Ry]
        SCF total energy convergence threshold (Ry).  1e-8 gives forces
        accurate to ~0.1 meV/Å, appropriate for IFC regression.
    pw_cmd : str or None
        Command used to run pw.x.  Examples:
        ``'pw.x'``, ``'/path/to/pw.x'``,
        ``'mpirun -np 8 pw.x'``,
        ``'srun -n 8 pw.x'``.
        If None, tries ``$QE_PW_CMD``, then searches PATH, then falls back to
        ``~/Softwares/qe-7.5/bin/pw.x``.
    workdir : str or Path
        Base directory for per-snapshot scratch subdirectories.  Created
        automatically.  Each snapshot is placed in ``workdir/snap_NNNN/``.
    prefix : str
        QE ``prefix`` used for temporary files inside each snapshot directory.
    n_parallel : int
        Maximum number of pw.x processes running simultaneously.  Default 1
        (sequential).  Safe to increase if pw.x is configured to use fewer
        cores than available on the node.
    extra_input_data : dict, optional
        Additional QE namelist overrides.  Nested dict, e.g.
        ``{'electrons': {'mixing_beta': 0.3}}``.  Keys in the top-level QE
        namelists (control, system, electrons, ions, cell) will be merged
        with (and override) the defaults computed from the other parameters.
    """

    def __init__(
        self,
        species: List[str],
        pseudopotentials: Dict[str, str],
        pseudo_dir: Union[str, Path],
        ecutwfc: float = 40.0,
        ecutrho: Optional[float] = None,
        kpts: Tuple[int, int, int] = (2, 2, 2),
        smearing: Optional[str] = "marzari-vanderbilt",
        degauss: float = 0.02,
        conv_thr: float = 1.0e-8,
        pw_cmd: Optional[str] = None,
        workdir: Union[str, Path] = "qe_scratch",
        prefix: str = "sqtc",
        n_parallel: int = 1,
        extra_input_data: Optional[Dict] = None,
    ):
        self.species = list(species)
        self.pseudopotentials = dict(pseudopotentials)
        self.pseudo_dir = Path(pseudo_dir).expanduser().resolve()
        self.ecutwfc = float(ecutwfc)
        self.ecutrho = float(ecutrho) if ecutrho is not None else 8.0 * self.ecutwfc
        self.kpts = tuple(kpts)
        self.smearing = smearing
        self.degauss = float(degauss)
        self.conv_thr = float(conv_thr)
        self.pw_cmd = pw_cmd or self._find_pw()
        self.workdir = Path(workdir).expanduser()
        self.prefix = prefix
        self.extra_input_data = extra_input_data or {}

        self.workdir.mkdir(parents=True, exist_ok=True)

        # Semaphore: limit concurrent pw.x processes.
        self._sem = threading.Semaphore(max(1, n_parallel))

        # Thread-safe snapshot counter.
        self._counter_lock = threading.Lock()
        self._n_calls: int = 0

        # Result cache: avoid two pw.x runs for the same positions.
        self._cache_lock = threading.Lock()
        self._cache_key: Optional[bytes] = None
        self._cache_forces: Optional[np.ndarray] = None
        self._cache_energy: Optional[float] = None

        # Validate pseudo_dir and pseudopotentials at construction time.
        self._validate_pseudos()

    # ── Setup helpers ─────────────────────────────────────────────────────────

    @staticmethod
    def _find_pw() -> str:
        """Locate pw.x: env var → PATH → known default location."""
        cmd = os.environ.get("QE_PW_CMD")
        if cmd:
            return cmd
        import shutil
        found = shutil.which("pw.x")
        if found:
            return found
        default = Path.home() / "Softwares" / "qe-7.5" / "bin" / "pw.x"
        if default.exists():
            return str(default)
        raise FileNotFoundError(
            "pw.x not found.  Set $QE_PW_CMD, add pw.x to $PATH, or pass "
            "pw_cmd='/path/to/pw.x' to QEForceCalculator."
        )

    def _validate_pseudos(self) -> None:
        """Warn early if a UPF file is missing."""
        missing = []
        for elem, fname in self.pseudopotentials.items():
            p = self.pseudo_dir / fname
            if not p.exists():
                missing.append(str(p))
        if missing:
            raise FileNotFoundError(
                "Pseudopotential file(s) not found:\n"
                + "\n".join(f"  {m}" for m in missing)
                + f"\nSet pseudo_dir to the directory containing the UPF files."
            )

    def _build_input_data(self) -> Dict:
        """Construct the QE input_data dict (namelist sections)."""
        system: Dict = {
            "ecutwfc": self.ecutwfc,
            "ecutrho": self.ecutrho,
        }
        if self.smearing is not None:
            system["occupations"] = "smearing"
            system["smearing"] = self.smearing
            system["degauss"] = self.degauss
        else:
            system["occupations"] = "fixed"

        data: Dict = {
            "control": {
                "calculation": "scf",
                "tstress": False,
                "tprnfor": True,   # forces are required
                "disk_io": "none", # avoid writing large restart files
            },
            "system": system,
            "electrons": {
                "conv_thr": self.conv_thr,
                "mixing_beta": 0.4,
                "electron_maxstep": 200,
            },
        }

        # Merge user overrides (deep-merge at the section level).
        for section, overrides in self.extra_input_data.items():
            if section in data:
                data[section].update(overrides)
            else:
                data[section] = dict(overrides)

        return data

    # ── Core calculation ──────────────────────────────────────────────────────

    def _make_atoms(self, positions: np.ndarray, cell: np.ndarray):
        from ase import Atoms

        n = len(positions)
        n_template = len(self.species)
        if n == n_template:
            symbols = self.species
        elif n % n_template == 0:
            symbols = self.species * (n // n_template)
        else:
            raise ValueError(
                f"QEForceCalculator: positions length ({n}) is not a multiple "
                f"of the species template length ({n_template}).  "
                f"Pass species for one formula unit, e.g. species=['Al'] or "
                f"species=['Na', 'Cl']."
            )

        return Atoms(symbols=symbols, positions=positions, cell=cell, pbc=True)

    def _make_calculator(self, snap_dir: Path):
        """Return a fresh ASE Espresso calculator for one snapshot."""
        try:
            from ase.calculators.espresso import Espresso, EspressoProfile
        except ImportError as exc:
            raise ImportError(
                "ASE is not installed or too old (need ≥ 3.23 for EspressoProfile). "
                "Install with: pip install ase"
            ) from exc

        snap_dir.mkdir(parents=True, exist_ok=True)

        profile = EspressoProfile(
            command=self.pw_cmd,
            pseudo_dir=str(self.pseudo_dir),
        )

        return Espresso(
            profile=profile,
            pseudopotentials=self.pseudopotentials,
            input_data=self._build_input_data(),
            kpts=self.kpts,
            koffset=(0, 0, 0),
            directory=str(snap_dir),
        )

    def _run_dft(
        self, positions: np.ndarray, cell: np.ndarray
    ) -> Tuple[np.ndarray, float]:
        """
        Run a single pw.x SCF calculation.

        Returns
        -------
        forces : (N, 3) ndarray [eV/Å]
        energy : float [eV]
        """
        with self._counter_lock:
            snap_n = self._n_calls
            self._n_calls += 1

        snap_dir = self.workdir / f"snap_{snap_n:04d}"
        atoms = self._make_atoms(positions, cell)
        atoms.calc = self._make_calculator(snap_dir)

        with self._sem:
            energy = float(atoms.get_potential_energy())
            forces = np.array(atoms.get_forces(), dtype=float)

        return forces, energy

    def _run_cached(
        self, positions: np.ndarray, cell: np.ndarray
    ) -> Tuple[np.ndarray, float]:
        """Return cached result or run a new DFT calculation."""
        key = positions.tobytes() + cell.tobytes()
        with self._cache_lock:
            if key == self._cache_key:
                assert self._cache_forces is not None
                return self._cache_forces, self._cache_energy  # type: ignore[return-value]

        forces, energy = self._run_dft(positions, cell)

        with self._cache_lock:
            self._cache_key = key
            self._cache_forces = forces
            self._cache_energy = energy

        return forces, energy

    # ── Public interface (matches mock_forces contract) ───────────────────────

    def compute(self, positions: np.ndarray, cell: np.ndarray) -> np.ndarray:
        """
        Compute atomic forces.

        Parameters
        ----------
        positions : (N, 3) ndarray [Å]
        cell      : (3, 3) ndarray [Å]  (rows = lattice vectors)

        Returns
        -------
        forces : (N, 3) ndarray [eV/Å]
        """
        forces, _ = self._run_cached(positions, cell)
        return forces

    def energy(self, positions: np.ndarray, cell: np.ndarray) -> float:
        """
        Compute total DFT energy.

        Parameters
        ----------
        positions : (N, 3) ndarray [Å]
        cell      : (3, 3) ndarray [Å]

        Returns
        -------
        energy : float [eV]
        """
        _, energy = self._run_cached(positions, cell)
        return energy

    def __repr__(self) -> str:
        return (
            f"QEForceCalculator("
            f"species={self.species}, "
            f"ecutwfc={self.ecutwfc:.0f} Ry, "
            f"kpts={self.kpts}, "
            f"n_calls={self._n_calls})"
        )
