"""Quantum ESPRESSO DFPT driver for Born charges and LO--TO corrections."""

from __future__ import annotations

from dataclasses import dataclass
import hashlib
import json
from pathlib import Path
import re
import shlex
import subprocess
from typing import Sequence

import numpy as np


_FLOAT = r"[-+]?(?:\d+(?:\.\d*)?|\.\d+)(?:[EeDd][-+]?\d+)?"


def _numbers(line: str) -> list[float]:
    return [float(value.replace("D", "E").replace("d", "e"))
            for value in re.findall(_FLOAT, line)]


def parse_ph_dielectric_born(text: str, n_atoms: int) -> tuple[np.ndarray, np.ndarray]:
    """Parse epsilon-infinity and Born tensors from a QE ph.x output string."""
    lines = text.splitlines()
    dielectric_start = next(
        (i for i, line in enumerate(lines)
         if "Dielectric constant in cartesian axis" in line),
        None,
    )
    if dielectric_start is None:
        raise ValueError("ph.x output does not contain the dielectric tensor")
    dielectric_rows = []
    for line in lines[dielectric_start + 1:]:
        values = _numbers(line)
        if len(values) >= 3:
            dielectric_rows.append(values[-3:])
        if len(dielectric_rows) == 3:
            break
    if len(dielectric_rows) != 3:
        raise ValueError("Could not parse three dielectric-tensor rows")

    born_start = next(
        (i for i, line in enumerate(lines)
         if "Effective charges" in line and "cartesian" in line),
        None,
    )
    if born_start is None:
        raise ValueError("ph.x output does not contain Born effective charges")

    tensors: dict[int, list[list[float]]] = {}
    current_atom = None
    for line in lines[born_start + 1:]:
        atom_match = re.search(r"atom\s+(\d+)", line, flags=re.IGNORECASE)
        if atom_match:
            current_atom = int(atom_match.group(1)) - 1
            tensors.setdefault(current_atom, [])
            continue
        if current_atom is not None and re.search(r"\bE[xyz]\b", line, re.IGNORECASE):
            values = _numbers(line)
            if len(values) >= 3:
                tensors[current_atom].append(values[-3:])
                if len(tensors[current_atom]) == 3:
                    current_atom = None
        if len(tensors) == n_atoms and all(len(rows) == 3 for rows in tensors.values()):
            break

    missing = [atom for atom in range(n_atoms)
               if atom not in tensors or len(tensors[atom]) != 3]
    if missing:
        raise ValueError(f"Could not parse Born tensors for atom indices {missing}")
    dielectric = np.asarray(dielectric_rows, dtype=float)
    born = np.asarray([tensors[index] for index in range(n_atoms)], dtype=float)
    return dielectric, born


def parse_ph_gamma_frequencies(text: str) -> np.ndarray:
    """Parse the final Gamma-point frequencies in THz from ``ph.x`` output."""
    frequencies = []
    pattern = re.compile(
        rf"freq\s*\(\s*\d+\s*\)\s*=\s*({_FLOAT})\s*\[THz\]",
        flags=re.IGNORECASE,
    )
    for match in pattern.finditer(text):
        frequencies.append(
            float(match.group(1).replace("D", "E").replace("d", "e"))
        )
    if not frequencies:
        raise ValueError("ph.x output does not contain Gamma frequencies in THz")
    return np.asarray(frequencies, dtype=float)


@dataclass
class DFPTNACResult:
    """Validated DFPT data in the convention used by PhononCalculator."""

    dielectric_raw: np.ndarray
    dielectric: np.ndarray
    born_raw: np.ndarray
    born: np.ndarray
    charge_neutrality_residual_raw: np.ndarray
    charge_neutrality_residual_corrected: np.ndarray
    scf_output: Path
    ph_output: Path

    def save(self, directory: str | Path, provenance: dict) -> None:
        target = Path(directory)
        target.mkdir(parents=True, exist_ok=True)
        np.savez_compressed(
            target / "nac_tensors.npz",
            dielectric_raw=self.dielectric_raw,
            dielectric=self.dielectric,
            born_raw=self.born_raw,
            born=self.born,
            charge_neutrality_residual_raw=self.charge_neutrality_residual_raw,
            charge_neutrality_residual_corrected=self.charge_neutrality_residual_corrected,
        )
        serializable = dict(provenance)
        serializable.update({
            "dielectric_eigenvalues": np.linalg.eigvalsh(self.dielectric).tolist(),
            "born_charge_neutrality_max_abs_raw": float(
                np.max(np.abs(self.charge_neutrality_residual_raw))
            ),
            "born_charge_neutrality_max_abs_corrected": float(
                np.max(np.abs(self.charge_neutrality_residual_corrected))
            ),
            "born_convention": (
                "rows are electric-field/polarization direction and columns are "
                "force/displacement direction; NAC uses q @ Z*"
            ),
        })
        (target / "nac_metadata.json").write_text(json.dumps(serializable, indent=2))


class QEDFPTNAC:
    """Run a persistent primitive-cell SCF followed by Gamma-point ph.x DFPT."""

    def __init__(
        self,
        species: Sequence[str],
        pseudopotentials: dict[str, str],
        pseudo_dir: str | Path,
        ecutwfc: float,
        ecutrho: float,
        kpts: tuple[int, int, int],
        pw_cmd: str,
        ph_cmd: str,
        workdir: str | Path,
        prefix: str,
        smearing: str | None = None,
        degauss: float = 0.0,
        conv_thr: float = 1.0e-10,
        tr2_ph: float = 1.0e-14,
        extra_input_data: dict | None = None,
    ):
        self.species = list(species)
        self.pseudopotentials = dict(pseudopotentials)
        self.pseudo_dir = Path(pseudo_dir).expanduser().resolve()
        self.ecutwfc = float(ecutwfc)
        self.ecutrho = float(ecutrho)
        self.kpts = tuple(int(v) for v in kpts)
        self.pw_cmd = str(pw_cmd)
        self.ph_cmd = str(ph_cmd)
        self.workdir = Path(workdir).expanduser().resolve()
        self.prefix = str(prefix)
        self.smearing = smearing
        self.degauss = float(degauss)
        self.conv_thr = float(conv_thr)
        self.tr2_ph = float(tr2_ph)
        self.extra_input_data = extra_input_data or {}
        self.workdir.mkdir(parents=True, exist_ok=True)
        for element, filename in self.pseudopotentials.items():
            if not (self.pseudo_dir / filename).is_file():
                raise FileNotFoundError(self.pseudo_dir / filename)

    def _symbols(self, n_atoms: int) -> list[str]:
        if n_atoms == len(self.species):
            return self.species
        if n_atoms % len(self.species):
            raise ValueError("Atom count is not a multiple of the species template")
        return self.species * (n_atoms // len(self.species))

    def _write_scf_input(self, positions: np.ndarray, cell: np.ndarray, path: Path) -> None:
        from ase import Atoms
        from ase.io.espresso import write_espresso_in

        system = {"ecutwfc": self.ecutwfc, "ecutrho": self.ecutrho}
        if self.smearing is None:
            system["occupations"] = "fixed"
        else:
            system.update({
                "occupations": "smearing",
                "smearing": self.smearing,
                "degauss": self.degauss,
            })
        data = {
            "control": {
                "calculation": "scf",
                "prefix": self.prefix,
                "pseudo_dir": str(self.pseudo_dir),
                "outdir": str(self.workdir / "scratch"),
                "disk_io": "low",
                "tprnfor": True,
            },
            "system": system,
            "electrons": {
                "conv_thr": self.conv_thr,
                "mixing_beta": 0.4,
                "electron_maxstep": 300,
            },
        }
        for section, settings in self.extra_input_data.items():
            data.setdefault(section, {}).update(settings)
        atoms = Atoms(
            symbols=self._symbols(len(positions)),
            positions=np.asarray(positions, dtype=float),
            cell=np.asarray(cell, dtype=float),
            pbc=True,
        )
        with path.open("w") as handle:
            write_espresso_in(
                handle, atoms, input_data=data,
                pseudopotentials=self.pseudopotentials,
                kpts=self.kpts, koffset=(0, 0, 0),
            )

    def _write_ph_input(self, path: Path) -> None:
        path.write_text(
            "&inputph\n"
            f"  tr2_ph = {self.tr2_ph:.16g}\n"
            f"  prefix = '{self.prefix}'\n"
            f"  outdir = '{self.workdir / 'scratch'}'\n"
            f"  fildyn = '{self.workdir / (self.prefix + '.dynG')}'\n"
            "  epsil = .true.\n"
            "  trans = .true.\n"
            "  ldisp = .false.\n"
            "/\n"
            "0.0 0.0 0.0\n"
        )

    @staticmethod
    def _run(command: str, input_path: Path, output_path: Path, cwd: Path) -> None:
        with input_path.open("rb") as stdin, output_path.open("wb") as stdout:
            completed = subprocess.run(
                shlex.split(command), stdin=stdin, stdout=stdout,
                stderr=subprocess.STDOUT, cwd=cwd, check=False,
            )
        if completed.returncode != 0:
            tail = "\n".join(output_path.read_text(errors="replace").splitlines()[-40:])
            raise RuntimeError(
                f"Command failed with exit code {completed.returncode}: {command}\n{tail}"
            )

    def run(
        self,
        positions: np.ndarray,
        cell: np.ndarray,
        enforce_charge_neutrality: bool = True,
        reuse_existing: bool = False,
    ) -> DFPTNACResult:
        scf_input = self.workdir / "scf.in"
        scf_output = self.workdir / "scf.out"
        ph_input = self.workdir / "ph.in"
        ph_output = self.workdir / "ph.out"
        (self.workdir / "scratch").mkdir(parents=True, exist_ok=True)
        self._write_scf_input(positions, cell, scf_input)
        self._write_ph_input(ph_input)
        if reuse_existing:
            if not scf_output.is_file() or not ph_output.is_file():
                raise FileNotFoundError(
                    "reuse_existing=True requires existing scf.out and ph.out"
                )
        else:
            self._run(self.pw_cmd, scf_input, scf_output, self.workdir)
            self._run(self.ph_cmd, ph_input, ph_output, self.workdir)

        dielectric_raw, born_raw = parse_ph_dielectric_born(
            ph_output.read_text(errors="replace"), len(positions)
        )
        dielectric = 0.5 * (dielectric_raw + dielectric_raw.T)
        if np.min(np.linalg.eigvalsh(dielectric)) <= 0.0:
            raise ValueError("DFPT electronic dielectric tensor is not positive definite")
        residual_raw = np.sum(born_raw, axis=0)
        born = born_raw.copy()
        if enforce_charge_neutrality:
            born -= residual_raw[None, :, :] / len(born)
        residual_corrected = np.sum(born, axis=0)

        result = DFPTNACResult(
            dielectric_raw=dielectric_raw,
            dielectric=dielectric,
            born_raw=born_raw,
            born=born,
            charge_neutrality_residual_raw=residual_raw,
            charge_neutrality_residual_corrected=residual_corrected,
            scf_output=scf_output,
            ph_output=ph_output,
        )
        pseudo_hashes = {
            element: {
                "filename": filename,
                "sha256": hashlib.sha256(
                    (self.pseudo_dir / filename).read_bytes()
                ).hexdigest(),
            }
            for element, filename in self.pseudopotentials.items()
        }
        result.save(self.workdir, {
            "backend": "Quantum ESPRESSO pw.x + ph.x DFPT",
            "pw_cmd": self.pw_cmd,
            "ph_cmd": self.ph_cmd,
            "prefix": self.prefix,
            "ecutwfc_Ry": self.ecutwfc,
            "ecutrho_Ry": self.ecutrho,
            "kpts": list(self.kpts),
            "conv_thr_Ry": self.conv_thr,
            "tr2_ph": self.tr2_ph,
            "enforce_charge_neutrality": enforce_charge_neutrality,
            "reused_existing_outputs": bool(reuse_existing),
            "pseudopotentials": pseudo_hashes,
        })
        return result
