"""
NexusMD — docked complex export.

Writes receptor + selected pose as a single PDB that Discovery Studio,
PyMOL, Chimera or MOE will open directly, with the ligand as HETATM/LIG
on its own chain so the visualiser recognises it as the ligand.

No new dependencies: pure text handling of the files Vina already wrote.

Install: save as app/services/complex_export.py
"""

import logging
from pathlib import Path
from typing import Optional, Tuple

logger = logging.getLogger("nexusmd.export")

# AutoDock atom type -> element. PDBQT carries AD types, not elements.
AD_TO_ELEMENT = {
    "A": "C", "C": "C", "N": "N", "NA": "N", "NS": "N",
    "O": "O", "OA": "O", "OS": "O", "S": "S", "SA": "S",
    "H": "H", "HD": "H", "HS": "H", "P": "P", "F": "F",
    "Cl": "CL", "CL": "CL", "Br": "BR", "BR": "BR", "I": "I",
    "Mg": "MG", "MG": "MG", "Ca": "CA", "CA": "CA", "Fe": "FE",
    "FE": "FE", "Zn": "ZN", "ZN": "ZN", "Mn": "MN", "MN": "MN",
}


def _element_from_pdbqt(line: str) -> str:
    """The AutoDock type is the last whitespace-separated token."""
    parts = line.rstrip().split()
    if parts:
        ad = parts[-1]
        return AD_TO_ELEMENT.get(ad, AD_TO_ELEMENT.get(ad.upper(), ad[:2].upper()))
    return "C"


def extract_pose(pdbqt_path: Path, mode: int = 1) -> Tuple[list, Optional[float]]:
    """Pull one MODEL out of a Vina output PDBQT.

    Returns (atom_lines, affinity). Modes are 1-indexed, as Vina numbers them.
    """
    if not pdbqt_path.exists():
        raise FileNotFoundError(f"pose file not found: {pdbqt_path.name}")

    current, atoms, affinity = 0, [], None
    found_affinity = None
    with pdbqt_path.open() as fh:
        for line in fh:
            if line.startswith("MODEL"):
                current += 1
                if current == mode:
                    atoms, found_affinity = [], None
            elif line.startswith("REMARK VINA RESULT") and current == mode:
                try:
                    found_affinity = float(line.split()[3])
                except (IndexError, ValueError):
                    pass
            elif line.startswith(("ATOM", "HETATM")) and current == mode:
                atoms.append(line.rstrip("\n"))
            elif line.startswith("ENDMDL") and current == mode:
                break

    # single-model file with no MODEL records at all
    if current == 0 and mode == 1:
        with pdbqt_path.open() as fh:
            for line in fh:
                if line.startswith("REMARK VINA RESULT"):
                    try:
                        found_affinity = float(line.split()[3])
                    except (IndexError, ValueError):
                        pass
                elif line.startswith(("ATOM", "HETATM")):
                    atoms.append(line.rstrip("\n"))

    if not atoms:
        raise ValueError(f"no atoms found for mode {mode} in {pdbqt_path.name}")
    return atoms, found_affinity


def build_complex(
    receptor_pdb: Path,
    pose_pdbqt: Path,
    mode: int = 1,
    ligand_name: str = "LIG",
    header_note: str = "",
) -> str:
    """Merge receptor and one docked pose into a single PDB string."""
    if not receptor_pdb.exists():
        raise FileNotFoundError(f"receptor not found: {receptor_pdb.name}")

    pose_atoms, affinity = extract_pose(pose_pdbqt, mode)

    out = [
        "REMARK   1 NexusMD docked complex",
        f"REMARK   1 LIGAND     {ligand_name[:60]}",
        f"REMARK   1 POSE MODE  {mode}",
    ]
    if affinity is not None:
        out.append(f"REMARK   1 AFFINITY   {affinity:.2f} kcal/mol (AutoDock Vina)")
    if header_note:
        out.append(f"REMARK   1 {header_note[:70]}")
    out.append("REMARK   1 Ligand is chain Z, residue LIG 1, as HETATM records.")

    serial = 0
    for line in receptor_pdb.read_text().splitlines():
        if line.startswith(("ATOM", "HETATM")):
            serial += 1
            out.append(f"{line[:6]}{serial:5d}{line[11:]}".rstrip())
        elif line.startswith("TER"):
            serial += 1
            out.append(f"TER   {serial:5d}")
    out.append(f"TER   {serial + 1:5d}")
    serial += 1

    # unique per-element atom names (C1, C2, O1 ...); duplicate names confuse
    # some visualisers into merging atoms
    counts: dict = {}
    for line in pose_atoms:
        serial += 1
        element = _element_from_pdbqt(line)
        counts[element] = counts.get(element, 0) + 1
        name = f"{element}{counts[element]}"[:4]
        x, y, z = line[30:38], line[38:46], line[46:54]
        out.append(
            f"HETATM{serial:5d} {name:<4} LIG Z   1    "
            f"{x}{y}{z}  1.00  0.00          {element:>2}"
        )

    out.append("END")
    return "\n".join(out) + "\n"


def write_complex(
    receptor_pdb: Path,
    pose_pdbqt: Path,
    destination: Path,
    mode: int = 1,
    ligand_name: str = "LIG",
) -> Path:
    destination.write_text(build_complex(receptor_pdb, pose_pdbqt, mode, ligand_name))
    return destination
