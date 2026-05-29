"""
NexusMD — Complex Parser Service

Parses protein-ligand complex files (PDB, PDBQT, mmCIF) and validates
that they contain both protein and ligand atoms.

Public API
----------
parse_complex_file(file_path)  → dict with name, protein_atoms, ligand_atoms, pdb_text
validate_complex(pdb_text)     → True or raises ValueError
pdbqt_to_pdb(pdbqt_text)      → PDB-format string
"""

import logging
import re
from pathlib import Path
from typing import Dict, Any

logger = logging.getLogger("nexusmd.complex_parser")

# Standard amino-acid residue names (same set used in md_service)
_STANDARD_RESIDUES = frozenset({
    "ALA", "ARG", "ASN", "ASP", "CYS", "GLN", "GLU", "GLY", "HIS",
    "ILE", "LEU", "LYS", "MET", "PHE", "PRO", "SER", "THR", "TRP",
    "TYR", "VAL", "HID", "HIE", "HIP", "CYX", "ACE", "NME",
    # Common modified residues
    "MSE", "SEC", "PYL", "SEP", "TPO", "PTR",
})

# Water / ion residue names to ignore when counting ligand atoms
_SOLVENT_RESIDUES = frozenset({
    "HOH", "WAT", "TIP", "SOL", "NA", "CL", "MG", "ZN", "CA",
    "K", "FE", "MN", "CU", "NI", "CO", "CD", "HG",
})

# Supported file extensions
SUPPORTED_EXTENSIONS = {".pdb", ".pdbqt", ".cif", ".mmcif"}


# ── Public API ─────────────────────────────────────────────────────────────

def parse_complex_file(file_path: Path) -> Dict[str, Any]:
    """
    Read a PDB / PDBQT / mmCIF complex file and return a summary dict.

    Parameters
    ----------
    file_path : Path
        Path to the complex file.

    Returns
    -------
    dict
        {
          "name":          str   — filename stem,
          "protein_atoms": int   — number of protein heavy atoms,
          "ligand_atoms":  int   — number of ligand heavy atoms,
          "pdb_text":      str   — PDB-format text (converted if needed),
        }

    Raises
    ------
    ValueError
        If the file format is unsupported, the file cannot be parsed, or
        the complex fails validation.
    """
    suffix = file_path.suffix.lower()
    if suffix not in SUPPORTED_EXTENSIONS:
        raise ValueError(
            f"Unsupported file format '{suffix}'. "
            f"Supported formats: {', '.join(sorted(SUPPORTED_EXTENSIONS))}"
        )

    try:
        raw_text = file_path.read_text(encoding="utf-8", errors="replace")
    except Exception as exc:
        raise ValueError(f"Cannot read file '{file_path.name}': {exc}") from exc

    if not raw_text.strip():
        raise ValueError(f"File '{file_path.name}' is empty.")

    # Convert to PDB format
    if suffix == ".pdbqt":
        pdb_text = pdbqt_to_pdb(raw_text)
    elif suffix in (".cif", ".mmcif"):
        pdb_text = _cif_to_pdb(raw_text, file_path.name)
    else:
        pdb_text = raw_text

    # Validate and count atoms
    validate_complex(pdb_text)
    protein_atoms, ligand_atoms = _count_atoms(pdb_text)

    return {
        "name": file_path.stem,
        "protein_atoms": protein_atoms,
        "ligand_atoms": ligand_atoms,
        "pdb_text": pdb_text,
    }


def validate_complex(pdb_text: str) -> bool:
    """
    Validate that a PDB-format string contains both protein and ligand atoms.

    Rules
    -----
    - At least 10 ATOM/HETATM records total
    - At least one standard amino-acid residue (protein)
    - At least one non-standard, non-solvent residue (ligand)

    Returns True if valid, raises ValueError with a descriptive message if not.
    """
    protein_atoms, ligand_atoms = _count_atoms(pdb_text)
    total = protein_atoms + ligand_atoms

    if total < 10:
        raise ValueError(
            f"Complex contains only {total} atoms (minimum 10 required). "
            "Ensure the file is a complete protein-ligand complex."
        )
    if protein_atoms == 0:
        raise ValueError(
            "Complex must contain protein atoms. "
            "No standard amino-acid residues (ATOM records) were found."
        )
    if ligand_atoms == 0:
        raise ValueError(
            "Complex must contain ligand atoms. "
            "No non-standard residues (HETATM records) were found. "
            "Ensure the file contains both protein and ligand coordinates."
        )
    return True


def pdbqt_to_pdb(pdbqt_text: str) -> str:
    """
    Convert AutoDock PDBQT format to standard PDB format.

    PDBQT files have two extra columns after the standard PDB columns:
      - partial charge (float)
      - AutoDock atom type (string, e.g. 'C', 'HD', 'OA')

    This function strips those extra columns and returns valid PDB text.

    Parameters
    ----------
    pdbqt_text : str
        Raw PDBQT file content.

    Returns
    -------
    str
        PDB-format text.
    """
    output_lines = []
    for line in pdbqt_text.splitlines():
        record = line[:6].strip().upper() if len(line) >= 6 else ""

        if record in ("ATOM", "HETATM"):
            # Standard PDB columns are columns 1-66 (0-indexed: 0-65)
            # PDBQT adds charge and atom type after column 66
            # We keep the first 66 characters and optionally the element (cols 77-78)
            pdb_line = line[:66].rstrip()
            # Pad to 80 chars if needed for element symbol
            if len(line) > 66:
                # Try to extract element from PDBQT atom type (last token)
                tokens = line[66:].split()
                if tokens:
                    # Last token is the AutoDock atom type; first char is element
                    ad_type = tokens[-1] if len(tokens) >= 2 else tokens[0]
                    element = re.sub(r"[^A-Za-z]", "", ad_type)[:2].upper()
                    # Pad line to column 76 and add element in cols 77-78
                    pdb_line = pdb_line.ljust(76) + element.rjust(2)
            output_lines.append(pdb_line)

        elif record in ("BRANCH", "ENDBRANCH", "TORSDOF", "ROOT", "ENDROOT"):
            # AutoDock-specific records — skip
            continue

        elif record in ("REMARK", "MODEL", "ENDMDL", "END", "CONECT", "TER",
                        "HEADER", "TITLE", "COMPND", "SOURCE", "SEQRES",
                        "HELIX", "SHEET", "SSBOND", "LINK", "CRYST1",
                        "ORIGX1", "ORIGX2", "ORIGX3", "SCALE1", "SCALE2",
                        "SCALE3", "ANISOU"):
            output_lines.append(line)

        else:
            # Pass through unknown records
            output_lines.append(line)

    return "\n".join(output_lines) + "\n"


# ── Internal helpers ───────────────────────────────────────────────────────

def _count_atoms(pdb_text: str):
    """
    Count protein and ligand heavy atoms in PDB-format text.

    Returns (protein_atom_count, ligand_atom_count).
    """
    protein_atoms = 0
    ligand_atoms = 0

    for line in pdb_text.splitlines():
        record = line[:6].strip().upper() if len(line) >= 6 else ""

        if record == "ATOM":
            # Standard ATOM records are always protein
            protein_atoms += 1

        elif record == "HETATM":
            # HETATM: check residue name (cols 17-20, 0-indexed)
            if len(line) >= 20:
                res_name = line[17:20].strip().upper()
            else:
                res_name = ""

            if res_name in _SOLVENT_RESIDUES:
                continue  # skip water / ions
            if res_name in _STANDARD_RESIDUES:
                # Some files use HETATM for modified residues — count as protein
                protein_atoms += 1
            else:
                ligand_atoms += 1

    return protein_atoms, ligand_atoms


def _cif_to_pdb(cif_text: str, filename: str) -> str:
    """
    Convert mmCIF format to PDB format using pdbfixer (if available),
    falling back to a lightweight line-by-line parser.

    Parameters
    ----------
    cif_text : str
        Raw mmCIF file content.
    filename : str
        Original filename (used for error messages).

    Returns
    -------
    str
        PDB-format text.
    """
    # Try pdbfixer first (most reliable)
    try:
        import tempfile, os
        import io as _io
        from pdbfixer import PDBFixer
        import openmm.app as _app

        with tempfile.NamedTemporaryFile(
            suffix=".cif", delete=False, mode="w", encoding="utf-8"
        ) as tmp:
            tmp.write(cif_text)
            tmp_path = tmp.name

        try:
            fixer = PDBFixer(filename=tmp_path)
            out = _io.StringIO()
            _app.PDBFile.writeFile(fixer.topology, fixer.positions, out)
            return out.getvalue()
        finally:
            os.unlink(tmp_path)

    except ImportError:
        logger.debug("pdbfixer not available — using lightweight CIF parser")
    except Exception as exc:
        logger.warning(f"pdbfixer CIF conversion failed ({exc}) — trying lightweight parser")

    # Lightweight fallback: extract _atom_site loop and convert to PDB ATOM/HETATM
    return _cif_to_pdb_lightweight(cif_text, filename)


def _cif_to_pdb_lightweight(cif_text: str, filename: str) -> str:
    """
    Minimal mmCIF → PDB converter that handles the _atom_site loop.
    Sufficient for well-formed CIF files from the PDB.
    """
    lines = cif_text.splitlines()

    # Find the _atom_site loop
    in_loop = False
    headers: list = []
    atom_lines: list = []

    i = 0
    while i < len(lines):
        line = lines[i].strip()

        if line.lower() == "loop_":
            # Check if next non-empty lines are _atom_site fields
            j = i + 1
            candidate_headers = []
            while j < len(lines) and lines[j].strip().startswith("_atom_site."):
                candidate_headers.append(lines[j].strip().lower())
                j += 1
            if candidate_headers:
                in_loop = True
                headers = candidate_headers
                i = j
                # Collect data lines until next loop_ or # or _
                while i < len(lines):
                    dl = lines[i].strip()
                    if not dl or dl.startswith("#") or dl.startswith("_") or dl.lower() == "loop_":
                        break
                    atom_lines.append(dl)
                    i += 1
                break
        i += 1

    if not headers or not atom_lines:
        raise ValueError(
            f"Cannot parse mmCIF file '{filename}': no _atom_site loop found. "
            "Install pdbfixer for robust CIF support."
        )

    # Build column index map
    col = {h.replace("_atom_site.", ""): idx for idx, h in enumerate(headers)}

    required = {"group_pdb", "id", "type_symbol", "label_atom_id",
                "label_comp_id", "label_asym_id", "label_seq_id",
                "cartn_x", "cartn_y", "cartn_z"}
    missing = required - set(col.keys())
    if missing:
        raise ValueError(
            f"mmCIF file '{filename}' is missing required _atom_site columns: "
            f"{', '.join(sorted(missing))}. Install pdbfixer for robust CIF support."
        )

    pdb_lines = []
    for raw in atom_lines:
        # Simple tokeniser (does not handle quoted strings with spaces)
        tokens = raw.split()
        if len(tokens) < len(headers):
            continue

        try:
            record   = tokens[col["group_pdb"]].upper()          # ATOM / HETATM
            serial   = tokens[col["id"]][:5]
            atom_nm  = tokens[col["label_atom_id"]][:4]
            res_name = tokens[col["label_comp_id"]][:3].upper()
            chain    = tokens[col["label_asym_id"]][:1]
            res_seq  = tokens[col["label_seq_id"]][:4]
            x        = float(tokens[col["cartn_x"]])
            y        = float(tokens[col["cartn_y"]])
            z        = float(tokens[col["cartn_z"]])
            element  = tokens[col["type_symbol"]][:2].upper()

            # Occupancy and B-factor (optional)
            occ   = float(tokens[col["occupancy"]]) if "occupancy" in col else 1.0
            bfac  = float(tokens[col["b_iso_or_equiv"]]) if "b_iso_or_equiv" in col else 0.0

            # Format atom name: 4-char, left-padded for 1-char elements
            if len(atom_nm) < 4:
                atom_nm = f" {atom_nm:<3}" if len(element) == 1 else f"{atom_nm:<4}"

            pdb_line = (
                f"{record:<6}{int(serial):>5} {atom_nm:<4} {res_name:<3} "
                f"{chain}{int(res_seq):>4}    "
                f"{x:>8.3f}{y:>8.3f}{z:>8.3f}"
                f"{occ:>6.2f}{bfac:>6.2f}          "
                f"{element:>2}"
            )
            pdb_lines.append(pdb_line)
        except (ValueError, IndexError, KeyError):
            continue  # skip malformed lines

    if not pdb_lines:
        raise ValueError(
            f"No valid atom records could be extracted from '{filename}'. "
            "The file may be malformed or use an unsupported CIF dialect."
        )

    return "\n".join(pdb_lines) + "\nEND\n"
