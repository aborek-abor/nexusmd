"""NexusMD — Protein File Parser & AlphaFold Integration

Provides:
  parse_pdb_file(file_path)         — parse a local PDB file
  parse_cif_file(file_path)         — parse a local mmCIF file
  fetch_alphafold_structure(uid)    — download & parse an AlphaFold structure
  search_alphafold(query)           — search UniProt for matching proteins
"""

import logging
import re
import time
from pathlib import Path
from typing import List, Optional

import httpx

logger = logging.getLogger("nexusmd.protein_parser")

PDB_CACHE_DIR = Path(__file__).parent.parent.parent / "data" / "pdb_cache"
PDB_CACHE_DIR.mkdir(parents=True, exist_ok=True)

AF_FILE_BASE = "https://alphafold.ebi.ac.uk/files"
UNIPROT_SEARCH = "https://rest.uniprot.org/uniprotkb/search"


# ── PDB parser ─────────────────────────────────────────────────

def parse_pdb_file(file_path: Path) -> dict:
    """Parse a PDB file and return a summary dict.

    Returns::

        {
            "name":     str,   # COMPND title or filename stem
            "chains":   int,   # number of unique chain IDs
            "residues": int,   # number of unique (chain, resseq) pairs
            "pdb_text": str,   # full file content
        }

    Raises:
        FileNotFoundError – file does not exist
        ValueError        – file has no ATOM/HETATM records
    """
    file_path = Path(file_path)
    if not file_path.exists():
        raise FileNotFoundError(f"PDB file not found: {file_path}")

    pdb_text = file_path.read_text(encoding="utf-8", errors="replace")
    if not pdb_text.strip():
        raise ValueError("PDB file is empty")

    # Validate: must have at least one ATOM record
    atom_lines = [l for l in pdb_text.splitlines()
                  if l.startswith("ATOM") or l.startswith("HETATM")]
    if not atom_lines:
        raise ValueError("PDB file contains no ATOM or HETATM records — not a valid structure file")

    # Extract protein name from COMPND or TITLE records
    name = _extract_pdb_name(pdb_text, file_path.stem)

    # Count unique chains and residues
    chains: set = set()
    residues: set = set()
    for line in atom_lines:
        if len(line) >= 26:
            chain_id = line[21].strip()
            res_seq  = line[22:26].strip()
            if chain_id:
                chains.add(chain_id)
            if chain_id and res_seq:
                residues.add((chain_id, res_seq))

    logger.info(
        f"Parsed PDB {file_path.name}: name={name!r}, "
        f"chains={len(chains)}, residues={len(residues)}"
    )
    return {
        "name":     name,
        "chains":   len(chains),
        "residues": len(residues),
        "pdb_text": pdb_text,
    }


def _extract_pdb_name(pdb_text: str, fallback: str) -> str:
    """Extract a human-readable name from PDB header records."""
    # Try COMPND MOLECULE field first
    compnd_match = re.search(
        r"^COMPND\s+(?:\d+\s+)?MOLECULE:\s*(.+?)(?:;|$)",
        pdb_text,
        re.MULTILINE | re.IGNORECASE,
    )
    if compnd_match:
        candidate = compnd_match.group(1).strip().rstrip(";").strip()
        if candidate and candidate.upper() not in ("NULL", "NONE", ""):
            return candidate

    # Try TITLE record
    title_lines = [
        l[10:].strip()
        for l in pdb_text.splitlines()
        if l.startswith("TITLE")
    ]
    if title_lines:
        title = " ".join(title_lines).strip()
        if title:
            return title

    return fallback.upper()


# ── mmCIF parser ───────────────────────────────────────────────

def parse_cif_file(file_path: Path) -> dict:
    """Parse an mmCIF file and return a summary dict.

    Attempts to use BioPython's MMCIF2Dict for robust parsing.
    Falls back to a lightweight regex-based approach if BioPython
    is not installed.

    Returns the same shape as :func:`parse_pdb_file`.

    Raises:
        FileNotFoundError – file does not exist
        ValueError        – file has no coordinate data
    """
    file_path = Path(file_path)
    if not file_path.exists():
        raise FileNotFoundError(f"mmCIF file not found: {file_path}")

    cif_text = file_path.read_text(encoding="utf-8", errors="replace")
    if not cif_text.strip():
        raise ValueError("mmCIF file is empty")

    # Validate: must contain _atom_site data
    if "_atom_site" not in cif_text:
        raise ValueError(
            "mmCIF file contains no _atom_site records — not a valid structure file"
        )

    name = _extract_cif_name(cif_text, file_path.stem)
    chains, residues = _extract_cif_chains_residues(cif_text)

    logger.info(
        f"Parsed CIF {file_path.name}: name={name!r}, "
        f"chains={chains}, residues={residues}"
    )
    return {
        "name":     name,
        "chains":   chains,
        "residues": residues,
        "pdb_text": cif_text,   # kept as CIF; callers that need PDB can convert
    }


def _extract_cif_name(cif_text: str, fallback: str) -> str:
    """Extract protein name from mmCIF _struct.title or _entry.id."""
    # _struct.title
    title_match = re.search(
        r"_struct\.title\s+['\"]?(.+?)['\"]?\s*\n",
        cif_text,
        re.IGNORECASE,
    )
    if title_match:
        candidate = title_match.group(1).strip().strip("'\"")
        if candidate and candidate not in (".", "?"):
            return candidate

    # _entry.id
    entry_match = re.search(r"_entry\.id\s+(\S+)", cif_text, re.IGNORECASE)
    if entry_match:
        return entry_match.group(1).strip().strip("'\"").upper()

    return fallback.upper()


def _extract_cif_chains_residues(cif_text: str):
    """Count unique chains and residues from _atom_site loop in mmCIF."""
    # Try BioPython first for accuracy
    try:
        from Bio.PDB.MMCIF2Dict import MMCIF2Dict  # type: ignore
        import tempfile, os

        with tempfile.NamedTemporaryFile(
            mode="w", suffix=".cif", delete=False, encoding="utf-8"
        ) as tmp:
            tmp.write(cif_text)
            tmp_path = tmp.name

        try:
            mmcif_dict = MMCIF2Dict(tmp_path)
            chain_ids = mmcif_dict.get("_atom_site.auth_asym_id") or \
                        mmcif_dict.get("_atom_site.label_asym_id") or []
            seq_ids   = mmcif_dict.get("_atom_site.auth_seq_id") or \
                        mmcif_dict.get("_atom_site.label_seq_id") or []
            chains    = set(chain_ids)
            residues  = set(zip(chain_ids, seq_ids))
            return len(chains), len(residues)
        finally:
            os.unlink(tmp_path)

    except ImportError:
        logger.debug("BioPython not available — using regex CIF parser")
    except Exception as exc:
        logger.debug(f"BioPython CIF parse failed: {exc} — falling back to regex")

    # Lightweight regex fallback
    chains: set = set()
    residues: set = set()

    # Find column indices from the loop_ header
    col_chain = col_seq = None
    in_atom_loop = False
    col_index = 0
    header_cols: list = []

    for line in cif_text.splitlines():
        stripped = line.strip()
        if stripped == "loop_":
            in_atom_loop = False
            header_cols = []
            col_index = 0
            continue
        if stripped.startswith("_atom_site."):
            in_atom_loop = True
            header_cols.append(stripped)
            col_index += 1
            continue
        if in_atom_loop and stripped.startswith("_"):
            in_atom_loop = False
            continue
        if in_atom_loop and stripped and not stripped.startswith("#"):
            if col_chain is None:
                # Resolve column indices once
                for i, col in enumerate(header_cols):
                    if col in ("_atom_site.auth_asym_id", "_atom_site.label_asym_id"):
                        col_chain = i
                    if col in ("_atom_site.auth_seq_id", "_atom_site.label_seq_id"):
                        col_seq = i
            if col_chain is not None:
                parts = stripped.split()
                if len(parts) > col_chain:
                    chain = parts[col_chain]
                    chains.add(chain)
                    if col_seq is not None and len(parts) > col_seq:
                        residues.add((chain, parts[col_seq]))

    return len(chains) or 1, len(residues) or 0


# ── AlphaFold structure fetch ──────────────────────────────────

async def fetch_alphafold_structure(uniprot_id: str) -> dict:
    """Download an AlphaFold predicted structure and return a summary dict.

    Tries model versions v4 → v3 → v2.  Caches the PDB file locally in
    ``data/pdb_cache/AF_{uniprot_id}.pdb``.

    Returns::

        {
            "name":     str,
            "pdb_text": str,
            "plddt":    float | None,   # mean pLDDT from B-factor column
            "source":   "alphafold",
        }

    Raises:
        ValueError – structure not found on AlphaFold DB
    """
    uniprot_id = uniprot_id.upper().strip()
    cached = PDB_CACHE_DIR / f"AF_{uniprot_id}.pdb"

    pdb_text: Optional[str] = None

    if cached.exists() and cached.stat().st_size > 1000:
        logger.info(f"AlphaFold cache hit: {uniprot_id}")
        pdb_text = cached.read_text(encoding="utf-8", errors="replace")
    else:
        async with httpx.AsyncClient(timeout=30, follow_redirects=True) as client:
            for ver in [4, 3, 2]:
                url = f"{AF_FILE_BASE}/AF-{uniprot_id}-F1-model_v{ver}.pdb"
                try:
                    r = await client.get(url)
                    if r.status_code == 200:
                        cached.write_bytes(r.content)
                        pdb_text = r.text
                        logger.info(f"AlphaFold PDB downloaded: {uniprot_id} v{ver}")
                        break
                    logger.debug(f"AF v{ver} returned {r.status_code} for {uniprot_id}")
                except Exception as exc:
                    logger.debug(f"AF download attempt failed (v{ver}): {exc}")

    if pdb_text is None:
        raise ValueError(
            f"AlphaFold structure not found for UniProt ID '{uniprot_id}'. "
            "Verify the ID is correct and the protein has an AlphaFold prediction."
        )

    # Extract name from PDB header
    name = _extract_pdb_name(pdb_text, f"AF-{uniprot_id}")

    # Compute mean pLDDT from B-factor column of ATOM records
    plddt = _extract_mean_plddt(pdb_text)

    return {
        "name":     name,
        "pdb_text": pdb_text,
        "plddt":    plddt,
        "source":   "alphafold",
    }


def _extract_mean_plddt(pdb_text: str) -> Optional[float]:
    """Compute mean pLDDT score from the B-factor column of ATOM records.

    In AlphaFold PDB files the per-residue pLDDT is stored in the
    B-factor (temperature factor) column (columns 61-66).
    """
    scores: list = []
    for line in pdb_text.splitlines():
        if not line.startswith("ATOM"):
            continue
        try:
            bfactor = float(line[60:66].strip())
            scores.append(bfactor)
        except (ValueError, IndexError):
            continue
    if not scores:
        return None
    return round(sum(scores) / len(scores), 2)


# ── UniProt / AlphaFold search ─────────────────────────────────

async def search_alphafold(query: str) -> List[dict]:
    """Search UniProt for proteins matching *query* (gene name, protein name, etc.).

    Returns up to 5 results, each with::

        {
            "uniprot_id":   str,
            "gene_name":    str,
            "protein_name": str,
            "organism":     str,
        }

    Returns an empty list if no results are found or the API is unreachable.
    """
    query = query.strip()
    if not query:
        return []

    params = {
        "query":  query,
        "format": "json",
        "size":   "5",
        "fields": "accession,gene_names,protein_name,organism_name",
    }

    try:
        async with httpx.AsyncClient(timeout=15) as client:
            r = await client.get(UNIPROT_SEARCH, params=params)
            if r.status_code != 200:
                logger.warning(f"UniProt search returned {r.status_code} for query={query!r}")
                return []
            data = r.json()
    except Exception as exc:
        logger.error(f"UniProt search failed for query={query!r}: {exc}")
        return []

    results: List[dict] = []
    for entry in data.get("results", []):
        uniprot_id = entry.get("primaryAccession", "")
        if not uniprot_id:
            continue

        # Gene name
        gene_names = entry.get("genes", [])
        gene_name = ""
        if gene_names:
            gn = gene_names[0]
            gene_name = (
                gn.get("geneName", {}).get("value", "")
                or (gn.get("synonyms") or [{}])[0].get("value", "")
            )

        # Protein name
        pn_block = entry.get("proteinDescription", {})
        rec_name = pn_block.get("recommendedName", {})
        protein_name = (
            rec_name.get("fullName", {}).get("value", "")
            or (pn_block.get("submissionNames") or [{}])[0]
               .get("fullName", {}).get("value", "")
        )

        # Organism
        organism = entry.get("organism", {}).get("scientificName", "")

        results.append({
            "uniprot_id":   uniprot_id,
            "gene_name":    gene_name,
            "protein_name": protein_name,
            "organism":     organism,
        })

    logger.info(f"UniProt search for {query!r}: {len(results)} results")
    return results
