"""
NexusMD — Protein Fetch Service
Real RCSB PDB REST API + AlphaFold EBI API + coordinate download
"""

import logging
from pathlib import Path
from typing import Optional

import httpx

logger = logging.getLogger("nexusmd.protein")

PDB_CACHE_DIR = Path(__file__).parent.parent.parent / "data" / "pdb_cache"
PDB_CACHE_DIR.mkdir(parents=True, exist_ok=True)

RCSB_REST   = "https://data.rcsb.org/rest/v1/core/entry"
RCSB_FILE   = "https://files.rcsb.org/download"
AF_API      = "https://alphafold.ebi.ac.uk/api/prediction"
AF_FILE_BASE = "https://alphafold.ebi.ac.uk/files"


# ── RCSB PDB ──────────────────────────────────────
async def fetch_pdb_info(pdb_id: str) -> Optional[dict]:
    """Fetch metadata from RCSB REST API."""
    pdb_id = pdb_id.upper().strip()
    async with httpx.AsyncClient(timeout=15) as client:
        try:
            r = await client.get(f"{RCSB_REST}/{pdb_id}")
            if r.status_code != 200:
                return None
            d = r.json()

            # Extract useful fields
            title = d.get("struct", {}).get("title", "—")
            method = (d.get("exptl") or [{}])[0].get("method", "—")
            resolution = None
            refine = d.get("refine") or []
            if refine:
                resolution = refine[0].get("ls_d_res_high")

            chains = [e.get("asym_id") for e in (d.get("entity_poly") or [])]
            
            # Ligands from non-polymer entities
            ligands = []
            for e in (d.get("chem_comp") or []):
                if e.get("type") not in ("L-peptide linking", "peptide linking", "RNA linking", "DNA linking"):
                    ligands.append(e.get("id", ""))

            organism = None
            source = (d.get("rcsb_entity_source_organism") or [{}])
            if source:
                organism = source[0].get("scientific_name")

            return {
                "pdb_id": pdb_id,
                "title": title,
                "method": method,
                "resolution_a": resolution,
                "organism": organism,
                "chain_ids": [c for c in chains if c],
                "ligands": [l for l in ligands if l],
                "pdb_url": f"{RCSB_FILE}/{pdb_id}.pdb",
                "source": "RCSB",
            }
        except Exception as e:
            logger.error(f"RCSB fetch failed for {pdb_id}: {e}")
            return None


async def download_pdb_file(pdb_id: str) -> Optional[Path]:
    """Download full PDB coordinate file and cache it."""
    pdb_id = pdb_id.upper().strip()
    cached = PDB_CACHE_DIR / f"{pdb_id}.pdb"
    if cached.exists() and cached.stat().st_size > 1000:
        logger.info(f"PDB cache hit: {pdb_id}")
        return cached

    async with httpx.AsyncClient(timeout=30, follow_redirects=True) as client:
        for url in [
            f"{RCSB_FILE}/{pdb_id}.pdb",
            f"https://files.rcsb.org/pub/pdb/data/structures/divided/pdb/{pdb_id[1:3].lower()}/pdb{pdb_id.lower()}.ent.gz",
        ]:
            try:
                r = await client.get(url)
                if r.status_code == 200:
                    content = r.content
                    # Decompress if gzipped
                    if url.endswith(".gz"):
                        import gzip
                        content = gzip.decompress(content)
                    cached.write_bytes(content)
                    logger.info(f"PDB downloaded: {pdb_id} ({len(content)} bytes)")
                    return cached
            except Exception as e:
                logger.debug(f"PDB download attempt failed ({url}): {e}")

    return None


async def fetch_pdb_file_content(pdb_id: str) -> Optional[str]:
    """Return PDB file as string (downloads if needed)."""
    path = await download_pdb_file(pdb_id)
    if path and path.exists():
        return path.read_text(errors="replace")
    return None


# ── AlphaFold EBI ─────────────────────────────────
async def fetch_alphafold_info(uniprot_id: str) -> Optional[dict]:
    """Fetch AlphaFold prediction metadata from EBI API."""
    uniprot_id = uniprot_id.upper().strip()
    async with httpx.AsyncClient(timeout=20) as client:
        try:
            r = await client.get(f"{AF_API}/{uniprot_id}")
            if r.status_code != 200:
                return None
            data = r.json()
            if not data:
                return None
            entry = data[0]
            uniprot_desc = entry.get("uniprotDescription") or entry.get("proteinFullName") or "—"
            plddt = entry.get("globalMetricValue")
            pdb_url = entry.get("pdbUrl") or f"{AF_FILE_BASE}/AF-{uniprot_id}-F1-model_v6.pdb"
            return {
                "uniprot_id": uniprot_id,
                "description": uniprot_desc,
                "mean_plddt": round(plddt, 2) if plddt else None,
                "pdb_url": pdb_url,
                "source": "AlphaFold",
            }
        except Exception as e:
            logger.error(f"AlphaFold fetch failed for {uniprot_id}: {e}")
            return None


async def download_alphafold_pdb(uniprot_id: str) -> Optional[Path]:
    """Download an AlphaFold predicted structure, resolving the URL properly."""
    uniprot_id = uniprot_id.upper().strip()
    cached = PDB_CACHE_DIR / f"AF_{uniprot_id}.pdb"
    if cached.exists() and cached.stat().st_size > 1000:
        return cached

    urls = []
    async with httpx.AsyncClient(timeout=30, follow_redirects=True) as client:
        # 1. ask the API where this model actually lives
        try:
            r = await client.get(f"{AF_API}/{uniprot_id}")
            if r.status_code == 200:
                data = r.json() or []
                entries = [e for e in (data if isinstance(data, list) else [data])
                           if isinstance(e, dict)]
                # exact accession first, isoforms after
                entries.sort(
                    key=lambda e: (e.get("uniprotAccession", "").upper() != uniprot_id)
                )
                for entry in entries:
                    if entry.get("pdbUrl"):
                        urls.append(entry["pdbUrl"])
            else:
                logger.warning("AlphaFold API %s -> HTTP %s", uniprot_id, r.status_code)
        except Exception as e:
            logger.error("AlphaFold API failed for %s: %s", uniprot_id, e)

        # 2. guessed filenames as a fallback, newest version first
        urls += [f"{AF_FILE_BASE}/AF-{uniprot_id}-F1-model_v{v}.pdb" for v in (6, 5, 4)]

        for url in urls:
            try:
                r = await client.get(url)
                if r.status_code == 200 and b"ATOM" in r.content:
                    cached.write_bytes(r.content)
                    logger.info("AlphaFold PDB for %s from %s", uniprot_id, url)
                    return cached
                logger.debug("AF %s -> HTTP %s", url, r.status_code)
            except Exception as e:
                logger.debug("AF download attempt failed: %s", e)

    logger.error("no AlphaFold structure available for %s", uniprot_id)
    return None

async def fetch_alphafold_pdb_content(uniprot_id: str) -> Optional[str]:
    """Return AlphaFold PDB as string."""
    path = await download_alphafold_pdb(uniprot_id)
    if path and path.exists():
        return path.read_text(errors="replace")
    return None
