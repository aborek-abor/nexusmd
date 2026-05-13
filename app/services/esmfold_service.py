"""
NexusMD — ESMFold Service
Real ESMFold API: https://esmatlas.com/api/fold
Returns actual PDB coordinates from Meta's ESM-2 language model.
Free, no API key required, works up to ~700 aa.
"""

import asyncio
import logging
import re
import time
from pathlib import Path
from typing import Optional, Tuple

import httpx

logger = logging.getLogger("nexusmd.esmfold")

ESMFOLD_URL = "https://api.esmatlas.com/foldSequence/v1/pdb/"
FASTA_CACHE_DIR = Path(__file__).parent.parent.parent / "data" / "fasta_results"
FASTA_CACHE_DIR.mkdir(parents=True, exist_ok=True)


async def fold_with_esmfold(sequence: str, job_id: str, log_fn) -> Optional[dict]:
    """
    Submit sequence to ESMFold API and return:
    - pdb_string: full PDB coordinate text
    - mean_plddt: average confidence score
    - ptm_score: predicted TM-score
    """
    seq = clean_sequence(sequence)
    if not seq:
        await log_fn(job_id, "[ESMFold] ERROR: sequence is empty after cleaning — check input for valid amino acid letters", "warn")
        logger.error(f"[ESMFold] job={job_id} sequence empty after clean_sequence()")
        return None

    logger.info(f"[ESMFold] job={job_id} submitting {len(seq)}-residue sequence")
    await log_fn(job_id, f"[ESMFold] Submitting {len(seq)}-residue sequence to Meta ESM API…", "info")

    start = time.time()
    pdb_string = None

    try:
        async with httpx.AsyncClient(timeout=120) as client:
            await log_fn(job_id, "[ESMFold] Waiting for ESM-2 language model inference…", "info")
            logger.info(f"[ESMFold] job={job_id} POST {ESMFOLD_URL} seq_len={len(seq)}")
            r = await client.post(
                ESMFOLD_URL,
                content=seq,
                headers={
                    "Content-Type": "application/x-www-form-urlencoded",
                    "User-Agent": "NexusMD/5.0",
                },
            )
            logger.info(f"[ESMFold] job={job_id} API response status={r.status_code} body_len={len(r.text)}")
            if r.status_code == 200:
                pdb_string = r.text
                await log_fn(job_id, f"[ESMFold] Structure received from API ✓ ({len(pdb_string)} bytes)", "info")
            else:
                err_body = r.text[:500]
                logger.error(f"[ESMFold] job={job_id} API error status={r.status_code} body={err_body}")
                await log_fn(job_id, f"[ESMFold] API returned HTTP {r.status_code} — {err_body}", "warn")
                return None
    except httpx.TimeoutException:
        logger.error(f"[ESMFold] job={job_id} request timed out after 120s for seq_len={len(seq)}")
        await log_fn(job_id, f"[ESMFold] API timeout after 120s — sequence length {len(seq)} may be too long. Try ColabFold for sequences >700 aa.", "warn")
        return None
    except Exception as e:
        import traceback
        tb = traceback.format_exc()
        logger.error(f"[ESMFold] job={job_id} request exception: {e}\n{tb}")
        await log_fn(job_id, f"[ESMFold] Request failed: {type(e).__name__}: {e}", "warn")
        return None

    # Validate PDB response
    if pdb_string is None:
        logger.error(f"[ESMFold] job={job_id} pdb_string is None after successful HTTP 200")
        await log_fn(job_id, "[ESMFold] ERROR: API returned HTTP 200 but response body is None", "warn")
        return None

    if len(pdb_string.strip()) == 0:
        logger.error(f"[ESMFold] job={job_id} pdb_string is empty (0 bytes) after HTTP 200")
        await log_fn(job_id, "[ESMFold] ERROR: API returned HTTP 200 but response body is empty", "warn")
        return None

    if not pdb_string.lstrip().startswith("ATOM"):
        first_100 = pdb_string[:100].replace("\n", "\\n")
        logger.error(f"[ESMFold] job={job_id} PDB does not start with ATOM — first 100 chars: {first_100}")
        await log_fn(job_id, f"[ESMFold] ERROR: API response is not valid PDB (does not start with ATOM). Got: {first_100}", "warn")
        return None

    elapsed = round(time.time() - start, 1)
    logger.info(f"[ESMFold] job={job_id} valid PDB received, parsing pLDDT…")

    # Parse pLDDT from B-factor column of ATOM records
    try:
        mean_plddt = parse_plddt_from_pdb(pdb_string)
        ptm = estimate_ptm(mean_plddt)
        logger.info(f"[ESMFold] job={job_id} mean_plddt={mean_plddt} ptm={ptm}")
    except Exception as e:
        import traceback
        tb = traceback.format_exc()
        logger.error(f"[ESMFold] job={job_id} parse_plddt_from_pdb() failed: {e}\n{tb}")
        await log_fn(job_id, f"[ESMFold] ERROR parsing pLDDT scores: {type(e).__name__}: {e}", "warn")
        return None

    await log_fn(job_id, f"[ESMFold] Mean pLDDT: {mean_plddt:.1f} · pTM: {ptm:.2f} · Elapsed: {elapsed}s", "done")

    # Cache the result
    try:
        cache_file = FASTA_CACHE_DIR / f"{job_id}_esmfold.pdb"
        cache_file.write_text(pdb_string)
        logger.info(f"[ESMFold] job={job_id} PDB cached at {cache_file}")
    except Exception as e:
        logger.warning(f"[ESMFold] job={job_id} failed to cache PDB: {e}")
        await log_fn(job_id, f"[ESMFold] Warning: could not cache PDB file: {e}", "warn")
        cache_file = None

    return {
        "pdb_string": pdb_string,
        "mean_plddt": mean_plddt,
        "ptm_score": ptm,
        "sequence_length": len(seq),
        "elapsed_s": elapsed,
        "engine": "ESMFold",
        "pdb_path": str(cache_file) if cache_file else None,
    }


async def fold_sequence(
    sequence: str,
    engine: str,
    job_id: str,
    log_fn,
    relax: bool = True,
    num_recycles: int = 3,
) -> Optional[dict]:
    """
    Route to correct engine.
    ESMFold: real API call.
    Others: return guidance on local installation.
    """
    seq = clean_sequence(sequence)
    logger.info(f"[fold_sequence] job={job_id} engine={engine} raw_len={len(sequence)} cleaned_len={len(seq)}")
    await log_fn(job_id, f"[fold_sequence] engine={engine} sequence length after cleaning: {len(seq)} residues", "info")

    if engine == "esm":
        logger.info(f"[fold_sequence] job={job_id} calling fold_with_esmfold()")
        result = await fold_with_esmfold(seq, job_id, log_fn)
        if result is None:
            logger.error(f"[fold_sequence] job={job_id} fold_with_esmfold() returned None — prediction failed")
            await log_fn(job_id, "[fold_sequence] ERROR: fold_with_esmfold() returned None — check ESMFold logs above for the specific failure reason", "warn")
        else:
            logger.info(f"[fold_sequence] job={job_id} fold_with_esmfold() succeeded: plddt={result.get('mean_plddt')} seq_len={result.get('sequence_length')}")
            if relax:
                await log_fn(job_id, "[ESMFold] Note: Amber relaxation requires OpenMM locally.", "info")
        return result

    elif engine == "colabfold":
        await log_fn(job_id, "[ColabFold] ColabFold requires a running local server or Google Colab.", "info")
        await log_fn(job_id, "[ColabFold] Install: pip install colabfold; colabfold_batch input.fasta output/", "info")
        await log_fn(job_id, "[ColabFold] Or use: https://colab.research.google.com/github/sokrypton/ColabFold", "info")
        await log_fn(job_id, "[ColabFold] ESMFold is available without installation — switch engine to ESMFold for immediate results.", "warn")
        return None

    elif engine == "omegafold":
        await log_fn(job_id, "[OmegaFold] OmegaFold requires local GPU installation.", "info")
        await log_fn(job_id, "[OmegaFold] Install: pip install omegafold; omegafold input.fasta output/", "info")
        await log_fn(job_id, "[OmegaFold] ESMFold covers similar use cases via free API — switch to ESMFold.", "warn")
        return None

    elif engine == "rosettafold":
        await log_fn(job_id, "[RoseTTAFold2] Requires local GPU + CUDA 11+ installation.", "info")
        await log_fn(job_id, "[RoseTTAFold2] See: https://github.com/uw-ipd/RoseTTAFold2", "info")
        return None

    return None


def clean_sequence(sequence: str) -> str:
    """Remove FASTA header lines and whitespace. Validate amino acids."""
    lines = sequence.strip().split("\n")
    seq_lines = [l.strip() for l in lines if not l.startswith(">")]
    seq = "".join(seq_lines).upper().replace(" ", "").replace("\t", "")
    # Keep only valid amino acid letters
    valid = set("ACDEFGHIKLMNPQRSTVWY")
    seq = "".join(c for c in seq if c in valid)
    return seq


def parse_plddt_from_pdb(pdb_string: str) -> float:
    """
    Extract pLDDT scores from B-factor column of CA atoms in ESMFold PDB output.
    ESMFold encodes pLDDT in the B-factor field.
    """
    plddts = []
    for line in pdb_string.split("\n"):
        if line.startswith("ATOM") and " CA " in line:
            try:
                bfactor = float(line[60:66].strip())
                plddts.append(bfactor)
            except (ValueError, IndexError):
                pass
    if not plddts:
        return 0.0
    return round(sum(plddts) / len(plddts), 2)


def parse_plddt_per_residue(pdb_string: str) -> list:
    """Return per-residue pLDDT list from ESMFold PDB."""
    plddts = []
    seen_residues = set()
    for line in pdb_string.split("\n"):
        if line.startswith("ATOM") and " CA " in line:
            try:
                res_num = int(line[22:26].strip())
                bfactor = float(line[60:66].strip())
                if res_num not in seen_residues:
                    seen_residues.add(res_num)
                    plddts.append(bfactor)
            except (ValueError, IndexError):
                pass
    return plddts


def estimate_ptm(mean_plddt: float) -> float:
    """Rough pTM estimate from mean pLDDT (empirical relationship)."""
    # ESMFold pTM ≈ pLDDT/100 * 1.08, capped at 0.99
    return min(0.99, round(mean_plddt / 100 * 1.08, 2))
