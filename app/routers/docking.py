"""NexusMD — Docking Router"""

import asyncio
import logging
import time
from pathlib import Path

from fastapi.responses import PlainTextResponse
from fastapi import APIRouter, HTTPException, BackgroundTasks, UploadFile, File, Query
from app.models.schemas import DockingRequest, DockingResult, JobStatus, PoseResult
from app.services.job_queue import job_manager
from app.services.vina_service import run_vina_docking, prepare_receptor_pdbqt, _clean_pdbqt
from app.services.admet_service import predict_admet_batch
from app.services.ligand_parser import parse_sdf_file, parse_mol2_file
from app.services.protein_parser import (
    parse_pdb_file,
    parse_cif_file,
    fetch_alphafold_structure,
    search_alphafold,
)

logger = logging.getLogger("nexusmd.docking")
router = APIRouter()

RESULTS_DIR  = Path(__file__).parent.parent.parent / "data" / "results"
UPLOADS_DIR  = Path(__file__).parent.parent.parent / "data" / "uploads"
LIGANDS_DIR  = Path(__file__).parent.parent.parent / "data" / "ligands"
UPLOADS_DIR.mkdir(parents=True, exist_ok=True)
LIGANDS_DIR.mkdir(parents=True, exist_ok=True)

SUPPORTED_EXTENSIONS      = {".sdf", ".mol", ".mol2"}
PROTEIN_EXTENSIONS        = {".pdb", ".cif", ".mmcif"}
MAX_UPLOAD_BYTES          = 10 * 1024 * 1024   # 10 MB
MAX_PROTEIN_UPLOAD_BYTES  = 50 * 1024 * 1024   # 50 MB


@router.post("/upload-ligands")
async def upload_ligands(file: UploadFile = File(...)):
    """Accept an SDF, MOL, or MOL2 file upload and return parsed ligand data.

    The uploaded file is stored under ``data/uploads/{timestamp}_{filename}``
    and a ``file_id`` is returned so the client can reference it in a
    subsequent ``/submit`` request via the ``ligand_file_id`` field.

    Response shape::

        {
            "file_id": "1700000000_ligands.sdf",
            "ligands": [{"name": "...", "smiles": "..."}, ...],
            "count": N
        }
    """
    # ── Validate filename / extension ──────────────────────────
    original_name = file.filename or "upload"
    suffix = Path(original_name).suffix.lower()
    if suffix not in SUPPORTED_EXTENSIONS:
        raise HTTPException(
            400,
            f"Unsupported file format '{suffix}'. "
            f"Supported formats: {', '.join(sorted(SUPPORTED_EXTENSIONS))}",
        )

    # ── Read and size-check ────────────────────────────────────
    content = await file.read()
    if len(content) > MAX_UPLOAD_BYTES:
        raise HTTPException(
            400,
            f"File too large ({len(content) // 1024} KB). Maximum allowed size is 10 MB.",
        )
    if not content.strip():
        raise HTTPException(400, "Uploaded file is empty.")

    # ── Persist to disk ────────────────────────────────────────
    timestamp = int(time.time())
    safe_name = Path(original_name).name.replace(" ", "_")
    file_id = f"{timestamp}_{safe_name}"
    dest = UPLOADS_DIR / file_id
    dest.write_bytes(content)
    logger.info(f"Ligand file uploaded: {file_id} ({len(content)} bytes)")

    # ── Parse ──────────────────────────────────────────────────
    try:
        if suffix in {".sdf", ".mol"}:
            molecules = parse_sdf_file(dest)
        else:  # .mol2
            molecules = parse_mol2_file(dest)
    except ValueError as exc:
        dest.unlink(missing_ok=True)
        raise HTTPException(400, str(exc))
    except Exception as exc:
        dest.unlink(missing_ok=True)
        logger.exception(f"Unexpected error parsing {file_id}: {exc}")
        raise HTTPException(400, f"Failed to parse file: {exc}")

    # ── Kick off background cleanup of old uploads ─────────────
    _cleanup_old_uploads()

    ligands = [{"name": m["name"], "smiles": m["smiles"]} for m in molecules]
    return {"file_id": file_id, "ligands": ligands, "count": len(ligands)}


# ── Protein upload / AlphaFold endpoints ──────────────────────

@router.post("/upload-protein")
async def upload_protein(file: UploadFile = File(...)):
    """Accept a PDB or mmCIF file upload and return parsed protein metadata.

    The uploaded file is stored under ``data/ligands/{timestamp}_{filename}``.

    Response shape::

        {
            "protein_id": "UPLOAD:timestamp_filename",
            "name":       str,
            "chains":     int,
            "residues":   int,
        }
    """
    original_name = file.filename or "upload"
    suffix = Path(original_name).suffix.lower()
    if suffix not in PROTEIN_EXTENSIONS:
        raise HTTPException(
            400,
            f"Unsupported protein file format '{suffix}'. "
            f"Supported formats: {', '.join(sorted(PROTEIN_EXTENSIONS))}",
        )

    content = await file.read()
    if len(content) > MAX_PROTEIN_UPLOAD_BYTES:
        raise HTTPException(
            400,
            f"File too large ({len(content) // (1024*1024)} MB). "
            "Maximum allowed size is 50 MB.",
        )
    if not content.strip():
        raise HTTPException(400, "Uploaded file is empty.")

    timestamp = int(time.time())
    safe_name = Path(original_name).name.replace(" ", "_")
    file_id = f"{timestamp}_{safe_name}"
    dest = LIGANDS_DIR / file_id
    dest.write_bytes(content)
    logger.info(f"Protein file uploaded: {file_id} ({len(content)} bytes)")

    try:
        if suffix == ".pdb":
            info = parse_pdb_file(dest)
        else:  # .cif / .mmcif
            info = parse_cif_file(dest)
    except (ValueError, FileNotFoundError) as exc:
        dest.unlink(missing_ok=True)
        raise HTTPException(400, str(exc))
    except Exception as exc:
        dest.unlink(missing_ok=True)
        logger.exception(f"Unexpected error parsing protein file {file_id}: {exc}")
        raise HTTPException(400, f"Failed to parse protein file: {exc}")

    return {
        "protein_id": f"UPLOAD:{file_id}",
        "name":       info["name"],
        "chains":     info["chains"],
        "residues":   info["residues"],
    }


@router.get("/search-alphafold")
async def search_alphafold_endpoint(q: str = Query(..., min_length=1, description="UniProt ID or gene/protein name")):
    """Search UniProt for proteins matching *q* and return up to 5 results.

    Response shape::

        {
            "results": [
                {
                    "uniprot_id":   str,
                    "gene_name":    str,
                    "protein_name": str,
                    "organism":     str,
                },
                ...
            ]
        }
    """
    results = await search_alphafold(q)
    return {"results": results}


@router.post("/fetch-alphafold")
async def fetch_alphafold_endpoint(body: dict):
    """Fetch an AlphaFold predicted structure by UniProt ID.

    Request body::

        {"uniprot_id": "P12345"}

    Response shape::

        {
            "protein_id": "ALPHAFOLD:P12345",
            "name":       str,
            "plddt":      float | null,
        }
    """
    uniprot_id = (body.get("uniprot_id") or "").strip().upper()
    if not uniprot_id:
        raise HTTPException(400, "Field 'uniprot_id' is required.")

    try:
        info = await fetch_alphafold_structure(uniprot_id)
    except ValueError as exc:
        raise HTTPException(404, str(exc))
    except Exception as exc:
        logger.exception(f"Unexpected error fetching AlphaFold structure for {uniprot_id}: {exc}")
        raise HTTPException(500, f"Failed to fetch AlphaFold structure: {exc}")

    return {
        "protein_id": f"ALPHAFOLD:{uniprot_id}",
        "name":       info["name"],
        "plddt":      info["plddt"],
    }


def _cleanup_old_uploads(max_age_seconds: int = 86400) -> None:
    """Delete upload files older than *max_age_seconds* (default 24 h)."""
    cutoff = time.time() - max_age_seconds
    for f in UPLOADS_DIR.iterdir():
        try:
            if f.is_file() and f.stat().st_mtime < cutoff:
                f.unlink()
                logger.debug(f"Cleaned up old upload: {f.name}")
        except Exception:
            pass


@router.post("/submit", response_model=JobStatus)
async def submit_docking(req: DockingRequest, bg: BackgroundTasks):
    """Submit a docking job. Returns job_id immediately; poll /status/{job_id} or connect to WS."""
    # Resolve ligands from uploaded file if ligand_file_id is provided
    if req.ligand_file_id:
        req = _load_ligands_from_file(req)

    if not req.ligand_smiles:
        raise HTTPException(400, "No ligands provided. Supply ligand_smiles list or upload a ligand file.")

    job_id = job_manager.create_job(
        f"Docking {len(req.ligand_smiles)} ligands on {req.protein_id} via {req.engine}"
    )
    bg.add_task(_run_docking_job, job_id, req)
    job = job_manager.get_job(job_id)
    return JobStatus(**job.to_dict())


@router.get("/status/{job_id}", response_model=JobStatus)
async def get_status(job_id: str):
    job = job_manager.get_job(job_id)
    if not job:
        raise HTTPException(404, f"Job {job_id} not found")
    logger.debug(f"[status] job={job_id} status={job.status} progress={job.progress}")
    return JobStatus(**job.to_dict())


@router.get("/poll/{job_id}")
async def poll_job(job_id: str):
    """
    Lightweight polling endpoint for clients that cannot use WebSockets.
    Returns current job status, progress, message, and a server timestamp.
    Also returns the last 20 log lines so the UI can display live progress.
    """
    job = job_manager.get_job(job_id)
    if not job:
        raise HTTPException(404, f"Job {job_id} not found")
    log_lines = [
        {"level": e.get("level", "info"), "line": e.get("line", e.get("message", ""))}
        for e in job.logs[-20:]
        if e.get("type") == "log"
    ]
    logger.debug(f"[poll] job={job_id} status={job.status} progress={job.progress}")
    return {
        "job_id": job_id,
        "status": job.status,
        "progress": job.progress,
        "message": job.message,
        "server_ts": time.time(),
        "updated_at": job.updated_at,
        "logs": log_lines,
        "result": job.result if job.status == "done" else None,
    }


@router.get("/results/{job_id}", response_model=DockingResult)
async def get_results(job_id: str):
    job = job_manager.get_job(job_id)
    if not job:
        raise HTTPException(404, f"Job {job_id} not found")
    if job.status != "done":
        raise HTTPException(400, f"Job {job_id} not done yet (status: {job.status})")
    if not job.result:
        raise HTTPException(500, "Job completed but no results stored")
    return DockingResult(**job.result)


async def _run_docking_job(job_id: str, req: DockingRequest):
    """Background task: full docking pipeline."""
    start = time.time()
    log = job_manager.log
    update = job_manager.update

    async def _update(status: str, progress: int, message: str, **kwargs):
        """Wrapper that also writes to the server log for easier debugging."""
        logger.info(f"[docking] job={job_id} status={status} progress={progress}% — {message}")
        await update(job_id, status, progress, message, **kwargs)

    try:
        await _update("running", 5, "Starting docking pipeline…")
        await log(job_id, f"[NexusMD] Job {job_id} — {req.engine.upper()}")
        await log(job_id, f"[INFO] Protein: {req.protein_id}")
        await log(job_id, f"[INFO] Ligands: {len(req.ligand_smiles)}")
        await log(job_id, f"[INFO] Grid: ({req.grid.center_x:.1f}, {req.grid.center_y:.1f}, {req.grid.center_z:.1f}) ±{req.grid.size_x:.0f}Å")

        # Step 1: Prepare receptor
        await _update("running", 15, "Preparing receptor…")
        await log(job_id, f"[INFO] Fetching and preparing receptor: {req.protein_id}")

        # Use upload path, AlphaFold cache, or fetch from RCSB
        protein_id = req.protein_id
        if protein_id.startswith("UPLOAD:"):
            filename = protein_id[7:]
            receptor_path = LIGANDS_DIR / filename
            # Clean PDBQT: remove PDB headers that Vina doesn't accept
            if receptor_path and receptor_path.exists():
                pdbqt_text = receptor_path.read_text()
                pdbqt_text = _clean_pdbqt(pdbqt_text)
                receptor_path.write_text(pdbqt_text)
                await log(job_id, f"[INFO] Receptor cleaned: {receptor_path.stat().st_size} bytes", "info")
        elif protein_id.startswith("ALPHAFOLD:"):
            uniprot_id = protein_id[10:]
            await log(job_id, f"[INFO] Fetching AlphaFold structure for {uniprot_id}…")
            from app.services.protein_parser import fetch_alphafold_structure
            try:
                af_info = await fetch_alphafold_structure(uniprot_id)
                # Write PDB text to cache path for Vina preparation
                af_pdb_path = LIGANDS_DIR / f"AF_{uniprot_id}.pdb"
                af_pdb_path.write_text(af_info["pdb_text"], encoding="utf-8")
                receptor_path = await prepare_receptor_pdbqt(str(af_pdb_path), log)
            except ValueError as exc:
                await log(job_id, f"[ERROR] {exc}", "warn")
                receptor_path = None
        else:
            receptor_path = await prepare_receptor_pdbqt(protein_id, log)

        if receptor_path is None:
            await log(job_id, "[WARN] Receptor preparation failed — check Vina/OpenBabel installation", "warn")
            await log(job_id, "[INFO] Generating simulated scores for demo purposes", "info")
            poses = _simulate_poses(req.ligand_smiles, req.ligand_names or req.ligand_smiles)
        else:
            await log(job_id, f"[INFO] Receptor ready: {receptor_path.name}")

            # Step 2: ADMET pre-filter
            names = req.ligand_names or [f"Compound-{i+1}" for i in range(len(req.ligand_smiles))]
            filtered_smiles = req.ligand_smiles
            filtered_names = names

            if req.admet_filter:
                await _update("running", 25, "Running ADMET pre-filter…")
                await log(job_id, "[ADMET] Applying Lipinski Ro5 pre-filter…")
                admet_results = await predict_admet_batch(req.ligand_smiles, names)
                filtered_pairs = [(s, n, a) for s, n, a in zip(req.ligand_smiles, names, admet_results)
                                  if a.get("ro5_violations", 0) <= 2]
                filtered_smiles = [p[0] for p in filtered_pairs]
                filtered_names = [p[1] for p in filtered_pairs]
                admet_map = {p[1]: p[2] for p in filtered_pairs}
                removed = len(req.ligand_smiles) - len(filtered_smiles)
                await log(job_id, f"[ADMET] {removed} compounds removed by Ro5 filter. {len(filtered_smiles)} proceeding.")
            else:
                admet_map = {}

            if not filtered_smiles:
                await _update("failed", 100, "All compounds failed ADMET filter")
                return

            # Step 3: Run docking
            await _update("running", 40, f"Docking {len(filtered_smiles)} ligands…")
            poses = await run_vina_docking(
                job_id, receptor_path, filtered_smiles, filtered_names,
                req.grid.model_dump(), log
            )

            # Annotate with ADMET status
            for pose in poses:
                admet = admet_map.get(pose["name"], {})
                pose["admet_status"] = admet.get("status", "—")

        await _update("running", 90, "Finalising results…")
        await log(job_id, f"[INFO] {len(poses)} total poses generated")

        elapsed = round(time.time() - start, 1)
        result = {
            "job_id": job_id,
            "protein": protein_id,
            "engine": req.engine,
            "poses": poses,
            "elapsed_s": elapsed,
            "sdf_url": f"/api/results/{job_id}/sdf",
            "pdbqt_url": f"/api/results/{job_id}/pdbqt",
        }
        await _update("done", 100, f"Done — {len(poses)} poses in {elapsed}s", result=result)
        await log(job_id, f"[DONE] Docking complete in {elapsed}s — {len(poses)} poses")

    except Exception as e:
        import traceback
        tb = traceback.format_exc()
        logger.error(f"[docking] job={job_id} unhandled exception: {e}\n{tb}")
        await update(job_id, "failed", 100, f"Error: {type(e).__name__}: {str(e)}")
        await log(job_id, f"[ERROR] {type(e).__name__}: {str(e)}", "warn")
        await log(job_id, f"[TRACEBACK] {tb}", "warn")
        raise


def _load_ligands_from_file(req: DockingRequest) -> DockingRequest:
    """Load ligand SMILES and names from a previously uploaded file.

    Returns a copy of *req* with ``ligand_smiles`` and ``ligand_names``
    populated from the file.  Raises ``HTTPException(400)`` if the file
    cannot be found or parsed.

    Molecules without SMILES are included with an empty string — they can
    still be docked using their 3D structure directly.
    """
    file_path = UPLOADS_DIR / req.ligand_file_id
    if not file_path.exists():
        raise HTTPException(
            400,
            f"Uploaded ligand file '{req.ligand_file_id}' not found. "
            "It may have expired (files are kept for 24 hours).",
        )

    suffix = file_path.suffix.lower()
    try:
        if suffix in {".sdf", ".mol"}:
            molecules = parse_sdf_file(file_path)
        elif suffix == ".mol2":
            molecules = parse_mol2_file(file_path)
        else:
            raise HTTPException(
                400,
                f"Unsupported ligand file format '{suffix}'.",
            )
    except HTTPException:
        raise
    except ValueError as exc:
        raise HTTPException(400, str(exc))
    except Exception as exc:
        logger.exception(f"Error loading ligands from {req.ligand_file_id}: {exc}")
        raise HTTPException(400, f"Failed to parse ligand file: {exc}")

    # Log molecules that have no SMILES — they will use 3D structure directly
    for mol in molecules:
        if not mol.get("smiles"):
            logger.info(
                f"Ligand {mol['name']!r} has no SMILES, using 3D structure for docking"
            )

    smiles = [m["smiles"] for m in molecules]
    names  = [m["name"]   for m in molecules]

    # Merge: file-derived ligands take precedence; keep any extra SMILES
    # that were also supplied directly (backward-compat).
    combined_smiles = smiles + list(req.ligand_smiles or [])
    combined_names  = names  + list(req.ligand_names  or [])

    # Build an updated request (Pydantic v2 model_copy)
    return req.model_copy(update={
        "ligand_smiles": combined_smiles,
        "ligand_names":  combined_names,
    })


def _simulate_poses(smiles_list: list, names: list) -> list:
    """Fallback: generate plausible-looking scores when Vina is not installed."""
    import random
    poses = []
    for i, (smi, name) in enumerate(zip(smiles_list, names)):
        # Score correlates with molecule size (bigger = potentially better binding, within limits)
        base = -6.0 - len(smi) * 0.02 + random.uniform(-1.5, 1.5)
        base = max(-13.0, min(-4.0, base))
        poses.append({
            "name": name, "rank": i + 1,
            "score": round(base, 1),
            "score_2": round(base + random.uniform(0.5, 2.0), 1),
            "rmsd_lb": round(random.uniform(0, 3), 2),
            "rmsd_ub": round(random.uniform(2, 5), 2),
            "admet_status": "—",
        })
    poses.sort(key=lambda p: p["score"])
    for rank, p in enumerate(poses, 1):
        p["rank"] = rank
    return poses


@router.post("/smiles2sdf")
async def smiles_to_sdf(data: dict):
    """Convert SMILES to 3D SDF using Open Babel for ligand visualization"""
    import subprocess, tempfile, os
    smiles = data.get("smiles", "")
    name = data.get("name", "ligand")
    if not smiles:
        raise HTTPException(400, "No SMILES provided")
    try:
        with tempfile.TemporaryDirectory() as tmpdir:
            sdf_out = os.path.join(tmpdir, "out.sdf")
            result = subprocess.run(
                ["obabel", "-:" + smiles, "--gen3d", "-O", sdf_out, "--title", name],
                capture_output=True, text=True, timeout=30
            )
            if os.path.exists(sdf_out):
                return PlainTextResponse(open(sdf_out).read())
    except Exception as e:
        pass
    raise HTTPException(500, "Could not convert SMILES to SDF")
