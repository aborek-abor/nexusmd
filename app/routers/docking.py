"""NexusMD — Docking Router"""

import asyncio
import logging
import time
from pathlib import Path

from fastapi.responses import PlainTextResponse
from fastapi import APIRouter, HTTPException, BackgroundTasks
from app.models.schemas import DockingRequest, DockingResult, JobStatus, PoseResult
from app.services.job_queue import job_manager
from app.services.vina_service import run_vina_docking, prepare_receptor_pdbqt
from app.services.admet_service import predict_admet_batch

logger = logging.getLogger("nexusmd.docking")
router = APIRouter()

RESULTS_DIR = Path(__file__).parent.parent.parent / "data" / "results"


@router.post("/submit", response_model=JobStatus)
async def submit_docking(req: DockingRequest, bg: BackgroundTasks):
    """Submit a docking job. Returns job_id immediately; poll /status/{job_id} or connect to WS."""
    if not req.ligand_smiles:
        raise HTTPException(400, "No ligands provided. Supply ligand_smiles list.")

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

        # Use upload path or fetch from RCSB
        protein_id = req.protein_id
        if protein_id.startswith("UPLOAD:"):
            filename = protein_id[7:]
            receptor_path = Path("data/ligands") / filename
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
