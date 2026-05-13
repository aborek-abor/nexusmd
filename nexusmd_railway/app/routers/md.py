"""NexusMD — Molecular Dynamics Router

Endpoints:
  POST /api/md/submit                              — submit an MD job
  GET  /api/md/status/{job_id}                     — poll job status
  GET  /api/md/results/{job_id}                    — fetch completed results
  GET  /api/md/results/{job_id}/download/trajectory — download DCD file
"""

import io
import logging
import time
from pathlib import Path

from fastapi import APIRouter, BackgroundTasks, HTTPException
from fastapi.responses import StreamingResponse

from app.models.md_schemas import MDRequest, MDResult
from app.models.schemas import JobStatus
from app.services.job_queue import job_manager

logger = logging.getLogger("nexusmd.md")

router = APIRouter()

# Results are written under  <repo_root>/data/results/<job_id>/
RESULTS_DIR = Path(__file__).parent.parent.parent / "data" / "results"


# ── Submit ─────────────────────────────────────────────────────────────────

@router.post("/submit", response_model=JobStatus)
async def submit_md(req: MDRequest, bg: BackgroundTasks):
    """
    Submit a molecular-dynamics simulation job.

    The docking job identified by *docking_job_id* must already be complete;
    its best-scoring PDBQT pose is used as the starting structure.
    Returns a ``JobStatus`` immediately — poll ``/status/{job_id}`` or
    connect to the WebSocket at ``/ws/{job_id}`` for live progress.
    """
    # Validate that the referenced docking job exists and is done
    docking_job = job_manager.get_job(req.docking_job_id)
    if not docking_job:
        raise HTTPException(
            status_code=404,
            detail=f"Docking job '{req.docking_job_id}' not found.",
        )
    if docking_job.status != "done":
        raise HTTPException(
            status_code=400,
            detail=(
                f"Docking job '{req.docking_job_id}' is not complete "
                f"(current status: {docking_job.status})."
            ),
        )

    job_id = job_manager.create_job(
        f"MD simulation {req.duration_ns} ns / {req.force_field} / "
        f"{req.solvation} solvent (from docking job {req.docking_job_id})"
    )
    bg.add_task(_run_md_job, job_id, req)
    job = job_manager.get_job(job_id)
    return JobStatus(**job.to_dict())


# ── Status ─────────────────────────────────────────────────────────────────

@router.get("/status/{job_id}", response_model=JobStatus)
async def get_md_status(job_id: str):
    """Return the current status of an MD job."""
    job = job_manager.get_job(job_id)
    if not job:
        raise HTTPException(status_code=404, detail=f"MD job '{job_id}' not found.")
    return JobStatus(**job.to_dict())


# ── Results ────────────────────────────────────────────────────────────────

@router.get("/results/{job_id}", response_model=MDResult)
async def get_md_results(job_id: str):
    """
    Return the full results for a completed MD job, including energy and
    RMSD analysis and a URL to download the DCD trajectory.
    """
    job = job_manager.get_job(job_id)
    if not job:
        raise HTTPException(status_code=404, detail=f"MD job '{job_id}' not found.")
    if job.status != "done":
        raise HTTPException(
            status_code=400,
            detail=f"MD job '{job_id}' is not complete yet (status: {job.status}).",
        )
    if not job.result:
        raise HTTPException(
            status_code=500,
            detail="Job completed but no results were stored.",
        )
    return MDResult(**job.result)


# ── Trajectory download ────────────────────────────────────────────────────

@router.get("/results/{job_id}/download/trajectory")
async def download_trajectory(job_id: str):
    """Download the DCD trajectory file for a completed MD job."""
    traj_path = RESULTS_DIR / job_id / "trajectory.dcd"
    if not traj_path.exists():
        raise HTTPException(
            status_code=404,
            detail=f"Trajectory file not found for MD job '{job_id}'.",
        )
    content = traj_path.read_bytes()
    return StreamingResponse(
        io.BytesIO(content),
        media_type="application/octet-stream",
        headers={
            "Content-Disposition": f'attachment; filename="{job_id}_trajectory.dcd"'
        },
    )


# ── Background task ────────────────────────────────────────────────────────

async def _run_md_job(job_id: str, req: MDRequest):
    """
    Full MD pipeline executed as a FastAPI background task.

    Steps:
      1. Locate the best-scoring PDBQT from the docking job
      2. Prepare the OpenMM system (force field + solvation)
      3. Run production MD with periodic logging
      4. Persist trajectory to disk (and optionally to the S3 bucket)
      5. Store analysis results in the job record
    """
    start = time.time()
    log = job_manager.log
    update = job_manager.update

    try:
        await update(job_id, "running", 5, "Initialising MD pipeline…")
        await log(job_id, f"[MD] Job {job_id} — {req.duration_ns} ns simulation")
        await log(job_id, f"[MD] Force field : {req.force_field}")
        await log(job_id, f"[MD] Solvation   : {req.solvation}")
        await log(job_id, f"[MD] Temperature : {req.temperature_K} K")
        await log(job_id, f"[MD] Timestep    : {req.timestep_fs} fs")

        # ── Step 1: locate starting structure ─────────────────────────────
        await update(job_id, "running", 10, "Locating docked complex…")
        pdbqt_path = _find_docking_pdbqt(req.docking_job_id)
        if pdbqt_path is None:
            await update(
                job_id, "failed", 100,
                f"No PDBQT file found for docking job '{req.docking_job_id}'. "
                "Ensure the docking job completed successfully and produced output files."
            )
            return
        await log(job_id, f"[MD] Starting structure: {pdbqt_path.name}", "info")

        # ── Step 2: prepare OpenMM system ─────────────────────────────────
        await update(job_id, "running", 20, "Preparing MD system (force field + solvation)…")
        try:
            from app.services.md_service import prepare_md_system
            padding_nm = req.padding_angstrom / 10.0  # Å → nm
            system, topology, positions = prepare_md_system(
                pdbqt_path=pdbqt_path,
                force_field=req.force_field,
                solvation=req.solvation,
                padding_nm=padding_nm,
                temperature_K=req.temperature_K,
            )
        except RuntimeError as exc:
            await update(job_id, "failed", 100, f"System preparation failed: {exc}")
            await log(job_id, f"[ERROR] {exc}", "warn")
            return
        except Exception as exc:
            await update(job_id, "failed", 100, f"Unexpected error during system setup: {exc}")
            await log(job_id, f"[ERROR] {exc}", "warn")
            return

        await log(job_id, "[MD] System ready — starting simulation", "info")

        # ── Step 3: run MD ────────────────────────────────────────────────
        await update(job_id, "running", 30, f"Running {req.duration_ns} ns MD simulation…")
        results_dir = RESULTS_DIR / job_id
        results_dir.mkdir(parents=True, exist_ok=True)

        try:
            from app.services.md_service import run_md_simulation
            analysis = await run_md_simulation(
                job_id=job_id,
                system=system,
                topology=topology,
                positions=positions,
                duration_ns=req.duration_ns,
                temperature_K=req.temperature_K,
                timestep_fs=req.timestep_fs,
                log_fn=log,
                results_dir=results_dir,
            )
        except RuntimeError as exc:
            await update(job_id, "failed", 100, f"Simulation failed: {exc}")
            await log(job_id, f"[ERROR] {exc}", "warn")
            return
        except Exception as exc:
            await update(job_id, "failed", 100, f"Simulation crashed: {exc}")
            await log(job_id, f"[ERROR] {exc}", "warn")
            return

        # ── Step 4: persist trajectory to bucket (best-effort) ────────────
        await update(job_id, "running", 90, "Saving trajectory…")
        traj_path = results_dir / "trajectory.dcd"
        if traj_path.exists():
            try:
                from app.services.storage_service import upload_job_results
                await upload_job_results(job_id, results_dir)
            except Exception as exc:
                logger.warning(f"[MD] Bucket upload failed (results still local): {exc}")

        # ── Step 5: store results ─────────────────────────────────────────
        elapsed = round(time.time() - start, 1)
        result = {
            "job_id": job_id,
            "docking_job_id": req.docking_job_id,
            "duration_ns": req.duration_ns,
            "temperature_K": req.temperature_K,
            "force_field": req.force_field,
            "solvation": req.solvation,
            "elapsed_s": elapsed,
            "trajectory_url": f"/api/md/results/{job_id}/download/trajectory",
            "analysis": analysis,
        }
        await update(
            job_id, "done", 100,
            f"MD simulation complete — {req.duration_ns} ns in {elapsed}s",
            result=result,
        )
        await log(job_id, f"[DONE] MD job {job_id} finished in {elapsed}s", "info")

    except Exception as exc:
        elapsed = round(time.time() - start, 1)
        await update(job_id, "failed", 100, f"Unhandled error: {exc}")
        await log(job_id, f"[ERROR] {exc}", "warn")
        logger.exception(f"MD job {job_id} failed after {elapsed}s")
        raise


# ── Helpers ────────────────────────────────────────────────────────────────

def _find_docking_pdbqt(docking_job_id: str) -> Path | None:
    """
    Return the path to the best-scoring docked PDBQT for *docking_job_id*.

    Preference order:
      1. out_0.pdbqt  (rank-1 pose from Vina)
      2. Any out_*.pdbqt
      3. Any lig_*.pdbqt (prepared ligand, no docking output)
    Returns None if no PDBQT file is found.
    """
    job_dir = RESULTS_DIR / docking_job_id
    if not job_dir.exists():
        logger.warning(f"[MD] Docking job directory not found: {job_dir}")
        return None

    # Prefer rank-1 Vina output
    best = job_dir / "out_0.pdbqt"
    if best.exists():
        return best

    # Fall back to any output PDBQT
    out_files = sorted(job_dir.glob("out_*.pdbqt"))
    if out_files:
        return out_files[0]

    # Last resort: prepared ligand PDBQT
    lig_files = sorted(job_dir.glob("lig_*.pdbqt"))
    if lig_files:
        return lig_files[0]

    return None
