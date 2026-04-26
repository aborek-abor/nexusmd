"""NexusMD — FASTA / ESMFold Router"""
import time
from pathlib import Path
from fastapi import APIRouter, HTTPException, BackgroundTasks
from fastapi.responses import PlainTextResponse
from app.models.schemas import FASTARequest, FASTAResult, JobStatus
from app.services.esmfold_service import fold_sequence, parse_plddt_per_residue
from app.services.job_queue import job_manager

router = APIRouter()
FASTA_DIR = Path(__file__).parent.parent.parent / "data" / "fasta_results"
FASTA_DIR.mkdir(parents=True, exist_ok=True)


@router.post("/fold", response_model=JobStatus)
async def fold_fasta(req: FASTARequest, bg: BackgroundTasks):
    """Submit FASTA sequence for structure prediction. Returns job_id."""
    if not req.sequence or len(req.sequence.replace("\n","").replace(" ","")) < 10:
        raise HTTPException(400, "Sequence too short (min 10 residues)")
    job_id = job_manager.create_job(f"ESMFold prediction — {req.engine}")
    bg.add_task(_fold_job, job_id, req)
    job = job_manager.get_job(job_id)
    return JobStatus(**job.to_dict())


@router.get("/result/{job_id}")
async def get_fold_result(job_id: str):
    job = job_manager.get_job(job_id)
    if not job:
        raise HTTPException(404, f"Job {job_id} not found")
    if job.status != "done":
        return {"status": job.status, "progress": job.progress, "message": job.message}
    return job.result


@router.get("/pdb/{job_id}", response_class=PlainTextResponse)
async def download_fold_pdb(job_id: str):
    """Download the predicted PDB file."""
    path = FASTA_DIR / f"{job_id}_esmfold.pdb"
    if not path.exists():
        raise HTTPException(404, "PDB not ready yet")
    return path.read_text()


@router.get("/plddt/{job_id}")
async def get_plddt(job_id: str):
    """Return per-residue pLDDT scores."""
    path = FASTA_DIR / f"{job_id}_esmfold.pdb"
    if not path.exists():
        raise HTTPException(404, "Structure not ready")
    pdb = path.read_text()
    plddts = parse_plddt_per_residue(pdb)
    mean = round(sum(plddts) / len(plddts), 2) if plddts else 0.0
    return {"job_id": job_id, "mean_plddt": mean, "per_residue": plddts}


async def _fold_job(job_id: str, req: FASTARequest):
    start = time.time()
    try:
        await job_manager.update(job_id, "running", 5, "Starting structure prediction…")
        result = await fold_sequence(
            req.sequence, req.engine, job_id,
            job_manager.log, req.relax, req.num_recycles,
        )
        if result is None:
            await job_manager.update(job_id, "failed", 100, "Prediction failed — see log for details")
            return

        elapsed = round(time.time() - start, 1)
        result_dict = {
            "header": req.header or ">Predicted",
            "sequence_length": result["sequence_length"],
            "mean_plddt": result["mean_plddt"],
            "ptm_score": result["ptm_score"],
            "pdb_url": f"/api/fasta/pdb/{job_id}",
            "plddt_url": f"/api/fasta/plddt/{job_id}",
            "engine": result["engine"],
            "elapsed_s": elapsed,
        }
        await job_manager.update(job_id, "done", 100,
            f"Done — pLDDT {result['mean_plddt']:.1f}", result=result_dict)
    except Exception as e:
        await job_manager.update(job_id, "failed", 100, f"Error: {e}")
        await job_manager.log(job_id, f"[ERROR] {e}", "warn")
        raise
