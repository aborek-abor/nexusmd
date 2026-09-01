"""
NexusMD — export router.

  GET /api/results/{job_id}/complex?pose_file=out_3.pdbqt&mode=1
      -> receptor + pose as one PDB, ready for Discovery Studio / PyMOL / Chimera
  GET /api/results/{job_id}/ligand?pose_file=out_3.pdbqt&mode=1
      -> the pose alone, as PDBQT
  GET /api/results/{job_id}/receptor
      -> the prepared receptor PDB
  GET /api/results/{job_id}/md_package
      -> zip: complex, protein, ligand SDF, docking metadata, and scripts
         that run and analyse an MD simulation of the pose

Install:
  1. save as app/routers/export.py
  2. in app/main.py, beside the other include_router calls, add:

        from app.routers import export
        app.include_router(export.router, prefix="/api", tags=["export"])

  3. in app/services/vina_service.py -> run_vina_single, right after
        poses = parse_vina_output(output_lines, name)
     add:
        for _p in poses:
            _p["pose_file"] = output.name
     so the frontend knows which file each pose came from.
"""

from pathlib import Path

from fastapi import APIRouter, HTTPException
from fastapi.responses import PlainTextResponse

from fastapi.responses import Response

from app.services.complex_export import build_complex, extract_pose
from app.services.md_package import PackageError, build_md_package
from app.services.job_queue import job_manager
from app.services.vina_service import PDB_CACHE_DIR, RESULTS_DIR

router = APIRouter()


def _job_and_dir(job_id: str):
    job = job_manager.get_job(job_id)
    if not job:
        raise HTTPException(404, f"Job {job_id} not found")
    job_dir = RESULTS_DIR / job_id
    if not job_dir.is_dir():
        raise HTTPException(404, f"No result files for job {job_id}")
    return job, job_dir


def _safe_pose(job_dir: Path, pose_file: str) -> Path:
    """pose_file must be a bare filename inside this job's directory."""
    if "/" in pose_file or "\\" in pose_file or pose_file.startswith("."):
        raise HTTPException(400, "pose_file must be a plain filename")
    path = (job_dir / pose_file).resolve()
    if not str(path).startswith(str(job_dir.resolve())):
        raise HTTPException(400, "pose_file escapes the job directory")
    if not path.exists():
        raise HTTPException(404, f"{pose_file} not found for this job")
    return path


def _receptor_for(job) -> Path:
    protein = (job.result or {}).get("protein") if job.result else None
    if not protein:
        raise HTTPException(400, "job has no recorded protein")
    for candidate in (f"{protein.upper()}_clean.pdb", f"{protein.upper()}.pdb"):
        path = PDB_CACHE_DIR / candidate
        if path.exists():
            return path
    raise HTTPException(404, f"prepared receptor for {protein} is no longer on disk")


@router.get("/results/{job_id}/complex", response_class=PlainTextResponse)
def download_complex(job_id: str, pose_file: str, mode: int = 1, name: str = "LIG"):
    """Receptor + one docked pose, merged into a single PDB."""
    job, job_dir = _job_and_dir(job_id)
    pose = _safe_pose(job_dir, pose_file)
    receptor = _receptor_for(job)
    try:
        pdb = build_complex(receptor, pose, mode=mode, ligand_name=name)
    except (ValueError, FileNotFoundError) as e:
        raise HTTPException(422, str(e))

    safe = "".join(c if c.isalnum() or c in "-_" else "_" for c in name)[:40] or "ligand"
    return PlainTextResponse(
        pdb,
        headers={"Content-Disposition": f'attachment; filename="{job_id}_{safe}_mode{mode}.pdb"'},
    )


@router.get("/results/{job_id}/ligand", response_class=PlainTextResponse)
def download_ligand(job_id: str, pose_file: str, mode: int = 1, name: str = "ligand"):
    """One pose on its own, as PDBQT."""
    job, job_dir = _job_and_dir(job_id)
    pose = _safe_pose(job_dir, pose_file)
    try:
        atoms, affinity = extract_pose(pose, mode)
    except (ValueError, FileNotFoundError) as e:
        raise HTTPException(422, str(e))

    header = f"REMARK  NexusMD pose  {name}  mode {mode}"
    if affinity is not None:
        header += f"  affinity {affinity:.2f} kcal/mol"
    safe = "".join(c if c.isalnum() or c in "-_" else "_" for c in name)[:40] or "ligand"
    return PlainTextResponse(
        header + "\n" + "\n".join(atoms) + "\n",
        headers={"Content-Disposition": f'attachment; filename="{safe}_mode{mode}.pdbqt"'},
    )


@router.get("/results/{job_id}/receptor", response_class=PlainTextResponse)
def download_receptor(job_id: str):
    """The prepared receptor, as used for docking."""
    job, _ = _job_and_dir(job_id)
    receptor = _receptor_for(job)
    return PlainTextResponse(
        receptor.read_text(),
        headers={"Content-Disposition": f'attachment; filename="{receptor.name}"'},
    )


@router.get("/results/{job_id}/md_package")
def download_md_package(job_id: str, pose_file: str, mode: int = 1, name: str = "LIG"):
    """Everything needed to simulate this pose somewhere with a GPU."""
    job, job_dir = _job_and_dir(job_id)
    pose = _safe_pose(job_dir, pose_file)
    receptor = _receptor_for(job)

    result = job.result or {}
    score = None
    for p in result.get("poses", []):
        if p.get("name") == name:
            score = p.get("score")
            break

    try:
        blob = build_md_package(
            receptor_pdb=receptor,
            pose_pdbqt=pose,
            ligand_name=name,
            mode=mode,
            job_id=job_id,
            protein_id=result.get("protein", ""),
            score=score,
            grid=result.get("grid"),
            engine=result.get("engine", "AutoDock Vina"),
        )
    except PackageError as e:
        raise HTTPException(422, str(e))
    except (ValueError, FileNotFoundError) as e:
        raise HTTPException(422, str(e))

    safe = "".join(c if c.isalnum() or c in "-_" else "_" for c in name)[:40] or "ligand"
    return Response(
        content=blob,
        media_type="application/zip",
        headers={"Content-Disposition":
                 f'attachment; filename="nexusmd_md_{safe}_mode{mode}.zip"'},
    )
