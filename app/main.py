"""
NexusMD Backend — FastAPI Server
Railway deployment: serves frontend static files + full API
"""

import asyncio
import logging
import os
import subprocess
import time
import uuid
from contextlib import asynccontextmanager
from pathlib import Path
from typing import Optional

import httpx
from fastapi import FastAPI, WebSocket, WebSocketDisconnect, HTTPException, BackgroundTasks
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse, JSONResponse, HTMLResponse

from app.routers import docking, proteins, admet, pockets, scaffold, mmgbsa, fasta
from app.services.job_queue import job_manager
from app.models.schemas import HealthResponse
from app.models.database import init_db

# ── Logging ───────────────────────────────────────
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
)
logger = logging.getLogger("nexusmd")

# ── Paths ──────────────────────────────────────────
BASE_DIR   = Path(__file__).parent.parent
DATA_DIR   = BASE_DIR / "data"
FRONTEND_DIR = BASE_DIR / "frontend"

for d in [DATA_DIR / "results", DATA_DIR / "pdb_cache", DATA_DIR / "ligands", BASE_DIR / "logs"]:
    d.mkdir(parents=True, exist_ok=True)

# ── App lifespan ───────────────────────────────────
@asynccontextmanager
async def lifespan(app: FastAPI):
    logger.info("NexusMD starting up on Railway…")
    # Initialise database tables (no-op if already exist)
    init_db()
    logger.info("  Database: tables initialised")
    vina_ok = _check_binary("vina")
    obabel_ok = _check_binary("obabel")
    logger.info(f"  Vina:   {'✓ found' if vina_ok else '✗ not found (simulation mode)'}")
    if obabel_ok:
        logger.info("  OBabel: ✓ found")
    else:
        logger.warning("  OBabel: ✗ not found — SMILES→3D and PDBQT conversion will be unavailable; docking will use simulation mode. Install openbabel or set OBABEL_BINARY env var.")
    logger.info(f"  Frontend: {FRONTEND_DIR} ({'exists' if FRONTEND_DIR.exists() else 'MISSING'})")
    await job_manager.start()
    yield
    logger.info("NexusMD shutting down…")
    await job_manager.stop()

# ── App ─────────────────────────────────────────────
app = FastAPI(
    title="NexusMD API",
    version="5.0.0",
    description="Drug Discovery Platform — Real docking, ADMET, ESMFold, PDB",
    lifespan=lifespan,
    docs_url="/api/docs",
    redoc_url="/api/redoc",
)

# ── CORS — allow all origins (public platform) ─────
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# ── API Routers ────────────────────────────────────
app.include_router(docking.router,  prefix="/api/docking",  tags=["Docking"])
app.include_router(proteins.router, prefix="/api/protein",  tags=["Protein"])
app.include_router(admet.router,    prefix="/api/admet",    tags=["ADMET"])
app.include_router(pockets.router,  prefix="/api/pockets",  tags=["Pockets"])
app.include_router(scaffold.router, prefix="/api/scaffold", tags=["Scaffold"])
app.include_router(mmgbsa.router,   prefix="/api/mmgbsa",   tags=["MM-GBSA"])
app.include_router(fasta.router,    prefix="/api/fasta",    tags=["FASTA"])

# ── Health ─────────────────────────────────────────
def _check_binary(name: str) -> bool:
    """
    Check if a binary is available. Tries in order:
    1. The bare name (relies on PATH)
    2. The env-var override (e.g. OBABEL_BINARY / VINA_BINARY)
    3. Common install locations (/usr/bin, /usr/local/bin)
    Logs the specific failure reason so startup output is actionable.
    """
    # 1. Try bare name via PATH
    try:
        result = subprocess.run(
            [name, "--version"], capture_output=True, timeout=5
        )
        if result.returncode == 0:
            return True
        logger.debug(f"  {name} via PATH returned non-zero exit code {result.returncode}")
    except FileNotFoundError:
        logger.debug(f"  {name} not found in PATH")
    except Exception as e:
        logger.debug(f"  {name} PATH check failed: {e}")

    # 2. Try env-var override
    env_path = os.environ.get(f"{name.upper()}_BINARY", "")
    if env_path:
        try:
            result = subprocess.run([env_path, "--version"], capture_output=True, timeout=5)
            if result.returncode == 0:
                return True
            logger.debug(f"  {name} at {env_path} (env var) returned exit code {result.returncode}")
        except FileNotFoundError:
            logger.debug(f"  {name} not found at env-var path: {env_path}")
        except Exception as e:
            logger.debug(f"  {name} env-var path check failed: {e}")

    # 3. Try common install locations
    for candidate in (f"/usr/bin/{name}", f"/usr/local/bin/{name}"):
        try:
            result = subprocess.run([candidate, "--version"], capture_output=True, timeout=5)
            if result.returncode == 0:
                logger.info(f"  {name} found at {candidate} (not in PATH — consider adding to PATH)")
                return True
        except FileNotFoundError:
            pass
        except Exception:
            pass

    logger.warning(f"  {name} not found in PATH, env var, or common locations (/usr/bin, /usr/local/bin)")
    return False

@app.get("/api/health", response_model=HealthResponse)
async def health():
    vina_ok   = _check_binary("vina")
    obabel_ok = _check_binary("obabel")
    redis_ok  = False
    try:
        from app.services.job_queue import job_manager
        redis_ok = job_manager.redis_connected if hasattr(job_manager, 'redis_connected') else False
    except Exception:
        pass
    return HealthResponse(
        status="ok",
        vina=vina_ok,
        obabel=obabel_ok,
        redis=redis_ok,
        version="5.0.0",
        environment="railway",
    )

# ── WebSocket ──────────────────────────────────────
@app.websocket("/ws/{job_id}")
async def websocket_job(websocket: WebSocket, job_id: str):
    await websocket.accept()
    logger.info(f"WebSocket connected for job {job_id}")
    try:
        job = job_manager.get_job(job_id)
        if not job:
            await websocket.send_json({"type": "error", "job_id": job_id, "message": f"Job {job_id} not found"})
            await websocket.close()
            return
        async for msg in job_manager.subscribe(job_id):
            await websocket.send_json(msg)
    except WebSocketDisconnect:
        logger.info(f"WebSocket disconnected for job {job_id}")
    except Exception as e:
        logger.error(f"WebSocket error for job {job_id}: {e}")

# ── Serve frontend ─────────────────────────────────
# All non-API routes return the SPA index.html
@app.get("/", response_class=HTMLResponse)
async def serve_frontend():
    index = FRONTEND_DIR / "index.html"
    if index.exists():
        return HTMLResponse(content=index.read_text(encoding="utf-8"))
    return HTMLResponse("<h1>NexusMD</h1><p>Frontend not found. Deploy frontend/index.html</p>", status_code=404)

@app.get("/{path:path}", response_class=HTMLResponse)
async def serve_spa(path: str):
    # Don't catch API routes
    if path.startswith("api/"):
        raise HTTPException(status_code=404)
    index = FRONTEND_DIR / "index.html"
    if index.exists():
        return HTMLResponse(content=index.read_text(encoding="utf-8"))
    raise HTTPException(status_code=404, detail="Frontend not found")
