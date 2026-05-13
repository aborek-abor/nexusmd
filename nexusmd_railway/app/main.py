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

from app.routers import docking, proteins, admet, pockets, scaffold, mmgbsa, fasta, md
from app.services.job_queue import job_manager
from app.models.schemas import HealthResponse

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
    vina_ok = _check_binary("vina")
    obabel_ok = _check_binary("obabel")
    logger.info(f"  Vina:   {'✓ found' if vina_ok else '✗ not found (simulation mode)'}")
    logger.info(f"  OBabel: {'✓ found' if obabel_ok else '✗ not found'}")
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
app.include_router(md.router,       prefix="/api/md",       tags=["MD"])

# ── Health ─────────────────────────────────────────
def _check_binary(name: str) -> bool:
    try:
        result = subprocess.run(
            [name, "--version"], capture_output=True, timeout=5
        )
        return result.returncode == 0
    except Exception:
        try:
            path = os.environ.get(f"{name.upper()}_BINARY", f"/usr/local/bin/{name}")
            result = subprocess.run([path, "--version"], capture_output=True, timeout=5)
            return result.returncode == 0
        except Exception:
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
    try:
        async for msg in job_manager.stream_job(job_id):
            await websocket.send_json(msg)
    except WebSocketDisconnect:
        pass
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
