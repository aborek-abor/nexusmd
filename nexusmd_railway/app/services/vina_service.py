"""
NexusMD — AutoDock Vina Docking Service
Real subprocess calls to Vina binary.
Handles: SMILES → 3D → PDBQT → Vina → parse results
"""

import asyncio
import logging
import os
import re
import subprocess
import tempfile
import time
from pathlib import Path
from typing import List, Optional, Tuple

logger = logging.getLogger("nexusmd.vina")

VINA_BINARY = os.environ.get("VINA_BINARY", "vina")
OBABEL_BINARY = os.environ.get("OBABEL_BINARY", "obabel")
DATA_DIR = Path(__file__).parent.parent.parent / "data"
RESULTS_DIR = DATA_DIR / "results"
PDB_CACHE_DIR = DATA_DIR / "pdb_cache"


async def run_vina_docking(
    job_id: str,
    protein_pdbqt: Path,
    ligand_smiles: List[str],
    ligand_names: List[str],
    grid: dict,
    log_fn,
) -> List[dict]:
    """
    Run AutoDock Vina for a list of ligands against a prepared receptor.
    Returns list of pose dicts with scores.
    """
    job_dir = RESULTS_DIR / job_id
    job_dir.mkdir(parents=True, exist_ok=True)

    all_poses = []
    n = len(ligand_smiles)

    for i, (smi, name) in enumerate(zip(ligand_smiles, ligand_names)):
        await log_fn(job_id, f"[Vina] ({i+1}/{n}) Preparing ligand: {name}", "info")

        # Convert SMILES → 3D SDF → PDBQT using Open Babel
        lig_sdf = job_dir / f"lig_{i}.sdf"
        lig_pdbqt = job_dir / f"lig_{i}.pdbqt"
        out_pdbqt = job_dir / f"out_{i}.pdbqt"

        sdf_ok = await smiles_to_3d_sdf(smi, lig_sdf, name)
        if not sdf_ok:
            await log_fn(job_id, f"[WARN] Could not generate 3D for {name} — skipping", "warn")
            continue

        pdbqt_ok = await sdf_to_pdbqt(lig_sdf, lig_pdbqt)
        if not pdbqt_ok:
            await log_fn(job_id, f"[WARN] PDBQT conversion failed for {name} — skipping", "warn")
            continue

        await log_fn(job_id, f"[Vina] Running docking: {name}", "info")
        poses = await run_vina_single(
            protein_pdbqt, lig_pdbqt, out_pdbqt, grid, name, job_id, log_fn
        )
        all_poses.extend(poses)

    # Sort by score (most negative = best)
    all_poses.sort(key=lambda p: p["score"])
    # Re-rank
    for rank, pose in enumerate(all_poses, 1):
        pose["rank"] = rank

    # Write combined SDF
    await write_combined_sdf(all_poses, job_dir / "poses.sdf")
    await log_fn(job_id, f"[Vina] Docking complete — {len(all_poses)} poses", "done")

    # Persist results to bucket for durable storage across container restarts
    try:
        from app.services.storage_service import upload_job_results
        await upload_job_results(job_id, job_dir)
    except Exception as e:
        logger.warning(f"[Storage] Bucket upload failed (results still available locally): {e}")

    return all_poses



async def smiles_to_3d_sdf(smiles: str, output_path: Path, name: str = "LIG") -> bool:
    """Convert SMILES to 3D SDF using Open Babel."""
    try:
        result = await asyncio.create_subprocess_exec(
            OBABEL_BINARY,
            f"-:{smiles}",
            "--gen3d", "--ff", "MMFF94",
            "-O", str(output_path),
            "-p", "7.4",
            "--title", name,
            stdout=asyncio.subprocess.PIPE,
            stderr=asyncio.subprocess.PIPE,
        )
        stdout, stderr = await asyncio.wait_for(result.communicate(), timeout=30)
        return output_path.exists() and output_path.stat().st_size > 0
    except Exception as e:
        logger.error(f"SMILES→SDF failed for {name}: {e}")
        return False


async def sdf_to_pdbqt(sdf_path: Path, pdbqt_path: Path) -> bool:
    """Convert SDF to PDBQT using Open Babel (adds Gasteiger charges)."""
    try:
        result = await asyncio.create_subprocess_exec(
            OBABEL_BINARY,
            str(sdf_path),
            "-O", str(pdbqt_path),
            "--partialcharge", "gasteiger",
            stdout=asyncio.subprocess.PIPE,
            stderr=asyncio.subprocess.PIPE,
        )
        await asyncio.wait_for(result.communicate(), timeout=30)
        return pdbqt_path.exists() and pdbqt_path.stat().st_size > 0
    except Exception as e:
        logger.error(f"SDF→PDBQT failed: {e}")
        return False


async def run_vina_single(
    receptor: Path,
    ligand: Path,
    output: Path,
    grid: dict,
    name: str,
    job_id: str,
    log_fn,
) -> List[dict]:
    """Run Vina for a single ligand and parse output."""
    cmd = [
        VINA_BINARY,
        "--receptor", str(receptor),
        "--ligand", str(ligand),
        "--out", str(output),
        "--center_x", str(grid.get("center_x", 0)),
        "--center_y", str(grid.get("center_y", 0)),
        "--center_z", str(grid.get("center_z", 0)),
        "--size_x", str(grid.get("size_x", 22)),
        "--size_y", str(grid.get("size_y", 22)),
        "--size_z", str(grid.get("size_z", 22)),
        "--exhaustiveness", str(grid.get("exhaustiveness", 8)),
        "--num_modes", str(grid.get("num_modes", 9)),
        "--energy_range", str(grid.get("energy_range", 3.0)),
    ]

    try:
        proc = await asyncio.create_subprocess_exec(
            *cmd,
            stdout=asyncio.subprocess.PIPE,
            stderr=asyncio.subprocess.STDOUT,
        )

        output_lines = []
        async for line in proc.stdout:
            decoded = line.decode("utf-8", errors="replace").rstrip()
            output_lines.append(decoded)
            # Stream relevant lines to WebSocket
            if any(kw in decoded for kw in ["mode", "-----", "Refining", "Writing"]):
                await log_fn(job_id, f"[Vina] {decoded}", "info")

        await asyncio.wait_for(proc.wait(), timeout=300)
        return parse_vina_output(output_lines, name)

    except asyncio.TimeoutError:
        await log_fn(job_id, f"[WARN] Vina timed out for {name}", "warn")
        return []
    except FileNotFoundError:
        await log_fn(job_id, f"[ERROR] Vina binary not found at '{VINA_BINARY}'. Install AutoDock Vina.", "warn")
        return []
    except Exception as e:
        logger.error(f"Vina error for {name}: {e}")
        return []


def parse_vina_output(lines: List[str], name: str) -> List[dict]:
    """
    Parse Vina stdout table:
       mode |   affinity | dist from best mode
            | (kcal/mol) | rmsd l.b.| rmsd u.b.
    -----+------------+----------+----------
       1         -9.1      0.000      0.000
    """
    poses = []
    in_table = False
    for line in lines:
        if "-----" in line and "mode" not in line:
            in_table = True
            continue
        if in_table:
            parts = line.split()
            if len(parts) >= 4:
                try:
                    mode = int(parts[0])
                    affinity = float(parts[1])
                    rmsd_lb = float(parts[2])
                    rmsd_ub = float(parts[3])
                    poses.append({
                        "name": name,
                        "rank": mode,
                        "score": affinity,
                        "score_2": None,
                        "rmsd_lb": rmsd_lb,
                        "rmsd_ub": rmsd_ub,
                        "admet_status": None,
                    })
                except (ValueError, IndexError):
                    pass
    return poses


async def write_combined_sdf(poses: List[dict], output_path: Path):
    """Write a simple SDF record for each pose (placeholder atoms)."""
    lines = []
    for p in poses:
        lines.append(f"\n  NexusMD   3D\n\n")
        lines.append("  0  0  0  0  0  0  0  0  0  0999 V2000\n")
        lines.append("M  END\n")
        lines.append(f"> <COMPOUND_NAME>\n{p['name']}\n\n")
        lines.append(f"> <DOCKING_SCORE>\n{p['score']}\n\n")
        lines.append(f"> <RMSD_LB>\n{p.get('rmsd_lb', 0)}\n\n")
        lines.append("$$$$\n")
    output_path.write_text("".join(lines))


async def prepare_receptor_pdbqt(pdb_id: str, log_fn) -> Optional[Path]:
    """
    Download PDB and convert to PDBQT using Open Babel.
    Returns path to PDBQT file, or None on failure.
    """
    pdb_path = PDB_CACHE_DIR / f"{pdb_id}.pdb"
    pdbqt_path = PDB_CACHE_DIR / f"{pdb_id}.pdbqt"

    if pdbqt_path.exists():
        await log_fn(None, f"[INFO] Using cached receptor: {pdb_id}", "info")
        return pdbqt_path

    # Download PDB
    await log_fn(None, f"[INFO] Downloading PDB: {pdb_id}", "info")
    import httpx
    try:
        async with httpx.AsyncClient(timeout=30) as client:
            r = await client.get(
                f"https://files.rcsb.org/download/{pdb_id}.pdb",
                follow_redirects=True,
            )
            if r.status_code != 200:
                return None
            pdb_path.write_bytes(r.content)
    except Exception as e:
        logger.error(f"PDB download failed for {pdb_id}: {e}")
        return None

    # Convert to PDBQT
    try:
        result = await asyncio.create_subprocess_exec(
            OBABEL_BINARY,
            str(pdb_path),
            "-O", str(pdbqt_path),
            "--partialcharge", "gasteiger",
            "-xr",  # remove non-receptor atoms
            stdout=asyncio.subprocess.PIPE,
            stderr=asyncio.subprocess.PIPE,
        )
        await asyncio.wait_for(result.communicate(), timeout=60)
        if pdbqt_path.exists() and pdbqt_path.stat().st_size > 0:
            return pdbqt_path
    except Exception as e:
        logger.error(f"Receptor PDBQT conversion failed: {e}")

    return None
