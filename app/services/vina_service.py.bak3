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

    return all_poses


OBABEL_TIMEOUT = int(os.environ.get("NEXUS_OBABEL_TIMEOUT", "90"))


def _largest_fragment(smiles: str) -> str:
    """Salts/hydrates arrive as 'parent.counterion.O.O'. Keep the parent."""
    parts = [p for p in (smiles or "").split(".") if p.strip()]
    if len(parts) <= 1:
        return (smiles or "").strip()
    return max(parts, key=lambda p: (sum(c.isalpha() for c in p), len(p)))


async def _run_tool(cmd: list, timeout: int, label: str):
    """Run a subprocess and actually kill it if it overruns."""
    try:
        proc = await asyncio.create_subprocess_exec(
            *cmd,
            stdout=asyncio.subprocess.PIPE,
            stderr=asyncio.subprocess.PIPE,
        )
    except FileNotFoundError:
        return False, f"{cmd[0]} not found on PATH"
    try:
        _, stderr = await asyncio.wait_for(proc.communicate(), timeout=timeout)
        return True, (stderr or b"").decode("utf-8", errors="replace")
    except asyncio.TimeoutError:
        try:
            proc.kill()
            await proc.wait()
        except Exception:
            pass
        return False, f"{label} exceeded {timeout}s and was killed"


OBABEL_TIMEOUT = int(os.environ.get("NEXUS_OBABEL_TIMEOUT", "90"))


def _largest_fragment(smiles: str) -> str:
    """Salts/hydrates arrive as 'parent.counterion.O.O'. Keep the parent."""
    parts = [p for p in (smiles or "").split(".") if p.strip()]
    if len(parts) <= 1:
        return (smiles or "").strip()
    return max(parts, key=lambda p: (sum(c.isalpha() for c in p), len(p)))


async def _run_tool(cmd: list, timeout: int, label: str):
    """Run a subprocess and actually kill it if it overruns."""
    try:
        proc = await asyncio.create_subprocess_exec(
            *cmd,
            stdout=asyncio.subprocess.PIPE,
            stderr=asyncio.subprocess.PIPE,
        )
    except FileNotFoundError:
        return False, f"{cmd[0]} not found on PATH"
    try:
        _, stderr = await asyncio.wait_for(proc.communicate(), timeout=timeout)
        return True, (stderr or b"").decode("utf-8", errors="replace")
    except asyncio.TimeoutError:
        try:
            proc.kill()
            await proc.wait()
        except Exception:
            pass
        return False, f"{label} exceeded {timeout}s and was killed"


async def smiles_to_3d_sdf(smiles: str, output_path: Path, name: str = "LIG") -> bool:
    """SMILES -> 3D SDF. Desalts, then escalates gen3d quality only if needed."""
    parent = _largest_fragment(smiles)
    if not parent:
        logger.error("no usable SMILES for %s", name)
        return False
    if parent != (smiles or "").strip():
        logger.info("desalted %s: %s -> %s", name, smiles, parent)

    # cheapest first; a version that rejects the speed argument falls through
    attempts = [
        (["--gen3d", "fastest"], min(OBABEL_TIMEOUT, 60)),
        (["--gen3d", "fast"],    min(OBABEL_TIMEOUT, 90)),
        (["--gen3d"],            OBABEL_TIMEOUT),
    ]
    last_err = ""
    for flags, timeout in attempts:
        if output_path.exists():
            try:
                output_path.unlink()
            except Exception:
                pass
        cmd = [OBABEL_BINARY, f"-:{parent}", "-O", str(output_path)] + flags + [
            "-p", "7.4", "--title", name,
        ]
        ok, err = await _run_tool(cmd, timeout, f"obabel {' '.join(flags)}")
        if output_path.exists() and output_path.stat().st_size > 0:
            return True
        last_err = (err or "no output produced").strip()

    logger.error("SMILES->SDF failed for %s (%s): %s", name, parent[:60], last_err[:300])
    return False


async def sdf_to_pdbqt(sdf_path: Path, pdbqt_path: Path) -> bool:
    """SDF -> PDBQT with Gasteiger charges."""
    cmd = [
        OBABEL_BINARY,
        str(sdf_path),
        "-O", str(pdbqt_path),
        "--partialcharge", "gasteiger",
    ]
    ok, err = await _run_tool(cmd, min(OBABEL_TIMEOUT, 60), "obabel sdf->pdbqt")
    if pdbqt_path.exists() and pdbqt_path.stat().st_size > 0:
        return True
    logger.error("SDF->PDBQT failed for %s: %s", sdf_path.name, (err or "no output").strip()[:300])
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
    """Convert docked PDBQT poses to SDF using Open Babel for real 3D coordinates."""
    job_dir = output_path.parent
    combined_lines = []

    for i, p in enumerate(poses):
        out_pdbqt = job_dir / f"out_{i}.pdbqt"
        if out_pdbqt.exists():
            # Convert best pose PDBQT to SDF using Open Babel
            tmp_sdf = job_dir / f"pose_{i}.sdf"
            try:
                result = await asyncio.create_subprocess_exec(
                    OBABEL_BINARY, str(out_pdbqt),
                    "-O", str(tmp_sdf),
                    stdout=asyncio.subprocess.PIPE,
                    stderr=asyncio.subprocess.PIPE,
                )
                await asyncio.wait_for(result.communicate(), timeout=15)
                if tmp_sdf.exists() and tmp_sdf.stat().st_size > 10:
                    sdf_content = tmp_sdf.read_text()
                    # Add metadata fields
                    sdf_content = sdf_content.rstrip().rstrip("$$$$").rstrip()
                    sdf_content += f"\n> <COMPOUND_NAME>\n{p['name']}\n\n"
                    sdf_content += f"> <DOCKING_SCORE>\n{p['score']}\n\n"
                    sdf_content += f"> <RMSD_LB>\n{p.get('rmsd_lb', 0)}\n\n"
                    sdf_content += "$$$$\n"
                    combined_lines.append(sdf_content)
                    continue
            except Exception:
                pass

        # Fallback: write minimal SDF with metadata only
        combined_lines.append(
            f"{p['name']}\n  NexusMD\n\n"
            "  0  0  0  0  0  0  0  0  0  0999 V2000\n"
            f"M  END\n> <COMPOUND_NAME>\n{p['name']}\n\n"
            f"> <DOCKING_SCORE>\n{p['score']}\n\n"
            f"> <RMSD_LB>\n{p.get('rmsd_lb', 0)}\n\n"
            "$$$$\n"
        )

    output_path.write_text("".join(combined_lines))


async def prepare_receptor_pdbqt(pdb_id: str, log_fn) -> Optional[Path]:
    """Download a structure and convert it to PDBQT.

    Accepts 4-character RCSB codes and UniProt accessions (served by AlphaFold).
    Returns None only after saying exactly why, never silently.
    """
    ident = (pdb_id or "").strip().upper()
    if not ident:
        await log_fn(None, "[ERROR] No protein identifier supplied", "warn")
        return None

    PDB_CACHE_DIR.mkdir(parents=True, exist_ok=True)
    pdb_path = PDB_CACHE_DIR / f"{ident}.pdb"
    pdbqt_path = PDB_CACHE_DIR / f"{ident}.pdbqt"

    if pdbqt_path.exists() and pdbqt_path.stat().st_size > 0:
        await log_fn(None, f"[INFO] Using cached receptor: {ident}", "info")
        return pdbqt_path

    if len(ident) == 4:
        sources = [(f"https://files.rcsb.org/download/{ident}.pdb", "RCSB")]
    else:
        sources = [
            (f"https://alphafold.ebi.ac.uk/files/AF-{ident}-F1-model_v4.pdb", "AlphaFold v4"),
            (f"https://alphafold.ebi.ac.uk/files/AF-{ident}-F1-model_v3.pdb", "AlphaFold v3"),
        ]

    import httpx

    content = None
    for url, label in sources:
        try:
            await log_fn(None, f"[INFO] Fetching {ident} from {label}", "info")
            async with httpx.AsyncClient(timeout=60) as client:
                r = await client.get(url, follow_redirects=True)
            if r.status_code == 200 and b"ATOM" in r.content:
                content = r.content
                await log_fn(None, f"[INFO] {label} returned {len(content)//1024} KB", "info")
                break
            await log_fn(None, f"[WARN] {label} returned HTTP {r.status_code} for {ident}", "warn")
            logger.error("structure fetch %s -> HTTP %s", url, r.status_code)
        except Exception as e:
            await log_fn(None, f"[WARN] {label} fetch failed for {ident}: {e}", "warn")
            logger.error("structure fetch %s failed: %s", url, e)

    if content is None:
        await log_fn(
            None,
            f"[ERROR] No structure available for '{ident}'. RCSB needs a 4-character "
            "PDB code; AlphaFold needs a UniProt accession such as P19484.",
            "warn",
        )
        return None

    pdb_path.write_bytes(content)

    # keep protein atoms only: no waters, no ligands, first model of an ensemble
    clean_path = PDB_CACHE_DIR / f"{ident}_clean.pdb"
    kept = 0
    with pdb_path.open() as src, clean_path.open("w") as dst:
        for line in src:
            if line.startswith(("ATOM", "TER")):
                dst.write(line)
                kept += 1
            elif line.startswith("ENDMDL"):
                break
    if kept < 50:
        await log_fn(None, f"[ERROR] {ident} has only {kept} protein atoms after cleaning", "warn")
        return None

    try:
        proc = await asyncio.create_subprocess_exec(
            OBABEL_BINARY,
            str(clean_path),
            "-O", str(pdbqt_path),
            "--partialcharge", "gasteiger",
            "-xr",
            stdout=asyncio.subprocess.PIPE,
            stderr=asyncio.subprocess.PIPE,
        )
        _, stderr = await asyncio.wait_for(proc.communicate(), timeout=180)
        if pdbqt_path.exists() and pdbqt_path.stat().st_size > 0:
            await log_fn(
                None,
                f"[INFO] Receptor ready: {pdbqt_path.name} "
                f"({pdbqt_path.stat().st_size // 1024} KB, {kept} atoms)",
                "info",
            )
            return pdbqt_path
        detail = (stderr or b"").decode(errors="replace")[:300]
        await log_fn(None, f"[ERROR] Open Babel produced no PDBQT: {detail}", "warn")
        logger.error("obabel produced no pdbqt for %s: %s", ident, detail)
    except asyncio.TimeoutError:
        await log_fn(None, "[ERROR] Receptor conversion timed out after 180s", "warn")
        logger.error("obabel timeout for %s", ident)
    except Exception as e:
        await log_fn(None, f"[ERROR] Receptor conversion failed: {e}", "warn")
        logger.error("receptor conversion failed for %s: %s", ident, e)

    return None

