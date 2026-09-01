"""
NexusMD — MD package builder.

Assembles everything needed to simulate one docked pose elsewhere: the
complex, the protein, the ligand with real bond orders, the docking
parameters it came from, and scripts that run and analyse the simulation.

The ligand is rebuilt from the Vina output PDBQT with Meeko, which recovers
bond orders and formal charges from the REMARK SMILES that Meeko itself
wrote during preparation. That matters: a ligand read back from PDB
coordinates alone has no bond orders, and a force field cannot parameterise
what it cannot perceive.

Install: save as app/services/md_package.py
Templates live in app/templates/md/.
"""

import io
import json
import logging
import zipfile
from datetime import datetime, timezone
from pathlib import Path
from typing import Optional

from app.services.complex_export import build_complex, extract_pose

logger = logging.getLogger("nexusmd.md")

TEMPLATE_DIR = Path(__file__).parent.parent / "templates" / "md"
SCRIPTS = ("run_md.py", "analyse.py", "README.md", "NexusMD_MD.ipynb")


class PackageError(RuntimeError):
    pass


def pose_to_sdf(pose_pdbqt: Path, mode: int = 1, name: str = "LIG") -> str:
    """Rebuild the docked pose as an SDF with correct bond orders."""
    try:
        from meeko import PDBQTMolecule, RDKitMolCreate
        from rdkit import Chem
    except ImportError as e:
        raise PackageError(f"Meeko/RDKit unavailable: {e}")

    try:
        pmol = PDBQTMolecule.from_file(str(pose_pdbqt), skip_typing=True)
        mols = RDKitMolCreate.from_pdbqt_mol(pmol)
    except Exception as e:
        raise PackageError(f"could not rebuild ligand from {pose_pdbqt.name}: {e}")

    if not mols or mols[0] is None:
        raise PackageError(f"Meeko returned no molecule for {pose_pdbqt.name}")

    mol = mols[0]
    conf_id = mode - 1
    if conf_id >= mol.GetNumConformers():
        conf_id = 0
        logger.warning("pose mode %s missing, exporting mode 1", mode)

    mol.SetProp("_Name", name)
    buf = io.StringIO()
    writer = Chem.SDWriter(buf)
    writer.write(mol, confId=conf_id)
    writer.close()
    return buf.getvalue()


def _receptor_only(receptor_pdb: Path) -> str:
    keep = [l for l in receptor_pdb.read_text().splitlines()
            if l.startswith(("ATOM", "TER"))]
    return "\n".join(keep) + "\nEND\n"


def build_md_package(
    receptor_pdb: Path,
    pose_pdbqt: Path,
    ligand_name: str,
    mode: int = 1,
    job_id: str = "",
    protein_id: str = "",
    score: Optional[float] = None,
    grid: Optional[dict] = None,
    engine: str = "AutoDock Vina",
) -> bytes:
    """Return the package as zip bytes."""
    missing = [s for s in SCRIPTS if not (TEMPLATE_DIR / s).exists()]
    if missing:
        raise PackageError(f"missing templates in {TEMPLATE_DIR}: {', '.join(missing)}")

    complex_pdb = build_complex(
        receptor_pdb, pose_pdbqt, mode=mode, ligand_name=ligand_name,
        header_note=f"job {job_id} target {protein_id}",
    )
    ligand_sdf = pose_to_sdf(pose_pdbqt, mode=mode, name=ligand_name)
    _, affinity = extract_pose(pose_pdbqt, mode)

    meta = {
        "generated": datetime.now(timezone.utc).isoformat(timespec="seconds"),
        "job_id": job_id,
        "protein": protein_id,
        "ligand": ligand_name,
        "pose_mode": mode,
        "vina_affinity_kcal_mol": affinity if affinity is not None else score,
        "engine": engine,
        "grid": grid or {},
        "note": (
            "Coordinates are the docked pose exactly as scored. The ligand SDF "
            "carries bond orders recovered by Meeko from the preparation SMILES."
        ),
    }

    safe = "".join(c if c.isalnum() or c in "-_" else "_" for c in ligand_name)[:40] or "ligand"
    root = f"nexusmd_md_{safe}_mode{mode}"

    buf = io.BytesIO()
    with zipfile.ZipFile(buf, "w", zipfile.ZIP_DEFLATED) as z:
        z.writestr(f"{root}/complex.pdb", complex_pdb)
        z.writestr(f"{root}/protein.pdb", _receptor_only(receptor_pdb))
        z.writestr(f"{root}/ligand.sdf", ligand_sdf)
        z.writestr(f"{root}/docking.json", json.dumps(meta, indent=2))
        for script in SCRIPTS:
            z.writestr(f"{root}/{script}", (TEMPLATE_DIR / script).read_text())
    return buf.getvalue()
