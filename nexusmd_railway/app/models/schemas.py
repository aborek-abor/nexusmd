"""
NexusMD — Pydantic Schemas
All request / response models for the API
"""

from typing import Any, Dict, List, Optional
from pydantic import BaseModel, Field


# ── Health ────────────────────────────────────────
class HealthResponse(BaseModel):
    status: str
    vina: bool
    obabel: bool = False
    redis: bool
    version: str
    environment: str = "local"
    timestamp: float = 0.0


# ── Docking ───────────────────────────────────────
class GridConfig(BaseModel):
    center_x: float = 0.0
    center_y: float = 0.0
    center_z: float = 0.0
    size_x: float = 22.0
    size_y: float = 22.0
    size_z: float = 22.0
    exhaustiveness: int = Field(8, ge=1, le=64)
    num_modes: int = Field(9, ge=1, le=20)
    energy_range: float = 3.0


class DockingRequest(BaseModel):
    protein_id: str                         # PDB ID or "UPLOAD:<filename>"
    ligand_smiles: Optional[List[str]] = [] # SMILES strings
    ligand_names: Optional[List[str]] = []  # display names
    engine: str = "vina"                    # vina | gnina
    grid: GridConfig = GridConfig()
    admet_filter: bool = True
    pharmacophore_filter: bool = False


class PoseResult(BaseModel):
    rank: int
    name: str
    score: float                 # kcal/mol
    score_2: Optional[float]
    rmsd_lb: Optional[float]
    rmsd_ub: Optional[float]
    admet_status: Optional[str]


class DockingResult(BaseModel):
    job_id: str
    protein: str
    engine: str
    poses: List[PoseResult]
    elapsed_s: float
    sdf_url: Optional[str]
    pdbqt_url: Optional[str]


class JobStatus(BaseModel):
    job_id: str
    status: str          # queued | running | done | failed
    progress: int        # 0-100
    message: Optional[str]
    result: Optional[DockingResult]
    created_at: float
    updated_at: float


# ── Protein ───────────────────────────────────────
class ProteinInfo(BaseModel):
    pdb_id: str
    title: Optional[str]
    method: Optional[str]
    resolution_a: Optional[float]
    organism: Optional[str]
    chain_ids: Optional[List[str]]
    ligands: Optional[List[str]]
    pdb_url: str
    source: str = "RCSB"


class AlphaFoldInfo(BaseModel):
    uniprot_id: str
    description: Optional[str]
    mean_plddt: Optional[float]
    pdb_url: str
    source: str = "AlphaFold"


# ── ADMET ─────────────────────────────────────────
class ADMETRequest(BaseModel):
    smiles: List[str]
    names: Optional[List[str]] = []


class ADMETProperty(BaseModel):
    name: str
    smiles: str
    mw: Optional[float]
    logp: Optional[float]
    hbd: Optional[int]
    hba: Optional[int]
    tpsa: Optional[float]
    rotatable_bonds: Optional[int]
    # pkCSM predictions
    gi_absorption: Optional[str]       # "High" | "Low"
    bbb_permeant: Optional[bool]
    pgp_substrate: Optional[bool]
    cyp1a2_inhibitor: Optional[bool]
    cyp2c19_inhibitor: Optional[bool]
    cyp2c9_inhibitor: Optional[bool]
    cyp2d6_inhibitor: Optional[bool]
    cyp3a4_inhibitor: Optional[bool]
    log_kp_skin: Optional[float]
    # Toxicity
    ames_toxicity: Optional[bool]
    max_tolerated_dose: Optional[float]
    herg_inhibitor: Optional[bool]
    oral_rat_acute_ld50: Optional[float]
    hepatotoxicity: Optional[bool]
    # Lipinski
    ro5_violations: int = 0
    druglike: bool = True
    # Overall
    status: str = "Pass"            # Pass | Warn | Fail
    flags: List[str] = []


class ADMETResponse(BaseModel):
    results: List[ADMETProperty]
    source: str = "pkCSM"


# ── Pocket Detection ──────────────────────────────
class PocketRequest(BaseModel):
    pdb_id: str
    algorithm: str = "fpocket"   # fpocket | p2rank | consensus
    known_ligand: Optional[str] = None


class PocketSite(BaseModel):
    rank: int
    pocket_id: str
    druggability_score: float
    volume_a3: float
    hydrophobicity: float
    residues: List[str]
    center_x: float
    center_y: float
    center_z: float
    algorithm: str


class PocketResponse(BaseModel):
    pdb_id: str
    algorithm: str
    pockets: List[PocketSite]
    elapsed_s: float


# ── Scaffold Hopping ──────────────────────────────
class ScaffoldRequest(BaseModel):
    query_smiles: str
    threshold: float = Field(0.4, ge=0.0, le=1.0)
    fingerprint: str = "morgan2"   # morgan2 | rdkit | maccs
    search_chembl: bool = True
    search_zinc: bool = False
    max_results: int = 50


class ScaffoldHit(BaseModel):
    name: str
    smiles: Optional[str]
    tanimoto: float
    source: str
    mw: Optional[float]
    logp: Optional[float]


class ScaffoldResponse(BaseModel):
    query: str
    threshold: float
    hits: List[ScaffoldHit]
    elapsed_s: float


# ── MM-GBSA ───────────────────────────────────────
class MMGBSARequest(BaseModel):
    job_id: str          # docking job to rescore
    gb_model: str = "igb2"
    top_n: int = 20
    entropy_correction: bool = True
    md_frames: int = 50


class MMGBSAEntry(BaseModel):
    name: str
    delta_g_bind: float
    delta_evdw: float
    delta_eelec: float
    delta_ggb: float
    delta_gsa: float
    vina_score: float
    rank_change: int
    confidence: str     # High | Moderate | Low


class MMGBSAResponse(BaseModel):
    job_id: str
    results: List[MMGBSAEntry]
    top_hit: Optional[str]
    elapsed_s: float


# ── FASTA / ESMFold ──────────────────────────────
class FASTARequest(BaseModel):
    sequence: str
    header: Optional[str] = ""
    engine: str = "esm"     # esm | colabfold | omegafold
    relax: bool = True
    num_recycles: int = 3


class FASTAResult(BaseModel):
    header: str
    sequence_length: int
    mean_plddt: float
    ptm_score: Optional[float]
    pdb_string: str          # full PDB coordinates
    pdb_url: str             # download URL
    engine: str
    elapsed_s: float
