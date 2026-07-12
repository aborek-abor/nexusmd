"""
NexusMD — Molecular Dynamics Pydantic Schemas
Request / response models for the MD simulation API.
"""

from typing import Any, Dict, List, Optional
from pydantic import BaseModel, Field, model_validator


class MDRequest(BaseModel):
    docking_job_id: Optional[str] = Field(
        default=None,
        description="Job ID of the completed docking run to use as starting structure"
    )
    complex_id: Optional[str] = Field(
        default=None,
        description=(
            "ID of an uploaded protein-ligand complex (from POST /upload-complex). "
            "Format: 'UPLOAD:<timestamp>_<filename>'. "
            "Provide either docking_job_id OR complex_id, not both."
        )
    )
    duration_ns: float = Field(
        ..., gt=0, le=200, description="Simulation length in nanoseconds (max 200 ns)"
    )
    temperature_K: float = Field(
        default=310.0, ge=250, le=400,
        description="Simulation temperature in Kelvin (physiological default: 310 K)"
    )
    force_field: str = Field(
        default="AMBER14",
        description="Force field to apply: AMBER14, CHARMM36, or OPLS-AA"
    )
    solvation: str = Field(
        default="explicit",
        description="Solvation model: 'explicit' (TIP3P water) or 'implicit' (OBC2)"
    )
    timestep_fs: float = Field(
        default=2.0, ge=0.5, le=4.0,
        description="Integration timestep in femtoseconds"
    )
    padding_angstrom: float = Field(
        default=10.0, ge=5, le=20,
        description="Water box padding around the solute in Ångströms (explicit solvation only)"
    )

    @model_validator(mode="after")
    def _require_one_source(self) -> "MDRequest":
        has_docking = bool(self.docking_job_id)
        has_complex = bool(self.complex_id)
        if not has_docking and not has_complex:
            raise ValueError(
                "Provide either 'docking_job_id' (docking-based MD) "
                "or 'complex_id' (uploaded complex MD)."
            )
        if has_docking and has_complex:
            raise ValueError(
                "Provide either 'docking_job_id' or 'complex_id', not both."
            )
        return self


class EnergyStats(BaseModel):
    mean: float
    std: float
    min: float
    max: float


class MDAnalysis(BaseModel):
    rmsd_ligand: EnergyStats
    rmsd_protein: EnergyStats
    potential_energy: EnergyStats
    kinetic_energy: EnergyStats
    total_energy: EnergyStats
    # Timeseries: list of [time_ps, potential_kJ, kinetic_kJ]
    energy_timeseries: List[List[float]]
    # Timeseries: list of [time_ps, rmsd_ligand_A, rmsd_protein_A]
    rmsd_timeseries: List[List[float]]


class MDResult(BaseModel):
    job_id: str
    docking_job_id: Optional[str] = None   # set for docking-based MD jobs
    complex_id: Optional[str] = None        # set for uploaded-complex MD jobs
    duration_ns: float
    temperature_K: float
    force_field: str
    solvation: str
    elapsed_s: float
    trajectory_url: str
    analysis: Dict[str, Any]  # MDAnalysis-shaped dict with energy/RMSD stats and timeseries
