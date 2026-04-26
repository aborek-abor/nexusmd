"""NexusMD — Protein Router"""
import time
from fastapi import APIRouter, HTTPException
from fastapi.responses import PlainTextResponse
from app.services.protein_service import (
    fetch_pdb_info, fetch_pdb_file_content,
    fetch_alphafold_info, fetch_alphafold_pdb_content,
)

router = APIRouter()

@router.get("/{pdb_id}/info")
async def get_pdb_info(pdb_id: str):
    info = await fetch_pdb_info(pdb_id.upper())
    if not info:
        raise HTTPException(404, f"PDB entry {pdb_id} not found")
    return info

@router.get("/{pdb_id}/pdb", response_class=PlainTextResponse)
async def get_pdb_file(pdb_id: str):
    content = await fetch_pdb_file_content(pdb_id.upper())
    if not content:
        raise HTTPException(404, f"Could not download PDB file for {pdb_id}")
    return content

@router.get("/alphafold/{uniprot_id}/info")
async def get_af_info(uniprot_id: str):
    info = await fetch_alphafold_info(uniprot_id.upper())
    if not info:
        raise HTTPException(404, f"AlphaFold entry {uniprot_id} not found")
    return info

@router.get("/alphafold/{uniprot_id}/pdb", response_class=PlainTextResponse)
async def get_af_pdb(uniprot_id: str):
    content = await fetch_alphafold_pdb_content(uniprot_id.upper())
    if not content:
        raise HTTPException(404, f"Could not download AlphaFold PDB for {uniprot_id}")
    return content
