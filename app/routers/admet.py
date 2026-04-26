"""NexusMD — ADMET Router"""
from fastapi import APIRouter, HTTPException
from app.models.schemas import ADMETRequest, ADMETResponse
from app.services.admet_service import predict_admet_batch

router = APIRouter()

@router.post("/predict", response_model=ADMETResponse)
async def predict_admet(req: ADMETRequest):
    if not req.smiles:
        raise HTTPException(400, "No SMILES provided")
    names = req.names or [f"Compound-{i+1}" for i in range(len(req.smiles))]
    results = await predict_admet_batch(req.smiles, names)
    source = results[0].get("source", "local_rules") if results else "local_rules"
    return ADMETResponse(results=results, source=source)
