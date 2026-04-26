"""NexusMD — Scaffold Hopping Router (RDKit Tanimoto)"""
import time, logging
from fastapi import APIRouter, HTTPException
from app.models.schemas import ScaffoldRequest, ScaffoldResponse, ScaffoldHit

router = APIRouter()
logger = logging.getLogger("nexusmd.scaffold")

@router.post("/search", response_model=ScaffoldResponse)
async def scaffold_search(req: ScaffoldRequest):
    start = time.time()
    hits = await _tanimoto_search(req.query_smiles, req.threshold, req.fingerprint, req.max_results)
    return ScaffoldResponse(
        query=req.query_smiles, threshold=req.threshold,
        hits=hits, elapsed_s=round(time.time() - start, 2),
    )

async def _tanimoto_search(query_smi: str, threshold: float, fp_type: str, max_n: int):
    """Use RDKit if available; fall back to string-similarity."""
    try:
        from rdkit import Chem
        from rdkit.Chem import AllChem, DataStructs
        from app.data.compound_library import LIBRARY_SMILES

        mol = Chem.MolFromSmiles(query_smi)
        if mol is None:
            return []
        if fp_type == "morgan2":
            query_fp = AllChem.GetMorganFingerprintAsBitVect(mol, 2, nBits=2048)
        else:
            from rdkit.Chem import RDKFingerprint
            query_fp = RDKFingerprint(mol)

        hits = []
        for entry in LIBRARY_SMILES:
            try:
                m = Chem.MolFromSmiles(entry["smiles"])
                if m is None: continue
                if fp_type == "morgan2":
                    fp = AllChem.GetMorganFingerprintAsBitVect(m, 2, nBits=2048)
                else:
                    fp = RDKFingerprint(m)
                sim = DataStructs.TanimotoSimilarity(query_fp, fp)
                if sim >= threshold:
                    hits.append(ScaffoldHit(
                        name=entry["name"], smiles=entry["smiles"],
                        tanimoto=round(sim, 4), source=entry["source"],
                        mw=entry.get("mw"), logp=entry.get("logp"),
                    ))
            except Exception:
                pass
        hits.sort(key=lambda h: -h.tanimoto)
        return hits[:max_n]

    except ImportError:
        logger.warning("RDKit not installed — using string similarity fallback")
        return _string_similarity_fallback(query_smi, threshold, max_n)

def _string_similarity_fallback(query: str, threshold: float, max_n: int):
    """Jaccard similarity on SMILES character n-grams as fallback."""
    from app.data.compound_library import LIBRARY_SMILES
    def ngrams(s, n=3):
        return set(s[i:i+n] for i in range(len(s)-n+1))
    qng = ngrams(query)
    hits = []
    for entry in LIBRARY_SMILES:
        smi = entry.get("smiles", "")
        eng = ngrams(smi)
        if not qng or not eng: continue
        sim = len(qng & eng) / len(qng | eng)
        if sim >= threshold:
            hits.append(ScaffoldHit(
                name=entry["name"], smiles=smi,
                tanimoto=round(sim, 4), source=entry["source"],
                mw=entry.get("mw"), logp=entry.get("logp"),
            ))
    hits.sort(key=lambda h: -h.tanimoto)
    return hits[:max_n]
