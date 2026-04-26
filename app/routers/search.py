import httpx
from fastapi import APIRouter, Query, HTTPException

router = APIRouter()

@router.get("/pdb")
async def search_pdb(q: str = Query(..., min_length=1)):
    try:
        payload = {
            "query": {"type": "terminal", "service": "full_text", "parameters": {"value": q}},
            "return_type": "entry",
            "request_options": {"paginate": {"start": 0, "rows": 8}}
        }
        async with httpx.AsyncClient(timeout=10) as client:
            r = await client.post("https://search.rcsb.org/rcsbsearch/v2/query", json=payload)
            if not r.is_success:
                raise HTTPException(status_code=502, detail="RCSB unavailable")
            hits = r.json().get("result_set", [])
            ids = [h["identifier"] for h in hits[:8]]

        results = []
        async with httpx.AsyncClient(timeout=10) as client:
            for pdb_id in ids:
                try:
                    meta = await client.get(f"https://data.rcsb.org/rest/v1/core/entry/{pdb_id}")
                    if meta.is_success:
                        d = meta.json()
                        res = d.get("refine", [{}])[0].get("ls_d_res_high")
                        results.append({
                            "id": pdb_id,
                            "title": d.get("struct", {}).get("title", pdb_id),
                            "method": d.get("exptl", [{}])[0].get("method", "X-RAY"),
                            "resolution": round(float(res), 2) if res else None
                        })
                except Exception:
                    results.append({"id": pdb_id, "title": pdb_id, "method": "-", "resolution": None})

        return {"results": results}
    except Exception as e:
        raise HTTPException(status_code=502, detail=str(e))
