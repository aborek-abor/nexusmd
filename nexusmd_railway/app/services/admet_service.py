"""
NexusMD — ADMET Service
Real pkCSM API integration + SwissADME-inspired local rules fallback.
pkCSM: https://biosig.lab.uq.edu.au/pkcsm/prediction
"""

import asyncio
import logging
import math
import re
import time
from typing import Dict, List, Optional

import httpx

logger = logging.getLogger("nexusmd.admet")

PKCSM_URL = "https://biosig.lab.uq.edu.au/pkcsm/prediction"
SWISSADME_URL = "https://swissadme.ch/swissadme/result.php"


async def predict_admet_batch(smiles_list: List[str], names: List[str]) -> List[dict]:
    """
    Predict ADMET for a list of SMILES.
    Tries pkCSM API first; falls back to local rule-based model.
    """
    results = []
    for smi, name in zip(smiles_list, names):
        result = await predict_single(smi, name)
        results.append(result)
        await asyncio.sleep(0.3)   # be polite to API
    return results


async def predict_single(smiles: str, name: str) -> dict:
    """Try pkCSM; fall back to local rules."""
    # Try pkCSM first
    pkcsm = await _query_pkcsm(smiles)
    if pkcsm:
        return _format_pkcsm(pkcsm, smiles, name)

    # Local rule-based fallback (SwissADME-inspired)
    return _local_admet(smiles, name)


async def _query_pkcsm(smiles: str) -> Optional[dict]:
    """Query pkCSM web API."""
    payload = {
        "smiles": smiles,
        "server": "pkCSM",
    }
    try:
        async with httpx.AsyncClient(timeout=15) as client:
            # pkCSM expects a form POST
            r = await client.post(
                PKCSM_URL,
                data={"smiles": smiles},
                headers={"Accept": "application/json"},
            )
            if r.status_code == 200:
                data = r.json()
                if isinstance(data, list) and data:
                    return data[0]
                if isinstance(data, dict):
                    return data
    except Exception as e:
        logger.debug(f"pkCSM query failed: {e}")
    return None


def _format_pkcsm(data: dict, smiles: str, name: str) -> dict:
    """Map pkCSM API response to our schema."""
    props = _calc_physicochemical(smiles)
    flags = []

    gi = data.get("Absorption_Caco2_permeability", 0)
    bbb = data.get("Distribution_BBB_permeability", -99)
    herg = data.get("Toxicity_hERG_inhibitor", False)
    ames = data.get("Toxicity_AMES_toxicity", False)
    hep = data.get("Toxicity_Hepatotoxicity", False)

    gi_abs = "High" if gi > -5.15 else "Low"
    bbb_perm = bbb > -1.0
    herg_inh = bool(herg)
    ames_tox = bool(ames)

    if herg_inh: flags.append("hERG inhibitor")
    if ames_tox: flags.append("Ames mutagenic")
    if hep: flags.append("Hepatotoxic")
    if gi_abs == "Low": flags.append("Low GI absorption")
    if props["ro5_violations"] >= 2: flags.append(f"{props['ro5_violations']} Ro5 violations")

    status = "Fail" if (herg_inh or ames_tox or props["ro5_violations"] >= 3) else \
             "Warn" if (props["ro5_violations"] >= 1 or hep or gi_abs == "Low") else "Pass"

    return {
        "name": name, "smiles": smiles, **props,
        "gi_absorption": gi_abs, "bbb_permeant": bbb_perm,
        "pgp_substrate": bool(data.get("Absorption_P-glycoprotein_substrate", False)),
        "cyp1a2_inhibitor": bool(data.get("Metabolism_CYP1A2_inhibitor", False)),
        "cyp2c19_inhibitor": bool(data.get("Metabolism_CYP2C19_inhibitor", False)),
        "cyp2c9_inhibitor": bool(data.get("Metabolism_CYP2C9_inhibitor", False)),
        "cyp2d6_inhibitor": bool(data.get("Metabolism_CYP2D6_inhibitor", False)),
        "cyp3a4_inhibitor": bool(data.get("Metabolism_CYP3A4_inhibitor", False)),
        "log_kp_skin": data.get("Absorption_Skin_permeability", None),
        "ames_toxicity": ames_tox,
        "max_tolerated_dose": data.get("Toxicity_Maximum_tolerated_dose_(human)", None),
        "herg_inhibitor": herg_inh,
        "oral_rat_acute_ld50": data.get("Toxicity_Oral_Rat_Acute_LD50", None),
        "hepatotoxicity": bool(hep),
        "status": status, "flags": flags, "source": "pkCSM",
    }


def _local_admet(smiles: str, name: str) -> dict:
    """
    Local rule-based ADMET prediction.
    Based on Lipinski Ro5, Veber rules, and toxicophore patterns.
    More reliable than random — uses actual SMILES analysis.
    """
    props = _calc_physicochemical(smiles)
    mw = props["mw"]
    logp = props["logp"]
    hbd = props["hbd"]
    hba = props["hba"]
    tpsa = props["tpsa"]
    rotb = props["rotatable_bonds"]
    flags = []

    # GI absorption — Veber rules: TPSA ≤ 140, rotatable bonds ≤ 10
    gi_absorption = "High" if (tpsa <= 140 and rotb <= 10 and mw <= 500) else "Low"

    # BBB — small, lipophilic, low TPSA
    bbb_permeant = (mw < 400 and logp > 1 and tpsa < 90 and hbd <= 3)

    # P-gp substrate — large, many H-bond features
    pgp_substrate = (mw > 400 and hba > 6)

    # CYP inhibition patterns (rough rule-based)
    has_imidazole = "n1ccnc1" in smiles.lower() or "C1=CN=CN=C1" in smiles
    has_quinoline = "c1ccc2ncccc2c1" in smiles.lower()
    cyp3a4_inhibitor = (logp > 4 or mw > 500 or has_imidazole)
    cyp2d6_inhibitor = (hbd <= 1 and logp > 2 and _count_nitrogen(smiles) >= 1)
    cyp2c9_inhibitor = _has_acidic_group(smiles)
    cyp1a2_inhibitor = (has_quinoline or _count_aromatic_rings(smiles) >= 3)
    cyp2c19_inhibitor = (logp > 3 and mw < 350)

    # hERG — planar aromatics + basic nitrogen
    herg_inhibitor = (_count_aromatic_rings(smiles) >= 2 and
                      _count_nitrogen(smiles) >= 1 and logp > 2)

    # Ames — check toxicophore SMARTS patterns
    ames_toxicity = _check_ames_patterns(smiles)

    # Hepatotoxicity — large, lipophilic, reactive
    hepatotoxicity = (mw > 500 or (logp > 5 and tpsa < 60))

    # Skin permeability (LogKp estimate: Potts-Guy model)
    log_kp = 0.71 * logp - 0.0061 * mw - 6.3

    if herg_inhibitor: flags.append("hERG inhibitor risk")
    if ames_toxicity: flags.append("Ames mutagenic risk")
    if hepatotoxicity: flags.append("Hepatotoxicity risk")
    if gi_absorption == "Low": flags.append("Low GI absorption")
    if props["ro5_violations"] >= 2: flags.append(f"{props['ro5_violations']} Lipinski violations")
    if pgp_substrate: flags.append("P-gp substrate")
    if bbb_permeant: flags.append("BBB penetrant")

    status = "Fail" if (herg_inhibitor or ames_toxicity or props["ro5_violations"] >= 3) else \
             "Warn" if (props["ro5_violations"] >= 1 or hepatotoxicity or gi_absorption == "Low") else "Pass"

    return {
        "name": name, "smiles": smiles, **props,
        "gi_absorption": gi_absorption, "bbb_permeant": bbb_permeant,
        "pgp_substrate": pgp_substrate,
        "cyp1a2_inhibitor": cyp1a2_inhibitor, "cyp2c19_inhibitor": cyp2c19_inhibitor,
        "cyp2c9_inhibitor": cyp2c9_inhibitor, "cyp2d6_inhibitor": cyp2d6_inhibitor,
        "cyp3a4_inhibitor": cyp3a4_inhibitor,
        "log_kp_skin": round(log_kp, 2),
        "ames_toxicity": ames_toxicity,
        "max_tolerated_dose": None,
        "herg_inhibitor": herg_inhibitor,
        "oral_rat_acute_ld50": None,
        "hepatotoxicity": hepatotoxicity,
        "status": status, "flags": flags, "source": "local_rules",
    }


def _calc_physicochemical(smiles: str) -> dict:
    """Calculate basic physicochemical properties from SMILES."""
    # Atom-count based MW estimation
    atom_weights = {'C':12,'N':14,'O':16,'S':32,'F':19,'Cl':35.5,'Br':80,'I':127,'P':31}
    mw = sum(atom_weights.get(c, 12) for c in smiles if c.isalpha() and c.isupper())
    # H count estimation: 2*(nC) + 2 + nN - nRings*2
    ring_count = smiles.count('1') + smiles.count('2') + smiles.count('3')
    mw += max(0, smiles.upper().count('C') + smiles.upper().count('N') - ring_count) * 1

    # logP: XLogP-style rough estimate
    logp = round(
        0.53 * smiles.upper().count('C') -
        0.26 * smiles.upper().count('N') -
        0.36 * smiles.upper().count('O') -
        0.15 * smiles.upper().count('S') +
        0.12 * smiles.upper().count('F') +
        0.20 * smiles.upper().count('CL') -
        0.5 * smiles.count('=O') -
        0.3, 2)

    # H-bond donors: OH + NH
    hbd = len(re.findall(r'[ON]H|[Nn]H', smiles))
    # H-bond acceptors: O + N (aromatic/non-aromatic)
    hba = len(re.findall(r'[NOno]', smiles))
    # TPSA estimate
    tpsa = round(hbd * 20.2 + hba * 13.1, 1)
    tpsa = min(tpsa, 300)
    # Rotatable bonds estimate
    rotb = max(0, len(re.findall(r'[^=][CNSO][^=]', smiles)) - 3)

    # Ro5 violations
    viol = sum([mw > 500, logp > 5, hbd > 5, hba > 10])

    return {
        "mw": round(mw, 1), "logp": logp,
        "hbd": hbd, "hba": hba, "tpsa": tpsa,
        "rotatable_bonds": rotb, "ro5_violations": viol,
        "druglike": viol <= 1,
    }


def _count_nitrogen(smiles: str) -> int:
    return len(re.findall(r'[Nn]', smiles))

def _count_aromatic_rings(smiles: str) -> int:
    return len(re.findall(r'c1[a-z]+1', smiles))

def _has_acidic_group(smiles: str) -> bool:
    return bool(re.search(r'C\(=O\)O|S\(=O\)\(=O\)O|P\(=O\)', smiles))

def _check_ames_patterns(smiles: str) -> bool:
    """Check for known mutagenic toxicophore patterns."""
    MUTAGENIC_PATTERNS = [
        r'N=O',           # nitroso
        r'N\(=O\)=O',    # nitro
        r'C=O.*C=O',     # 1,2-dicarbonyl
        r'\[N\+\]',       # quaternary N
        r'c1ccncc1',     # pyridinium-like
        r'Cl.*Cl',        # polychlorinated
    ]
    return any(re.search(p, smiles) for p in MUTAGENIC_PATTERNS)
