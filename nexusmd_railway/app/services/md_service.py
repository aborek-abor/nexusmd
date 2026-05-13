"""
NexusMD — Molecular Dynamics Service
OpenMM-based MD simulation pipeline for docked ligand-protein complexes.

Workflow:
  1. Load docked complex from PDBQT (converted to PDB via pdbfixer)
  2. Apply force field and solvation
  3. Energy-minimise, equilibrate, then run production MD
  4. Save DCD trajectory and return RMSD / energy analysis
"""

import logging
import math
import statistics
import time
from pathlib import Path
from typing import Any, Callable, Dict, Optional, Tuple

logger = logging.getLogger("nexusmd.md")

# ── Force-field file maps ──────────────────────────────────────────────────
_FF_FILES: Dict[str, list] = {
    "AMBER14": ["amber14-all.xml", "amber14/tip3pfb.xml"],
    "CHARMM36": ["charmm36.xml", "charmm36/water.xml"],
    "OPLS-AA": ["oplsaa.xml"],
}

# Implicit solvent model (used when solvation == "implicit")
_IMPLICIT_SOLVENT = "OBC2"


# ── Public API ─────────────────────────────────────────────────────────────

def prepare_md_system(
    pdbqt_path: Path,
    force_field: str = "AMBER14",
    solvation: str = "explicit",
    padding_nm: float = 1.0,
    temperature_K: float = 310.0,
):
    """
    Load a ligand-protein complex from a PDBQT file, apply the chosen force
    field and solvation model, and return an (openmm.System, topology,
    positions) tuple ready for simulation.

    Parameters
    ----------
    pdbqt_path : Path
        Path to the docked complex PDBQT file.
    force_field : str
        One of AMBER14, CHARMM36, OPLS-AA.
    solvation : str
        'explicit' (TIP3P water box) or 'implicit' (OBC2 GB).
    padding_nm : float
        Water-box padding in nanometres (explicit only).
    temperature_K : float
        Temperature used to set the implicit-solvent reference (implicit only).

    Returns
    -------
    tuple : (system, topology, positions)
    """
    try:
        import openmm as mm
        import openmm.app as app
        import openmm.unit as unit
        from pdbfixer import PDBFixer
    except ImportError as exc:
        raise RuntimeError(
            "OpenMM / PDBFixer not installed. "
            "Add openmm and pdbfixer to requirements.txt."
        ) from exc

    logger.info(f"[MD] Loading complex from {pdbqt_path}")

    # ── Convert PDBQT → PDB via PDBFixer ──────────────────────────────────
    fixer = PDBFixer(filename=str(pdbqt_path))
    fixer.findMissingResidues()
    fixer.findNonstandardResidues()
    fixer.replaceNonstandardResidues()
    fixer.removeHeterogens(keepWater=False)
    fixer.findMissingAtoms()
    fixer.addMissingAtoms()
    fixer.addMissingHydrogens(7.0)  # physiological pH

    topology = fixer.topology
    positions = fixer.positions

    # ── Force field ────────────────────────────────────────────────────────
    ff_files = _FF_FILES.get(force_field.upper(), _FF_FILES["AMBER14"])
    logger.info(f"[MD] Force field: {force_field} → {ff_files}")

    try:
        forcefield = app.ForceField(*ff_files)
    except Exception as exc:
        logger.warning(f"[MD] Force field {force_field} load failed ({exc}), falling back to AMBER14")
        forcefield = app.ForceField(*_FF_FILES["AMBER14"])

    # ── Solvation ──────────────────────────────────────────────────────────
    if solvation == "explicit":
        logger.info(f"[MD] Adding explicit TIP3P water box (padding={padding_nm:.2f} nm)")
        modeller = app.Modeller(topology, positions)
        modeller.addSolvent(
            forcefield,
            model="tip3p",
            padding=padding_nm * unit.nanometers,
            ionicStrength=0.15 * unit.molar,
        )
        topology = modeller.topology
        positions = modeller.positions

        system = forcefield.createSystem(
            topology,
            nonbondedMethod=app.PME,
            nonbondedCutoff=1.0 * unit.nanometers,
            constraints=app.HBonds,
        )
    else:
        # Implicit OBC2
        logger.info("[MD] Using implicit OBC2 solvent")
        system = forcefield.createSystem(
            topology,
            nonbondedMethod=app.NoCutoff,
            constraints=app.HBonds,
            implicitSolvent=app.OBC2,
            soluteDielectric=1.0,
            solventDielectric=78.5,
        )

    logger.info(f"[MD] System created: {system.getNumParticles()} particles")
    return system, topology, positions


async def run_md_simulation(
    job_id: str,
    system,
    topology,
    positions,
    duration_ns: float,
    temperature_K: float,
    timestep_fs: float,
    log_fn: Callable,
    results_dir: Path,
) -> Dict[str, Any]:
    """
    Run a Langevin-dynamics MD simulation and return an analysis dict.

    Parameters
    ----------
    job_id : str
        Used for log messages and output file naming.
    system : openmm.System
        Prepared OpenMM system (from prepare_md_system).
    topology : openmm.app.Topology
        Topology matching the system.
    positions : list
        Initial atomic positions.
    duration_ns : float
        Production run length in nanoseconds.
    temperature_K : float
        Simulation temperature in Kelvin.
    timestep_fs : float
        Integration timestep in femtoseconds.
    log_fn : async callable
        Async function (job_id, message, level) for streaming log lines.
    results_dir : Path
        Directory where trajectory.dcd will be written.

    Returns
    -------
    dict
        Analysis dict with energy/RMSD stats and timeseries.
    """
    try:
        import openmm as mm
        import openmm.app as app
        import openmm.unit as unit
    except ImportError as exc:
        raise RuntimeError("OpenMM not installed.") from exc

    results_dir.mkdir(parents=True, exist_ok=True)
    traj_path = results_dir / "trajectory.dcd"

    # ── Integrator ────────────────────────────────────────────────────────
    temperature = temperature_K * unit.kelvin
    timestep = timestep_fs * unit.femtoseconds
    friction = 1.0 / unit.picoseconds

    integrator = mm.LangevinMiddleIntegrator(temperature, friction, timestep)
    integrator.setRandomNumberSeed(42)

    # ── Platform selection (GPU → CPU fallback) ───────────────────────────
    platform = _select_platform()
    await log_fn(job_id, f"[MD] Using OpenMM platform: {platform.getName()}", "info")

    simulation = app.Simulation(topology, system, integrator, platform)
    simulation.context.setPositions(positions)

    # ── Energy minimisation ───────────────────────────────────────────────
    await log_fn(job_id, "[MD] Energy minimisation…", "info")
    simulation.minimizeEnergy(maxIterations=1000)

    # ── NVT equilibration (100 ps) ────────────────────────────────────────
    equil_steps = int(100_000 / timestep_fs * 0.1)  # 100 ps worth of steps
    equil_steps = max(equil_steps, 500)
    await log_fn(job_id, f"[MD] NVT equilibration ({equil_steps} steps)…", "info")
    simulation.context.setVelocitiesToTemperature(temperature)
    simulation.step(equil_steps)

    # ── Production run ────────────────────────────────────────────────────
    total_steps = int(duration_ns * 1_000_000 / timestep_fs)  # ns → fs → steps
    log_interval_steps = int(1_000 / timestep_fs)             # log every 1 ps
    log_interval_steps = max(log_interval_steps, 1)

    await log_fn(
        job_id,
        f"[MD] Production run: {duration_ns} ns / {total_steps} steps "
        f"(log every {log_interval_steps} steps = 1 ps)",
        "info",
    )

    # DCD reporter
    simulation.reporters.append(
        app.DCDReporter(str(traj_path), log_interval_steps)
    )

    # Collect energy / RMSD data
    energy_timeseries: list = []   # [time_ps, pot_kJ, kin_kJ]
    rmsd_timeseries: list = []     # [time_ps, rmsd_lig_A, rmsd_prot_A]

    # Capture initial positions for RMSD reference
    state0 = simulation.context.getState(getPositions=True)
    ref_positions = state0.getPositions(asNumpy=True)

    steps_done = 0
    report_every = log_interval_steps

    while steps_done < total_steps:
        chunk = min(report_every, total_steps - steps_done)
        simulation.step(chunk)
        steps_done += chunk

        state = simulation.context.getState(
            getEnergy=True, getPositions=True
        )
        time_ps = steps_done * timestep_fs / 1000.0
        pot_kj = state.getPotentialEnergy().value_in_unit(
            unit.kilojoules_per_mole
        )
        kin_kj = state.getKineticEnergy().value_in_unit(
            unit.kilojoules_per_mole
        )
        energy_timeseries.append([round(time_ps, 3), round(pot_kj, 2), round(kin_kj, 2)])

        # Lightweight RMSD estimate (Cα / heavy-atom displacement)
        cur_pos = state.getPositions(asNumpy=True)
        rmsd_lig, rmsd_prot = _compute_rmsd_split(
            ref_positions, cur_pos, topology
        )
        rmsd_timeseries.append([round(time_ps, 3), round(rmsd_lig, 3), round(rmsd_prot, 3)])

        # Progress log every ~10 % of the run
        pct = steps_done / total_steps
        if steps_done % max(1, total_steps // 10) < report_every:
            await log_fn(
                job_id,
                f"[MD] {pct*100:.0f}% — t={time_ps:.1f} ps  "
                f"Epot={pot_kj:.0f} kJ/mol  Ekin={kin_kj:.0f} kJ/mol  "
                f"RMSD_lig={rmsd_lig:.2f} Å",
                "info",
            )

    await log_fn(job_id, f"[MD] Simulation complete. Trajectory: {traj_path}", "info")

    # ── Build analysis dict ───────────────────────────────────────────────
    analysis = _build_analysis(energy_timeseries, rmsd_timeseries)
    return analysis


def analyze_trajectory(
    trajectory_path: Path,
    topology,
    ref_positions=None,
) -> Dict[str, Any]:
    """
    Post-hoc trajectory analysis using MDTraj (if available).
    Falls back to a lightweight numpy-free summary when MDTraj is absent.

    Parameters
    ----------
    trajectory_path : Path
        Path to the DCD trajectory file.
    topology : openmm.app.Topology
        Topology object matching the trajectory.
    ref_positions : optional
        Reference positions for RMSD calculation.

    Returns
    -------
    dict
        Summary statistics (mean, std, min, max) for RMSD and energies.
    """
    try:
        import mdtraj as mdt
        import openmm.app as app

        # Write topology to a temporary PDB for MDTraj
        import tempfile, os
        with tempfile.NamedTemporaryFile(suffix=".pdb", delete=False) as tmp:
            tmp_pdb = tmp.name

        try:
            import openmm.app as app
            with open(tmp_pdb, "w") as fh:
                app.PDBFile.writeFile(topology, ref_positions or [], fh)

            traj = mdt.load(str(trajectory_path), top=tmp_pdb)

            # Protein Cα RMSD
            ca_idx = traj.topology.select("name CA")
            if len(ca_idx) > 0:
                rmsd_prot = mdt.rmsd(traj, traj, 0, atom_indices=ca_idx) * 10  # nm → Å
            else:
                rmsd_prot = [0.0] * len(traj)

            # Ligand heavy-atom RMSD (non-protein, non-water)
            lig_idx = traj.topology.select("not protein and not water")
            if len(lig_idx) > 0:
                rmsd_lig = mdt.rmsd(traj, traj, 0, atom_indices=lig_idx) * 10
            else:
                rmsd_lig = [0.0] * len(traj)

            return {
                "rmsd_ligand": _stats(list(rmsd_lig)),
                "rmsd_protein": _stats(list(rmsd_prot)),
                "frames": len(traj),
                "source": "mdtraj",
            }
        finally:
            os.unlink(tmp_pdb)

    except ImportError:
        logger.debug("MDTraj not available — skipping post-hoc trajectory analysis")
        return {"source": "unavailable", "note": "Install mdtraj for post-hoc analysis"}
    except Exception as exc:
        logger.warning(f"[MD] Trajectory analysis failed: {exc}")
        return {"source": "error", "error": str(exc)}


# ── Internal helpers ───────────────────────────────────────────────────────

def _select_platform():
    """Return the fastest available OpenMM platform."""
    try:
        import openmm as mm
        for name in ("CUDA", "OpenCL", "CPU"):
            try:
                platform = mm.Platform.getPlatformByName(name)
                logger.info(f"[MD] Platform selected: {name}")
                return platform
            except Exception:
                continue
        return mm.Platform.getPlatformByName("Reference")
    except Exception:
        import openmm as mm
        return mm.Platform.getPlatformByName("Reference")


def _compute_rmsd_split(ref_pos, cur_pos, topology) -> Tuple[float, float]:
    """
    Compute per-atom RMSD for ligand and protein atoms separately.
    Uses a simple all-atom RMSD without superposition (fast, no numpy required
    beyond what OpenMM already provides).
    """
    try:
        import numpy as np

        ref = np.array([[v.x, v.y, v.z] for v in ref_pos])
        cur = np.array([[v.x, v.y, v.z] for v in cur_pos])

        prot_idx = []
        lig_idx = []
        for atom in topology.atoms():
            res = atom.residue
            if res.name in ("HOH", "WAT", "TIP", "SOL"):
                continue
            # Standard amino-acid residues → protein
            if res.name in _STANDARD_RESIDUES:
                prot_idx.append(atom.index)
            else:
                lig_idx.append(atom.index)

        def rmsd(idx):
            if not idx:
                return 0.0
            diff = ref[idx] - cur[idx]
            return float(np.sqrt(np.mean(np.sum(diff ** 2, axis=1)))) * 10  # nm → Å

        return rmsd(lig_idx), rmsd(prot_idx)

    except Exception:
        return 0.0, 0.0


def _build_analysis(
    energy_timeseries: list,
    rmsd_timeseries: list,
) -> Dict[str, Any]:
    """Compute summary statistics from collected timeseries data."""
    pot_vals = [row[1] for row in energy_timeseries]
    kin_vals = [row[2] for row in energy_timeseries]
    tot_vals = [p + k for p, k in zip(pot_vals, kin_vals)]
    lig_vals = [row[1] for row in rmsd_timeseries]
    prot_vals = [row[2] for row in rmsd_timeseries]

    return {
        "rmsd_ligand": _stats(lig_vals),
        "rmsd_protein": _stats(prot_vals),
        "potential_energy": _stats(pot_vals),
        "kinetic_energy": _stats(kin_vals),
        "total_energy": _stats(tot_vals),
        "energy_timeseries": energy_timeseries,
        "rmsd_timeseries": rmsd_timeseries,
    }


def _stats(values: list) -> Dict[str, float]:
    """Return mean/std/min/max for a list of floats."""
    if not values:
        return {"mean": 0.0, "std": 0.0, "min": 0.0, "max": 0.0}
    mean = statistics.mean(values)
    std = statistics.pstdev(values) if len(values) > 1 else 0.0
    return {
        "mean": round(mean, 4),
        "std": round(std, 4),
        "min": round(min(values), 4),
        "max": round(max(values), 4),
    }


# Standard amino-acid residue names (used for protein/ligand atom splitting)
_STANDARD_RESIDUES = frozenset({
    "ALA", "ARG", "ASN", "ASP", "CYS", "GLN", "GLU", "GLY", "HIS",
    "ILE", "LEU", "LYS", "MET", "PHE", "PRO", "SER", "THR", "TRP",
    "TYR", "VAL", "HID", "HIE", "HIP", "CYX", "ACE", "NME",
})
