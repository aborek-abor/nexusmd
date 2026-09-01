#!/usr/bin/env python
"""
NexusMD — molecular dynamics of a docked complex.

Everything here is already configured for the pose you exported: the ligand
is parameterised from its own SDF (correct bond orders, correct protonation
as docked), the protein is repaired and protonated, and the box is built
around the complex.

Usage
-----
    python run_md.py                 # 10 ns production, default settings
    python run_md.py --ns 50         # longer
    python run_md.py --ns 2 --quick  # short test run, coarse output

On a GPU this does roughly 50-150 ns/day for a system this size. On CPU it
is 50-100x slower; use --ns 0.2 if you have no GPU, purely to check the
pipeline runs.

Outputs
-------
    system.pdb        solvated topology (needed by analyse.py)
    trajectory.dcd    production trajectory
    production.log    energies, temperature, volume, progress
    md_summary.json   what was actually run
"""

import argparse
import json
import sys
import time
from pathlib import Path

HERE = Path(__file__).parent


def log(msg):
    print(f"[NexusMD-MD] {msg}", flush=True)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--protein", default=str(HERE / "protein.pdb"))
    ap.add_argument("--ligand", default=str(HERE / "ligand.sdf"))
    ap.add_argument("--ns", type=float, default=10.0, help="production length, nanoseconds")
    ap.add_argument("--equil-ps", type=float, default=200.0, help="equilibration, picoseconds")
    ap.add_argument("--padding", type=float, default=1.0, help="solvent padding, nm")
    ap.add_argument("--salt", type=float, default=0.15, help="NaCl concentration, molar")
    ap.add_argument("--temp", type=float, default=310.0, help="kelvin")
    ap.add_argument("--frames", type=int, default=1000, help="frames to write in production")
    ap.add_argument("--forcefield", default="amber14-all.xml")
    ap.add_argument("--water", default="amber14/tip3pfb.xml")
    ap.add_argument("--ligand-ff", default="openff", choices=["openff", "gaff"])
    ap.add_argument("--platform", default=None, help="CUDA / OpenCL / CPU (default: fastest)")
    ap.add_argument("--quick", action="store_true", help="looser settings for a smoke test")
    args = ap.parse_args()

    import openmm
    from openmm import unit, app, LangevinMiddleIntegrator, MonteCarloBarostat
    from openmm.app import PDBFile, Modeller, Simulation, DCDReporter, StateDataReporter
    from pdbfixer import PDBFixer
    from openff.toolkit import Molecule
    from openmmforcefields.generators import SMIRNOFFTemplateGenerator, GAFFTemplateGenerator

    t_start = time.time()

    # ---------------------------------------------------------------- ligand
    log(f"reading ligand: {args.ligand}")
    ligand = Molecule.from_file(args.ligand, file_format="sdf", allow_undefined_stereo=True)
    if isinstance(ligand, list):
        ligand = ligand[0]
    log(f"ligand: {ligand.n_atoms} atoms, net charge {ligand.total_charge}")

    # ---------------------------------------------------------------- protein
    log(f"repairing protein: {args.protein}")
    fixer = PDBFixer(filename=args.protein)
    fixer.findMissingResidues()
    fixer.findNonstandardResidues()
    fixer.replaceNonstandardResidues()
    fixer.removeHeterogens(keepWater=False)
    fixer.findMissingAtoms()
    fixer.addMissingAtoms()
    fixer.addMissingHydrogens(7.4)
    log(f"protein: {fixer.topology.getNumResidues()} residues, "
        f"{fixer.topology.getNumAtoms()} atoms after repair")

    # ---------------------------------------------------------------- force field
    log(f"building force field ({args.ligand_ff} for the ligand)")
    if args.ligand_ff == "openff":
        generator = SMIRNOFFTemplateGenerator(molecules=ligand)
    else:
        generator = GAFFTemplateGenerator(molecules=ligand)

    forcefield = app.ForceField(args.forcefield, args.water)
    forcefield.registerTemplateGenerator(generator.generator)

    # ---------------------------------------------------------------- assemble
    modeller = Modeller(fixer.topology, fixer.positions)
    lig_top = ligand.to_topology().to_openmm()
    lig_pos = ligand.conformers[0].to_openmm()
    modeller.add(lig_top, lig_pos)
    log(f"complex: {modeller.topology.getNumAtoms()} atoms")

    log(f"solvating: {args.padding} nm padding, {args.salt} M NaCl")
    modeller.addSolvent(
        forcefield,
        padding=args.padding * unit.nanometer,
        ionicStrength=args.salt * unit.molar,
        neutralize=True,
    )
    n_atoms = modeller.topology.getNumAtoms()
    log(f"solvated system: {n_atoms} atoms")

    system = forcefield.createSystem(
        modeller.topology,
        nonbondedMethod=app.PME,
        nonbondedCutoff=1.0 * unit.nanometer,
        constraints=app.HBonds,
        rigidWater=True,
        hydrogenMass=1.5 * unit.amu,          # allows a 4 fs step
    )
    dt = 0.004 * unit.picoseconds
    integrator = LangevinMiddleIntegrator(args.temp * unit.kelvin, 1.0 / unit.picosecond, dt)
    system.addForce(MonteCarloBarostat(1.0 * unit.bar, args.temp * unit.kelvin, 25))

    if args.platform:
        platform = openmm.Platform.getPlatformByName(args.platform)
        simulation = Simulation(modeller.topology, system, integrator, platform)
    else:
        simulation = Simulation(modeller.topology, system, integrator)
    log(f"platform: {simulation.context.getPlatform().getName()}")

    simulation.context.setPositions(modeller.positions)

    # ---------------------------------------------------------------- minimise
    log("minimising")
    simulation.minimizeEnergy(maxIterations=5000)
    state = simulation.context.getState(getEnergy=True)
    log(f"energy after minimisation: {state.getPotentialEnergy()}")

    with open(HERE / "system.pdb", "w") as fh:
        PDBFile.writeFile(
            simulation.topology,
            simulation.context.getState(getPositions=True).getPositions(),
            fh,
            keepIds=True,
        )
    log("wrote system.pdb (topology for analyse.py)")

    # ---------------------------------------------------------------- equilibrate
    equil_steps = max(1, int((args.equil_ps * unit.picoseconds) / dt))
    log(f"equilibrating {args.equil_ps} ps ({equil_steps} steps) at {args.temp} K")
    simulation.context.setVelocitiesToTemperature(args.temp * unit.kelvin)
    simulation.reporters.append(
        StateDataReporter(sys.stdout, max(1, equil_steps // 5), step=True,
                          temperature=True, potentialEnergy=True, speed=True)
    )
    simulation.step(equil_steps)
    simulation.reporters.clear()

    # ---------------------------------------------------------------- produce
    prod_steps = max(1, int((args.ns * unit.nanoseconds) / dt))
    stride = max(1, prod_steps // max(1, args.frames))
    log(f"production: {args.ns} ns = {prod_steps} steps, frame every {stride} steps")

    simulation.reporters.append(DCDReporter(str(HERE / "trajectory.dcd"), stride))
    simulation.reporters.append(
        StateDataReporter(str(HERE / "production.log"), stride, step=True, time=True,
                          potentialEnergy=True, kineticEnergy=True, totalEnergy=True,
                          temperature=True, volume=True, density=True, speed=True,
                          remainingTime=True, totalSteps=prod_steps)
    )
    simulation.reporters.append(
        StateDataReporter(sys.stdout, max(1, prod_steps // 20), step=True, time=True,
                          temperature=True, speed=True, remainingTime=True,
                          totalSteps=prod_steps)
    )
    simulation.step(prod_steps)

    elapsed = time.time() - t_start
    summary = {
        "protein": args.protein,
        "ligand": args.ligand,
        "ligand_atoms": ligand.n_atoms,
        "ligand_charge": float(ligand.total_charge.magnitude)
        if hasattr(ligand.total_charge, "magnitude") else float(ligand.total_charge),
        "system_atoms": n_atoms,
        "forcefield": args.forcefield,
        "water_model": args.water,
        "ligand_forcefield": args.ligand_ff,
        "temperature_K": args.temp,
        "timestep_fs": 4.0,
        "equilibration_ps": args.equil_ps,
        "production_ns": args.ns,
        "frames_written": prod_steps // stride,
        "platform": simulation.context.getPlatform().getName(),
        "wall_seconds": round(elapsed, 1),
        "ns_per_day": round(args.ns / (elapsed / 86400.0), 1) if elapsed > 0 else None,
        "openmm_version": openmm.version.version,
    }
    (HERE / "md_summary.json").write_text(json.dumps(summary, indent=2))
    log(f"done in {elapsed/60:.1f} min — {summary['ns_per_day']} ns/day")
    log("next: python analyse.py")


if __name__ == "__main__":
    main()
