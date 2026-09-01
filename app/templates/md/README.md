# NexusMD — MD package

Everything in this folder describes one docked pose. The system is already
set up: the ligand carries the bond orders and coordinates it was docked
with, the protein is the same structure Vina used, and the scripts know how
to put them together.

## Files

| file | what it is |
|---|---|
| `complex.pdb` | receptor + pose in one file — open this in Discovery Studio, PyMOL or Chimera |
| `protein.pdb` | receptor alone, as prepared for docking |
| `ligand.sdf` | the docked pose with correct bond orders, rebuilt from the Vina output |
| `docking.json` | the score, grid box, engine and run parameters this pose came from |
| `run_md.py` | builds the solvated system and runs the simulation |
| `analyse.py` | turns the trajectory into CSVs, figures and a summary |
| `NexusMD_MD.ipynb` | the same two steps as a Colab notebook |

## Running it on Google Colab (free GPU)

Upload this folder to Colab, open `NexusMD_MD.ipynb`, set the runtime to a
GPU (Runtime → Change runtime type → T4 GPU), and run the cells. The first
cell installs the toolchain through conda, which takes about five minutes.

A system of this size does roughly 50–150 ns/day on a T4, so 10 ns is
around two to four hours. Colab disconnects idle sessions, so either stay on
the tab or run shorter blocks and restart from the checkpoint.

## Running it locally

The toolchain is conda-only — `openff-toolkit` has no pip wheel:

```bash
conda create -n nexusmd -c conda-forge python=3.11 openmm openff-toolkit \
    openmmforcefields pdbfixer rdkit mdanalysis prolif matplotlib pandas
conda activate nexusmd

python run_md.py --ns 10       # simulate
python analyse.py              # analyse
```

Check `nvidia-smi` first. Without a GPU, use `--ns 0.2` to confirm the
pipeline works and run the real thing somewhere with hardware.

## What comes out

`analysis/` contains RMSD of protein and ligand over time, per-residue RMSF
with pocket residues marked, radius of gyration, ligand centre-of-mass
displacement and minimum contact distance, hydrogen bonds with the
percentage of frames each survives, a ProLIF interaction fingerprint per
frame with an occupancy chart and timeline heatmap, and `summary.md` stating
the numbers in words.

## Reading the result

The question MD answers after docking is whether the pose is real. A ligand
RMSD that settles below about 2 Å and stays there means the pose held. A
climbing RMSD with the centre of mass drifting past 5 Å means the ligand
left, and the docking score was optimistic.

Contacts matter more than the score. A hydrogen bond present in 90% of
frames is a binding determinant; one appearing in 15% is noise, and
reporting it as an interaction overstates what the simulation showed.

## Honest limits

Ten nanoseconds is a short simulation. It tests whether a pose is stable,
not whether a compound binds — that needs free-energy methods, replicates,
and far more sampling. One trajectory is one sample of a stochastic process;
three independent runs with different seeds (`--ns 10` three times) tell you
much more than one run three times as long.

The starting pose came from rigid-receptor docking, so if Vina put the
ligand in the wrong place, MD will faithfully simulate it being in the wrong
place.
