#!/usr/bin/env python
"""
NexusMD — trajectory analysis.

Reads system.pdb + trajectory.dcd from run_md.py and produces, in ./analysis:

    rmsd.csv / rmsd.png                protein backbone and ligand RMSD vs time
    rmsf.csv / rmsf.png                per-residue fluctuation, pocket marked
    rgyr.csv / rgyr.png                radius of gyration
    ligand_displacement.csv/.png       ligand centre-of-mass drift, min distance
    hbonds.csv                         every protein-ligand H-bond event
    hbond_occupancy.csv / .png         each pair, and the % of frames it holds
    interactions.csv                   ProLIF fingerprint, per frame
    interaction_occupancy.csv / .png   contact type per residue, % of frames
    interaction_timeline.png           heatmap: which contacts, when
    summary.md                         the numbers, in words

Usage:
    python analyse.py
    python analyse.py --skip 10          # analyse every 10th frame
    python analyse.py --pocket 4.5       # pocket = residues within 4.5 A of ligand
"""

import argparse
import warnings
from pathlib import Path

warnings.filterwarnings("ignore")

HERE = Path(__file__).parent
OUT = HERE / "analysis"


def log(msg):
    print(f"[analyse] {msg}", flush=True)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--topology", default=str(HERE / "system.pdb"))
    ap.add_argument("--trajectory", default=str(HERE / "trajectory.dcd"))
    ap.add_argument("--ligand-resname", default="UNK")
    ap.add_argument("--skip", type=int, default=1)
    ap.add_argument("--pocket", type=float, default=5.0, help="pocket cutoff, angstrom")
    ap.add_argument("--no-prolif", action="store_true")
    args = ap.parse_args()

    import numpy as np
    import pandas as pd
    import MDAnalysis as mda
    from MDAnalysis.analysis import rms, align, hydrogenbonds
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    OUT.mkdir(exist_ok=True)
    plt.rcParams.update({"figure.dpi": 140, "font.size": 9,
                         "axes.spines.top": False, "axes.spines.right": False})

    u = mda.Universe(args.topology, args.trajectory)
    log(f"{len(u.trajectory)} frames, {len(u.atoms)} atoms")

    # ---- locate the ligand -------------------------------------------------
    ligand = u.select_atoms(f"resname {args.ligand_resname}")
    if len(ligand) == 0:
        candidates = [r for r in u.residues
                      if r.resname not in ("HOH", "WAT", "NA", "CL", "SOD", "CLA")
                      and not r.atoms.select_atoms("protein")]
        if candidates:
            biggest = max(candidates, key=lambda r: len(r.atoms))
            ligand = biggest.atoms
            log(f"ligand auto-detected as resname {biggest.resname} ({len(ligand)} atoms)")
        else:
            raise SystemExit("could not find the ligand — pass --ligand-resname")
    else:
        log(f"ligand: {args.ligand_resname}, {len(ligand)} atoms")

    protein = u.select_atoms("protein")

    # PDB topologies carry no bond records, and both the hydrogen-bond analysis
    # and the interaction fingerprint need connectivity. Guess it once, for the
    # protein and ligand only (guessing across solvent would be slow and useless).
    try:
        if not hasattr(u, "bonds") or len(u.bonds) == 0:
            log("topology has no bonds — guessing for protein + ligand")
            (protein + ligand).guess_bonds()
            log(f"guessed {len(u.bonds)} bonds")
    except Exception as e:
        log(f"bond guessing failed ({e}) — H-bond and fingerprint analysis may be skipped")

    times_ns = np.array([ts.time / 1000.0 for ts in u.trajectory[::args.skip]])
    if times_ns.max() == 0:
        times_ns = np.arange(len(times_ns), dtype=float)

    # ---- align on the protein, then RMSD ----------------------------------
    log("aligning on protein backbone")
    align.AlignTraj(u, u, select="protein and backbone", in_memory=True).run()

    R = rms.RMSD(u, u, select="protein and backbone",
                 groupselections=[f"index {ligand.indices[0]}:{ligand.indices[-1]}"],
                 ref_frame=0).run(step=args.skip)
    arr = R.results.rmsd
    df_rmsd = pd.DataFrame({
        "time_ns": arr[:, 1] / 1000.0 if arr[:, 1].max() > 0 else np.arange(len(arr)),
        "protein_backbone_rmsd_A": arr[:, 2],
        "ligand_rmsd_A": arr[:, 3] if arr.shape[1] > 3 else np.nan,
    })
    df_rmsd.to_csv(OUT / "rmsd.csv", index=False)

    fig, ax = plt.subplots(figsize=(6, 3.2))
    ax.plot(df_rmsd["time_ns"], df_rmsd["protein_backbone_rmsd_A"], lw=1.2, label="protein backbone")
    if df_rmsd["ligand_rmsd_A"].notna().any():
        ax.plot(df_rmsd["time_ns"], df_rmsd["ligand_rmsd_A"], lw=1.2, label="ligand")
    ax.set_xlabel("time (ns)"); ax.set_ylabel("RMSD (Å)"); ax.legend(frameon=False)
    ax.set_title("Structural drift from the docked pose")
    fig.tight_layout(); fig.savefig(OUT / "rmsd.png"); plt.close(fig)
    log(f"RMSD: protein mean {df_rmsd['protein_backbone_rmsd_A'].mean():.2f} Å, "
        f"ligand mean {df_rmsd['ligand_rmsd_A'].mean():.2f} Å")

    # ---- RMSF, with pocket residues marked --------------------------------
    ca = protein.select_atoms("name CA")
    F = rms.RMSF(ca).run(step=args.skip)
    pocket = u.select_atoms(f"protein and around {args.pocket} group lig",
                            lig=ligand, updating=False)
    pocket_resids = sorted(set(pocket.residues.resids))
    df_rmsf = pd.DataFrame({
        "resid": ca.resids, "resname": ca.resnames, "rmsf_A": F.results.rmsf,
        "in_pocket": [r in pocket_resids for r in ca.resids],
    })
    df_rmsf.to_csv(OUT / "rmsf.csv", index=False)

    fig, ax = plt.subplots(figsize=(7, 3.2))
    ax.plot(df_rmsf["resid"], df_rmsf["rmsf_A"], lw=0.9, color="#555")
    hit = df_rmsf[df_rmsf["in_pocket"]]
    ax.scatter(hit["resid"], hit["rmsf_A"], s=14, color="#d1495b", zorder=3,
               label=f"within {args.pocket} Å of ligand")
    ax.set_xlabel("residue"); ax.set_ylabel("RMSF (Å)"); ax.legend(frameon=False)
    ax.set_title("Per-residue flexibility")
    fig.tight_layout(); fig.savefig(OUT / "rmsf.png"); plt.close(fig)
    log(f"RMSF: {len(pocket_resids)} pocket residues, "
        f"mean {hit['rmsf_A'].mean():.2f} Å vs {df_rmsf['rmsf_A'].mean():.2f} Å overall")

    # ---- radius of gyration and ligand drift ------------------------------
    rg, com_disp, min_dist = [], [], []
    ref_com = None
    for ts in u.trajectory[::args.skip]:
        rg.append(protein.radius_of_gyration())
        com = ligand.center_of_mass()
        if ref_com is None:
            ref_com = com
        com_disp.append(float(np.linalg.norm(com - ref_com)))
        d = np.linalg.norm(
            ligand.positions[:, None, :] - protein.positions[None, :, :], axis=-1)
        min_dist.append(float(d.min()))

    pd.DataFrame({"time_ns": times_ns[:len(rg)], "rgyr_A": rg}).to_csv(OUT / "rgyr.csv", index=False)
    fig, ax = plt.subplots(figsize=(6, 2.6))
    ax.plot(times_ns[:len(rg)], rg, lw=1.2, color="#3d5a80")
    ax.set_xlabel("time (ns)"); ax.set_ylabel("Rg (Å)"); ax.set_title("Protein compactness")
    fig.tight_layout(); fig.savefig(OUT / "rgyr.png"); plt.close(fig)

    df_disp = pd.DataFrame({"time_ns": times_ns[:len(com_disp)],
                            "ligand_com_displacement_A": com_disp,
                            "min_ligand_protein_distance_A": min_dist})
    df_disp.to_csv(OUT / "ligand_displacement.csv", index=False)
    fig, ax = plt.subplots(figsize=(6, 2.8))
    ax.plot(df_disp["time_ns"], df_disp["ligand_com_displacement_A"], lw=1.2,
            color="#ee6c4d", label="COM displacement")
    ax.plot(df_disp["time_ns"], df_disp["min_ligand_protein_distance_A"], lw=1.2,
            color="#3d5a80", label="min contact distance")
    ax.set_xlabel("time (ns)"); ax.set_ylabel("Å"); ax.legend(frameon=False)
    ax.set_title("Did the ligand stay in the site?")
    fig.tight_layout(); fig.savefig(OUT / "ligand_displacement.png"); plt.close(fig)
    log(f"ligand COM drift: {com_disp[-1]:.2f} Å at end, max {max(com_disp):.2f} Å")

    # ---- hydrogen bonds ---------------------------------------------------
    log("hydrogen bonds (protein <-> ligand)")
    lig_sel = f"index {ligand.indices[0]}:{ligand.indices[-1]}"
    try:
        # name-based selections: PDB topologies carry no charges, so the
        # guess_hydrogens/guess_acceptors helpers cannot be used here
        H = hydrogenbonds.HydrogenBondAnalysis(
            universe=u,
            between=[lig_sel, "protein"],
            donors_sel="name N* O* S*",
            hydrogens_sel="name H*",
            acceptors_sel="name O* N* S*",
            d_a_cutoff=3.5, d_h_a_angle_cutoff=140,
        )
        H.run(step=args.skip)
        hb = H.results.hbonds
        n_frames = len(times_ns)
        rows = []
        for donor_ix, hydrogen_ix, acceptor_ix, count in _hbond_pairs(hb):
            d_at, a_at = u.atoms[int(donor_ix)], u.atoms[int(acceptor_ix)]
            rows.append({
                "donor": f"{d_at.resname}{d_at.resid}:{d_at.name}",
                "acceptor": f"{a_at.resname}{a_at.resid}:{a_at.name}",
                "frames": int(count),
                "occupancy_pct": round(100.0 * count / max(1, n_frames), 1),
            })
        df_hb = pd.DataFrame(rows).sort_values("occupancy_pct", ascending=False)
        pd.DataFrame(hb, columns=["frame", "donor_ix", "hydrogen_ix", "acceptor_ix",
                                  "distance", "angle"]).to_csv(OUT / "hbonds.csv", index=False)
        df_hb.to_csv(OUT / "hbond_occupancy.csv", index=False)

        if len(df_hb):
            top = df_hb.head(12).iloc[::-1]
            fig, ax = plt.subplots(figsize=(6, max(2.2, 0.32 * len(top))))
            ax.barh([f"{r.donor} → {r.acceptor}" for r in top.itertuples()],
                    top["occupancy_pct"], color="#3d8361")
            ax.set_xlabel("% of frames"); ax.set_xlim(0, 100)
            ax.set_title("Hydrogen bond persistence")
            fig.tight_layout(); fig.savefig(OUT / "hbond_occupancy.png"); plt.close(fig)
        log(f"hydrogen bonds: {len(df_hb)} distinct pairs, "
            f"{(df_hb['occupancy_pct'] > 50).sum() if len(df_hb) else 0} present in >50% of frames")
    except Exception as e:
        log(f"hydrogen bond analysis skipped: {e}")
        df_hb = None

    # ---- ProLIF interaction fingerprint -----------------------------------
    df_occ = None
    if not args.no_prolif:
        try:
            import prolif
            log("interaction fingerprint (ProLIF)")
            prot_sel = u.select_atoms("protein")
            # ProLIF converts through RDKit, which needs explicit hydrogens.
            # An OpenMM-written system.pdb has them; a stripped PDB does not.
            if len(prot_sel.select_atoms("name H*")) == 0:
                raise ValueError(
                    "protein selection has no hydrogens — ProLIF cannot infer "
                    "chemistry. Use the system.pdb written by run_md.py, which "
                    "is protonated, rather than a bare crystal structure.")
            fp = prolif.Fingerprint(
                ["Hydrophobic", "HBDonor", "HBAcceptor", "PiStacking",
                 "Anionic", "Cationic", "CationPi", "PiCation", "VdWContact"]
            )
            fp.run(u.trajectory[::args.skip], ligand, prot_sel, progress=False)
            df_if = fp.to_dataframe()
            df_if.to_csv(OUT / "interactions.csv")

            if df_if.shape[1] == 0:
                log("no protein-ligand contacts detected in any frame")
                raise ValueError("empty fingerprint")

            occ = (df_if.mean() * 100).reset_index()
            # column index has 3 levels (ligand, residue, interaction) plus the value
            occ.columns = list(occ.columns[:-1]) + ["occupancy_pct"]
            occ = occ.rename(columns={occ.columns[-3]: "residue",
                                      occ.columns[-2]: "interaction"})
            occ = occ[["residue", "interaction", "occupancy_pct"]].sort_values(
                "occupancy_pct", ascending=False)
            occ["occupancy_pct"] = occ["occupancy_pct"].round(1)
            occ.to_csv(OUT / "interaction_occupancy.csv", index=False)
            df_occ = occ

            top = occ.head(15).iloc[::-1]
            fig, ax = plt.subplots(figsize=(6.4, max(2.4, 0.34 * len(top))))
            colors = {"Hydrophobic": "#e9c46a", "HBDonor": "#2a9d8f", "HBAcceptor": "#264653",
                      "PiStacking": "#e76f51", "Anionic": "#8ab17d", "Cationic": "#5b8e7d",
                      "VdWContact": "#adb5bd"}
            ax.barh([f"{r.residue}  {r.interaction}" for r in top.itertuples()],
                    top["occupancy_pct"],
                    color=[colors.get(r.interaction, "#888") for r in top.itertuples()])
            ax.set_xlabel("% of frames"); ax.set_xlim(0, 100)
            ax.set_title("Which contacts actually persist")
            fig.tight_layout(); fig.savefig(OUT / "interaction_occupancy.png"); plt.close(fig)

            mat = df_if.T
            fig, ax = plt.subplots(figsize=(7, max(2.5, 0.22 * len(mat))))
            ax.imshow(mat.values.astype(float), aspect="auto", cmap="Greens",
                      interpolation="nearest")
            ax.set_yticks(range(len(mat)))
            ax.set_yticklabels([f"{i[1]} {i[2]}" for i in mat.index], fontsize=6)
            ax.set_xlabel("frame"); ax.set_title("Interaction timeline")
            fig.tight_layout(); fig.savefig(OUT / "interaction_timeline.png"); plt.close(fig)
            log(f"interactions: {len(occ)} contacts, "
                f"{(occ['occupancy_pct'] > 50).sum()} present in >50% of frames")
        except Exception as e:
            log(f"ProLIF analysis skipped: {e}")

    # ---- summary ----------------------------------------------------------
    lines = [
        "# MD analysis summary", "",
        f"Frames analysed: {len(times_ns)} (stride {args.skip})",
        f"Simulated time covered: {times_ns.max():.2f} ns" if times_ns.max() else "",
        "",
        "## Stability", "",
        f"Protein backbone RMSD averaged {df_rmsd['protein_backbone_rmsd_A'].mean():.2f} Å "
        f"and ended at {df_rmsd['protein_backbone_rmsd_A'].iloc[-1]:.2f} Å.",
    ]
    if df_rmsd["ligand_rmsd_A"].notna().any():
        lig_end = df_rmsd["ligand_rmsd_A"].iloc[-1]
        verdict = ("the pose held" if lig_end < 2.5 else
                   "the ligand shifted substantially" if lig_end < 5 else
                   "the ligand left the docked position")
        lines.append(f"Ligand RMSD ended at {lig_end:.2f} Å — {verdict}.")
    lines.append(f"Ligand centre of mass moved {com_disp[-1]:.2f} Å from its docked position "
                 f"(maximum {max(com_disp):.2f} Å).")
    if df_hb is not None and len(df_hb):
        lines += ["", "## Hydrogen bonds", ""]
        for r in df_hb.head(8).itertuples():
            lines.append(f"- {r.donor} → {r.acceptor}: {r.occupancy_pct}% of frames")
    if df_occ is not None and len(df_occ):
        lines += ["", "## Persistent contacts", ""]
        for r in df_occ.head(10).itertuples():
            lines.append(f"- {r.residue} · {r.interaction}: {r.occupancy_pct}%")
    lines += ["", "Contacts below ~30% occupancy are usually transient and should not be "
              "reported as binding determinants.", ""]
    (OUT / "summary.md").write_text("\n".join(l for l in lines if l is not None))
    log(f"wrote {len(list(OUT.iterdir()))} files to {OUT}")


def _hbond_pairs(hb):
    """Collapse the raw hbond table into unique (donor, hydrogen, acceptor) counts."""
    import numpy as np
    if len(hb) == 0:
        return []
    key = hb[:, 1:4]
    uniq, counts = np.unique(key, axis=0, return_counts=True)
    return [(u[0], u[1], u[2], c) for u, c in zip(uniq, counts)]


if __name__ == "__main__":
    main()
