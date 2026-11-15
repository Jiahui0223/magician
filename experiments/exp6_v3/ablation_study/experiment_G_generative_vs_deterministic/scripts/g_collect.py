"""Experiment G -- Table 3 (generative vs deterministic) from the closed-loop records (protocol G-v1 section 8; CPU only).
D0 / G0: work_dirs/exp6_v3/experiment_G/closed_loop/<ARM>/<scene>_s<k>/ (episode + official replay metrics; the replay must equal the
stored curve bitwise, 101 values, 100 moves; otherwise the episode does not count). G1 = archived Table 3 V3-T0-K1D2, G4 = archived Table 1
V3-T0-D1, G1 noise replicates V3-T0-K1D2-N{1,2,3} (cvpr_lib.load_manifest; not rerun). Metrics = cvpr_lib (AUC@T = trapz(C[0..T]) / T,
Final = C[100], scene = mean of the 5 starts, group = scene-equal mean); paired scene bootstrap (10,000, seed 0, 95 % percentile) vs G4 and
vs G1; wins / losses; Train-6 / Dev-5 / Test-3. Mean decision time per step from the planner logs.
Writes results/table3/table3_G.{csv,md,tex}, table3_G_splits.csv, table3_G_paired.csv, table3_G.json, per_scene.csv.
Usage: python g_collect.py [--allow-incomplete]"""
import argparse
import glob
import json
import os
import sys

import numpy as np

R = "/bsuscratch/jiahuizhang/projects/occupancy/seen2scene"
sys.path.insert(0, os.path.join(R, "experiments/cvpr_results/scripts"))
import cvpr_lib as L  # noqa: E402

GD = os.path.join(R, "experiments/exp6_v3/ablation_study/experiment_G_generative_vs_deterministic")
WG = os.path.join(R, "work_dirs/exp6_v3/experiment_G")
OUT = os.path.join(GD, "results/table3")
NAMES = ["AUC@20", "AUC@50", "AUC@100", "Final"]
ROWS = [("D0", "Deterministic U-Net (D0)", "–", 1), ("G1", "Generative flow (G1)", "✓", 1), ("G4", "Ours (G4)", "✓", 4),
        ("G0", "Diagnostic: flow weights, zero-noise decoding (G0)", "(det. sampling)", 1)]
ARCHIVE = {"G1": "V3-T0-K1D2", "G4": "V3-T0-D1"}
NOISE_REPS = ["V3-T0-K1D2-N1", "V3-T0-K1D2-N2", "V3-T0-K1D2-N3"]


def load_new(arm):
    """-> ({(scene, start): record}, problems, decision seconds per step)."""
    E, bad, dec = {}, [], []
    for s in L.SCENE_ORDER:
        for k in range(L.N_START):
            o = os.path.join(WG, "closed_loop", arm, f"{s}_s{k}")
            tag = f"V2M_{s}_s{k}_r0"
            pe, pm = os.path.join(o, "episodes", f"ep_{tag}.json"), os.path.join(o, "metrics", "V2M", f"{tag}.json")
            if not (os.path.exists(pe) and os.path.exists(pm)):
                continue
            ep, mt = json.load(open(pe)), json.load(open(pm))
            ok = (mt.get("official_equal_stored") and mt["cov_official"] == ep["coverage"] and ep.get("moves") == L.MOVES
                  and len(ep["coverage"]) == L.MOVES + 1 and ep["scene"] == s and int(ep["start"]) == k)
            if not ok:
                bad.append(f"{arm} {s} s{k}: replay / length / header check failed")
                continue
            E[(s, k)] = {"model": arm, "scene": s, "start": k, "split": L.SPLIT[s], "coverage_official": [float(x) for x in mt["cov_official"]],
                         "episode_file": os.path.relpath(pe, R), "metrics_file": os.path.relpath(pm, R)}
            pl = os.path.join(o, "planner_logs", f"pl_{tag}.json")
            if os.path.exists(pl):
                p = json.load(open(pl))
                dec.append(p["decision_s_total"] / max(1, len(p["steps"])))
    return E, bad, dec


def archived_decision(model):
    dec = []
    for f in glob.glob(os.path.join(L.T1W, "closed_loop", "outdoor", model, "V2M", "planner_logs", "pl_*.json")):
        p = json.load(open(f))
        dec.append(p["decision_s_total"] / max(1, len(p["steps"])))
    return dec


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--allow-incomplete", action="store_true")
    a = ap.parse_args()
    E, problems, dec, ncomp = {}, {}, {}, {}
    for arm in ("D0", "G0"):
        E[arm], problems[arm], dec[arm] = load_new(arm)
    for arm, m in ARCHIVE.items():
        E[arm], problems[arm], dec[arm] = L.load_manifest(m), [], archived_decision(m)
    tabs = {arm: L.scene_table(E[arm], NAMES) for arm in E}
    for arm in E:
        ncomp[arm] = {"episodes": len(E[arm]), "complete_scenes": len(tabs[arm])}
    missing = [arm for arm in ("D0", "G1", "G4") if ncomp[arm]["complete_scenes"] < 14]
    if missing and not a.allow_incomplete:
        sys.exit(f"incomplete arms {missing}: {ncomp} (use --allow-incomplete for a partial table)")
    groups = ["All-14", "Train-6", "Dev-5", "Test-3"]
    val = {arm: {g: {n: L.group_mean(tabs[arm], L.GROUPS[g], n) for n in NAMES} for g in groups} for arm in E}
    paired = {}
    for arm in E:
        for ref in ("G4", "G1"):
            if arm == ref:
                continue
            for g in groups:
                for n in NAMES:
                    paired[(arm, ref, g, n)] = L.paired(tabs[arm], tabs[ref], L.GROUPS[g], n)
    reps = {}
    for m in NOISE_REPS:
        t = L.scene_table(L.load_manifest(m), NAMES)
        reps[m] = {n: L.group_mean(t, L.GROUPS["All-14"], n)[0] for n in NAMES}
    os.makedirs(OUT, exist_ok=True)
    # ------------------------------------------------------------------ main table (All-14)
    hdr = ["arm", "method", "generative", "K", "n_scenes", "n_episodes"] + NAMES + [f"d_{n}_vs_G4" for n in NAMES] + [f"ci_{n}_vs_G4" for n in NAMES] + \
          [f"d_{n}_vs_G1" for n in NAMES] + [f"ci_{n}_vs_G1" for n in NAMES] + ["wins_losses_AUC@100_vs_G4", "wins_losses_AUC@100_vs_G1", "decision_s_per_step"]
    rows = []
    for arm, disp, gen, K in ROWS:
        v = val[arm]["All-14"]
        r = [arm, disp, gen, K, v["AUC@100"][1], ncomp[arm]["episodes"]] + [v[n][0] for n in NAMES]
        for ref in ("G4", "G1"):
            r += [None if arm == ref else paired[(arm, ref, "All-14", n)][0] for n in NAMES]
            r += [None if arm == ref else f"[{paired[(arm, ref, 'All-14', n)][1]:+.3f}, {paired[(arm, ref, 'All-14', n)][2]:+.3f}]"
                  if paired[(arm, ref, "All-14", n)][1] is not None else None for n in NAMES]
        r += [None if arm == "G4" else f"{paired[(arm, 'G4', 'All-14', 'AUC@100')][4]}/{paired[(arm, 'G4', 'All-14', 'AUC@100')][5]}",
              None if arm == "G1" else f"{paired[(arm, 'G1', 'All-14', 'AUC@100')][4]}/{paired[(arm, 'G1', 'All-14', 'AUC@100')][5]}",
              float(np.mean(dec[arm])) if dec[arm] else None]
        rows.append(r)
    L.write_csv(os.path.join(OUT, "table3_G.csv"), hdr, rows)
    # ------------------------------------------------------------------ splits + paired long format
    L.write_csv(os.path.join(OUT, "table3_G_splits.csv"), ["arm", "group", "n_scenes"] + NAMES,
                [[arm, g, val[arm][g]["AUC@100"][1]] + [val[arm][g][n][0] for n in NAMES] for arm, *_ in ROWS for g in groups])
    L.write_csv(os.path.join(OUT, "table3_G_paired.csv"), ["arm", "ref", "group", "metric", "mean_diff", "ci_lo", "ci_hi", "n_scenes", "wins", "losses"],
                [[k[0], k[1], k[2], k[3], *v] for k, v in sorted(paired.items())])
    L.write_csv(os.path.join(OUT, "per_scene.csv"), ["arm", "scene", "split"] + NAMES,
                [[arm, s, L.SPLIT[s]] + [tabs[arm][s][n] for n in NAMES] for arm, *_ in ROWS for s in L.SCENE_ORDER if s in tabs[arm]])
    # ------------------------------------------------------------------ markdown + LaTeX
    f3 = lambda x: L.fmt(x)  # noqa: E731

    def cell(arm, n):
        x = val[arm]["All-14"][n][0]
        if arm == "G4" or x is None:
            return f3(x)
        d, lo, hi = paired[(arm, "G4", "All-14", n)][:3]
        return f"{f3(x)} ({d:+.3f} [{lo:+.3f}, {hi:+.3f}])"
    md = ["# Table 3 — Generative vs deterministic (Experiment G, protocol G-v1)", "",
          "Macarons++ All-14 × 5 official starts × 100 moves, official coverage, scene-equal means. In brackets: paired scene bootstrap difference vs **G4** "
          "(10,000 draws, seed 0, 95 % CI). G1 / G4 = archived Table 3 / Table 1 rows (not rerun).", "",
          "| Method | Generative | K | AUC@20 | AUC@50 | AUC@100 | Final |", "|---|---|---|---|---|---|---|"]
    for arm, disp, gen, K in ROWS:
        md.append(f"| {disp} | {gen} | {K} | " + " | ".join(cell(arm, n) for n in NAMES) + " |")
    md += ["", "## Paired differences vs G1 (All-14)", "", "| Arm − G1 | " + " | ".join(NAMES) + " | wins/losses AUC@100 |", "|---|" + "---|" * (len(NAMES) + 1)]
    for arm in ("D0", "G0", "G4"):
        p = [paired[(arm, "G1", "All-14", n)] for n in NAMES]
        md.append(f"| {arm} | " + " | ".join(f"{x[0]:+.3f} [{x[1]:+.3f}, {x[2]:+.3f}]" if x[0] is not None else "—" for x in p) + f" | {p[2][4]}/{p[2][5]} |")
    md += ["", "## Splits", "", "| Arm | Group | n | " + " | ".join(NAMES) + " |", "|---|---|---|" + "---|" * len(NAMES)]
    for arm, *_ in ROWS:
        for g in groups:
            md.append(f"| {arm} | {g} | {val[arm][g]['AUC@100'][1]} | " + " | ".join(f3(val[arm][g][n][0]) for n in NAMES) + " |")
    md += ["", "G1 noise-map replicates (All-14): " + "; ".join(f"{m.split('-')[-1]} " + " / ".join(f3(reps[m][n]) for n in NAMES) for m in NOISE_REPS),
           "", "Decision time per step (s, mean over episodes): " + ", ".join(f"{arm} {np.mean(dec[arm]):.2f}" for arm, *_ in ROWS if dec[arm]),
           "", f"Completeness: {json.dumps(ncomp)}; problems: {json.dumps({k: v for k, v in problems.items() if v})}"]
    open(os.path.join(OUT, "table3_G.md"), "w").write("\n".join(md) + "\n")
    tex = [r"\begin{tabular}{lcccccc}", r"\toprule", r"Method & Generative & $K$ & AUC@20 & AUC@50 & AUC@100 & Final \\", r"\midrule"]
    for arm, disp, gen, K in ROWS[:3]:
        gtex = "--" if gen == "–" else ("\\checkmark" if gen == "✓" else gen)
        tex.append(disp.split(" (")[0] + f" & {gtex} & {K} & " + " & ".join(f3(val[arm]["All-14"][n][0]) for n in NAMES) + " \\\\")
    tex += [r"\bottomrule", r"\end{tabular}"]
    open(os.path.join(OUT, "table3_G.tex"), "w").write("\n".join(tex) + "\n")
    js = {"protocol": "G-v1", "values": {arm: {g: {n: val[arm][g][n][0] for n in NAMES} for g in groups} for arm in E},
          "paired": {"|".join(k): v for k, v in paired.items()}, "noise_replicates_G1": reps, "completeness": ncomp, "problems": problems,
          "decision_s_per_step": {arm: (float(np.mean(dec[arm])) if dec[arm] else None) for arm in E}, "archive": ARCHIVE}
    json.dump(js, open(os.path.join(OUT, "table3_G.json"), "w"), indent=1, default=float)
    print("\n".join(md))


if __name__ == "__main__":
    main()
