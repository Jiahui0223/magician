"""Table 3 v2 (2026-11-16, user request): the K x planner table of make_table3.py restricted to the Unseen-6 scenes, i.e. the same outdoor
evaluation units as the main Table 1 v4 (alhambra, neuschwanstein, eiffel, bridge, barts, sestino_museum: the 6 Macarons++ scenes that
neither MAGICIAN / MACARONS nor our model were trained on; = All-14 minus MAGICIAN's 8 training scenes, make_table1_v4.py / make_fig3_v3.py).
Same 8 closed-loop cells, same loader (make_table3.data(): manifest COMPLETE rows, replay-verified), same metrics / layout / bold rule; only
the scene set changes (6 scenes x 5 official starts = 30 episodes per cell). Paired effects (scene bootstrap, 10,000, seed 0) are recomputed
on Unseen-6. The All-14 version (make_table3.py -> table3_hypothesis_planning.*) is unchanged.
Outputs fig/table3/: table3_hypothesis_planning_v2.{csv,tex,pdf,png}, table3_v2_effects.csv. Run with CUDA_VISIBLE_DEVICES="".
--k1-mean (v3, 2026-11-16): the K = 1 cells (One-Step and CDGR) are the per-scene mean over noise maps 0-3 (method ids V3-T0-K1D{1,2} and
-N1..N3, the replicates run for Table 3 on 2026-11-12), i.e. the same four maps K = 4 uses; a single map is one draw (Unseen-6 AUC@20 of
K=1+CDGR spans 0.412-0.438 across maps). Outputs table3_hypothesis_planning_v3.* and table3_v3_effects.csv.
--k12-mean (v4, 2026-11-16, T3K2): as --k1-mean and in addition the K = 2 cells are the per-scene mean of the two K = 2 runs with noise maps
{0, 1} (V3-T0-K2D{1,2}, EXP9) and {2, 3} (V3-T0-K2D{1,2}-N2, kd9b noise offset 2), so the K = 1 and K = 2 rows average runs that together cover
noise maps 0-3, the maps K = 4 uses; K = 8 stays one run (maps 0-7). Rule fixed before the N2 runs. Outputs table3_hypothesis_planning_v4.*,
table3_v4_effects.csv, table3_v4_k2_replicates.csv (maps {0,1} / {2,3} / mean next to K = 4, Unseen-6 and All-14) and, once V3-T0-K2D{1,2}-N2
are complete on all 14 scenes, the All-14 version of the same rule: table3_v4_all14.{csv,md} + table3_v4_all14_effects.csv."""
import os
import sys

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
import make_table3 as M  # noqa: E402

L, RD = M.L, M.RD
MAG_TRAIN = ("dunnottar", "pisa", "bannerman", "redeemer", "pantheon", "fushimi", "liberty", "colosseum")
UNSEEN6 = [s for s in L.SCENE_ORDER if s not in MAG_TRAIN]
assert UNSEEN6 == ["alhambra", "neuschwanstein", "eiffel", "bridge", "barts", "sestino_museum"], UNSEEN6
M.NOTE = ("Macarons++ Unseen-6 (alhambra, neuschwanstein, eiffel, bridge, barts, sestino museum: scenes no method was trained on, as in Table 1): "
          "6 scenes × 5 official starts, 100 steps, official coverage, scene-equal means; every cell = 30 real closed-loop episodes with the same "
          "frozen V3-T0 checkpoint, candidates, value_cov and noise keys (K uses the nested noise maps 0..K-1). Only K and the planner change. "
          "Shaded: full model (K = 4, CDGR = Table 1 main row). Bold: highest value per metric over all 8 cells (3 decimals). Paired effects: "
          "table3_v2_effects.csv.")


K1_REPS = {1: ["V3-T0-K1D1", "V3-T0-K1D1-N1", "V3-T0-K1D1-N2", "V3-T0-K1D1-N3"],
           2: ["V3-T0-K1D2", "V3-T0-K1D2-N1", "V3-T0-K1D2-N2", "V3-T0-K1D2-N3"]}
NOTE_V3 = ("Macarons++ Unseen-6 (alhambra, neuschwanstein, eiffel, bridge, barts, sestino museum: scenes no method was trained on, as in Table 1): "
           "6 scenes × 5 official starts, 100 steps, official coverage, scene-equal means; same frozen V3-T0 checkpoint, candidates, value_cov and "
           "noise keys in every cell; only K and the planner change. K = 4 uses noise maps 0-3; the K = 1 rows are averaged over the same four "
           "noise maps (4 × 30 episodes; a single map varies by up to 0.026 in AUC@20). Shaded: full model (K = 4, CDGR = Table 1 main row). "
           "Bold: highest value per metric over all 8 cells (3 decimals). Paired effects: table3_v3_effects.csv.")
K2_REPS = {1: ["V3-T0-K2D1", "V3-T0-K2D1-N2"], 2: ["V3-T0-K2D2", "V3-T0-K2D2-N2"]}     # noise maps {0, 1} and {2, 3}
NOTE_V4 = ("Macarons++ Unseen-6 (alhambra, neuschwanstein, eiffel, bridge, barts, sestino museum: scenes no method was trained on, as in Table 1): "
           "6 scenes × 5 official starts, 100 steps, official coverage, scene-equal means; same frozen V3-T0 checkpoint, candidates, value_cov and "
           "noise keys in every cell; only K and the planner change. K = 4 uses noise maps 0-3; the K = 1 rows average four single-map runs (maps 0, 1, 2, 3; "
           "4 × 30 episodes) and the K = 2 rows two runs (maps {0,1} and {2,3}; 2 × 30 episodes), so the runs of each of these rows together cover "
           "noise maps 0-3, the maps K = 4 uses; K = 8 is one run (maps 0-7). Shaded: full model (K = 4, CDGR = Table 1 main row). "
           "Bold: highest value per metric over all 8 cells (3 decimals). Paired effects: table3_v4_effects.csv.")
TAG_NOTE = {"v2": "", "v3": " (mean of N0-N3)", "v4": " (mean of N0-N3)"}


def rep_mean(ids, need):
    """Per-scene mean over the runs `ids` on the scenes complete in every run; asserts that the scenes `need` are complete."""
    reps = [L.scene_table(L.load_manifest(m), M.METS) for m in ids]
    ok = [s for s in L.SCENE_ORDER if all(s in r for r in reps)]
    miss = [s for s in need if s not in ok]
    assert not miss, f"{ids}: incomplete scenes {miss}"
    return {s: {n: float(np.mean([r[s][n] for r in reps])) for n in M.METS} for s in ok}, reps


def k2_outputs(tab, k2reps):
    """table3_v4_k2_replicates.csv (Unseen-6 and, when complete, All-14) and the All-14 version of the v4 rule (table3_v4_all14.*)."""
    rows = []
    groups = [("Unseen-6", UNSEEN6)]
    full14 = all(len(r) == 14 for d in (1, 2) for r in k2reps[d])
    if full14:
        groups.append(("All-14", list(L.SCENE_ORDER)))
    for gname, ss in groups:
        for d, short, _ in M.PLAN:
            r01, r23 = k2reps[d]
            for n in M.METS:
                v01, v23 = (float(np.mean([r[s][n] for s in ss])) for r in (r01, r23))
                k4 = float(np.mean([tab[(4, d)][s][n] for s in ss]))
                k1 = float(np.mean([tab[(1, d)][s][n] for s in ss]))
                dd = np.array([(r01[s][n] + r23[s][n]) / 2 - tab[(4, d)][s][n] for s in ss])
                m, lo, hi = L.boot_ci(dd)
                d2 = np.array([r23[s][n] - r01[s][n] for s in ss])
                m2, lo2, hi2 = L.boot_ci(d2)
                rows.append([gname, len(ss), short, n, ";".join(K2_REPS[d]), f"{v01:.4f}", f"{v23:.4f}", f"{(v01 + v23) / 2:.4f}", f"{abs(v23 - v01):.4f}",
                             f"{m2:+.4f}", f"{lo2:+.4f}", f"{hi2:+.4f}", f"{k4:.4f}", f"{m:+.4f}", f"{lo:+.4f}", f"{hi:+.4f}", f"{int((dd > 0).sum())}/{int((dd < 0).sum())}", f"{k1:.4f}"])
    L.write_csv(os.path.join(M.OUT, "table3_v4_k2_replicates.csv"),
                ["scene_group", "n_scenes", "planner", "metric", "K2_method_ids(maps 01;23)", "K2_maps01", "K2_maps23", "K2_mean", "abs_diff_maps23_vs_01",
                 "maps23_minus_maps01", "ci95_lo", "ci95_hi", "K4_maps0123", "K2mean_minus_K4", "ci95_lo", "ci95_hi", "scenes_pos/neg", "K1_mean_of_maps0123"], rows)
    if not full14:
        print("All-14 v4 numbers NOT written: V3-T0-K2D{1,2}-N2 complete on " + ", ".join(f"{K2_REPS[d][1]} {len(k2reps[d][1])}/14" for d in (1, 2)) + " scenes")
        return rows
    V = {c: {n: L.boot_ci([tab[c][s][n] for s in L.SCENE_ORDER]) for n in M.METS} for c in M.CELL}
    out = []
    for d, short, _ in M.PLAN:
        for K in M.KS:
            c = (K, d)
            lab = M.CELL[c] + {1: " (mean of maps 0,1,2,3)", 2: " (mean of maps {0,1},{2,3})"}.get(K, "")
            out.append([short, K, lab, {1: 280, 2: 140}.get(K, 70)] + [f"{V[c][n][0]:.4f}" for n in M.METS] + [f"{V[c][n][1]:.4f}" for n in M.METS] + [f"{V[c][n][2]:.4f}" for n in M.METS])
    L.write_csv(os.path.join(M.OUT, "table3_v4_all14.csv"), ["planner", "K", "method_id", "n_episodes"] + M.METS + [f"{n}_ci95_lo" for n in M.METS] + [f"{n}_ci95_hi" for n in M.METS], out)
    eff = effect_rows(tab, list(L.SCENE_ORDER))
    L.write_csv(os.path.join(M.OUT, "table3_v4_all14_effects.csv"), ["contrast", "metric", "mean_diff", "ci95_lo", "ci95_hi", "ci_excludes_0", "scenes_pos/neg"], eff)
    md = ["# Table 3 v4 rule on All-14 (generated by make_table3_v2.py --k12-mean; do not edit by hand)", "",
          "Macarons++ All-14 (14 scenes × 5 official starts, 100 steps, official coverage, scene-equal means). K = 4 uses noise maps 0-3; the K = 1 rows "
          "average four single-map runs (maps 0, 1, 2, 3; 4 × 70 episodes) and the K = 2 rows two runs (maps {0,1} and {2,3}; 2 × 70 episodes), so the runs of "
          "each of these rows together cover noise maps 0-3, the maps K = 4 uses; K = 8 is one run (maps 0-7). 95 % scene-bootstrap CIs in table3_v4_all14.csv, "
          "paired effects in table3_v4_all14_effects.csv.", "",
          "| Planner | K | " + " | ".join(M.METS) + " |", "|---|---|" + "---|" * len(M.METS)]
    for d, short, _ in M.PLAN:
        for K in M.KS:
            md.append(f"| {short} | {K} | " + " | ".join(f"{V[(K, d)][n][0]:.3f}" for n in M.METS) + " |")
    open(os.path.join(M.OUT, "table3_v4_all14.md"), "w").write("\n".join(md) + "\n")
    print("TABLE 3 v4 rule, All-14 mean")
    for d, short, _ in M.PLAN:
        for K in M.KS:
            print(f"  {short:9s} K={K}: " + "  ".join(f"{n} {V[(K, d)][n][0]:.3f}" for n in M.METS))
    return rows


def effect_rows(tab, scenes):
    def row(name, a, b, n):
        dd = np.array([tab[a][s][n] - tab[b][s][n] for s in scenes])
        m, lo, hi = L.boot_ci(dd)
        return [name, n, f"{m:+.4f}", f"{lo:+.4f}", f"{hi:+.4f}", "yes" if (lo > 0 or hi < 0) else "no", f"{int((dd > 0).sum())}/{int((dd < 0).sum())}"]
    eff = []
    for K in M.KS:
        for n in M.METS:
            eff.append(row(f"K={K}: CDGR - One-Step", (K, 2), (K, 1), n))
    for d, short, _ in M.PLAN:
        for K in (1, 2, 8):
            for n in M.METS:
                eff.append(row(f"{short}: K={K} - K=4", (K, d), (4, d), n))
    for n in M.METS:
        dd = np.array([np.mean([tab[(K, 2)][s][n] - tab[(K, 1)][s][n] for K in M.KS]) for s in scenes])
        m, lo, hi = L.boot_ci(dd)
        eff.append(["mean over K: CDGR - One-Step", n, f"{m:+.4f}", f"{lo:+.4f}", f"{hi:+.4f}", "yes" if (lo > 0 or hi < 0) else "no",
                    f"{int((dd > 0).sum())}/{int((dd < 0).sum())}"])
    return eff


def main(k1_mean=False, k12_mean=False):
    k1_mean = k1_mean or k12_mean
    E, tab = M.data()
    if k1_mean:                                                     # K = 1 cells := per-scene mean over noise maps 0..3 (the maps K = 4 uses)
        for d, ids in K1_REPS.items():
            reps = [L.scene_table(L.load_manifest(m), M.METS) for m in ids]
            assert all(len(r) == 14 for r in reps), [len(r) for r in reps]
            tab[(1, d)] = {s: {n: float(np.mean([r[s][n] for r in reps])) for n in M.METS} for s in L.SCENE_ORDER}
        M.NOTE = NOTE_V3
    if k12_mean:                                                    # K = 2 cells := per-scene mean of the maps {0,1} and {2,3} runs (together maps 0..3)
        k2reps = {}
        for d, ids in K2_REPS.items():
            tab[(2, d)], k2reps[d] = rep_mean(ids, UNSEEN6)
        M.NOTE = NOTE_V4
    V = {c: {n: L.boot_ci([tab[c][s][n] for s in UNSEEN6]) for n in M.METS} for c in M.CELL}
    rows = []
    tag = "v4" if k12_mean else ("v3" if k1_mean else "v2")
    for d, short, _ in M.PLAN:
        for K in M.KS:
            c = (K, d)
            lab = M.CELL[c] + (TAG_NOTE[tag] if K == 1 else (" (mean of maps {0,1},{2,3})" if (k12_mean and K == 2) else ""))
            ne = 120 if (k1_mean and K == 1) else (60 if (k12_mean and K == 2) else 30)
            rows.append([short, K, lab, ne] + [f"{V[c][n][0]:.4f}" for n in M.METS] + [f"{V[c][n][1]:.4f}" for n in M.METS] + [f"{V[c][n][2]:.4f}" for n in M.METS])
    L.write_csv(os.path.join(M.OUT, f"table3_hypothesis_planning_{tag}.csv"),
                ["planner", "K", "method_id", "n_episodes"] + M.METS + [f"{n}_ci95_lo" for n in M.METS] + [f"{n}_ci95_hi" for n in M.METS], rows)
    eff = []

    def row(name, a, b, n):
        dd = np.array([tab[a][s][n] - tab[b][s][n] for s in UNSEEN6])
        m, lo, hi = L.boot_ci(dd)
        return [name, n, f"{m:+.4f}", f"{lo:+.4f}", f"{hi:+.4f}", "yes" if (lo > 0 or hi < 0) else "no", f"{int((dd > 0).sum())}/{int((dd < 0).sum())}"]
    for K in M.KS:
        for n in M.METS:
            eff.append(row(f"K={K}: CDGR - One-Step", (K, 2), (K, 1), n))
    for d, short, _ in M.PLAN:
        for K in (1, 2, 8):
            for n in M.METS:
                eff.append(row(f"{short}: K={K} - K=4", (K, d), (4, d), n))
    for n in M.METS:
        dd = np.array([np.mean([tab[(K, 2)][s][n] - tab[(K, 1)][s][n] for K in M.KS]) for s in UNSEEN6])
        m, lo, hi = L.boot_ci(dd)
        eff.append(["mean over K: CDGR - One-Step", n, f"{m:+.4f}", f"{lo:+.4f}", f"{hi:+.4f}", "yes" if (lo > 0 or hi < 0) else "no",
                    f"{int((dd > 0).sum())}/{int((dd < 0).sum())}"])
    L.write_csv(os.path.join(M.OUT, f"table3_{tag}_effects.csv"), ["contrast", "metric", "mean_diff", "ci95_lo", "ci95_hi", "ci_excludes_0", "scenes_pos/neg"], eff)
    RD.save(M.draw(V), os.path.join(M.OUT, f"table3_hypothesis_planning_{tag}"))
    t = M.tex(V)
    t = t.replace("% Table 3 (K x planner), generated by experiments/cvpr_results/exp9_ablation_qualitative/scripts/table3_fig5/make_table3.py",
                  "% Table 3 v2 (K x planner, Unseen-6), generated by experiments/cvpr_results/exp9_ablation_qualitative/scripts/table3_fig5/make_table3_v2.py")
    t = t.replace(r"Macarons++ (All-14: 14 scenes $\times$ 5 official starts, 100 steps, official coverage, "
                  r"scene-equal means; every cell is 70 real closed-loop episodes)",
                  r"Macarons++ scenes unseen during training (Unseen-6, as in Table~1: 6 scenes $\times$ 5 official starts, 100 steps, official coverage, "
                  r"scene-equal means; every cell is 30 real closed-loop episodes" + (r"; the $K{=}1$ rows average the same four noise maps that $K{=}4$ uses, 120 episodes each" if (k1_mean and not k12_mean) else "")
                  + (r"; $K{=}4$ uses noise maps 0--3, the $K{=}1$ rows average four single-map runs (maps 0, 1, 2, 3; 120 episodes each) and the $K{=}2$ rows two runs "
                     r"(maps \{0,1\} and \{2,3\}; 60 episodes each), so the runs of each of these rows together cover noise maps 0--3; $K{=}8$ is one run (maps 0--7)" if k12_mean else "") + ")")
    assert "Unseen-6" in t, "caption replacement failed"
    if k12_mean:
        t = t.replace("% Table 3 v2 (K x planner, Unseen-6)", "% Table 3 v4 (K x planner, Unseen-6, K=1 = mean over noise maps 0-3, K=2 = mean of maps {0,1} and {2,3})")
    elif k1_mean:
        t = t.replace("% Table 3 v2 (K x planner, Unseen-6)", "% Table 3 v3 (K x planner, Unseen-6, K=1 = mean over noise maps 0-3)")
    open(os.path.join(M.OUT, f"table3_hypothesis_planning_{tag}.tex"), "w").write(t)
    if k12_mean:
        k2r = k2_outputs(tab, k2reps)
        print("K = 2 replicate (Unseen-6): maps {0,1} | maps {2,3} | mean | K = 4")
        for r in k2r:
            if r[0] == "Unseen-6":
                print(f"  {r[2]:9s} {r[3]:8s} {r[5]} | {r[6]} | {r[7]} | K4 {r[12]}  (mean - K4 {r[13]} [{r[14]}, {r[15]}])")
    print(f"TABLE 3 {tag} (Unseen-6 mean)")
    for d, short, _ in M.PLAN:
        for K in M.KS:
            print(f"  {short:9s} K={K}: " + "  ".join(f"{n} {V[(K, d)][n][0]:.3f}" for n in M.METS))
    print("effects:")
    for e in eff:
        if e[1] in ("AUC@20", "AUC@100", "Final"):
            print("  ", e)


if __name__ == "__main__":
    main(k1_mean="--k1-mean" in sys.argv, k12_mean="--k12-mean" in sys.argv)
