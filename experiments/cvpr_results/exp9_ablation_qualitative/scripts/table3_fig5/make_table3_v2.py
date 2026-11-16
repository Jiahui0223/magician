"""Table 3 v2 (2026-11-16, user request): the K x planner table of make_table3.py restricted to the Unseen-6 scenes, i.e. the same outdoor
evaluation units as the main Table 1 v4 (alhambra, neuschwanstein, eiffel, bridge, barts, sestino_museum: the 6 Macarons++ scenes that
neither MAGICIAN / MACARONS nor our model were trained on; = All-14 minus MAGICIAN's 8 training scenes, make_table1_v4.py / make_fig3_v3.py).
Same 8 closed-loop cells, same loader (make_table3.data(): manifest COMPLETE rows, replay-verified), same metrics / layout / bold rule; only
the scene set changes (6 scenes x 5 official starts = 30 episodes per cell). Paired effects (scene bootstrap, 10,000, seed 0) are recomputed
on Unseen-6. The All-14 version (make_table3.py -> table3_hypothesis_planning.*) is unchanged.
Outputs fig/table3/: table3_hypothesis_planning_v2.{csv,tex,pdf,png}, table3_v2_effects.csv. Run with CUDA_VISIBLE_DEVICES="".
--k1-mean (v3, 2026-11-16): the K = 1 cells (One-Step and CDGR) are the per-scene mean over noise maps 0-3 (method ids V3-T0-K1D{1,2} and
-N1..N3, the replicates run for Table 3 on 2026-11-12), i.e. the same four maps K = 4 uses; a single map is one draw (Unseen-6 AUC@20 of
K=1+CDGR spans 0.412-0.438 across maps). Outputs table3_hypothesis_planning_v3.* and table3_v3_effects.csv."""
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


def main(k1_mean=False):
    E, tab = M.data()
    if k1_mean:                                                     # K = 1 cells := per-scene mean over noise maps 0..3 (the maps K = 4 uses)
        for d, ids in K1_REPS.items():
            reps = [L.scene_table(L.load_manifest(m), M.METS) for m in ids]
            assert all(len(r) == 14 for r in reps), [len(r) for r in reps]
            tab[(1, d)] = {s: {n: float(np.mean([r[s][n] for r in reps])) for n in M.METS} for s in L.SCENE_ORDER}
        M.NOTE = NOTE_V3
    V = {c: {n: L.boot_ci([tab[c][s][n] for s in UNSEEN6]) for n in M.METS} for c in M.CELL}
    rows = []
    for d, short, _ in M.PLAN:
        for K in M.KS:
            c = (K, d)
            rows.append([short, K, M.CELL[c] + (" (mean of N0-N3)" if (k1_mean and K == 1) else ""), 120 if (k1_mean and K == 1) else 30] + [f"{V[c][n][0]:.4f}" for n in M.METS] + [f"{V[c][n][1]:.4f}" for n in M.METS] + [f"{V[c][n][2]:.4f}" for n in M.METS])
    tag = "v3" if k1_mean else "v2"
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
                  r"scene-equal means; every cell is 30 real closed-loop episodes" + (r"; the $K{=}1$ rows average the same four noise maps that $K{=}4$ uses, 120 episodes each" if k1_mean else "") + ")")
    assert "Unseen-6" in t, "caption replacement failed"
    if k1_mean:
        t = t.replace("% Table 3 v2 (K x planner, Unseen-6)", "% Table 3 v3 (K x planner, Unseen-6, K=1 = mean over noise maps 0-3)")
    open(os.path.join(M.OUT, f"table3_hypothesis_planning_{tag}.tex"), "w").write(t)
    print(f"TABLE 3 {tag} (Unseen-6 mean)")
    for d, short, _ in M.PLAN:
        for K in M.KS:
            print(f"  {short:9s} K={K}: " + "  ".join(f"{n} {V[(K, d)][n][0]:.3f}" for n in M.METS))
    print("effects:")
    for e in eff:
        if e[1] in ("AUC@20", "AUC@100", "Final"):
            print("  ", e)


if __name__ == "__main__":
    main(k1_mean="--k1-mean" in sys.argv)
