"""Experiment U -- extra CPU-only analysis for the final report (reads existing records and planner logs; changes no result).
  1. Random Walk reference (archived Table 1 row RANDOM, load_manifest) per group + paired U - Random (scene bootstrap 10,000, seed 0)
  2. per-scene AUC@20 / AUC@100 / Final for D1, C1, U1-U3, Random
  3. exploration breadth: distinct executed poses per episode (episode executed_keys)
  4. U planner behaviour from the step logs: share of steps whose U pick equals the C1 argmax of the same state (u_c1, same predictions),
     C1 score of the pick / max C1 score, support of the picked candidate (n_rc, n_r2, n_cells_union), fallback steps
  5. early stage: scene-equal mean curves at t = 5, 10, 15, 20 and the per-step U - C1 gap
  6. Figure U1b: the u_collect Figure U1 curves plus the Random reference (dashed)
Outputs results/closed_loop/final_extras.json, per_scene_with_random.csv, figures/U1b_coverage_curves_with_random.png."""
import glob
import json
import os
import sys

import numpy as np

R = "/bsuscratch/jiahuizhang/projects/occupancy/seen2scene"
HERE = os.path.dirname(os.path.abspath(__file__))
UD = os.path.dirname(HERE)
sys.path.insert(0, HERE)
sys.path.insert(0, os.path.join(R, "experiments", "cvpr_results", "scripts"))
import cvpr_lib as L  # noqa: E402
import u_collect as UC  # noqa: E402

OUT = os.path.join(UD, "results", "closed_loop")
N = ["AUC@10", "AUC@20", "Cov@20", "AUC@50", "AUC@100", "Final"]


def main():
    M = {"D1": L.load_manifest("V3-T0-D1"), "C1": L.load_manifest("V3-T0-K4D1"), "RANDOM": L.load_manifest("RANDOM")}
    for p in ("u1", "u2", "u3"):
        M[p], bad = UC.load_u(p)
        assert not bad, bad
    T = {k: L.scene_table(E, N) for k, E in M.items()}
    ex = {"groups": {}, "vs_random": {}, "breadth": {}, "behaviour": {}, "early": {}}
    for g in ("All-14", "Train-6", "Dev-5", "Test-3", "Dev+Test-8"):
        ex["groups"][g] = {k: {n: L.group_mean(T[k], L.GROUPS[g], n)[0] for n in N} for k in M}
        for p in ("u1", "u2", "u3", "C1", "D1"):
            ex["vs_random"].setdefault(g, {})[p] = {n: L.paired(T[p], T["RANDOM"], L.GROUPS[g], n) for n in ("AUC@20", "AUC@100", "Final")}
    rows = []
    for s in L.SCENE_ORDER:
        rows.append([s, L.SPLIT[s]] + [T[k][s][n] for k in ("D1", "C1", "u1", "u2", "u3", "RANDOM") for n in ("AUC@20", "AUC@100", "Final")])
    L.write_csv(os.path.join(OUT, "per_scene_with_random.csv"), ["scene", "split"] + [f"{k}_{n}" for k in ("D1", "C1", "u1", "u2", "u3", "RANDOM") for n in ("AUC@20", "AUC@100", "Final")], rows)
    # exploration breadth
    for k, E in M.items():
        d = []
        for r in E.values():
            ep = json.load(open(os.path.join(R, r["episode_file"])))
            keys = [json.dumps(x) for x in ep["executed_keys"]]
            d.append((r["scene"], len(set(keys)), sum(1 for i in range(1, len(keys)) if keys[i] == keys[i - 1])))
        per = {}
        for s, u, st in d:
            per.setdefault(s, []).append((u, st))
        ex["breadth"][k] = {"distinct_poses_scene_equal": float(np.mean([np.mean([a for a, _ in v]) for v in per.values()])),
                            "repeat_steps_scene_equal": float(np.mean([np.mean([b for _, b in v]) for v in per.values()]))}
    # U planner behaviour (closed-loop step logs)
    for p in ("u1", "u2", "u3"):
        same, ratio, nrc, nr2, ncell, fb, n = [], [], [], [], [], 0, 0
        for f in glob.glob(os.path.join(UC.WU, p, "*", "planner_logs", "*.steps.jsonl")):
            for line in open(f):
                s = json.loads(line)
                if not s.get("n"):
                    continue
                n += 1
                ch = s["choice"]
                fb += int(bool(ch.get("fallback")))
                j = ch["pick"]
                c1 = np.asarray(s["all_scores"]["u_c1"], float)
                same.append(float(c1.max() > 0 and j == int(np.flatnonzero(c1 == c1.max())[0])))
                if c1.max() > 0:
                    ratio.append(float(c1[j] / c1.max()))
                m = s["all_scores"]["u_metrics"][j]
                nrc.append(m["n_rc"]); nr2.append(m["n_r2"]); ncell.append(m["n_cells_union"])
        ex["behaviour"][p] = {"steps": n, "fallback_steps": fb, "pick_equals_C1_argmax": float(np.mean(same)), "c1_score_of_pick_over_max_mean": float(np.mean(ratio)),
                              "pick_n_rc_median": float(np.median(nrc)), "pick_n_r2_median": float(np.median(nr2)), "pick_n_r2_lt100_frac": float(np.mean(np.asarray(nr2) < 100)),
                              "pick_ncells_median": float(np.median(ncell))}
    # early stage
    curves = {k: L.mean_curve(E, L.GROUPS["All-14"])[0] for k, E in M.items()}
    ex["early"] = {k: {f"C@{t}": float(c[t]) for t in (1, 2, 3, 5, 10, 15, 20)} for k, c in curves.items()}
    gap = {p: (curves[p] - curves["C1"])[:21].tolist() for p in ("u1", "u2", "u3")}
    ex["early"]["gap_vs_C1_t0_20"] = gap
    json.dump(ex, open(os.path.join(OUT, "final_extras.json"), "w"), indent=1, default=float)
    # Figure U1b
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    col = {"D1": "#2a78d6", "C1": "#1baf7a", "u1": "#eb6834", "u2": "#6250d6", "u3": "#e34948", "RANDOM": "#777777"}
    lab = {"D1": "D1 (two-step coverage)", "C1": "C1 (one-step coverage)", "u1": "U1 hit disagreement", "u2": "U2 depth disagreement (U-best)",
           "u3": "U3 geometric disagreement", "RANDOM": "Random Walk (Table 1 reference)"}
    fig, axs = plt.subplots(1, 2, figsize=(11, 3.8))
    for ax, hi in zip(axs, (100, 20)):
        for k in ("D1", "C1", "u2", "u1", "u3", "RANDOM"):
            ax.plot(np.arange(hi + 1), curves[k][:hi + 1], lw=2, color=col[k], ls="--" if k == "RANDOM" else "-", label=lab[k])
        ax.set_xlabel("step"); ax.set_ylabel("official coverage (All-14, scene-equal)"); ax.set_title(f"steps 0-{hi}")
        ax.spines["top"].set_visible(False); ax.spines["right"].set_visible(False); ax.grid(color="#e6e6e6", lw=0.6)
    axs[0].legend(frameon=False, fontsize=7)
    fig.suptitle("Figure U1 -- closed-loop coverage, 14 scenes x 5 official starts, 100 steps (official MAGICIAN coverage)", fontsize=10)
    fig.tight_layout()
    fig.savefig(os.path.join(UD, "figures", "U1b_coverage_curves_with_random.png"), dpi=160)
    plt.close(fig)
    print(json.dumps({"breadth": ex["breadth"], "behaviour": ex["behaviour"], "early": {k: v for k, v in ex["early"].items() if k != "gap_vs_C1_t0_20"}}, indent=1))


if __name__ == "__main__":
    main()
