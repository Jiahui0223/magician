"""E2 Task 3 -- closed-loop comparison KSV vs D1 (CPU; read-only) -> results/closed_loop_summary.json, results/closed_loop_episodes.csv,
figures/closed_loop_curves.{png,pdf}.
KSV episodes: work_dirs/.../E2_observed_geometry/runs/ksv/episodes (simulator official coverage; rep r = planner noise key).
D1 references: rep 0 = CVPR Table 1 canonical D1 episodes (coverage_official); rep 1 / 2 = experiment E noise replicates
(work_dirs/.../experiment_E_stability_plateau/runs/rep/D1). Pairs share scene, start and rep (same noise keys, same GT seeds).
Stage 1: colosseum 5 starts x rep 0-2. Stage 2: Dev-5 rep 0 (scene = mean of 5 starts; scene-level bootstrap 10k seed 0).
Gate (configs/ksv_v1.yaml): Dev-5 (dAUC@100 >= +0.01 or dFinal >= +0.01) and dAUC@20 >= -0.005 and colosseum plateau count not higher."""
import csv
import glob
import json
import os

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
E2 = os.path.dirname(HERE)
R = os.path.dirname(os.path.dirname(os.path.dirname(os.path.dirname(os.path.dirname(E2)))))
W = os.path.join(R, "work_dirs", "exp6_v3", "ablation_study", "experiment_E_stability_plateau")
EP = os.path.join(R, "work_dirs", "cvpr_results", "table1", "episodes")
DEV = ["barts", "colosseum", "eiffel", "pantheon", "sestino_museum"]


def met(c):
    c = np.asarray(c, float)
    return {"auc20": float(np.trapz(c[:21]) / 20), "auc100": float(np.trapz(c) / 100), "final": float(c[100])}


def plan_time(ep_path, tag):
    p = os.path.join(os.path.dirname(os.path.dirname(ep_path)), "planner_logs", f"pl_{tag}.steps.jsonl")
    if not os.path.exists(p):
        return None
    return float(np.mean([json.loads(l)["decision_s"] for l in open(p)]))


def load():
    K, D = {}, {}
    for f in glob.glob(os.path.join(W, "E2_observed_geometry", "runs", "ksv", "episodes", "ep_*.json")):
        e = json.load(open(f))
        K[(e["scene"], int(e["start"]), int(e["rep"]))] = (e["coverage"], plan_time(f, e["tag"]))
    for s in DEV:
        for k in range(5):
            p = os.path.join(EP, "V3-D1-H3-NORISK", f"{s}_s{k}.json")
            if os.path.exists(p):
                r = json.load(open(p))
                D[(s, k, 0)] = (r["coverage_official"], plan_time(os.path.join(R, r["episode_file"]), f"V2M_{s}_s{k}_r0"))
    for f in glob.glob(os.path.join(W, "runs", "rep", "D1", "episodes", "ep_*.json")):
        e = json.load(open(f))
        D[(e["scene"], int(e["start"]), int(e["rep"]))] = (e["coverage"], plan_time(f, e["tag"]))
    return K, D


def boot(v, n=10000, seed=0):
    v = np.asarray(v, float)
    if len(v) < 2:
        return [None, None]
    b = v[np.random.default_rng(seed).integers(0, len(v), (n, len(v)))].mean(1)
    return [float(np.quantile(b, 0.025)), float(np.quantile(b, 0.975))]


def main():
    K, D = load()
    rows = []
    for key in sorted(set(K) & set(D)):
        mk, md = met(K[key][0]), met(D[key][0])
        rows.append({"scene": key[0], "start": key[1], "rep": key[2], **{f"ksv_{q}": mk[q] for q in mk}, **{f"d1_{q}": md[q] for q in md},
                     "ksv_plan_s": K[key][1], "d1_plan_s": D[key][1]})
    os.makedirs(os.path.join(E2, "results"), exist_ok=True)
    with open(os.path.join(E2, "results", "closed_loop_episodes.csv"), "w", newline="") as f:
        if rows:
            w = csv.DictWriter(f, fieldnames=list(rows[0]))
            w.writeheader()
            for r in rows:
                w.writerow(r)
    out = {"n_pairs": len(rows), "missing_ksv": sorted(f"{a}_s{b}_r{c}" for a, b, c in set(D) - set(K) if a == "colosseum" or c == 0)}
    col = [r for r in rows if r["scene"] == "colosseum"]
    out["stage1_colosseum"] = {"pairs": len(col), **{f"d_{q}": float(np.mean([r[f"ksv_{q}"] - r[f"d1_{q}"] for r in col])) if col else None for q in ("auc20", "auc100", "final")},
                               **{f"ci_{q}": boot([r[f"ksv_{q}"] - r[f"d1_{q}"] for r in col]) for q in ("auc20", "auc100", "final")},
                               "plateau_ksv": sum(r["ksv_final"] < 0.70 for r in col), "plateau_d1": sum(r["d1_final"] < 0.70 for r in col),
                               "ksv_mean": {q: float(np.mean([r[f"ksv_{q}"] for r in col])) if col else None for q in ("auc20", "auc100", "final")},
                               "d1_mean": {q: float(np.mean([r[f"d1_{q}"] for r in col])) if col else None for q in ("auc20", "auc100", "final")}}
    dev = [r for r in rows if r["rep"] == 0 and r["scene"] in DEV]
    scenes = [s for s in DEV if sum(1 for r in dev if r["scene"] == s) == 5]
    sc = {s: {q: (np.mean([r[f"ksv_{q}"] for r in dev if r["scene"] == s]), np.mean([r[f"d1_{q}"] for r in dev if r["scene"] == s])) for q in ("auc20", "auc100", "final")} for s in scenes}
    st2 = {"complete_scenes": scenes, "partial": len(scenes) < 5}
    for q in ("auc20", "auc100", "final"):
        d = [sc[s][q][0] - sc[s][q][1] for s in scenes]
        st2[f"ksv_{q}"] = float(np.mean([sc[s][q][0] for s in scenes])) if scenes else None
        st2[f"d1_{q}"] = float(np.mean([sc[s][q][1] for s in scenes])) if scenes else None
        st2[f"d_{q}"] = float(np.mean(d)) if d else None
        st2[f"ci_{q}"] = boot(d)
        st2[f"wins_{q}"] = int(sum(x > 1e-9 for x in d))
        st2[f"losses_{q}"] = int(sum(x < -1e-9 for x in d))
        st2[f"per_scene_{q}"] = {s: float(sc[s][q][0] - sc[s][q][1]) for s in scenes}
    st2["plateau_ksv"] = sum(r["ksv_final"] < 0.70 for r in dev)
    st2["plateau_d1"] = sum(r["d1_final"] < 0.70 for r in dev)
    st2["plan_s_ksv"] = float(np.mean([r["ksv_plan_s"] for r in dev if r["ksv_plan_s"]])) if dev else None
    st2["plan_s_d1"] = float(np.mean([r["d1_plan_s"] for r in dev if r["d1_plan_s"]])) if dev else None
    out["stage2_dev5"] = st2
    s1 = out["stage1_colosseum"]
    out["gate_all14"] = bool(scenes and len(scenes) == 5 and (st2["d_auc100"] >= 0.01 or st2["d_final"] >= 0.01) and st2["d_auc20"] >= -0.005
                             and s1["pairs"] == 15 and s1["plateau_ksv"] <= s1["plateau_d1"])
    json.dump(out, open(os.path.join(E2, "results", "closed_loop_summary.json"), "w"), indent=1, default=float)
    print(json.dumps({k: v for k, v in out.items() if k != "stage2_dev5"}, indent=1, default=float))
    print(json.dumps({k: v for k, v in st2.items() if not k.startswith("per_scene")}, indent=1, default=float))
    print("per scene", {q: {s: round(v, 3) for s, v in st2[f"per_scene_{q}"].items()} for q in ("auc20", "auc100", "final")})
    # figure: mean curves colosseum (15 pairs) and Dev-5 (rep 0)
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
        os.makedirs(os.path.join(E2, "figures"), exist_ok=True)
        fig, ax = plt.subplots(1, 2, figsize=(9, 3.4), sharey=True)
        for A, sel, title in ((ax[0], [k for k in K if k[0] == "colosseum" and k in D], "colosseum, 5 starts x 3 reps"),
                              (ax[1], [k for k in K if k[2] == 0 and k[0] in DEV and k in D], "Dev-5, rep 0")):
            if not sel:
                continue
            for src, col, lab in ((D, "#D55E00", "D1"), (K, "#0072B2", "D1 + KSV")):
                A.plot(np.mean([np.asarray(src[k][0]) for k in sel], 0), color=col, lw=1.6, label=lab)
            A.set_title(f"{title} ({len(sel)} episodes)", fontsize=9)
            A.set_xlabel("step")
            A.grid(alpha=0.3)
        ax[0].set_ylabel("official coverage")
        ax[0].legend(fontsize=8)
        fig.tight_layout()
        for ext in ("png", "pdf"):
            fig.savefig(os.path.join(E2, "figures", f"closed_loop_curves.{ext}"), dpi=200)
    except Exception as ex:
        print("figure skipped", ex)


if __name__ == "__main__":
    main()
