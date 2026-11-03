"""EXP Density A/B -- G3 closed-loop analysis (seen2scene env, CPU; FROZEN_PROTOCOL.md §4).

Configs <CKPT>_<INF>_<SCORE>: the 5 historical ones read from the Task 3 / E2 roots (allowed only after the G0 replay
passed), the 7 new ones from work_dirs/exp_density_ab/g3. Episode readers = task3 t34_analyze (indoor: C_obs curve of
episode.json, AUC = trapz / moves; outdoor: monotone coverage of the metrics file), plus MP3D stall (e2_g3_analyze.stall),
per-step gain, revisits, decision time, peak memory. Scene-level paired bootstrap (t34 boot, 10k, seed 0) of the
pre-registered pairs, wins / losses, per-scene values. Only configs with all episodes present are summarised.
Outputs results/g3_summary.json, results/g3_episodes.csv, figures/g3_curves_<dataset>.png
"""
import csv
import itertools
import json
import os
import sys

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
EXP = os.path.dirname(HERE)
REPO = os.path.dirname(os.path.dirname(EXP))
T3S = os.path.join(REPO, "experiments", "exp5_future_view", "task3_full_depth_p1")
sys.path.insert(0, os.path.join(T3S, "scripts"))
sys.path.insert(0, os.path.join(T3S, "e2_pilot", "scripts"))
import t34_analyze as T  # noqa: E402
import e2_g3_analyze as EG  # noqa: E402

A = T.A
W3 = T.W3
NEW = os.path.join(REPO, "work_dirs", "exp_density_ab", "g3")
HIST = {"B2_S10_S10": "B0", "B2_D100_S10": "B2", "B2_D100_D100": "B1", "E2S_S10_S10": "E2S", "E2D_D100_S10": "E2D"}
CK, DN = ("B2", "E2S", "E2D"), ("S10", "D100")
CFGS = [f"{m}_{i}_{s}" for m in CK for i in DN for s in DN]


def roots(cfg):
    if cfg in HIST:
        g = HIST[cfg]
        return (os.path.join(W3, "closed_loop", "indoor", g, "V2M"), os.path.join(W3, "closed_loop", "outdoor", g, "V2M"),
                os.path.join(W3, "metrics", "outdoor", g, "V2M"))
    return (os.path.join(NEW, "closed_loop", "indoor", cfg, "V2M"), os.path.join(NEW, "closed_loop", "outdoor", cfg, "V2M"),
            os.path.join(NEW, "metrics", "outdoor", cfg, "V2M"))


def ep_in(cfg, ep):
    d = os.path.join(roots(cfg)[0], ep["tag"])
    if not os.path.exists(os.path.join(d, "episode.json")):
        return None
    e = json.load(open(os.path.join(d, "episode.json")))
    pl = [json.loads(x) for x in open(os.path.join(d, "planner.jsonl"))]
    c = np.asarray(e["coverage"])
    ex = [tuple(p) for p in e["executed"]]
    pos = [p[:3] for p in ex]
    return {"auc": float(np.trapz(c, dx=1) / e["moves"]), "final": float(c[-1]), "curve": c.tolist(),
            **{f"cov{x}": float(c[x]) for x in (10, 20, 50, 100)}, "gain_per_step": float((c[-1] - c[0]) / e["moves"]),
            "revisit_pose": float(np.mean([ex[i] in set(ex[:i]) for i in range(1, len(ex))])),
            "revisit_position": float(np.mean([pos[i] in set(pos[:i]) for i in range(1, len(pos))])),
            "translations": int(sum(pos[i] != pos[i - 1] for i in range(1, len(pos)))),
            "stall": EG.stall(c), "time_per_move": float(np.mean([x["timing"]["decision_s"] for x in pl])),
            "peak_gb": float(max(x.get("peak_mem_gb", 0) for x in pl)), "executed": [list(p) for p in ex]}


def ep_out(cfg, ep):
    s, k = ep["scene"], ep["start"]
    tg = f"V2M_{s}_s{k}_r0"
    _, ro, rm = roots(cfg)
    pm, pe = os.path.join(rm, f"{tg}.json"), os.path.join(ro, "episodes", f"ep_{tg}.json")
    if not (os.path.exists(pm) and os.path.exists(pe)):
        return None
    e, m = json.load(open(pe)), json.load(open(pm))
    pl = json.load(open(os.path.join(ro, "planner_logs", f"pl_{tg}.json")))
    c = np.asarray(m["cov_mono"])
    keys = e["executed_keys"]
    return {"auc": float(m["auc_mono"]), "final": float(m["cov_mono_final"]), "curve": c.tolist(),
            "auc_official": float(m["auc_official"]), "final_official": float(m["cov_official_final"]),
            **{f"cov{x}": float(c[x]) for x in (10, 20, 50)}, "gain_per_step": float((c[-1] - c[0]) / (len(c) - 1)),
            "revisit_pose": float(np.mean([keys[i] in set(keys[:i]) for i in range(1, len(keys))])),
            "hover": int(e.get("n_hover", 0)), "time_per_move": float(np.mean([st["decision_s"] for st in pl["steps"]])),
            "peak_gb": float(pl.get("peak_mem_gb_this_process", 0))}


def pairs():
    out = []
    for m in CK:
        for s in DN:
            out.append((f"{m}_D100_{s}", f"{m}_S10_{s}", f"inference D100-S10 ({m}, score {s})"))
        for i in DN:
            out.append((f"{m}_{i}_D100", f"{m}_{i}_S10", f"scoring D100-S10 ({m}, inference {i})"))
    for i, s in itertools.product(DN, DN):
        out.append((f"E2D_{i}_{s}", f"E2S_{i}_{s}", f"E2D-E2S (inf {i}, score {s})"))
        out.append((f"E2S_{i}_{s}", f"B2_{i}_{s}", f"E2S-B2 (inf {i}, score {s})"))
        out.append((f"E2D_{i}_{s}", f"B2_{i}_{s}", f"E2D-B2 (inf {i}, score {s})"))
    out.append(("E2D_D100_S10", "E2S_S10_S10", "historical E2D - E2S (matched train/test)"))
    return out


def summarize(rows, dataset, avail):
    keys = ["auc", "final", "cov10", "cov20", "cov50", "gain_per_step", "revisit_pose", "time_per_move", "peak_gb"]
    keys += ["auc_official", "final_official", "hover"] if dataset == "outdoor" else ["cov100", "revisit_position", "translations", "stall"]
    scenes = sorted({r["scene"] for r in rows}, key=lambda s: (A.SPLIT_OF.get(s, ""), s))
    S = {"n_episodes": len(rows), "scenes": scenes, "configs": avail, "means": {}, "paired": {}, "per_scene": {}}
    for k in keys:
        for c in avail:
            S["means"].setdefault(k, {})[c] = T.boot([np.mean([r[c][k] for r in rows if r["scene"] == s]) for s in scenes])
            S["means"][k][c]["episode_mean"] = float(np.mean([r[c][k] for r in rows]))
    for hi, lo, fam in pairs():
        if hi not in avail or lo not in avail:
            continue
        for k in ("auc", "final", "cov20", "cov50", "gain_per_step", "revisit_pose", "time_per_move"):
            d = [np.mean([r[hi][k] - r[lo][k] for r in rows if r["scene"] == s]) for s in scenes]
            de = [r[hi][k] - r[lo][k] for r in rows]
            S["paired"][f"{hi} - {lo}|{k}"] = dict(T.boot(d), per_scene=dict(zip(scenes, map(float, d))), wins=int(sum(x > 0 for x in d)),
                                                    losses=int(sum(x < 0 for x in d)), ep_wins=int(sum(x > 0 for x in de)),
                                                    ep_losses=int(sum(x < 0 for x in de)), episode_boot=T.boot(de), family=fam)
    for s in scenes:
        S["per_scene"][s] = {c: {k: float(np.mean([r[c][k] for r in rows if r["scene"] == s])) for k in ("auc", "final")} for c in avail}
    if dataset == "indoor":
        S["stall_episodes"] = {c: int(sum(r[c]["stall"] >= 30 for r in rows)) for c in avail}
    S["mean_curve"] = {c: np.mean([r[c]["curve"] for r in rows], 0).tolist() for c in avail}
    return S


def main():
    out, csv_rows = {}, []
    for dataset in ("indoor", "outdoor"):
        rows = []
        for ep in A.episodes(dataset):
            r = {"dataset": dataset, "scene": ep["scene"], "start": ep["start"], "split": ep.get("split", "mp3d_val")}
            for c in CFGS:
                x = ep_in(c, ep) if dataset == "indoor" else ep_out(c, ep)
                if x is not None:
                    r[c] = x
            rows.append(r)
        avail = [c for c in CFGS if all(c in r for r in rows)]
        partial = {c: sum(c in r for r in rows) for c in CFGS if c not in avail}
        S = summarize(rows, dataset, avail)
        S["partial_configs"] = partial
        out[dataset] = S
        for r in rows:
            for c in avail:
                csv_rows.append([dataset, r["scene"], r["start"], c] + [r[c].get(k) for k in ("auc", "final", "cov20", "cov50", "gain_per_step",
                                                                                             "revisit_pose", "time_per_move", "peak_gb")])
        # curves
        plt = T.plt
        T.ST.setup()
        fig, ax = plt.subplots(1, 1, figsize=(9, 5.2))
        cols = {"B2": ("#86b6ef", "#1c5cab"), "E2S": ("#f6c49f", "#d9711c"), "E2D": ("#a7dfc3", "#1b8a55")}
        for c in avail:
            m, i, s = c.split("_")
            ax.plot(S["mean_curve"][c], color=cols[m][1 if i == "D100" else 0], lw=1.8, ls="-" if s == "S10" else "--",
                    label=f"{m} inf {i} / score {s}  AUC {S['means']['auc'][c]['mean']:.3f}")
        ax.set_xlabel("move")
        ax.set_ylabel("coverage")
        ax.set_title(f"{'MP3D (C_obs), 10 episodes' if dataset == 'indoor' else 'Macarons++ (monotone), 30 episodes'} -- solid: S10 scoring, dashed: D100 scoring",
                     loc="left", fontsize=10)
        ax.legend(fontsize=7.5, loc="lower right", ncol=1)
        T.ST.clean(ax, "both")
        fig.tight_layout()
        os.makedirs(os.path.join(EXP, "figures"), exist_ok=True)
        fig.savefig(os.path.join(EXP, "figures", f"g3_curves_{dataset}.png"), dpi=130)
        plt.close(fig)
    json.dump(out, open(os.path.join(EXP, "results", "g3_summary.json"), "w"), indent=1, default=float)
    with open(os.path.join(EXP, "results", "g3_episodes.csv"), "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["dataset", "scene", "start", "config", "auc", "final", "cov20", "cov50", "gain_per_step", "revisit_pose", "time_per_move_s", "peak_gb"])
        w.writerows(csv_rows)
    for d in ("indoor", "outdoor"):
        S = out[d]
        print(d, "available", S["configs"], "partial", S["partial_configs"])
        for c in S["configs"]:
            M = S["means"]
            print(f"  {c:14s} AUC {M['auc'][c]['mean']:.4f} Final {M['final'][c]['mean']:.4f} Cov@20 {M['cov20'][c]['mean']:.4f} "
                  f"Cov@50 {M['cov50'][c]['mean']:.4f} t/move {M['time_per_move'][c]['mean']:.2f}s peak {M['peak_gb'][c]['mean']:.1f}GB")


if __name__ == "__main__":
    main()
