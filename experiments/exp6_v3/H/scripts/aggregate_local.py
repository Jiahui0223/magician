"""EXP6 V3 / H -- local analysis of the H0-H3 closed loops (seen2scene env, CPU; generated, do not edit by hand).
Single data source = the CVPR Table 1 canonical records (work_dirs/cvpr_results/table1/episodes; H0 = the V3 row, H1-H3 =
V3-H1 / V3-H2 / V3-H3; B0 / B2 / MAG-B as external references), read-only; planner step logs (Table 1 g3 closed-loop roots)
for decisions / runtime. Metrics (Table 1 rules): Cov@k = C[k], AUC@k = trapz(C[0..k]) / k, k in {10, 20, 40, 50, 60, 80, 100},
Final = C[100]; scene = mean over its 5 starts (complete scenes only), split = scene-equal mean; paired differences per
scene (same starts), scene-level bootstrap 95% CI (10k, seed 0), wins / losses; groups train / dev / test / all14 (+
partial labels when incomplete). Protocols: official (primary) and mono (secondary), never mixed.
Outputs results/{per_episode,per_scene,by_split,coverage_curves,pairwise,decisions}.csv, results/summary.json,
reports/G3_{DEV,TRAIN,TEST,ALL14}.md, reports/RUNTIME.md, reports/MISSING_EPISODES.md. Usage: python aggregate_local.py"""
import csv
import glob
import json
import os

import numpy as np
import yaml

HERE = os.path.dirname(os.path.abspath(__file__))
H = os.path.dirname(HERE)
R = os.path.dirname(os.path.dirname(os.path.dirname(H)))
T1 = os.path.join(R, "experiments", "cvpr_results", "table1")
E = os.path.join(R, "work_dirs", "cvpr_results", "table1", "episodes")
G3 = os.path.join(R, "work_dirs", "cvpr_results", "table1", "g3", "closed_loop", "outdoor")
KS = (10, 20, 40, 50, 60, 80, 100)
MODELS = {"H0": "V3", "H1": "V3-H1", "H2": "V3-H2", "H3": "V3-H3", "B0": "B0", "B2": "B2", "MAG-B": "MAG-B"}
LABEL = {"H0": "V3-T1 / H0 (DistP1 Greedy)", "H1": "V3-H1 Coverage-Aligned", "H2": "V3-H2 Uncertainty-Aware", "H3": "V3-H3 Short-Horizon2",
         "B0": "B0 (V2-Mix S10/S10)", "B2": "B2 (V2-Mix D100/S10)", "MAG-B": "MAGICIAN Beam 10x10"}
PAIRS = [("H1", "H0"), ("H2", "H1"), ("H3", "H2"), ("H2", "H0"), ("H3", "H0"), ("H3", "H1")] + \
        [(h, x) for h in ("H0", "H1", "H2", "H3") for x in ("B0", "B2", "MAG-B")]
GROUPS = ("dev", "test", "train", "all14")
GN = {"dev": "Dev-5", "test": "Test-3（非盲测）", "train": "Train-6（训练场景）", "all14": "All-14"}


def met(c):
    c = np.asarray(c, float)
    out = {f"cov{k}": float(c[k]) for k in KS}
    out.update({f"auc{k}": float(np.trapz(c[:k + 1], dx=1.0) / k) for k in KS})
    out["final"] = float(c[100])
    return out


def boot(v, n=10000, seed=0):
    v = np.asarray(v, float)
    rng = np.random.default_rng(seed)
    b = v[rng.integers(0, len(v), (n, len(v)))].mean(1)
    return [float(np.quantile(b, 0.025)), float(np.quantile(b, 0.975))]


def main():
    sr = yaml.safe_load(open(os.path.join(T1, "SCENE_REGISTRY.yaml")))["scenes"]
    split = {s["local_id"]: s["split"] for s in sr if (s.get("runnable") or {}).get("overall")}
    scenes = {g: sorted(s for s in split if split[s] == g) for g in ("train", "dev", "test")}
    scenes["all14"] = sorted(split)
    rec = {}
    for h, mid in MODELS.items():
        for s in split:
            for k in range(5):
                p = os.path.join(E, mid, f"{s}_s{k}.json")
                if os.path.exists(p):
                    rec[(h, s, k)] = json.load(open(p))
    keys = [f"cov{k}" for k in KS] + [f"auc{k}" for k in KS] + ["final"]
    S = {"protocols": {}}
    os.makedirs(os.path.join(H, "results"), exist_ok=True)
    for prot in ("official", "mono"):
        M = {x: met(r[f"coverage_{prot}"]) for x, r in rec.items()}
        P = {"groups": {}, "pairs": {}}
        for g in GROUPS:
            blk = {}
            for h in MODELS:
                done = [s for s in scenes[g] if all((h, s, k) in M for k in range(5))]
                row = {"n_scenes": len(done), "n_expected": len(scenes[g]), "n_episodes": sum((h, s, k) in M for s in scenes[g] for k in range(5)),
                       "complete": len(done) == len(scenes[g])}
                for key in keys:
                    v = [np.mean([M[(h, s, k)][key] for k in range(5)]) for s in done]
                    row[key] = float(np.mean(v)) if v else None
                    row[key + "_ci"] = boot(v) if len(v) > 1 else [None, None]
                    row[key + "_per_scene"] = {s: float(np.mean([M[(h, s, k)][key] for k in range(5)])) for s in done}
                blk[h] = row
            P["groups"][g] = blk
            pr = []
            for a, b in PAIRS:
                both = [s for s in scenes[g] if all((a, s, k) in M and (b, s, k) in M for k in range(5))]
                q = {"a": a, "b": b, "n_scenes": len(both), "complete": len(both) == len(scenes[g])}
                for key in ("auc10", "auc20", "auc50", "auc100", "cov20", "cov50", "final"):
                    d = [np.mean([M[(a, s, k)][key] - M[(b, s, k)][key] for k in range(5)]) for s in both]
                    q[key] = {"d": float(np.mean(d)), "ci": boot(d) if len(d) > 1 else [None, None], "wins": int(sum(x > 0 for x in d)),
                              "losses": int(sum(x < 0 for x in d)), "per_scene": dict(zip(both, map(float, d)))} if d else None
                pr.append(q)
            P["pairs"][g] = pr
        S["protocols"][prot] = P
        suf = "" if prot == "official" else "_mono"
        with open(os.path.join(H, "results", f"per_episode{suf}.csv"), "w", newline="") as f:
            w = csv.writer(f)
            w.writerow(["policy", "model_id", "scene", "split", "start"] + keys)
            for (h, s, k) in sorted(M, key=lambda x: (list(MODELS).index(x[0]), x[1], x[2])):
                w.writerow([h, MODELS[h], s, split[s], k] + [f"{M[(h, s, k)][q]:.6f}" for q in keys])
        with open(os.path.join(H, "results", f"per_scene{suf}.csv"), "w", newline="") as f:
            w = csv.writer(f)
            w.writerow(["policy", "scene", "split"] + keys)
            for h in MODELS:
                for s in sorted(split):
                    if all((h, s, k) in M for k in range(5)):
                        w.writerow([h, s, split[s]] + [f"{np.mean([M[(h, s, k)][q] for k in range(5)]):.6f}" for q in keys])
        with open(os.path.join(H, "results", f"by_split{suf}.csv"), "w", newline="") as f:
            w = csv.writer(f)
            w.writerow(["policy", "group", "complete", "n_scenes", "n_episodes"] + keys)
            for g in GROUPS:
                for h in MODELS:
                    r = P["groups"][g][h]
                    w.writerow([h, g, r["complete"], r["n_scenes"], r["n_episodes"]] + [("" if r[q] is None else f"{r[q]:.6f}") for q in keys])
        with open(os.path.join(H, "results", f"pairwise{suf}.csv"), "w", newline="") as f:
            w = csv.writer(f)
            w.writerow(["group", "a", "b", "metric", "d", "ci_lo", "ci_hi", "wins", "losses", "n_scenes"])
            for g in GROUPS:
                for q in P["pairs"][g]:
                    for key in ("auc10", "auc20", "auc50", "auc100", "cov20", "cov50", "final"):
                        if q[key]:
                            w.writerow([g, q["a"], q["b"], key, f"{q[key]['d']:.6f}", q[key]["ci"][0], q[key]["ci"][1], q[key]["wins"], q[key]["losses"], q["n_scenes"]])
    with open(os.path.join(H, "results", "coverage_curves.csv"), "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["policy", "scene", "start", "protocol"] + [f"c{t}" for t in range(101)])
        for (h, s, k), r in sorted(rec.items(), key=lambda x: (list(MODELS).index(x[0][0]), x[0][1], x[0][2])):
            for prot in ("official", "mono"):
                w.writerow([h, s, k, prot] + [f"{x:.6f}" for x in r[f"coverage_{prot}"]])
    # decisions / runtime from the planner step logs (H0 = historical V3 roots of the records)
    D = {}
    with open(os.path.join(H, "results", "decisions.csv"), "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["policy", "scene", "start", "steps", "fallback_steps", "revisit_steps", "decision_s_median", "decision_s_p90", "decision_s_total", "peak_mem_gb",
                    "h3_c2_unique_mean", "h3_blocked_mean", "h3_overlap_mean"])
        for h in ("H0", "H1", "H2", "H3"):
            for s in sorted(split):
                for k in range(5):
                    if (h, s, k) not in rec:
                        continue
                    r = rec[(h, s, k)]
                    ep_dir = os.path.dirname(os.path.dirname(os.path.join(R, r["episode_file"])))
                    pl = glob.glob(os.path.join(ep_dir, "planner_logs", f"pl_V2M_{s}_s{k}_r0.steps.jsonl"))
                    if not pl:
                        continue
                    L = [json.loads(l) for l in open(pl[0])]
                    dec = np.asarray([x["decision_s"] for x in L])
                    fb = sum(1 for x in L if (x.get("choice") or {}).get("fallback"))
                    keys_ = r["executed_keys"]
                    rv = sum(1 for i in range(1, len(keys_)) if keys_[i] in keys_[:i])
                    h3 = [x["all_scores"].get("h3") for x in L if x.get("all_scores") and x["all_scores"].get("h3")]
                    ov = [p["overlap_mean"] for z in h3 for p in z["per_c1"] if p.get("overlap_mean") is not None]
                    row = [h, s, k, len(L), fb, rv, float(np.median(dec)), float(np.quantile(dec, 0.9)), float(dec.sum()), max(x["peak_mem_gb"] for x in L),
                           float(np.mean([z["n_c2_unique"] for z in h3])) if h3 else "", float(np.mean([z["blocked"] for z in h3])) if h3 else "", float(np.mean(ov)) if ov else ""]
                    w.writerow(row)
                    D.setdefault(h, []).append(row)
    S["decisions"] = {h: {"episodes": len(v), "fallback_rate": float(np.sum([x[4] for x in v]) / np.sum([x[3] for x in v])),
                          "revisit_steps_mean": float(np.mean([x[5] for x in v])), "decision_s_median": float(np.median([x[6] for x in v])),
                          "decision_s_p90_median": float(np.median([x[7] for x in v])), "planning_s_per_episode_median": float(np.median([x[8] for x in v])),
                          "peak_mem_gb_max": float(np.max([x[9] for x in v])),
                          "h3_c2_unique_mean": float(np.mean([x[10] for x in v])) if v[0][10] != "" else None,
                          "h3_overlap_mean": float(np.mean([x[12] for x in v if x[12] != ""])) if v[0][12] != "" else None} for h, v in D.items()}
    json.dump(S, open(os.path.join(H, "results", "summary.json"), "w"), indent=1)
    reports(S, scenes, split)
    print({g: {h: S["protocols"]["official"]["groups"][g][h]["n_episodes"] for h in ("H0", "H1", "H2", "H3")} for g in GROUPS})


def f3(x):
    return "—" if x is None else f"{x:.3f}"


def fd(q, key):
    if not q or not q.get(key):
        return "—"
    z = q[key]
    st = " ✱" if z["ci"][0] is not None and (z["ci"][0] > 0 or z["ci"][1] < 0) else ""
    ci = f"[{z['ci'][0]:+.3f}, {z['ci'][1]:+.3f}]" if z["ci"][0] is not None else "[—]"
    return f"{z['d']:+.3f} {ci} {z['wins']}/{z['losses']}{st}"


def reports(S, scenes, split):
    dec = S.get("decisions", {})
    for g, fn in (("dev", "G3_DEV.md"), ("train", "G3_TRAIN.md"), ("test", "G3_TEST.md"), ("all14", "G3_ALL14.md")):
        P, Pm = S["protocols"]["official"], S["protocols"]["mono"]
        blk = P["groups"][g]
        L = [f"# {fn[:-3]} — {GN[g]}（{len(scenes[g])} 个场景 × 5 个起点 × 100 步；自动生成，`scripts/aggregate_local.py`）", "",
             "**数据来源**：只读的 CVPR Table 1 规范记录。H0 = Table 1 的 V3 行；H1–H3 = V3-H1 / V3-H2 / V3-H3。",
             "",
             "**口径**：主口径为 MAGICIAN 官方覆盖，辅口径单调覆盖在文末，两者不混用。",
             "",
             "**统计**：场景等权；配对差为场景级 bootstrap 95% CI（10k，seed 0），并给出胜 / 负场景数；**✱** 表示 CI 不含 0。", "",
             "## 主表（官方覆盖）", "", "| model_id | 场景 | episodes | AUC@20 | Cov@20 | AUC@100 | Final | AUC@10 | Cov@50 | 每步决策中位数（s，共享 GPU） |",
             "|---|---|---|---|---|---|---|---|---|---|"]
        for h in MODELS:
            r = blk[h]
            tag = "" if r["complete"] else f"（partial {r['n_scenes']}/{r['n_expected']}）"
            L.append(f"| {LABEL[h]}{tag} | {r['n_scenes']} | {r['n_episodes']} | {f3(r['auc20'])} | {f3(r['cov20'])} | {f3(r['auc100'])} | {f3(r['final'])} | "
                     f"{f3(r['auc10'])} | {f3(r['cov50'])} | {f3(dec.get(h, {}).get('decision_s_median'))} |")
        L += ["", "## 配对差（官方覆盖）", "", "| A − B | ΔAUC@20 | ΔAUC@100 | ΔFinal | ΔAUC@10 | ΔCov@20 |", "|---|---|---|---|---|---|"]
        for q in P["pairs"][g]:
            L.append(f"| {q['a']} − {q['b']} | {fd(q, 'auc20')} | {fd(q, 'auc100')} | {fd(q, 'final')} | {fd(q, 'auc10')} | {fd(q, 'cov20')} |")
        L += ["", "## 完整 Cov@k / AUC@k（官方覆盖，均值）", "", "| model_id | " + " | ".join(f"Cov@{k}" for k in KS) + " | " + " | ".join(f"AUC@{k}" for k in KS) + " |",
              "|---|" + "---|" * (2 * len(KS))]
        for h in MODELS:
            r = blk[h]
            L.append(f"| {h} | " + " | ".join(f3(r[f'cov{k}']) for k in KS) + " | " + " | ".join(f3(r[f'auc{k}']) for k in KS) + " |")
        L += ["", "## 逐场景（官方覆盖；AUC@20 / AUC@100 / Final，5 个起点均值）", "", "| 场景 | " + " | ".join(("H0", "H1", "H2", "H3", "MAG-B")) + " |", "|---|" + "---|" * 5]
        for s in scenes[g]:
            cells = []
            for h in ("H0", "H1", "H2", "H3", "MAG-B"):
                r = blk[h]
                v = [r[f"{k}_per_scene"].get(s) for k in ("auc20", "auc100", "final")]
                cells.append(" / ".join(f3(x) for x in v))
            L.append(f"| {s} | " + " | ".join(cells) + " |")
        L += ["", "## 辅口径（单调覆盖）", "", "| model_id | AUC@20 | AUC@100 | Final |", "|---|---|---|---|"]
        for h in MODELS:
            r = Pm["groups"][g][h]
            L.append(f"| {h} | {f3(r['auc20'])} | {f3(r['auc100'])} | {f3(r['final'])} |")
        L += ["", "| A − B（单调） | ΔAUC@20 | ΔAUC@100 | ΔFinal |", "|---|---|---|---|"]
        for q in Pm["pairs"][g][:6]:
            L.append(f"| {q['a']} − {q['b']} | {fd(q, 'auc20')} | {fd(q, 'auc100')} | {fd(q, 'final')} |")
        open(os.path.join(H, "reports", fn), "w").write("\n".join(L) + "\n")
    # runtime + missing
    L = ["# RUNTIME — 规划开销（自动生成）", "",
         "**来源**：Table 1 正式运行的规划器步日志。运行时 GPU 是共享的（每卡 2–3 个 worker），所以这里的时间是上界。受控独占计时见 `RUNTIME_CONTROLLED`（若有）。", "",
         "| 策略 | episodes | 每步决策 中位数 / P90（s） | 每 episode 规划总时长中位数（s） | 规划器峰值显存最大值（GB） | 回退步比例 | 每 episode 重复访问步数 | H3 第二层视角数 / 重叠率 |", "|---|---|---|---|---|---|---|---|"]
    for h in ("H0", "H1", "H2", "H3"):
        d = dec.get(h)
        if d:
            L.append(f"| {h} | {d['episodes']} | {d['decision_s_median']:.2f} / {d['decision_s_p90_median']:.2f} | {d['planning_s_per_episode_median']:.0f} | {d['peak_mem_gb_max']:.1f} | "
                     f"{d['fallback_rate']:.3f} | {d['revisit_steps_mean']:.1f} | {f3(d['h3_c2_unique_mean'])} / {f3(d['h3_overlap_mean'])} |")
    L += ["", "*H0 的步日志来自历史 V3 运行（Table 1 复用的记录），硬件负载与 H1–H3 的正式运行不同。*"]
    open(os.path.join(H, "reports", "RUNTIME.md"), "w").write("\n".join(L) + "\n")
    miss = []
    for h in ("H0", "H1", "H2", "H3"):
        for s in sorted(split):
            for k in range(5):
                if not os.path.exists(os.path.join(E, MODELS[h], f"{s}_s{k}.json")):
                    miss.append(f"| {h} | {s} | {k} |")
    open(os.path.join(H, "reports", "MISSING_EPISODES.md"), "w").write("# MISSING_EPISODES（自动生成）\n\n期望：H0–H3 × All-14 × 5 个起点（每个策略 70 个）。sofia_church 为 PENDING，不在 All-14 中。缺失的 episode 不插值、不补值。\n\n"
                                                                        + ("| 策略 | 场景 | 起点 |\n|---|---|---|\n" + "\n".join(miss) if miss else "**无缺失。**") + "\n")


if __name__ == "__main__":
    main()
