"""EXP6 V5 / F1 -- G3 aggregation (seen2scene env, CPU; generated outputs, do not edit by hand).
Models: B0 parent (CVPR Table 1 canonical records, read-only; same simulator / starts / seeds, chain regression bitwise),
A2k / A5k / B2k / B5k (work_dirs/exp6_v5/F1/g3/episodes). Groups: pilot = Dev-3 x starts {0, 1} (pre-registered, 6
episodes), extended = Dev-3 x starts {0..4} (15 episodes). Metrics (Table 1 rules): Cov@k = C[k], AUC@k = trapz(C[0..k]) / k,
k in {10, 20, 40, 50, 60, 80, 100}, Final = C[100] (== Cov@100 checked). Scene = mean over its starts (only complete
scenes), group = equal-weight scene mean; paired differences per scene (same starts), scene-level bootstrap 95% CI
(10k, seed 0), wins / losses. Selection (f1_eval.yaml): per arm the higher extended Dev-3 AUC@20 of 2k / 5k (|d| < 1e-4 -> 5k).
Outputs results/g3_per_episode[_mono].csv, g3_per_scene[_mono].csv, g3_curves.csv, summary.json, checkpoints.csv and
reports/G3_PILOT.md, reports/G3_EXTENDED.md. Usage: python aggregate.py"""
import csv
import glob
import hashlib
import json
import os

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
F1 = os.path.dirname(HERE)
R = os.path.dirname(os.path.dirname(os.path.dirname(F1)))
W = os.path.join(R, "work_dirs", "exp6_v5", "F1")
T1E = os.path.join(R, "work_dirs", "cvpr_results", "table1", "episodes")
KS = (10, 20, 40, 50, 60, 80, 100)
SCENES = ("barts", "pantheon", "colosseum")
GROUPS = {"pilot": (0, 1), "extended": (0, 1, 2, 3, 4)}
MODELS = ("B0", "A2k", "A5k", "B2k", "B5k")
LABEL = {"B0": "B0 parent (V2-Mix EMA@17.5k)", "A2k": "F1-A Continue @2k", "A5k": "F1-A Continue @5k", "B2k": "F1-B Query @2k", "B5k": "F1-B Query @5k"}
STEPS = {"B0": 0, "A2k": 2000, "A5k": 5000, "B2k": 2000, "B5k": 5000}
PAIRS = (("B5k", "A5k"), ("B2k", "A2k"), ("B5k", "B0"), ("A5k", "B0"), ("B2k", "B0"), ("A2k", "B0"), ("A5k", "A2k"), ("B5k", "B2k"))


def metrics(c):
    c = np.asarray(c, float)
    out = {f"cov{k}": float(c[k]) for k in KS}
    out.update({f"auc{k}": float(np.trapz(c[:k + 1], dx=1.0) / k) for k in KS})
    out["final"] = float(c[100])
    assert out["final"] == out["cov100"]
    return out


def boot(v, n=10000, seed=0):
    v = np.asarray(v, float)
    if len(v) == 0:
        return [None, None]
    rng = np.random.default_rng(seed)
    b = v[rng.integers(0, len(v), (n, len(v)))].mean(1)
    return [float(np.quantile(b, 0.025)), float(np.quantile(b, 0.975))]


def load():
    E = {}
    for s in SCENES:
        for k in range(5):
            p = os.path.join(T1E, "B0", f"{s}_s{k}.json")
            if os.path.exists(p):
                E[("B0", s, k)] = json.load(open(p))
    for m in MODELS[1:]:
        for p in glob.glob(os.path.join(W, "g3", "episodes", m, "*.json")):
            r = json.load(open(p))
            E[(m, r["scene"], r["start"])] = r
    return E


def scene_vals(E, m, prot, starts, key):
    out = {}
    for s in SCENES:
        if all((m, s, k) in E for k in starts):
            out[s] = float(np.mean([metrics(E[(m, s, k)][f"coverage_{prot}"])[key] for k in starts]))
    return out


def summarize(E, prot):
    S = {"groups": {}}
    keys = [f"cov{k}" for k in KS] + [f"auc{k}" for k in KS] + ["final"]
    for g, starts in GROUPS.items():
        G = {"models": {}, "pairs": []}
        for m in MODELS:
            sv = {k: scene_vals(E, m, prot, starts, k) for k in keys}
            n_sc = len(sv["final"])
            row = {"n_scenes": n_sc, "n_episodes": sum((m, s, k) in E for s in SCENES for k in starts), "complete": n_sc == len(SCENES)}
            for k in keys:
                row[k] = float(np.mean(list(sv[k].values()))) if row["complete"] else None
                row[k + "_ci"] = boot(list(sv[k].values())) if row["complete"] else [None, None]
                row[k + "_per_scene"] = sv[k]
            G["models"][m] = row
        for a_, b_ in PAIRS:
            pr = {"a": a_, "b": b_}
            for k in ("auc20", "auc100", "final", "cov20", "auc50"):
                va, vb = scene_vals(E, a_, prot, starts, k), scene_vals(E, b_, prot, starts, k)
                if len(va) == len(SCENES) and len(vb) == len(SCENES):
                    d = [va[s] - vb[s] for s in SCENES]
                    pr[k] = {"d": float(np.mean(d)), "ci": boot(d), "wins": int(sum(x > 0 for x in d)), "losses": int(sum(x < 0 for x in d)),
                             "per_scene": dict(zip(SCENES, d))}
                    ep = [metrics(E[(a_, s, k_)][f"coverage_{prot}"])[k] - metrics(E[(b_, s, k_)][f"coverage_{prot}"])[k] for s in SCENES for k_ in starts]
                    pr[k]["episode_wins"], pr[k]["episode_losses"] = int(sum(x > 0 for x in ep)), int(sum(x < 0 for x in ep))
                else:
                    pr[k] = None
            G["pairs"].append(pr)
        S["groups"][g] = G
    ext = S["groups"]["extended"]["models"]
    sel = {}
    for arm in ("A", "B"):
        a2, a5 = ext[f"{arm}2k"]["auc20"], ext[f"{arm}5k"]["auc20"]
        sel[arm] = None if a2 is None or a5 is None else (f"{arm}2k" if a2 - a5 >= 1e-4 else f"{arm}5k")
    S["selection"] = {"rule": "per arm higher extended Dev-3 AUC@20 (official); |d| < 1e-4 -> 5k", "selected": sel}
    return S


def f3(x):
    return "—" if x is None else f"{x:.3f}"


def fd(p, k):
    if not p or p.get(k) is None:
        return "—"
    q = p[k]
    star = " ✱" if q["ci"][0] is not None and (q["ci"][0] > 0 or q["ci"][1] < 0) else ""
    return f"{q['d']:+.3f} [{q['ci'][0]:+.3f}, {q['ci'][1]:+.3f}] {q['wins']}/{q['losses']}（ep {q['episode_wins']}/{q['episode_losses']}）{star}"


def report(E, S, Smono, cost, g):
    G, Gm = S["groups"][g], Smono["groups"][g]
    starts = GROUPS[g]
    n_exp = len(SCENES) * len(starts)
    title = "G3_PILOT — 预注册 Pilot：Dev-3 × 起点 {0, 1} × 100 步" if g == "pilot" else "G3_EXTENDED — Dev-3 × 起点 {0..4} × 100 步"
    L = [f"# {title}（自动生成，`scripts/aggregate.py`）", "",
         "**口径**：",
         "- 主口径为 MAGICIAN 官方覆盖（与 Table 1 同一评价器，重放逐位一致）；",
         "- 场景等权，每个场景先对起点取均值；",
         "- 配对差为场景级 bootstrap 95% CI（10k，seed 0），并给出胜 / 负场景数及逐 episode 胜 / 负；",
         "- **只有 3 个场景，CI 只作探索性参考**。", "",
         "**B0 parent**：取自 CVPR Table 1 的规范记录。训练前已核对：F1 的启动链加 B0 权重重跑 barts_s0，与该记录逐位一致。", "",
         "## 主表（官方覆盖）", "",
         "| Model | Parent | Train/Infer/Score | Added Steps | Dev Episodes | AUC@20 | Cov@20 | AUC@100 | Final Cov@100 | Peak VRAM (GB) | s/Step |",
         "|---|---|---|---:|---:|---:|---:|---:|---:|---:|---:|"]
    for m in MODELS:
        r = G["models"][m]
        c = cost.get(m, {})
        L.append(f"| {LABEL[m]} | {'—' if m == 'B0' else 'B0 EMA@17.5k'} | S10/S10/S10 | {STEPS[m]} | {r['n_episodes']}/{n_exp} | {f3(r['auc20'])} | {f3(r['cov20'])} | "
                 f"{f3(r['auc100'])} | {f3(r['final'])} | {f3(c.get('peak_mem_gb'))} | {f3(c.get('decision_s'))} |")
    L += ["", "*Peak VRAM 为规划器进程的峰值显存；s/Step 为规划器每步决策时间的中位数（共享 GPU，只作数量级参考）。*", "",
          "## 配对差（官方覆盖）", "", "| A − B | ΔAUC@20 | ΔAUC@100 | ΔFinal | ΔCov@20 |", "|---|---|---|---|---|"]
    for p in G["pairs"]:
        L.append(f"| {p['a']} − {p['b']} | {fd(p, 'auc20')} | {fd(p, 'auc100')} | {fd(p, 'final')} | {fd(p, 'cov20')} |")
    L += ["", "## 完整指标（官方覆盖；均值 [场景 bootstrap CI]）", "",
          "| Model | " + " | ".join(f"Cov@{k}" for k in KS) + " | " + " | ".join(f"AUC@{k}" for k in KS) + " |", "|---|" + "---|" * (2 * len(KS))]
    for m in MODELS:
        r = G["models"][m]
        L.append(f"| {m} | " + " | ".join(f3(r[f'cov{k}']) for k in KS) + " | " + " | ".join(f3(r[f'auc{k}']) for k in KS) + " |")
    L += ["", "## 逐场景（官方覆盖，起点均值）", "", "| 场景 | " + " | ".join(f"{m} AUC@20 / AUC@100 / Final" for m in MODELS) + " |", "|---|" + "---|" * len(MODELS)]
    for s in SCENES:
        cells = []
        for m in MODELS:
            r = G["models"][m]
            v = [r[f"{k}_per_scene"].get(s) for k in ("auc20", "auc100", "final")]
            cells.append(" / ".join(f3(x) for x in v))
        L.append(f"| {s} | " + " | ".join(cells) + " |")
    L += ["", "## 逐 episode（官方覆盖：AUC@20 / AUC@100 / Final）", "", "| 场景 | 起点 | " + " | ".join(MODELS) + " |", "|---|---|" + "---|" * len(MODELS)]
    for s in SCENES:
        for k in starts:
            cells = []
            for m in MODELS:
                if (m, s, k) in E:
                    mm = metrics(E[(m, s, k)]["coverage_official"])
                    cells.append(f"{mm['auc20']:.3f} / {mm['auc100']:.3f} / {mm['final']:.3f}")
                else:
                    cells.append("未完成")
            L.append(f"| {s} | {k} | " + " | ".join(cells) + " |")
    L += ["", "## 辅口径（单调 S10 覆盖）", "", "| Model | AUC@20 | AUC@100 | Final |", "|---|---|---|---|"]
    for m in MODELS:
        r = Gm["models"][m]
        L.append(f"| {m} | {f3(r['auc20'])} | {f3(r['auc100'])} | {f3(r['final'])} |")
    L += ["", "| A − B（单调） | ΔAUC@20 | ΔAUC@100 | ΔFinal |", "|---|---|---|---|"]
    for p in Gm["pairs"][:2]:
        L.append(f"| {p['a']} − {p['b']} | {fd(p, 'auc20')} | {fd(p, 'auc100')} | {fd(p, 'final')} |")
    miss = [f"{m}:{s}:s{k}" for m in MODELS for s in SCENES for k in starts if (m, s, k) not in E]
    L += ["", f"**未完成的 episode**：{', '.join(miss) if miss else '无'}"]
    if g == "extended":
        L += ["", f"**checkpoint 选择（预注册规则，只用 Dev-3 扩展组）**：{S['selection']['selected']}"]
    return "\n".join(L) + "\n"


def main():
    E = load()
    S = {p: summarize(E, p) for p in ("official", "mono")}
    os.makedirs(os.path.join(F1, "results"), exist_ok=True)
    cost = {}
    for m in MODELS:
        ds = [E[x]["decision_s_median"] for x in E if x[0] == m and E[x].get("decision_s_median") is not None]
        mem = [E[x]["planner_peak_mem_gb"] for x in E if x[0] == m and E[x].get("planner_peak_mem_gb")]
        if m == "B0":                                        # Table 1 records carry the episode file only
            for x in E:
                if x[0] == "B0":
                    ep = json.load(open(os.path.join(R, E[x]["episode_file"])))
                    st = [y["planner"]["decision_s"] for y in ep["steps"] if y.get("planner") and y["planner"].get("decision_s") is not None]
                    ds.append(float(np.median(st)))
                    mem.append((ep.get("planner_bye") or {}).get("peak_mem_gb") or np.nan)
        cost[m] = {"decision_s": float(np.median(ds)) if ds else None, "peak_mem_gb": float(np.nanmax(mem)) if mem else None}
    for prot in ("official", "mono"):
        suf = "" if prot == "official" else "_mono"
        with open(os.path.join(F1, "results", f"g3_per_episode{suf}.csv"), "w", newline="") as f:
            w = csv.writer(f)
            w.writerow(["model", "scene", "start", "group"] + [f"cov{k}" for k in KS] + [f"auc{k}" for k in KS] + ["final"])
            for (m, s, k) in sorted(E, key=lambda x: (MODELS.index(x[0]), x[1], x[2])):
                mm = metrics(E[(m, s, k)][f"coverage_{prot}"])
                w.writerow([m, s, k, "pilot" if k in GROUPS["pilot"] else "extended_only"] + [f"{mm[f'cov{q}']:.6f}" for q in KS] + [f"{mm[f'auc{q}']:.6f}" for q in KS] + [f"{mm['final']:.6f}"])
        with open(os.path.join(F1, "results", f"g3_per_scene{suf}.csv"), "w", newline="") as f:
            w = csv.writer(f)
            w.writerow(["group", "model", "scene", "auc20", "auc100", "final", "cov20"])
            for g in GROUPS:
                for m in MODELS:
                    r = S[prot]["groups"][g]["models"][m]
                    for s in SCENES:
                        if s in r["final_per_scene"]:
                            w.writerow([g, m, s] + [f"{r[f'{k}_per_scene'][s]:.6f}" for k in ("auc20", "auc100", "final", "cov20")])
    with open(os.path.join(F1, "results", "g3_curves.csv"), "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["model", "scene", "start", "protocol"] + [f"c{t}" for t in range(101)])
        for (m, s, k) in sorted(E, key=lambda x: (MODELS.index(x[0]), x[1], x[2])):
            for prot in ("official", "mono"):
                w.writerow([m, s, k, prot] + [f"{x:.6f}" for x in E[(m, s, k)][f"coverage_{prot}"]])
    with open(os.path.join(F1, "results", "checkpoints.csv"), "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["model", "arm", "step", "path", "sha256"])
        for arm in ("A", "B"):
            for p in sorted(glob.glob(os.path.join(W, "runs", arm, "ema_*.pt"))):
                h = hashlib.sha256(open(p, "rb").read()).hexdigest()
                w.writerow([f"F1-{arm}", arm, int(os.path.basename(p)[4:10]), os.path.relpath(p, R), h])
    json.dump({"official": S["official"], "mono": S["mono"], "cost": cost, "n_records": len(E)}, open(os.path.join(F1, "results", "summary.json"), "w"), indent=1)
    for g, fn in (("pilot", "G3_PILOT.md"), ("extended", "G3_EXTENDED.md")):
        open(os.path.join(F1, "reports", fn), "w").write(report(E, S["official"], S["mono"], cost, g))
    print("records", len(E), "selection", S["official"]["selection"]["selected"])


if __name__ == "__main__":
    main()
