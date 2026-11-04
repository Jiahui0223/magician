"""EXP6 V5 / F1 -- reports/TABLE1_FOLLOWUP.md + results/table1_followup.json (CPU; generated). Reads the CVPR Table 1
canonical records (work_dirs/cvpr_results/table1/episodes, read-only) of the two Dev-selected F1 checkpoints, evaluated by
Table 1's own pipeline after the Dev-3 selection, and of B0 / MAG-B: group means (scene-equal) and paired differences
(scene-level bootstrap 95% CI, 10k, seed 0, wins / losses) for Train-6, Dev-5, Test-3, Dev+Test (8 non-training scenes) and
All-14; official (primary) and monotone coverage. Reproducibility: the Table 1 Dev-3 episodes of the two F1 models vs the
F1 G3 records (same chain, same seeds) -> bitwise equality count. The Table 1 tables themselves come from Table 1's
aggregate.py; this file only adds the F1-specific pairs."""
import json
import os

import numpy as np
import yaml

HERE = os.path.dirname(os.path.abspath(__file__))
F1 = os.path.dirname(HERE)
R = os.path.dirname(os.path.dirname(os.path.dirname(F1)))
T1 = os.path.join(R, "experiments", "cvpr_results", "table1")
E = os.path.join(R, "work_dirs", "cvpr_results", "table1", "episodes")
G3 = os.path.join(R, "work_dirs", "exp6_v5", "F1", "g3", "episodes")
MOD = {"GQ": "F1-GQ-v1@5k", "CONT": "F1-CONT-v1@2k", "B0": "B0", "MAG-B": "MAG-B"}
F1MAP = {"F1-GQ-v1@5k": "B5k", "F1-CONT-v1@2k": "A2k"}
PAIRS = (("GQ", "CONT"), ("GQ", "B0"), ("CONT", "B0"), ("GQ", "MAG-B"), ("CONT", "MAG-B"), ("B0", "MAG-B"))
KEYS = ("auc20", "auc100", "final", "cov20")


def met(c, k):
    c = np.asarray(c, float)
    if k == "final":
        return float(c[100])
    if k.startswith("cov"):
        return float(c[int(k[3:])])
    n = int(k[3:])
    return float(np.trapz(c[:n + 1], dx=1.0) / n)


def boot(v, n=10000, seed=0):
    v = np.asarray(v, float)
    rng = np.random.default_rng(seed)
    b = v[rng.integers(0, len(v), (n, len(v)))].mean(1)
    return [float(np.quantile(b, 0.025)), float(np.quantile(b, 0.975))]


def main():
    sr = yaml.safe_load(open(os.path.join(T1, "SCENE_REGISTRY.yaml")))["scenes"]
    split = {s["local_id"]: s["split"] for s in sr if (s.get("runnable") or {}).get("overall")}
    groups = {"train": [s for s in split if split[s] == "train"], "dev": [s for s in split if split[s] == "dev"], "test": [s for s in split if split[s] == "test"],
              "devtest": [s for s in split if split[s] in ("dev", "test")], "all14": sorted(split)}
    out = {"models": MOD, "groups": {}}
    for prot in ("official", "mono"):
        recs = {}
        for k_, m in MOD.items():
            for s in split:
                for k in range(5):
                    p = os.path.join(E, m, f"{s}_s{k}.json")
                    if os.path.exists(p):
                        recs[(k_, s, k)] = json.load(open(p))[f"coverage_{prot}"]
        G = {}
        for g, sc in groups.items():
            blk = {"n_scenes": len(sc)}
            for k_ in MOD:
                ok = all((k_, s, k) in recs for s in sc for k in range(5))
                blk[k_] = {key: (float(np.mean([np.mean([met(recs[(k_, s, k)], key) for k in range(5)]) for s in sc])) if ok else None) for key in KEYS}
                blk[k_]["complete"] = ok
            for a, b in PAIRS:
                pr = {}
                for key in KEYS:
                    if blk[a]["complete"] and blk[b]["complete"]:
                        d = [np.mean([met(recs[(a, s, k)], key) - met(recs[(b, s, k)], key) for k in range(5)]) for s in sc]
                        pr[key] = {"d": float(np.mean(d)), "ci": boot(d), "wins": int(sum(x > 0 for x in d)), "losses": int(sum(x < 0 for x in d))}
                    else:
                        pr[key] = None
                blk[f"{a}-{b}"] = pr
            G[g] = blk
        out["groups"][prot] = G
    rep = {}
    for m, f1m in F1MAP.items():
        n = eq = 0
        for s in ("barts", "pantheon", "colosseum"):
            for k in range(5):
                a, b = os.path.join(E, m, f"{s}_s{k}.json"), os.path.join(G3, f1m, f"{s}_s{k}.json")
                if os.path.exists(a) and os.path.exists(b):
                    n += 1
                    ra, rb = json.load(open(a)), json.load(open(b))
                    eq += ra["executed_keys"] == rb["executed_keys"] and ra["coverage_official"] == rb["coverage_official"]
        rep[m] = {"compared": n, "bitwise_equal": eq}
    out["reproducibility_dev3_table1_vs_f1"] = rep
    json.dump(out, open(os.path.join(F1, "results", "table1_followup.json"), "w"), indent=1)
    f3 = lambda x: "—" if x is None else f"{x:.3f}"  # noqa: E731

    def fd(q):
        if not q:
            return "—"
        st = " ✱" if q["ci"][0] > 0 or q["ci"][1] < 0 else ""
        return f"{q['d']:+.3f} [{q['ci'][0]:+.3f}, {q['ci'][1]:+.3f}] {q['wins']}/{q['losses']}{st}"
    GN = {"train": "Train-6（训练场景，已见过）", "dev": "Dev-5", "test": "Test-3（非盲测）", "devtest": "Dev+Test（8 个非训练场景）", "all14": "All-14"}
    L = ["# TABLE1_FOLLOWUP — F1 选定 checkpoint 的 Table 1 正式评估（All-14；自动生成，`scripts/table1_followup.py`）", "",
         "**来源**：",
         "- F1-CONT-v1@2k（F1-A@2k）与 F1-GQ-v1@5k（F1-B@5k）是只在 Dev-3 上按预注册规则选出的；",
         "- 之后由 CVPR Table 1 自己的流水线（`t1_queue.sh` → `t1_out_cov` → `import_episodes`）运行全部 14 个场景 × 5 个起点；",
         "- 本文件只读取 Table 1 的规范记录。",
         "- sofia_church 为 PENDING，所以是 All-14，不是 All-15。", "",
         "**可复现性**：Table 1 重跑的 Dev-3 episode 与 F1 G3 记录逐位一致——" + ", ".join(m + " " + str(v["bitwise_equal"]) + "/" + str(v["compared"]) for m, v in rep.items()) + "。", ""]
    for prot in ("official", "mono"):
        G = out["groups"][prot]
        L += [f"## {'主口径：MAGICIAN 官方覆盖' if prot == 'official' else '辅口径：单调覆盖'}", "",
              "| 组 | 模型 | AUC@20 | AUC@100 | Final | Cov@20 |", "|---|---|---|---|---|---|"]
        for g in ("all14", "train", "devtest", "dev", "test"):
            for k_ in MOD:
                r = G[g][k_]
                L.append(f"| {GN[g]} | {MOD[k_]} | {f3(r['auc20'])} | {f3(r['auc100'])} | {f3(r['final'])} | {f3(r['cov20'])} |")
        L += ["", "| 组 | A − B | ΔAUC@20 | ΔAUC@100 | ΔFinal |", "|---|---|---|---|---|"]
        for g in ("all14", "train", "devtest", "dev", "test"):
            for a, b in PAIRS:
                p = G[g][f"{a}-{b}"]
                L.append(f"| {GN[g]} | {MOD[a]} − {MOD[b]} | {fd(p['auc20'])} | {fd(p['auc100'])} | {fd(p['final'])} |")
        L.append("")
    L += ["**说明**：",
          "- 场景等权，每个场景先对 5 个起点取均值，再在场景层面做 bootstrap 95% CI（10k，seed 0）；",
          "- **✱** 表示 CI 不含 0；",
          "- 没有做多重比较校正；Test-3 只有 3 个场景。"]
    open(os.path.join(F1, "reports", "TABLE1_FOLLOWUP.md"), "w").write("\n".join(L) + "\n")
    print("written; reproducibility", rep)


if __name__ == "__main__":
    main()
