"""CVPR Table 1 -- aggregation of the canonical per-episode records (CPU; PROTOCOL.md §4-§6).

Input : work_dirs/cvpr_results/table1/episodes/<model>/<scene>_s<k>.json, one record per (model, scene, start):
        {"model", "scene", "start", "coverage_official": [C(0)..C(100)], "coverage_mono": [...] or null, "source", ...}
        experiments/cvpr_results/table1/SCENE_REGISTRY.yaml (scene -> split), MODEL_REGISTRY.yaml (model order / labels)
Metric : per curve c[0..100] -- Cov@k = c[k]; AUC@k = trapz(c[0..k], dx = 1) / k (the historical AUC@100 rule of
         task2/scripts/t25_out_cov.py, extended to every horizon); Final = c[100] (== Cov@100 by definition, checked).
Levels : episode -> scene (mean over its 5 starts; a scene counts only with all 5 starts) -> split / All-15 (equal weight
         over complete scenes; All-15 only when all 15 registry scenes are complete, otherwise a "partial-N" row).
Stats  : std over scenes, scene-level bootstrap 95% CI (10k, seed 0); paired difference vs MAG-B per scene (mean over the
         matched starts), bootstrap over scenes, wins / losses.
Output : results/*.csv, results/summary.json, TABLE1_FULL.csv, TABLE1*.md, TABLE1_LATEX.tex (all generated; never edit by hand)
Usage  : python aggregate.py [--protocol official|mono]   (official = primary, MAGICIAN-compatible)
"""
import argparse
import csv
import glob
import json
import os

import numpy as np
import yaml

HERE = os.path.dirname(os.path.abspath(__file__))
T1 = os.path.dirname(HERE)
REPO = os.path.dirname(os.path.dirname(os.path.dirname(T1)))
EPI = os.path.join(REPO, "work_dirs", "cvpr_results", "table1", "episodes")
KS = (10, 20, 40, 50, 60, 80, 100)
SPLITS = ("train", "dev", "test", "other")
REF = "MAG-B"
RUNNABLE = []          # scenes runnable under the frozen protocol (SCENE_REGISTRY runnable.overall) -> the All-14 group
N_START, T_MAX = 5, 100


def load_registries():
    sr = yaml.safe_load(open(os.path.join(T1, "SCENE_REGISTRY.yaml")))
    scenes = sr["scenes"] if isinstance(sr, dict) and "scenes" in sr else sr
    if isinstance(scenes, dict):
        scenes = [dict(v, local_id=v.get("local_id", k)) for k, v in scenes.items()]
    split = {}
    global RUNNABLE
    RUNNABLE = []
    for s in scenes:
        sp = str(s["split"]).lower()
        split[s["local_id"]] = "other" if sp.startswith("other") else sp
        if (s.get("runnable") or {}).get("overall"):
            RUNNABLE.append(s["local_id"])
    mr = yaml.safe_load(open(os.path.join(T1, "MODEL_REGISTRY.yaml")))
    models = mr["models"] if isinstance(mr, dict) and "models" in mr else mr
    if isinstance(models, dict):
        models = [dict(v, id=v.get("id", k)) for k, v in models.items()]
    order = [m["id"] for m in models]
    label = {m["id"]: m.get("label", m.get("name", m["id"])) for m in models}
    return split, order, label, models


def metrics(c):
    c = np.asarray(c, float)
    out = {f"cov{k}": float(c[k]) for k in KS}
    out.update({f"auc{k}": float(np.trapz(c[:k + 1], dx=1.0) / k) for k in KS})
    out["final"] = float(c[T_MAX])
    return out


def boot(vals, n=10000, seed=0):
    g = np.asarray(vals, float)
    if len(g) == 0:
        return [None, None]
    rng = np.random.default_rng(seed)
    bs = g[rng.integers(0, len(g), (n, len(g)))].mean(1)
    return [float(np.quantile(bs, 0.025)), float(np.quantile(bs, 0.975))]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--protocol", choices=["official", "mono"], default="official")
    a = ap.parse_args()
    key = f"coverage_{a.protocol}"
    split, order, label, models = load_registries()
    all_scenes = sorted(split)
    recs, problems = [], []
    for f in sorted(glob.glob(os.path.join(EPI, "*", "*.json"))):
        r = json.load(open(f))
        c = r.get(key)
        if c is None:
            continue
        if len(c) != T_MAX + 1 or not np.all(np.isfinite(c)) or min(c) < 0 or max(c) > 1.05:
            problems.append(f"{r['model']} {r['scene']} s{r['start']}: bad curve (len {len(c)})")
            continue
        if r["scene"] not in split:
            problems.append(f"{r['model']} {r['scene']}: scene not in registry")
            continue
        m = metrics(c)
        assert m["cov100"] == m["final"]
        recs.append(dict(model=r["model"], scene=r["scene"], start=int(r["start"]), split=split[r["scene"]], source=r.get("source"), curve=c, **m))
    present = [m for m in order if any(r["model"] == m for r in recs)]
    mk = [f"cov{k}" for k in KS] + [f"auc{k}" for k in KS] + ["final"]
    # ---- per scene
    ps = {}
    for r in recs:
        ps.setdefault((r["model"], r["scene"]), []).append(r)
    scene_rows = []
    for (m, s), rr in ps.items():
        starts = sorted(x["start"] for x in rr)
        row = {"model": m, "scene": s, "split": split[s], "n_starts": len(rr), "complete": starts == list(range(N_START))}
        for k in mk:
            v = [x[k] for x in rr]
            row[k], row[k + "_std"] = float(np.mean(v)), float(np.std(v, ddof=1)) if len(v) > 1 else 0.0
        scene_rows.append(row)
    S = {(r["model"], r["scene"]): r for r in scene_rows}
    # ---- per split / All-15
    groups = [(sp, [s for s in all_scenes if split[s] == sp]) for sp in SPLITS if any(split[s] == sp for s in all_scenes)] + \
        [("all14", sorted(RUNNABLE)), ("all15", all_scenes)]
    split_rows, pair_rows = [], []
    for m in present:
        for g, members in groups:
            done = [s for s in members if (m, s) in S and S[(m, s)]["complete"]]
            if not done:
                continue
            full = len(done) == len(members)
            name = g if full else f"{g}-partial{len(done)}of{len(members)}"
            row = {"model": m, "label": label[m], "group": name, "complete": full, "n_scenes": len(done),
                   "n_episodes": sum(S[(m, s)]["n_starts"] for s in done), "scenes": " ".join(done)}
            for k in mk:
                v = [S[(m, s)][k] for s in done]
                row[k], row[k + "_std"] = float(np.mean(v)), float(np.std(v, ddof=1)) if len(v) > 1 else 0.0
                row[k + "_ci"] = boot(v)
            split_rows.append(row)
            if m != REF:
                common = [s for s in done if (REF, s) in S and S[(REF, s)]["complete"]]
                if common:
                    pr = {"model": m, "ref": REF, "group": name if len(common) == len(done) else f"{name}-vsref{len(common)}", "n_scenes": len(common)}
                    for k in ("auc20", "auc50", "auc100", "cov20", "final"):
                        d = []
                        for s in common:
                            mine = {x["start"]: x[k] for x in ps[(m, s)]}
                            ref = {x["start"]: x[k] for x in ps[(REF, s)]}
                            d.append(float(np.mean([mine[i] - ref[i] for i in range(N_START)])))
                        pr[f"d_{k}"], pr[f"d_{k}_ci"] = float(np.mean(d)), boot(d)
                        pr[f"d_{k}_wins"], pr[f"d_{k}_losses"] = int(sum(x > 1e-12 for x in d)), int(sum(x < -1e-12 for x in d))
                        pr[f"d_{k}_per_scene"] = dict(zip(common, d))
                    pair_rows.append(pr)
    # ---- write
    R = os.path.join(T1, "results")
    os.makedirs(R, exist_ok=True)
    suf = "" if a.protocol == "official" else "_mono"

    def wcsv(name, rows, cols):
        with open(os.path.join(R, f"{name}{suf}.csv"), "w", newline="") as f:
            w = csv.writer(f)
            w.writerow(cols)
            for r in rows:
                w.writerow([json.dumps(r[c]) if isinstance(r.get(c), (list, dict)) else r.get(c) for c in cols])
    wcsv("per_episode", sorted(recs, key=lambda r: (order.index(r["model"]), r["scene"], r["start"])),
         ["model", "scene", "split", "start", "source"] + mk)
    wcsv("per_scene", sorted(scene_rows, key=lambda r: (order.index(r["model"]), r["split"], r["scene"])),
         ["model", "scene", "split", "n_starts", "complete"] + [x for k in mk for x in (k, k + "_std")])
    wcsv("per_split", split_rows, ["model", "label", "group", "complete", "n_scenes", "n_episodes"] + [x for k in mk for x in (k, k + "_std", k + "_ci")] + ["scenes"])
    wcsv("per_model", [r for r in split_rows if r["group"] == "all14" or r["group"] == "all15"], ["model", "label", "group", "complete", "n_scenes", "n_episodes"] + mk)
    with open(os.path.join(R, f"coverage_curves{suf}.csv"), "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["model", "scene", "split", "start"] + [f"c{t}" for t in range(T_MAX + 1)])
        for r in recs:
            w.writerow([r["model"], r["scene"], r["split"], r["start"]] + list(r["curve"]))
    wcsv("auc_horizons", split_rows, ["model", "group", "n_scenes"] + [f"auc{k}" for k in KS] + [f"auc{k}_ci" for k in KS])
    pc = ["model", "ref", "group", "n_scenes"] + [x for k in ("auc20", "auc50", "auc100", "cov20", "final") for x in (f"d_{k}", f"d_{k}_ci", f"d_{k}_wins", f"d_{k}_losses")]
    wcsv("pairwise_comparison", pair_rows, pc)
    json.dump({"protocol": a.protocol, "models": present, "problems": problems, "n_records": len(recs), "per_split": split_rows, "pairwise": pair_rows,
               "per_scene": scene_rows}, open(os.path.join(R, f"summary{suf}.json"), "w"), indent=1, default=float)
    if a.protocol == "official":
        tables(split_rows, pair_rows, scene_rows, present, label, split)
    print(f"protocol {a.protocol}: {len(recs)} episodes, models {present}, problems {len(problems)}")
    for p in problems[:20]:
        print("  PROBLEM", p)
    for r in split_rows:
        if r["group"] in ("all14", "all15") or "partial" in r["group"] or r["group"] in SPLITS:
            print(f"  {r['model']:6s} {r['group']:22s} n={r['n_scenes']:2d}/{r['n_episodes']:2d} AUC@20 {r['auc20']:.3f} AUC@100 {r['auc100']:.3f} Final {r['final']:.3f}")


def f3(x):
    return "—" if x is None else f"{x:.3f}"


def fci(r, k):
    lo, hi = r[k + "_ci"]
    return f"{r[k]:.3f} [{lo:.3f}, {hi:.3f}]" if lo is not None else f"{r[k]:.3f}"


def tables(split_rows, pair_rows, scene_rows, present, label, split):
    mk = [f"cov{k}" for k in KS] + [f"auc{k}" for k in KS] + ["final"]
    with open(os.path.join(T1, "TABLE1_FULL.csv"), "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["model", "label", "group", "complete", "n_scenes", "n_episodes"] + [x for k in mk for x in (k, k + "_ci_lo", k + "_ci_hi", k + "_std")])
        for r in split_rows:
            w.writerow([r["model"], r["label"], r["group"], r["complete"], r["n_scenes"], r["n_episodes"]] +
                       [x for k in mk for x in (round(r[k], 5), r[k + "_ci"][0], r[k + "_ci"][1], round(r[k + "_std"], 5))])
    gname = {"train": "Train", "dev": "Dev", "test": "Test", "other": "Other / Unassigned", "all14": "All-14（14 个可运行场景）", "all15": "All-15 — 不完整（sofia_church 缺失），只作审计，不是 All-15 结果"}
    order_g = ["train", "dev", "test", "other", "all14", "all15"]

    def rows_of(g):
        return [r for r in split_rows if r["group"] == g or r["group"].startswith(g + "-partial")]
    hdr = "*自动生成（`scripts/aggregate.py`，主协议 = MAGICIAN 官方覆盖）。不要手工修改。*\n"
    L = ["# TABLE1 — Macarons++ Active Mapping（主协议：MAGICIAN 官方 S10 覆盖）", "", hdr,
         "均值 [场景级 bootstrap 95% CI]；场景等权（每场景先平均 5 个起点）。`partialXofY` 表示该组场景不完整，不计入完整汇总。", ""]
    for g in order_g:
        rr = rows_of(g)
        if not rr:
            continue
        L += [f"## {gname[g]}", "", "| 模型 | 组 | 场景 / episode | AUC@20 | AUC@50 | AUC@100 | Cov@20 | Final |", "|---|---|---|---|---|---|---|---|"]
        for r in rr:
            L.append(f"| {r['label']} | {r['group']} | {r['n_scenes']} / {r['n_episodes']} | {fci(r, 'auc20')} | {fci(r, 'auc50')} | {fci(r, 'auc100')} | {fci(r, 'cov20')} | {fci(r, 'final')} |")
        L.append("")
    L += ["## 相对 MAG-B 的配对差（同场景、同起点；场景级 bootstrap 95% CI；胜/负场景数）", "", "| 模型 | 组 | 场景数 | ΔAUC@20 | ΔAUC@100 | ΔFinal |", "|---|---|---|---|---|---|"]
    for p in pair_rows:
        L.append(f"| {label[p['model']]} | {p['group']} | {p['n_scenes']} | " + " | ".join(
            f"{p[f'd_{k}']:+.3f} [{p[f'd_{k}_ci'][0]:+.3f}, {p[f'd_{k}_ci'][1]:+.3f}] {p[f'd_{k}_wins']}/{p[f'd_{k}_losses']}" for k in ("auc20", "auc100", "final")) + " |")
    open(os.path.join(T1, "TABLE1.md"), "w").write("\n".join(L) + "\n")
    # paper table: compact
    P = ["# TABLE1_PAPER — 正文紧凑版（主协议：MAGICIAN 官方覆盖）", "", hdr, "| 模型 | " + " | ".join(
        f"{gname[g]} AUC@20 | {gname[g]} AUC@50 | {gname[g]} AUC@100 | {gname[g]} Cov@20 | {gname[g]} Final" for g in ("dev", "test", "all14")) + " |",
         "|---|" + "---|" * 15]
    for m in present:
        cells = []
        for g in ("dev", "test", "all14"):
            r = next((x for x in split_rows if x["model"] == m and x["group"] == g), None)
            cells += [f3(r[k]) if r else "—" for k in ("auc20", "auc50", "auc100", "cov20", "final")]
        P.append(f"| {label[m]} | " + " | ".join(cells) + " |")
    P += ["", "Train 与 Other 组见 TABLE1.md / TABLE1_BY_SPLIT.md。"]
    open(os.path.join(T1, "TABLE1_PAPER.md"), "w").write("\n".join(P) + "\n")
    # by split (all metrics)
    B = ["# TABLE1_BY_SPLIT — 全部指标（主协议）", "", hdr]
    for g in order_g:
        rr = rows_of(g)
        if not rr:
            continue
        B += [f"## {gname[g]}", "", "| 模型 | 组 | 场景/ep | " + " | ".join(f"Cov@{k}" for k in KS) + " | " + " | ".join(f"AUC@{k}" for k in KS) + " | Final |",
              "|---|---|---|" + "---|" * (2 * len(KS) + 1)]
        for r in rr:
            B.append(f"| {r['label']} | {r['group']} | {r['n_scenes']}/{r['n_episodes']} | " + " | ".join(f3(r[f'cov{k}']) for k in KS) + " | " +
                     " | ".join(f3(r[f'auc{k}']) for k in KS) + f" | {f3(r['final'])} |")
        B.append("")
    open(os.path.join(T1, "TABLE1_BY_SPLIT.md"), "w").write("\n".join(B) + "\n")
    # by scene
    sc = sorted({r["scene"] for r in scene_rows}, key=lambda s: (order_g.index(split[s]) if split[s] in order_g else 9, s))
    C = ["# TABLE1_BY_SCENE — 逐场景（5 个起点的均值；AUC@20 / AUC@100 / Final）", "", hdr,
         "| 场景 | 划分 | " + " | ".join(label[m] for m in present) + " |", "|---|---|" + "---|" * len(present)]
    SS = {(r["model"], r["scene"]): r for r in scene_rows}
    for s in sc:
        cells = []
        for m in present:
            r = SS.get((m, s))
            cells.append("—" if r is None else (f"{r['auc20']:.3f} / {r['auc100']:.3f} / {r['final']:.3f}" + ("" if r["complete"] else f" ({r['n_starts']}/5)")))
        C.append(f"| {s} | {gname.get(split[s], split[s])} | " + " | ".join(cells) + " |")
    open(os.path.join(T1, "TABLE1_BY_SCENE.md"), "w").write("\n".join(C) + "\n")
    # LaTeX (paper table)
    X = ["% auto-generated by scripts/aggregate.py (primary protocol: MAGICIAN official coverage); do not edit",
         "\\begin{tabular}{l" + "ccc" * 3 + "}", "\\toprule",
         " & \\multicolumn{3}{c}{Dev} & \\multicolumn{3}{c}{Test} & \\multicolumn{3}{c}{All-14} \\\\",
         "Method & AUC@20 & AUC@100 & Final & AUC@20 & AUC@100 & Final & AUC@20 & AUC@100 & Final \\\\", "\\midrule"]
    for m in present:
        cells = []
        for g in ("dev", "test", "all14"):
            r = next((x for x in split_rows if x["model"] == m and x["group"] == g), None)
            cells += [f"{r[k]:.3f}" if r else "--" for k in ("auc20", "auc100", "final")]
        X.append(f"{label[m]} & " + " & ".join(cells) + " \\\\")
    X += ["\\bottomrule", "\\end{tabular}"]
    open(os.path.join(T1, "TABLE1_LATEX.tex"), "w").write("\n".join(X) + "\n")


if __name__ == "__main__":
    main()
