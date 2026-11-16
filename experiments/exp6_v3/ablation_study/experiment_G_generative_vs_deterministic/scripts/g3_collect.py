"""Experiment G3 -- FINAL TABLES (protocol G3-v1 sections 6-7; CPU only).
Outdoor (Macarons++, official coverage, 100 moves): D0 = experiment_G/closed_loop/D0, M-raw / M-depth = experiment_G/g3/closed_loop/{MRAW,MDEPTH}
(replay == stored, 101 values), G1 = Table 3 V3-T0-K1D2, G4 = Table 1 V3-T0-D1 (cvpr_lib.load_manifest). Groups All-14 / Dev+Test-8 / Unseen-6;
scene = mean of the 5 starts, group = scene-equal mean; paired scene bootstrap vs G4 (10,000, seed 0).
Indoor (MP3D-150, 150 moves, level C_obs): G4 = EXP7 OURS canonical records; D0IN / G1IN = G2 raw episodes and M-raw / M-depth = G3 raw episodes,
each re-validated with collect_mp3d150.validate (method OURS; D0 sha for D0IN) + same start as OURS + the G_MODE / G3_MODE line; level =
mean of the 2 starts; ALL-22 / HO-12 / VALSEL-10; paired level bootstrap vs G4 (collect_mp3d150.boot: 10,000, seed 0).
Also: per-step difference curves vs G4 with pointwise bootstrap bands and the crossover step (G2 item 3 definition), the pre-registered
verdicts (M-depth vs G4: AUC@100 outdoor / AUC@150 indoor primary; AUC@20 and Final secondary), and the fixed-state table if present.
Writes results/g3/{final_table_outdoor,final_table_indoor}.{md,csv,tex}, temporal_{outdoor,indoor}.csv, G3_TABLES.md, g3_tables.json and
figures/g3/temporal_vs_g4.{pdf,png}. Usage: python g3_collect.py [--allow-incomplete] [--workers 8]"""
import argparse
import csv
import json
import os
import sys
from multiprocessing import Pool

import numpy as np

R = "/bsuscratch/jiahuizhang/projects/occupancy/seen2scene"
HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
import g_collect as GC  # noqa: E402
import g2_indoor_collect as GI  # noqa: E402

L, C7 = GC.L, GI.C7
GD = GC.GD
WG = os.path.join(R, "work_dirs/exp6_v3/experiment_G")
OUT, FIG = os.path.join(GD, "results/g3"), os.path.join(GD, "figures/g3")
ROWS = [("D0", "– (deterministic, one forward)", "1"), ("G1", "single sample (noise map 0)", "1"),
        ("MRAW", "average (b, y) of the 4 samples, then decode", "4 → 1"), ("MDEPTH", "decode each, ≥ 2/4 vote, mean hit depth", "4 → 1"),
        ("G4", "score each sample, aggregate in D1 (Ours)", "4")]
NAME = {"D0": "D0", "G1": "G1", "MRAW": "M-raw", "MDEPTH": "M-depth", "G4": "Ours (G4)"}
OUT_NAMES = ["AUC@20", "AUC@50", "AUC@100", "Final"]
IN_NAMES = ["AUC@20", "AUC@50", "AUC@100", "AUC@150", "Final"]
OUT_GROUPS = {"All-14": list(L.SCENE_ORDER), "Dev+Test-8": [s for s in L.SCENE_ORDER if L.SPLIT[s] != "train"],
              "Unseen-6": ["alhambra", "neuschwanstein", "eiffel", "bridge", "barts", "sestino_museum"]}
T0_SHA = C7.T0_SHA


# ------------------------------------------------------------------------------------------------ outdoor
def load_dir(root, arm):
    keep = GC.WG
    GC.WG = root
    try:
        return GC.load_new(arm)
    finally:
        GC.WG = keep


def outdoor():
    E = {"D0": load_dir(WG, "D0")[0], "MRAW": load_dir(os.path.join(WG, "g3"), "MRAW")[0], "MDEPTH": load_dir(os.path.join(WG, "g3"), "MDEPTH")[0],
         "G1": L.load_manifest("V3-T0-K1D2"), "G4": L.load_manifest("V3-T0-D1")}
    tabs = {a: L.scene_table(E[a], OUT_NAMES) for a in E}
    res = {"n_episodes": {a: len(E[a]) for a in E}, "complete_scenes": {a: len(tabs[a]) for a in E}, "groups": {}}
    for g, sc in OUT_GROUPS.items():
        res["groups"][g] = {}
        for a, _, _ in ROWS:
            r = {n: L.group_mean(tabs[a], sc, n)[0] for n in OUT_NAMES}
            r["n_scenes"] = L.group_mean(tabs[a], sc, "Final")[1]
            if a != "G4":
                for n in OUT_NAMES:
                    m, lo, hi, ns, w, l_ = L.paired(tabs[a], tabs["G4"], sc, n)
                    r[f"d_{n}"], r[f"ci_{n}"], r[f"wl_{n}"] = m, (lo, hi), (w, l_)
            res["groups"][g][a] = r
    curves = {a: {s: np.mean([E[a][(s, k)]["coverage_official"] for k in range(L.N_START)], 0) for s in L.SCENE_ORDER
                  if all((s, k) in E[a] for k in range(L.N_START))} for a in E}
    return res, curves


# ------------------------------------------------------------------------------------------------ indoor
def check3(args):
    arm, r = args
    tag = f"{r['scene_id']}_S{r['start_id']}"
    p = os.path.join(WG, "g3", "indoor", "raw", arm, "V2M", tag, "episode.json")
    if not os.path.exists(p):
        return {"arm": arm, "row": r, "exists": False}
    row = dict(r, method="OURS", output_path=os.path.relpath(p, R))
    C7.T0_SHA = T0_SHA
    try:
        o = C7.validate(row)
    except Exception as e:                                   # noqa: BLE001
        return {"arm": arm, "row": r, "exists": True, "valid": False, "checks": {"validate_error": repr(e)}, "metrics": None}
    ref = json.load(open(os.path.join(C7.W7, "episodes", "mp3d150", "OURS", f"{tag}.json")))
    o["checks"]["V4_same_start"] = o["start"] == ref["start_pose"]
    d = os.path.dirname(p)
    err = open(os.path.join(d, "planner.err")).read()
    gl = [json.loads(x[len("G3_MODE "):]) for x in err.splitlines() if x.startswith("G3_MODE ")]
    info = json.load(open(os.path.join(d, "planner_info.json")))
    mode = {"MRAWIN": "raw", "MDEPTHIN": "depth"}[arm]
    o["checks"]["G3_mode_K1"] = bool(gl) and gl[0]["g3_mode"] == mode and gl[0]["K"] == 1 and info.get("g3_mode") == mode
    o["valid"] = all(o["checks"].values())
    o.update({"arm": arm, "exists": True})
    o.pop("executed", None)
    return o


def indoor(workers):
    rows = GI.ours_rows()
    jobs3 = [(a, r) for a in ("MRAWIN", "MDEPTHIN") for r in rows]
    jobs2 = [(a, r) for a in ("G1IN", "D0IN") for r in rows]
    with Pool(workers) as pool:
        res = pool.map(check3, jobs3) + pool.map(GI.check, jobs2)
    amap = {"MRAWIN": "MRAW", "MDEPTHIN": "MDEPTH", "G1IN": "G1", "D0IN": "D0"}
    ep, cov, status = {}, {}, {}
    for o in res:
        a = amap[o["arm"]]
        st = status.setdefault(a, {"exists": 0, "valid": 0, "invalid": []})
        if o["exists"]:
            st["exists"] += 1
            if o.get("valid"):
                st["valid"] += 1
                ep[(a, o["row"]["scene_id"], o["row"]["start_id"])] = o["metrics"]
                cov[(a, o["row"]["scene_id"], o["row"]["start_id"])] = np.asarray(o["coverage"], np.float64)
            else:
                st["invalid"].append([o["row"]["scene_id"], o["row"]["start_id"], o.get("checks")])
    for r in rows:
        g = json.load(open(os.path.join(C7.W7, "episodes", "mp3d150", "OURS", f"{r['scene_id']}_S{r['start_id']}.json")))
        if g["valid"]:
            ep[("G4", r["scene_id"], r["start_id"])] = g["metrics"]
            cov[("G4", r["scene_id"], r["start_id"])] = np.asarray(g["coverage"], np.float64)
    status["G4"] = {"valid": sum(1 for k in ep if k[0] == "G4")}
    need, split = {}, {}
    for r in rows:
        need.setdefault(r["scene_id"], set()).add(r["start_id"])
        split[r["scene_id"]] = r["split"]
    lv, lcurve = {}, {}
    for a, _, _ in ROWS:
        for s, st in need.items():
            if all((a, s, i) in ep for i in st):
                lv[(a, s)] = {k: float(np.mean([ep[(a, s, i)][k] for i in st])) for k in IN_NAMES}
                lcurve[(a, s)] = np.mean([cov[(a, s, i)] for i in st], 0)
    groups = {"ALL-22": sorted(need), "HO-12": sorted(s for s in need if split[s] == "HO"), "VALSEL-10": sorted(s for s in need if split[s] == "VALSEL")}
    out = {"status": status, "groups": {}}
    for g, Ls in groups.items():
        out["groups"][g] = {}
        for a, _, _ in ROWS:
            done = [s for s in Ls if (a, s) in lv]
            r = {k: (float(np.mean([lv[(a, s)][k] for s in done])) if done else None) for k in IN_NAMES}
            r["n_levels"] = len(done)
            if a != "G4":
                both = [s for s in Ls if (a, s) in lv and ("G4", s) in lv]
                for k in IN_NAMES:
                    if both:
                        dl = [lv[(a, s)][k] - lv[("G4", s)][k] for s in both]
                        lo, hi = C7.boot(dl)
                        r[f"d_{k}"], r[f"ci_{k}"], r[f"wl_{k}"] = float(np.mean(dl)), (lo, hi), (int(sum(x > 0 for x in dl)), int(sum(x < 0 for x in dl)))
                    else:
                        r[f"d_{k}"], r[f"ci_{k}"], r[f"wl_{k}"] = None, (None, None), (0, 0)
            out["groups"][g][a] = r
    curves = {a: {s: lcurve[(a, s)] for s in need if (a, s) in lcurve} for a, _, _ in ROWS}
    return out, curves, groups


# ------------------------------------------------------------------------------------------------ temporal
def temporal(curves, groups, T, n=10000, seed=0):
    rows, summ = [], []
    for g, sc in groups.items():
        for a, _, _ in ROWS:
            if a == "G4":
                continue
            ss = [s for s in sc if s in curves[a] and s in curves["G4"]]
            if not ss:
                continue
            Dm = np.stack([curves[a][s] - curves["G4"][s] for s in ss])
            rng = np.random.default_rng(seed)
            B = Dm[rng.integers(0, len(ss), (n, len(ss)))].mean(1)
            m, lo, hi = Dm.mean(0), np.percentile(B, 2.5, 0), np.percentile(B, 97.5, 0)
            for t in range(T + 1):
                rows.append([g, f"{a}-G4", t, m[t], lo[t], hi[t], len(ss)])
            cross = next((t for t in range(T + 1) if np.all(m[t:] <= 0)), None)
            summ.append({"group": g, "pair": f"{NAME[a]} − G4", "n": len(ss), "crossover_t": cross, "dC@20": float(m[20]), "dC@50": float(m[50]),
                         f"dC@{T}": float(m[T]), "band_above0": int((lo > 0).sum()), "band_below0": int((hi < 0).sum())})
    return rows, summ


def fmt_cell(r, k, a):
    v = r.get(k)
    if v is None:
        return "—"
    if a == "G4":
        return f"{v:.3f}"
    d, (lo, hi), (w, l_) = r.get(f"d_{k}"), r.get(f"ci_{k}"), r.get(f"wl_{k}")
    if d is None or lo is None:
        return f"{v:.3f}"
    return f"{v:.3f} ({d:+.3f} [{lo:+.3f}, {hi:+.3f}]; {w}/{l_})"


def write_table(name, res, names, gkey):
    md, rows = [], []
    for g, by in res["groups"].items():
        md += [f"### {g}", "", "| Method | Combination of K samples | K | " + " | ".join(names) + " |", "|---|---|---|" + "---|" * len(names)]
        for a, comb, K in ROWS:
            r = by[a]
            md.append(f"| {NAME[a]} | {comb} | {K} | " + " | ".join(fmt_cell(r, k, a) for k in names) + " |")
            rows.append([g, NAME[a], comb, K, r.get(f"n_{gkey}")] + [r.get(k) for k in names] + [r.get(f"d_{k}") for k in names] +
                        [r.get(f"ci_{k}") for k in names] + [r.get(f"wl_{k}") for k in names])
        md.append("")
    with open(os.path.join(OUT, f"{name}.csv"), "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["group", "method", "combination", "K", f"n_{gkey}"] + names + [f"d_{k}_vs_G4" for k in names] + [f"ci_{k}_vs_G4" for k in names] +
                   [f"wins_losses_{k}_vs_G4" for k in names])
        w.writerows(rows)
    g0 = list(res["groups"])[0]
    tex = [r"\begin{tabular}{llc" + "c" * len(names) + "}", r"\toprule", "Method & Combination of $K$ samples & $K$ & " + " & ".join(names) + r" \\", r"\midrule"]
    for a, comb, K in ROWS:
        r = res["groups"][g0][a]
        ctex = comb.replace("≥", "$\\ge$").replace("→", "$\\to$").replace("–", "--")
        ktex = K.replace("→", "$\\to$")
        tex.append(NAME[a] + " & " + ctex + " & " + ktex + " & " + " & ".join("--" if r.get(k) is None else f"{r[k]:.3f}" for k in names) + " \\\\")
    tex += [r"\bottomrule", r"\end{tabular}"]
    open(os.path.join(OUT, f"{name}.tex"), "w").write("\n".join(tex) + "\n")
    open(os.path.join(OUT, f"{name}.md"), "w").write("\n".join(md) + "\n")
    return md


def verdict(r, k):
    d, (lo, hi) = r.get(f"d_{k}"), r.get(f"ci_{k}", (None, None))
    if d is None or lo is None:
        return "未完成"
    if hi < 0:
        return "M-depth < G4 显著 → 保持 K 个样本分开（逐样本覆盖 + D1 聚合）有价值；把生成图取平均会损失"
    if lo > 0:
        return "M-depth > G4 显著 → 取平均在该指标上更好"
    return "CI 含 0 → 在该指标上把样本平均成一张图与 Ours 相当；分开使用样本的好处在该指标上没有显示出来"


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--allow-incomplete", action="store_true")
    ap.add_argument("--workers", type=int, default=8)
    a = ap.parse_args()
    os.makedirs(OUT, exist_ok=True)
    os.makedirs(FIG, exist_ok=True)
    o_res, o_curves = outdoor()
    i_res, i_curves, i_groups = indoor(a.workers)
    inc_o = [x for x in ("D0", "G1", "MRAW", "MDEPTH", "G4") if o_res["complete_scenes"][x] < 14]
    inc_i = [x for x in ("D0", "G1", "MRAW", "MDEPTH", "G4") if i_res["groups"]["ALL-22"][x]["n_levels"] < 22]
    if (inc_o or inc_i) and not a.allow_incomplete:
        print("incomplete outdoor", {x: o_res["complete_scenes"][x] for x in inc_o}, "indoor", {x: i_res["groups"]["ALL-22"][x]["n_levels"] for x in inc_i},
              "status", {k: {kk: vv for kk, vv in v.items() if kk != "invalid"} for k, v in i_res["status"].items()})
        for k, v in i_res["status"].items():
            for x in v.get("invalid", []):
                print("INVALID indoor", k, x)
        sys.exit(1)
    md = ["# G3 — Ours vs averaging the K = 4 generated maps: final tables (protocol G3-v1)", "",
          "Values: group mean; in brackets: paired difference vs Ours (G4) [95 % bootstrap CI, 10,000 draws, seed 0]; wins / losses (scenes or levels).", "",
          "## Outdoor (Macarons++, official coverage, 5 official starts, 100 moves)", ""]
    md += write_table("final_table_outdoor", o_res, OUT_NAMES, "scenes")
    md += ["## Indoor (MP3D-150, level C_obs, 22 levels × 2 starts, 150 moves)", ""]
    md += write_table("final_table_indoor", i_res, IN_NAMES, "levels")
    to_rows, to_sum = temporal(o_curves, OUT_GROUPS, 100)
    ti_rows, ti_sum = temporal(i_curves, i_groups, 150)
    for nm, rr in (("temporal_outdoor.csv", to_rows), ("temporal_indoor.csv", ti_rows)):
        with open(os.path.join(OUT, nm), "w", newline="") as f:
            w = csv.writer(f)
            w.writerow(["group", "pair", "t", "mean_diff", "ci_lo", "ci_hi", "n"])
            w.writerows(rr)
    md += ["## Coverage difference vs Ours over time (crossover = first step from which the method is never above Ours)", "",
           "| Domain | Group | Pair | crossover t | ΔC@20 | ΔC@50 | ΔC@end | steps band > 0 | steps band < 0 |", "|---|---|---|---|---|---|---|---|---|"]
    for dom, ss, T in (("outdoor", to_sum, 100), ("indoor", ti_sum, 150)):
        for s in ss:
            md.append(f"| {dom} | {s['group']} | {s['pair']} | {s['crossover_t']} | {s['dC@20']:+.4f} | {s['dC@50']:+.4f} | {s[f'dC@{T}']:+.4f} | {s['band_above0']} | {s['band_below0']} |")
    md.append("")
    # pre-registered verdicts
    vo = o_res["groups"]["All-14"]["MDEPTH"]
    vi = i_res["groups"]["ALL-22"]["MDEPTH"]
    md += ["## 预注册判定（协议 §7；主：M-depth vs G4）", "",
           f"- 室外 All-14 AUC@100（主）：{verdict(vo, 'AUC@100')}",
           f"- 室内 ALL-22 AUC@150（主）：{verdict(vi, 'AUC@150')}",
           f"- 次要（不升级为主结论）：室外 AUC@20 {verdict(vo, 'AUC@20')}；室外 Final {verdict(vo, 'Final')}；室内 AUC@20 {verdict(vi, 'AUC@20')}；室内 Final {verdict(vi, 'Final')}",
           f"- M-raw（第二个平均变体）：室外 AUC@100 {verdict(o_res['groups']['All-14']['MRAW'], 'AUC@100')}；室内 AUC@150 {verdict(i_res['groups']['ALL-22']['MRAW'], 'AUC@150')}", ""]
    fx = os.path.join(OUT, "fixed_table.md")
    if os.path.exists(fx):
        md += ["## Fixed states", ""] + open(fx).read().splitlines()[2:]
    md += ["", f"Completeness: outdoor episodes {o_res['n_episodes']}, complete scenes {o_res['complete_scenes']}; indoor "
           + json.dumps({k: {kk: vv for kk, vv in v.items() if kk != 'invalid'} for k, v in i_res["status"].items()})]
    open(os.path.join(OUT, "G3_TABLES.md"), "w").write("\n".join(md) + "\n")
    json.dump({"outdoor": o_res, "indoor": i_res, "temporal_outdoor": to_sum, "temporal_indoor": ti_sum}, open(os.path.join(OUT, "g3_tables.json"), "w"), indent=1, default=str)
    # temporal figure (All-14 / ALL-22)
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    col = {"D0-G4": "#2471a3", "G1-G4": "#7d3c98", "MRAW-G4": "#e67e22", "MDEPTH-G4": "#c0392b"}
    fig, axes = plt.subplots(1, 2, figsize=(11, 3.6))
    for ax, rows_, g, T in ((axes[0], to_rows, "All-14", 100), (axes[1], ti_rows, "ALL-22", 150)):
        for pair, c in col.items():
            rr = np.asarray([[x[2], x[3], x[4], x[5]] for x in rows_ if x[0] == g and x[1] == pair], float)
            if len(rr):
                ax.fill_between(rr[:, 0], rr[:, 2], rr[:, 3], color=c, alpha=0.15, lw=0)
                ax.plot(rr[:, 0], rr[:, 1], color=c, lw=1.5, label=NAME[pair.split("-")[0]] + " − Ours")
        ax.axhline(0, color="k", lw=0.7)
        ax.set_xlim(0, T)
        ax.set_title(f"{'Outdoor' if T == 100 else 'Indoor'} {g}", fontsize=10)
        ax.set_xlabel("step")
        ax.grid(alpha=0.25, lw=0.5)
    axes[0].set_ylabel("coverage difference vs Ours")
    axes[0].legend(frameon=False, fontsize=8)
    fig.tight_layout()
    for ext in ("pdf", "png"):
        fig.savefig(os.path.join(FIG, f"temporal_vs_g4.{ext}"), dpi=200)
    print("\n".join(md))


if __name__ == "__main__":
    main()
