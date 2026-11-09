"""Experiment XB / Task B3 -- aggregate the per-camera error strata written by xb_b3.py into results/B3_strata/{summary.json,summary.md}.
Per model (T1@none, XA_SR@none, XA_MI@none) and domain (indoor / outdoor Dev scenes / outdoor Train scenes): FNR = fn / (tp + fn), FPR = fp / (fp + tn),
MAE_tp = sae_tp / n_tp (runtime units) in the strata depth / nseg / boundary / lu, plus episode phase (early / middle / late by t)."""
import glob, json, os
import yaml
R = "/bsuscratch/jiahuizhang/projects/occupancy/seen2scene"
W = os.path.join(R, "work_dirs", "exp6_v3", "ablation_study", "experiment_XB_bottleneck_audit", "states", "xb_b3")
OUT = os.path.join(R, "experiments", "exp6_v3", "ablation_study", "experiment_XB_bottleneck_audit", "results", "B3_strata"); os.makedirs(OUT, exist_ok=True)
SPLIT = {s["local_id"]: s["split"] for s in yaml.safe_load(open(os.path.join(R, "experiments", "cvpr_results", "table1", "SCENE_REGISTRY.yaml")))["scenes"]}
LAB = {"depth": ["[0,2)", "[2,5)", "[5,10)", "[10,25)", "[25,45)", "[45,70)"], "nseg": ["1", "2", ">=3"], "boundary": ["[0,.1)", "[.1,.3)", "[.3,1]"], "lu": ["[0,.1)", "[.1,.3)", "[.3,1]"]}
MODELS = ("T1@none", "XA_SR@none", "XA_MI@none")
def dom_of(r):
    if r["domain"] == "indoor": return "indoor"
    return "outdoor_dev" if SPLIT.get(r["scene"], "").startswith("dev") else "outdoor_train"
def phase(r):
    T = 150 if r["domain"] == "indoor" else 100
    f = r["t"] / T; return "early" if f <= 0.2 else ("middle" if f <= 0.6 else "late")
Z = lambda: {"tp": 0, "fp": 0, "fn": 0, "tn": 0, "sae_tp": 0.0, "n_tp": 0, "n_lh": 0, "n_lm": 0, "cams": 0}
def add(a, c):
    for k in ("tp", "fp", "fn", "tn", "n_lh", "n_lm"): a[k] += int(c.get(k, 0))
    a["sae_tp"] += float(c.get("sae_tp", 0.0)); a["n_tp"] += int(c.get("n_tp", c.get("tp", 0))); a["cams"] += 1
def fin(a):
    o = dict(a); o["fnr"] = a["fn"] / max(1, a["tp"] + a["fn"]); o["fpr"] = a["fp"] / max(1, a["fp"] + a["tn"]); o["mae_tp"] = a["sae_tp"] / max(1, a["n_tp"])
    o["share_lh"] = None; return o
S = {m: {d: {"total": Z(), "phase": {p: Z() for p in ("early", "middle", "late")}, **{g: [Z() for _ in LAB[g]] for g in LAB}} for d in ("indoor", "outdoor_dev", "outdoor_train")} for m in MODELS}
n_rows = 0
for f in sorted(glob.glob(os.path.join(W, "*.jsonl"))):
    for l in open(f):
        r = json.loads(l); m, d = r["ckpt"], dom_of(r)
        if m not in S: continue
        n_rows += 1
        for c in (r["cams"].values() if isinstance(r["cams"], dict) else r["cams"]):
            add(S[m][d]["total"], c); add(S[m][d]["phase"][phase(r)], c)
            for g in LAB:
                for i, b in enumerate(c["strata"][g]): add(S[m][d][g][i], b)
out = {m: {d: {"total": fin(S[m][d]["total"]), "phase": {p: fin(v) for p, v in S[m][d]["phase"].items()}, **{g: [fin(v) for v in S[m][d][g]] for g in LAB}} for d in S[m]} for m in S}
# labelled-hit share per stratum (where the surfaces are)
for m in out:
    for d in out[m]:
        tot = max(1, out[m][d]["total"]["n_lh"])
        for g in LAB:
            for v in out[m][d][g]: v["share_lh"] = v["n_lh"] / tot
json.dump({"rows": n_rows, "labels": LAB, "models": out}, open(os.path.join(OUT, "summary.json"), "w"), indent=1)
L = [f"# B3 误差分层（{n_rows} 行；每格 = 该层的标签 UNKNOWN 射线，K = 4 个样本汇总）", "", "FNR = 真值命中而预测未命中；FPR = 真值无命中而预测命中；MAE = 真阳性射线的首次命中距离绝对误差（运行时单位：室内 = 米，室外 = 场景单位）；份额 = 该层占全部标签命中射线的比例。", ""]
for d in ("indoor", "outdoor_dev", "outdoor_train"):
    L += [f"## {d}", ""]
    L += ["| 模型 | 全部 FNR / FPR / MAE | early FNR | middle FNR | late FNR |", "|---|---|---|---|---|"]
    for m in MODELS:
        t = out[m][d]["total"]; ph = out[m][d]["phase"]
        L.append(f"| {m} | {t['fnr']:.3f} / {t['fpr']:.3f} / {t['mae_tp']:.2f} | {ph['early']['fnr']:.3f} | {ph['middle']['fnr']:.3f} | {ph['late']['fnr']:.3f} |")
    L.append("")
    for g in LAB:
        L += [f"### {d} — 按 {g}", "", "| 层 | 份额(T1) | " + " | ".join(f"{m} FNR / FPR / MAE" for m in MODELS) + " |", "|---|---|" + "---|" * len(MODELS)]
        for i, lab in enumerate(LAB[g]):
            if out[MODELS[0]][d][g][i]["n_lh"] + out[MODELS[0]][d][g][i]["tn"] == 0: continue
            L.append(f"| {lab} | {out[MODELS[0]][d][g][i]['share_lh']:.2f} | " + " | ".join(f"{out[m][d][g][i]['fnr']:.3f} / {out[m][d][g][i]['fpr']:.3f} / {out[m][d][g][i]['mae_tp']:.2f}" for m in MODELS) + " |")
        L.append("")
open(os.path.join(OUT, "summary.md"), "w").write("\n".join(L))
print("\n".join(L))
