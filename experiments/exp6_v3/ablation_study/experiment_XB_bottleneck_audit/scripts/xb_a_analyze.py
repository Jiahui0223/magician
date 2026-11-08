"""Experiment XB / Task A -- analysis of the intervention rows (CPU). Reads states/xb_a/<tag>.jsonl (CT row format, ckpt = BASE@IV) and <tag>.a1.json.
Per model and intervention, over the states of a layer (indoor / outdoor Dev / outdoor Train):
  prediction  precision / recall / FNR of HIT on labelled UNKNOWN rays, Brier, first-hit MAE on UNKNOWN rays (runtime z), novelty precision / recall
  decision    official Top-1 and regret vs truth (indoor: oracle gains of the candidates; outdoor: Experiment F official one-step gains of keys1),
              Spearman of the planner scores vs the SAME model's @none scores, Top-1 agreement with @none, share of states whose pick changed
  A1          residual / input RMS, attention entropy vs uniform, attention mass on BAND / EMPTY-only / F16 / F8 / local tokens
  regression  T1@none vs the CT g1 rows of A_40K (same states): planner scores bitwise
Output results/A_attention/{summary.json, per_state.csv}."""
import csv
import glob
import json
import os

import numpy as np
from scipy.stats import spearmanr

R = "/bsuscratch/jiahuizhang/projects/occupancy/seen2scene"
X = f"{R}/experiments/exp6_v3/ablation_study/experiment_XB_bottleneck_audit"
W = f"{R}/work_dirs/exp6_v3/ablation_study/experiment_XB_bottleneck_audit/states/xb_a"
FW = f"{R}/work_dirs/exp6_v3/ablation_study/experiment_F_stochastic_coverage_decision/states"
G1 = f"{R}/work_dirs/exp6_v3/ablation_study/experiment_CT_continued_training/states/g1"
OUT = f"{X}/results/A_attention"
DEV5 = {"colosseum", "barts", "pantheon", "eiffel", "sestino_museum"}
TOL = 1e-4


def layer_of(r):
    if r["domain"] == "indoor":
        return "indoor"
    return "outdoor_dev" if r["scene"] in DEV5 else "outdoor_train"


def truth_of(r, frows):
    """official one-step gains per candidate (nan where unknown) and the candidate index list the scores refer to"""
    if r["domain"] == "indoor":
        orc = np.full(r["n_legal"], np.nan)
        for j, g in zip(r["cand"], r["gain_pp"]) if "gain_pp" in r and isinstance(r["gain_pp"], list) and len(r["gain_pp"]) == len(r["cand"]) else []:
            orc[j] = g
        if "gain_pp" in r and len(r["gain_pp"]) == r["n_legal"]:
            orc = np.asarray([np.nan if x is None else x for x in r["gain_pp"]], float)
        return orc
    f = frows.get((r["tag"], r["t"]))
    if f is None:
        return None
    return np.asarray([np.nan if x is None else x for x in f["oracle"]], float)


def dec(score, orc, cand=None):
    s = np.asarray(score, float)
    if cand is not None and len(cand) < len(s):
        sub = np.full_like(s, -np.inf); sub[cand] = s[cand]; s = sub
    pk = int(np.flatnonzero(s == s.max())[0])
    if orc is None or not np.isfinite(orc).any():
        return {"pick": pk}
    m = np.nanmax(orc)
    o = {"pick": pk, "regret": float(m - orc[pk]) if np.isfinite(orc[pk]) else None, "top1": float(orc[pk] >= m - TOL) if np.isfinite(orc[pk]) else None}
    idx = np.flatnonzero(np.isfinite(orc))
    if len(idx) > 2 and np.ptp(s[idx]) > 0 and np.ptp(orc[idx]) > 0:
        rr = spearmanr(s[idx], orc[idx]).correlation
        o["rho_truth"] = float(rr) if np.isfinite(rr) else None
    return o


def cam_sums(cams):
    c = cams.values() if isinstance(cams, dict) else cams
    agg = {}
    for o in c:
        for k in ("tp", "fp", "fn", "tn", "n_lm", "n_lh", "brier", "sae_unk", "n_mae_unk", "new_tp", "new_fp", "new_fn", "n_true_new"):
            agg[k] = agg.get(k, 0.0) + float(o.get(k, 0.0) or 0.0)
    return agg


def main():
    os.makedirs(OUT, exist_ok=True)
    frows = {}
    for f in glob.glob(FW + "/*.jsonl"):
        for r in map(json.loads, open(f)):
            frows[(r["tag"], r["t"])] = r
    g1 = {}
    for f in glob.glob(G1 + "/*.jsonl"):
        for r in map(json.loads, open(f)):
            if r["ckpt"] == "A_40K":
                g1[(r["tag"], r["t"])] = r
    rows = [r for f in glob.glob(W + "/*.jsonl") for r in map(json.loads, open(f))]
    a1 = {}
    for f in glob.glob(W + "/*.a1.json"):
        tag = os.path.basename(f)[:-8]
        for n, v in json.load(open(f)).items():
            a1.setdefault(n, []).append(v)
    by_state = {}
    for r in rows:
        by_state.setdefault((r["tag"], r["t"]), {})[r["ckpt"]] = r
    per_state, summary = [], {}
    reg = {"n": 0, "bitwise": 0, "same_pick": 0}
    for key, d in sorted(by_state.items()):
        tag, t = key
        if "T1@none" in d and key in g1:
            a_, b_ = d["T1@none"]["score"], g1[key]["score"]
            reg["n"] += 1; reg["bitwise"] += int(a_ == b_); reg["same_pick"] += int(int(np.argmax(a_)) == int(np.argmax(b_)))
        for name, r in d.items():
            base, iv = name.split("@")
            ref = d.get(f"{base}@none")
            lay = layer_of(r)
            orc = truth_of(r, frows)
            cand = r.get("cand") if r["domain"] == "indoor" else None
            dd = dec(r["score"], orc, cand)
            row = {"tag": tag, "t": t, "layer": lay, "scene": r["scene"], "model": base, "iv": iv, **{f"dec_{k}": v for k, v in dd.items()}}
            if ref is not None and ref is not r:
                s0, s1 = np.asarray(ref["score"], float), np.asarray(r["score"], float)
                row["agree_top1_vs_none"] = int(int(np.argmax(s0)) == int(np.argmax(s1)))
                row["rho_vs_none"] = float(spearmanr(s0, s1).correlation) if len(s0) > 2 and np.ptp(s0) > 0 and np.ptp(s1) > 0 else None
                row["score_rel_change"] = float(np.abs(s1 - s0).mean() / max(1e-9, np.abs(s0).mean()))
            cs = cam_sums(r["cams"])
            row.update({"tp": cs["tp"], "fp": cs["fp"], "fn": cs["fn"], "tn": cs["tn"], "n_lm": cs["n_lm"], "brier_sum": cs["brier"], "sae_unk": cs["sae_unk"], "n_mae_unk": cs["n_mae_unk"],
                        "new_tp": cs["new_tp"], "new_fp": cs["new_fp"], "new_fn": cs["new_fn"]})
            per_state.append(row)
    with open(f"{OUT}/per_state.csv", "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=sorted({k for r in per_state for k in r}))
        w.writeheader(); w.writerows(per_state)
    for lay in ("indoor", "outdoor_dev", "outdoor_train"):
        for name in sorted({f"{r['model']}@{r['iv']}" for r in per_state}):
            base, iv = name.split("@")
            rs = [r for r in per_state if r["layer"] == lay and r["model"] == base and r["iv"] == iv]
            if not rs:
                continue
            tp, fp, fn = sum(r["tp"] for r in rs), sum(r["fp"] for r in rs), sum(r["fn"] for r in rs)
            lm = sum(r["n_lm"] for r in rs)
            o = {"n_states": len(rs), "precision": tp / max(1, tp + fp), "recall": tp / max(1, tp + fn), "fnr": fn / max(1, tp + fn), "brier": sum(r["brier_sum"] for r in rs) / max(1, lm),
                 "mae_unk": sum(r["sae_unk"] for r in rs) / max(1, sum(r["n_mae_unk"] for r in rs)),
                 "new_precision": sum(r["new_tp"] for r in rs) / max(1, sum(r["new_tp"] + r["new_fp"] for r in rs)), "new_recall": sum(r["new_tp"] for r in rs) / max(1, sum(r["new_tp"] + r["new_fn"] for r in rs))}
            v = [r["dec_regret"] for r in rs if r.get("dec_regret") is not None]
            o["regret"] = float(np.mean(v)) if v else None
            v = [r["dec_top1"] for r in rs if r.get("dec_top1") is not None]
            o["top1"] = float(np.mean(v)) if v else None
            v = [r["dec_rho_truth"] for r in rs if r.get("dec_rho_truth") is not None]
            o["rho_truth"] = float(np.mean(v)) if v else None
            if iv != "none":
                o["top1_agreement_with_none"] = float(np.mean([r["agree_top1_vs_none"] for r in rs if "agree_top1_vs_none" in r]))
                v = [r["rho_vs_none"] for r in rs if r.get("rho_vs_none") is not None]
                o["rho_scores_vs_none"] = float(np.mean(v)) if v else None
                o["score_rel_change"] = float(np.mean([r["score_rel_change"] for r in rs if "score_rel_change" in r]))
            summary.setdefault(lay, {})[name] = o
    a1s = {}
    for n, lst in a1.items():
        keys = sorted({k for v in lst for k in v if k != "n_calls"})
        a1s[n] = {k: float(np.mean([v[k] for v in lst if k in v])) for k in keys}
        a1s[n]["n_trajectories"] = len(lst)
    json.dump({"regression_T1_none_vs_CT_g1": reg, "summary": summary, "a1": a1s}, open(f"{OUT}/summary.json", "w"), indent=1)
    print("regression T1@none vs CT g1 A_40K:", reg)
    for lay, d in summary.items():
        print("==", lay)
        for n, o in d.items():
            print("  %-20s n %3d | prec %.3f rec %.3f fnr %.3f brier %.4f maeU %.3f | top1 %s regret %s | agree@none %s rho@none %s dscore %s" % (
                n, o["n_states"], o["precision"], o["recall"], o["fnr"], o["brier"], o["mae_unk"], None if o["top1"] is None else round(o["top1"], 3), None if o["regret"] is None else round(o["regret"], 3),
                None if "top1_agreement_with_none" not in o else round(o["top1_agreement_with_none"], 3), None if o.get("rho_scores_vs_none") is None else round(o["rho_scores_vs_none"], 3),
                None if "score_rel_change" not in o else round(o["score_rel_change"], 3)))
    print("== A1")
    for n, v in a1s.items():
        print("  %-20s" % n, {k: round(x, 3) for k, x in v.items() if k in ("ratio", "rms_in", "rms_res", "entropy", "entropy_uniform", "max_w", "mass_band", "mass_free_only", "mass_f16", "mass_f8", "mass_local", "n_keys_valid")})


if __name__ == "__main__":
    main()
