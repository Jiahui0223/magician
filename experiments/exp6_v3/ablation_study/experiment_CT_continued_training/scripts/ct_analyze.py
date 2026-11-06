"""Experiment CT -- analysis of the fixed-state rows (CPU). Reads states/<group>/*.jsonl and, for the outdoor truth, the Experiment F rows.
  CT1  prediction + decision metrics per checkpoint, layer (indoor, outdoor_dev, outdoor_train) and phase; paired differences vs A_40K (cluster bootstrap: floors indoor, scenes outdoor)
  CT3  A / B / C metrics by phase, decision-error analysis, BLOCKED statistics, counterfactual regret ladder
Output results/CT1_training_steps/fixed_summary.json, results/CT3_prediction_diagnostics/*.json, per-state CSV. Usage: python ct_analyze.py --groups g1,g2 [--cf cf_A_40K]"""
import argparse
import csv
import glob
import json
import os
import sys

import numpy as np
from scipy.stats import spearmanr

HERE = os.path.dirname(os.path.abspath(__file__))
FD = os.path.dirname(HERE)
R = os.path.dirname(os.path.dirname(os.path.dirname(os.path.dirname(FD))))
sys.path.insert(0, os.path.join(R, "experiments", "exp3b_macaronspp_transfer"))
import m_common as MC  # noqa: E402

W = os.path.join(R, "work_dirs", "exp6_v3", "ablation_study", "experiment_CT_continued_training", "states")
FW = os.path.join(R, "work_dirs", "exp6_v3", "ablation_study", "experiment_F_stochastic_coverage_decision", "states")
RES = os.path.join(FD, "results")
DEV5 = {"colosseum", "barts", "pantheon", "eiffel", "sestino_museum"}
PH = ("early", "middle", "late")
TOL = 1e-4
NB = 10000


def phase(t):
    return "early" if t < 20 else ("middle" if t < 50 else "late")


def layer(r):
    return "indoor" if r["domain"] == "indoor" else ("outdoor_dev" if r["scene"] in DEV5 else "outdoor_train")


def boot(d, cl, n=NB, seed=0):
    d, cl = np.asarray(d, float), np.asarray(cl)
    keys = sorted(set(cl))
    if len(d) == 0:
        return None
    tot = np.asarray([d[cl == k].sum() for k in keys])
    cnt = np.asarray([(cl == k).sum() for k in keys], float)
    o = {"d": float(tot.sum() / cnt.sum()), "n": int(cnt.sum()), "clusters": len(keys), "wins": int(sum(tot[i] > 1e-12 for i in range(len(keys)))), "losses": int(sum(tot[i] < -1e-12 for i in range(len(keys))))}
    if len(keys) >= 2:
        idx = np.random.default_rng(seed).integers(0, len(keys), (n, len(keys)))
        bs = tot[idx].sum(1) / cnt[idx].sum(1)
        o["ci"] = [float(np.quantile(bs, 0.025)), float(np.quantile(bs, 0.975))]
    return o


def sd(a, b):
    return float(a) / float(b) if b else None


def sumc(cams, keys=None):
    o = {}
    for c in cams:
        for k, v in c.items():
            if isinstance(v, (int, float)) and not isinstance(v, bool):
                o[k] = o.get(k, 0) + v
    return o


def pred_metrics(cams, u):
    s = sumc(cams)
    tp, fp, fn, tn = s.get("tp", 0), s.get("fp", 0), s.get("fn", 0), s.get("tn", 0)
    prec, rec = sd(tp, tp + fp), sd(tp, tp + fn)
    m = {"precision": prec, "recall": rec, "f1": (2 * prec * rec / (prec + rec)) if prec and rec else None, "fpr": sd(fp, fp + tn), "fnr": sd(fn, tp + fn), "accuracy": sd(tp + tn, tp + fp + fn + tn),
         "brier": sd(s.get("brier", 0), s.get("n_lm", 0)), "pred_hit_rate": sd(tp + fp, tp + fp + fn + tn), "label_hit_rate": sd(s.get("n_lh", 0), s.get("n_lm", 0)), "fp_known_share": sd(s.get("fp_known", 0), fp),
         "mae_tp": (s["sae_tp"] / tp / u) if tp else None, "bias_tp": (s["sse_tp"] / tp / u) if tp else None, "rel_tp": (s["sre_tp"] / tp) if tp else None,
         "mae_all": (s["sae_all"] / s["n_all"] / u) if s.get("n_all") else None, "mae_unk": (s["sae_unk"] / s["n_mae_unk"] / u) if s.get("n_mae_unk") else None,
         "new_precision": sd(s.get("new_tp", 0), s.get("new_tp", 0) + s.get("new_fp", 0)), "new_recall": sd(s.get("new_tp", 0), s.get("new_tp", 0) + s.get("new_fn", 0)),
         "pair": (s["pair_sum"] / s["n_allhit"] / u) if s.get("n_allhit") else None, "rel_std": sd(s.get("rel_std_sum", 0), s.get("n_allhit", 0)), "disagree": sd(s.get("n_disagree", 0), s.get("n_anyhit", 0)),
         "iv_cov": sd(s.get("iv_in", 0), s.get("n_iv", 0)), "best_err": (s["sae_best"] / s["n_best"] / u) if s.get("n_best") else None, "mean_err": (s["sae_mean"] / s["n_best"] / u) if s.get("n_best") else None,
         "within_best": sd(s.get("within_best", 0), s.get("n_best", 0)), "within_mean": sd(s.get("within_mean", 0), s.get("n_best", 0))}
    rv = [c["rho_spread_err"] for c in cams if c.get("rho_spread_err") is not None]
    m["rho_spread_err"] = float(np.mean(rv)) if rv else None
    n_r = sum(c["n_rays"] for c in cams)
    m["blocked_share"] = sum(c["cat_all"]["BLOCKED"] for c in cams) / n_r if n_r else None
    m["near_share"] = sum(c["cat_all"]["NEAR"] for c in cams) / n_r if n_r else None
    nu = sum(c["n_unk"] for c in cams)
    m["unk_share"] = nu / n_r if n_r else None
    m["inf_share_unk"] = sum(sum(c["cat_unk"][k] for k in ("INF_EMPTY", "INF_BAND", "INF_OOD", "INF_OTHER")) for c in cams) / nu if nu else None
    m["blocked_share_unk"] = sum(c["cat_unk"]["BLOCKED"] for c in cams) / nu if nu else None
    hist = np.sum([c["hist_tp"] for c in cams], 0) if cams else None
    m["_hist"] = None if hist is None else hist.tolist()
    return m


def pick_of(score, r, cand=None):
    s = np.asarray(score, float)
    if cand is not None:
        sub = s[cand]
        return cand[int(np.flatnonzero(sub == sub.max())[0])]
    if s.max() == 0.0:
        return MC.fallback_choice(r["scene"], int(r["start"]), int(r["t"]), len(s))
    return MC.argmax_lowest(s)


def dec_metrics(score, orc, r, cand=None, geo2=None):
    """decision metrics of one score list; orc nan where no truth"""
    pk = pick_of(score, r, cand)
    m = np.nanmax(orc)
    o = {"pick": pk, "off_regret": float(m - orc[pk]), "off_top1": float(orc[pk] >= m - TOL), "off_gain": float(orc[pk])}
    idx = np.flatnonzero(np.isfinite(orc))
    s = np.asarray(score, float)[idx]
    o["rho"] = None
    if len(idx) > 2 and np.ptp(s) > 0 and np.ptp(orc[idx]) > 0:
        rr = spearmanr(s, orc[idx]).correlation
        o["rho"] = float(rr) if np.isfinite(rr) else None
    if geo2 is not None:
        o["geo2_regret"] = float(geo2.max() - geo2[pk])
        o["geo2_top1"] = float(geo2[pk] >= geo2.max() - TOL)
    return o


def two_of(Vk, V12k, p1, n, ks):
    V1 = np.asarray(Vk, float)[:, ks].mean(1)
    if len(V12k) == 0:
        return V1
    V12 = np.asarray(V12k, float)[:, ks].mean(1)
    s = V1.copy()
    for i in range(n):
        c = np.flatnonzero(p1 == i)
        if len(c):
            s[i] = V12[c].max()
    return s


class State:
    def __init__(self, r, truth):
        self.r, self.key = r, (r["tag"], r["t"])
        self.domain, self.scene, self.t, self.ckpt, self.tag = r["domain"], r["scene"], r["t"], r["ckpt"], r["tag"]
        self.layer, self.phase = layer(r), phase(r["t"])
        self.u = 7.0 if r["domain"] == "indoor" else 1.0
        if r["domain"] == "indoor":
            self.cand = [int(c) for c in r["cand"]]
            n = r["n_legal"]
            self.orc = np.asarray([np.nan if x is None else x for x in r["gain_pp"]], float)
            self.score = np.asarray(r["score"], float)
            self.camlist = [r["cams"][str(j)] for j in self.cand]
            self.geo2 = None
            self.dec = dec_metrics(self.score, self.orc, r, self.cand)
            K1 = [dec_metrics(np.asarray(r["h0k"], float)[:, k], self.orc, r, self.cand) for k in range(4)]
        else:
            self.cand = None
            keys = [tuple(k) for k in r["keys1"]]
            n = len(keys)
            self.orc = np.asarray([np.nan if x is None else x for x in truth["oracle"]], float)
            gp = np.asarray(truth["geo_path"], float)
            idx = {k: i for i, k in enumerate(keys)}
            p1 = np.asarray([idx[tuple(a)] for a, _ in r["paths"]], int)
            g1 = np.asarray(truth["geo1"], float)
            self.geo2 = np.asarray([gp[p1 == i].max() if (p1 == i).any() else g1[i] for i in range(n)])
            self.score = np.asarray(r["score"], float)
            self.camlist = r["cams"]
            self.dec = dec_metrics(self.score, self.orc, r, None, self.geo2)
            K1 = [dec_metrics(two_of(r["V1k"], r["V12k"], p1, n, [k]), self.orc, r, None, self.geo2) for k in range(4)]
            self.dec_h0 = dec_metrics(np.asarray(r["h0"], float), self.orc, r, None, self.geo2) if r.get("h0") else None
            self.dec_one = dec_metrics(np.asarray(r["V1k"], float).mean(1), self.orc, r, None, self.geo2)
        self.dec["k1_off_regret"] = float(np.mean([x["off_regret"] for x in K1]))
        self.dec["k1_off_top1"] = float(np.mean([x["off_top1"] for x in K1]))
        if self.geo2 is not None:
            self.dec["k1_geo2_regret"] = float(np.mean([x["geo2_regret"] for x in K1]))
        pk = [x["pick"] for x in K1]
        self.dec["k1_flip"] = float(np.mean([pk[a] != pk[b] for a in range(4) for b in range(a + 1, 4)]))
        self.pred = pred_metrics(self.camlist, self.u)


def load(groups):
    rows = {}
    for g in groups:
        for f in sorted(glob.glob(os.path.join(W, g, "*.jsonl"))):
            for line in open(f):
                r = json.loads(line)
                rows[(r["ckpt"], r["tag"], r["t"])] = r
    tags = {k[1] for k in rows if rows[k]["domain"] == "outdoor"}
    truth = {}
    for tag in tags:
        for r in map(json.loads, open(os.path.join(FW, f"{tag}.jsonl"))):
            truth[(tag, r["t"])] = {k: r[k] for k in ("oracle", "geo1", "geo_path", "audit")}
    return {k: State(v, truth.get((k[1], k[2]))) for k, v in rows.items()}


DEC = ("off_regret", "off_top1", "rho", "geo2_regret", "geo2_top1", "k1_off_regret", "k1_off_top1", "k1_geo2_regret", "k1_flip")
PRED = ("precision", "recall", "f1", "fpr", "fnr", "accuracy", "brier", "pred_hit_rate", "label_hit_rate", "fp_known_share", "mae_tp", "bias_tp", "rel_tp", "mae_all", "mae_unk", "new_precision", "new_recall", "pair", "rel_std",
        "disagree", "iv_cov", "best_err", "mean_err", "within_best", "within_mean", "rho_spread_err", "blocked_share", "near_share", "unk_share", "inf_share_unk", "blocked_share_unk")


def table(S, ckpts, base="A_40K"):
    """{layer: {ckpt: {phase|all: {metric: mean}}, 'diff': ...}}"""
    out = {}
    for L in ("indoor", "outdoor_dev", "outdoor_train"):
        out[L] = {}
        keys = sorted({(k[1], k[2]) for k, s in S.items() if s.layer == L})
        if not keys:
            continue
        for c in ckpts:
            st = [S[(c, *k)] for k in keys if (c, *k) in S]
            if not st:
                continue
            row = {"n": len(st), "clusters": len({s.scene for s in st})}
            for ph in ("all",) + PH:
                ss = [s for s in st if ph == "all" or s.phase == ph]
                r = {"n": len(ss)}
                for m in DEC:
                    v = [s.dec.get(m) for s in ss if s.dec.get(m) is not None]
                    r[m] = float(np.mean(v)) if v else None
                for m in PRED:
                    v = [s.pred.get(m) for s in ss if s.pred.get(m) is not None]
                    r[m] = float(np.mean(v)) if v else None
                row[ph] = r
                if st[0].domain == "outdoor":
                    for nm, attr in (("h0", "dec_h0"), ("one", "dec_one")):
                        v = [getattr(s, attr)["off_regret"] for s in ss if getattr(s, attr)]
                        r[f"{nm}_off_regret"] = float(np.mean(v)) if v else None
            if c != base:
                d = {}
                pairs = [(S[(c, *k)], S[(base, *k)]) for k in keys if (c, *k) in S and (base, *k) in S]
                for ph in ("all",) + PH:
                    pp = [(a, b) for a, b in pairs if ph == "all" or a.phase == ph]
                    for m in ("off_regret", "off_top1", "rho", "geo2_regret", "k1_off_regret"):
                        v = [(a.dec[m] - b.dec[m], a.scene) for a, b in pp if a.dec.get(m) is not None and b.dec.get(m) is not None]
                        if v:
                            d.setdefault(ph, {})[m] = boot(*zip(*v))
                    for m in ("precision", "recall", "f1", "brier", "mae_tp", "rel_tp", "mae_all", "mae_unk", "new_precision", "new_recall", "pair", "rel_std", "disagree", "iv_cov", "best_err", "within_best"):
                        v = [(a.pred[m] - b.pred[m], a.scene) for a, b in pp if a.pred.get(m) is not None and b.pred.get(m) is not None]
                        if v:
                            d.setdefault(ph, {})[m] = boot(*zip(*v))
                    d.setdefault(ph, {})["changed_picks"] = int(sum(a.dec["pick"] != b.dec["pick"] for a, b in pp))
                row["diff_vs_" + base] = d
                row["per_scene_off_regret"] = {s_: float(np.mean([a.dec["off_regret"] - b.dec["off_regret"] for a, b in pairs if a.scene == s_])) for s_ in sorted({a.scene for a, _ in pairs})}
            out[L][c] = row
    return out


def decision_error(S, c):
    """CT3-D for checkpoint c, per layer and phase."""
    out = {}
    for L in ("indoor", "outdoor_dev", "outdoor_train"):
        for ph in ("all",) + PH:
            st = [s for k, s in S.items() if k[0] == c and s.layer == L and (ph == "all" or s.phase == ph)]
            if not st:
                continue
            rows = {"n": len(st), "n_wrong": 0}
            diffs = {m: [] for m in ("mae_unk", "fpr", "fnr", "new_fp_share", "new_fn_share", "new_pred_rays")}
            over = {m: [] for m in ("mae_unk", "fpr", "fnr", "new_fp_share", "new_fn_share")}
            st_err = {m: [] for m in ("mae_unk", "fpr", "fnr", "new_fp_share", "new_fn_share")}
            reg = []
            for s in st:
                idx = s.cand if s.cand is not None else list(range(len(s.camlist)))
                cam = {i: (s.r["cams"][str(i)] if s.cand is not None else s.camlist[i]) for i in idx}

                def camm(i):
                    c_ = cam[i]
                    tp, fp, fn, tn = c_["tp"], c_["fp"], c_["fn"], c_["tn"]
                    return {"mae_unk": (c_["sae_unk"] / c_["n_mae_unk"] / s.u) if c_["n_mae_unk"] else np.nan, "fpr": fp / (fp + tn) if fp + tn else np.nan, "fnr": fn / (tp + fn) if tp + fn else np.nan,
                            "new_fp_share": c_["new_fp"] / (c_["new_tp"] + c_["new_fp"]) if c_["new_tp"] + c_["new_fp"] else np.nan, "new_fn_share": c_["new_fn"] / (c_["new_tp"] + c_["new_fn"]) if c_["new_tp"] + c_["new_fn"] else np.nan,
                            "new_pred_rays": (c_["new_tp"] + c_["new_fp"]) / c_["K"]}
                M = {i: camm(i) for i in idx}
                sc = np.asarray([s.score[i] for i in idx], float)
                gn = np.asarray([s.orc[i] for i in idx], float)
                pk = s.dec["pick"]
                best = idx[int(np.nanargmax(gn))]
                sm = pred_like(s)
                for m in st_err:
                    st_err[m].append(sm[m])
                reg.append(s.dec["off_regret"])
                if s.dec["off_top1"] < 1:
                    rows["n_wrong"] += 1
                    for m in diffs:
                        a, b = M[pk][m], M[best][m]
                        if np.isfinite(a) and np.isfinite(b):
                            diffs[m].append((a - b, s.scene))
                if len(idx) >= 3 and np.ptp(gn) > 0 and np.ptp(sc) > 0:
                    rk = lambda x: np.argsort(np.argsort(x))  # noqa: E731
                    ov = rk(sc) - rk(gn)
                    for m in over:
                        e = np.asarray([M[i][m] for i in idx], float)
                        ok = np.isfinite(e)
                        if ok.sum() >= 3 and np.ptp(e[ok]) > 0 and np.ptp(ov[ok]) > 0:
                            rr = spearmanr(e[ok], ov[ok]).correlation
                            if np.isfinite(rr):
                                over[m].append((rr, s.scene))
            rows["picked_minus_best"] = {m: boot(*zip(*v)) if v else None for m, v in diffs.items()}
            rows["overestimate_vs_error_rho"] = {m: boot(*zip(*v)) if v else None for m, v in over.items()}
            rows["state_regret_vs_error_spearman"] = {}
            for m, v in st_err.items():
                v = np.asarray(v, float)
                r_ = np.asarray(reg)
                ok = np.isfinite(v)
                if ok.sum() > 5 and np.ptp(v[ok]) > 0 and np.ptp(r_[ok]) > 0:
                    rows["state_regret_vs_error_spearman"][m] = float(spearmanr(v[ok], r_[ok]).correlation)
            out.setdefault(L, {})[ph] = rows
    return out


def pred_like(s):
    x = sumc(s.camlist)
    tp, fp, fn, tn = x["tp"], x["fp"], x["fn"], x["tn"]
    return {"mae_unk": x["sae_unk"] / x["n_mae_unk"] / s.u if x["n_mae_unk"] else np.nan, "fpr": fp / (fp + tn) if fp + tn else np.nan, "fnr": fn / (tp + fn) if tp + fn else np.nan,
            "new_fp_share": x["new_fp"] / (x["new_tp"] + x["new_fp"]) if x["new_tp"] + x["new_fp"] else np.nan, "new_fn_share": x["new_fn"] / (x["new_tp"] + x["new_fn"]) if x["new_tp"] + x["new_fn"] else np.nan}


def mae_vs_regret(S, c, base="A_40K"):
    """state-level: change of prediction metrics vs change of regret (c - base); and scene-level pairs."""
    out = {}
    for L in ("indoor", "outdoor_dev", "outdoor_train"):
        pairs = [(s, S[(base, k[1], k[2])]) for k, s in S.items() if k[0] == c and s.layer == L and (base, k[1], k[2]) in S]
        if not pairs:
            continue
        r = {}
        dr = np.asarray([a.dec["off_regret"] - b.dec["off_regret"] for a, b in pairs])
        for m in ("mae_unk", "mae_tp", "new_precision", "new_recall", "precision", "recall"):
            dv = np.asarray([(a.pred[m] - b.pred[m]) if a.pred.get(m) is not None and b.pred.get(m) is not None else np.nan for a, b in pairs], float)
            ok = np.isfinite(dv)
            ch = ok & (dr != 0)
            r[m] = {"spearman_all": float(spearmanr(dv[ok], dr[ok]).correlation) if ok.sum() > 5 and np.ptp(dr[ok]) > 0 else None,
                    "spearman_changed": float(spearmanr(dv[ch], dr[ch]).correlation) if ch.sum() > 5 else None, "n_changed": int(ch.sum()),
                    "scenes": {sc: {"d_metric": float(np.nanmean([x for x, (a, _) in zip(dv, pairs) if a.scene == sc])), "d_regret": float(np.mean([x for x, (a, _) in zip(dr, pairs) if a.scene == sc]))} for sc in sorted({a.scene for a, _ in pairs})}}
        out[L] = r
    return out


def blocked(S, c):
    out = {}
    for k, s in S.items():
        if k[0] != c:
            continue
        for cam in s.camlist:
            o = out.setdefault(s.layer, {}).setdefault(s.scene, {"rays": 0, "BLOCKED": 0, "NEAR": 0, "HIT": 0, "NOHIT": 0, "KNOWN": 0, "OUT": 0, "INF": 0, "by_phase": {p: [0, 0] for p in PH}})
            o["rays"] += cam["n_rays"]
            for m in ("BLOCKED", "NEAR", "HIT", "NOHIT", "KNOWN", "OUT"):
                o[m] += cam["cat_all"][m]
            o["INF"] += sum(cam["cat_all"][m] for m in ("INF_EMPTY", "INF_BAND", "INF_OOD", "INF_OTHER"))
            o["by_phase"][s.phase][0] += cam["cat_all"]["BLOCKED"]
            o["by_phase"][s.phase][1] += cam["n_rays"]
    return out


def counterfactuals(groups, S, c):
    """regret ladder (official + geometric two-step outdoor; official indoor) for the cf group of checkpoint c, paired vs pred"""
    rows = []
    for g in groups:
        for f in sorted(glob.glob(os.path.join(W, g, "*.jsonl"))):
            rows += list(map(json.loads, open(f)))
    out = {}
    VARS = ("pred", "no_fp", "no_fn", "hit_oracle", "depth_oracle", "perfect", "perfect_all")
    data = {}
    for r in rows:
        if r["ckpt"] != c or (c, r["tag"], r["t"]) not in S:
            continue
        s = S[(c, r["tag"], r["t"])]
        for v in VARS:
            if r["domain"] == "indoor":
                d = dec_metrics(np.asarray([np.nan if x is None else x for x in r["cf"][v]["one"]], float), s.orc, r, s.cand)
                data.setdefault((s.layer, "one"), {}).setdefault(v, []).append((d, s))
            else:
                data.setdefault((s.layer, "two"), {}).setdefault(v, []).append((dec_metrics(np.asarray(r["cf"][v]["two"], float), s.orc, r, None, s.geo2), s))
                data.setdefault((s.layer, "one"), {}).setdefault(v, []).append((dec_metrics(np.asarray(r["cf"][v]["one"], float), s.orc, r, None, s.geo2), s))
    for (L, h), dd in data.items():
        o = {}
        for ph in ("all",) + PH:
            for v in VARS:
                it = [(d, s) for d, s in dd[v] if ph == "all" or s.phase == ph]
                if not it:
                    continue
                x = {"n": len(it), "off_regret": float(np.mean([d["off_regret"] for d, _ in it])), "off_top1": float(np.mean([d["off_top1"] for d, _ in it]))}
                if "geo2_regret" in it[0][0]:
                    x["geo2_regret"] = float(np.mean([d["geo2_regret"] for d, _ in it]))
                if v != "pred":
                    base = [d for d, s in dd["pred"] if ph == "all" or s.phase == ph]
                    x["d_off_regret"] = boot([d["off_regret"] - b["off_regret"] for (d, _), b in zip(it, base)], [s.scene for _, s in it])
                o.setdefault(ph, {})[v] = x
        out.setdefault(L, {})[h] = o
    return out


def write_csv(S, path):
    with open(path, "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["ckpt", "layer", "scene", "tag", "t", "phase", "pick", "off_regret", "off_top1", "rho", "geo2_regret", "k1_off_regret", "k1_flip", "precision", "recall", "mae_tp", "mae_unk", "new_precision", "new_recall", "pair", "disagree", "iv_cov", "blocked_share"])
        for k in sorted(S):
            s = S[k]
            w.writerow([k[0], s.layer, s.scene, s.tag, s.t, s.phase] + [s.dec.get(m) for m in ("pick", "off_regret", "off_top1", "rho", "geo2_regret", "k1_off_regret", "k1_flip")] + [s.pred.get(m) for m in ("precision", "recall", "mae_tp", "mae_unk", "new_precision", "new_recall", "pair", "disagree", "iv_cov", "blocked_share")])


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--groups", required=True)
    ap.add_argument("--cf", default="")
    a = ap.parse_args()
    S = load(a.groups.split(","))
    ck = sorted({k[0] for k in S}, key=lambda c: {"A_40K": 0, "B_Y50K": 1, "C_CT50K": 2, "C_CT60K": 3}.get(c, 9))
    print("states", {c: sum(k[0] == c for k in S) for c in ck})
    T = table(S, ck)
    os.makedirs(os.path.join(RES, "CT1_training_steps"), exist_ok=True)
    os.makedirs(os.path.join(RES, "CT3_prediction_diagnostics"), exist_ok=True)
    json.dump({"checkpoints": ck, "table": T}, open(os.path.join(RES, "CT1_training_steps", "fixed_summary.json"), "w"), indent=1, default=float)
    write_csv(S, os.path.join(RES, "CT1_training_steps", "fixed_states.csv"))
    reg = {}
    for k, s in S.items():
        for kk, vv in s.r.get("regression", {}).items():
            x = reg.setdefault(k[0], {}).setdefault(kk, [])
            x.append(vv)
    summary = {c: {kk: (sum(bool(x) for x in v), len(v)) if not any(isinstance(x, float) for x in v) else (max(v)) for kk, v in d.items()} for c, d in reg.items()}
    print("regression", summary)
    json.dump(summary, open(os.path.join(RES, "CT1_training_steps", "fixed_regression.json"), "w"), indent=1, default=float)
    D = {"decision_error": {c: decision_error(S, c) for c in ck}, "metric_vs_regret": {c: mae_vs_regret(S, c) for c in ck if c != "A_40K"}, "blocked": blocked(S, "A_40K")}
    if a.cf:
        groups = a.cf.split(",")
        D["counterfactuals"] = {c: counterfactuals(groups, S, c) for c in ck}
    json.dump(D, open(os.path.join(RES, "CT3_prediction_diagnostics", "diagnostics.json"), "w"), indent=1, default=float)
    for L in T:
        for c in ck:
            if c not in T[L]:
                continue
            r = T[L][c]["all"]
            d = T[L][c].get("diff_vs_A_40K", {}).get("all", {})
            f = lambda x: "-" if x is None else f"{x:.3f}"  # noqa: E731
            print(f"{L:14s} {c:8s} n={r['n']:4d} regret {f(r['off_regret'])} top1 {f(r['off_top1'])} rho {f(r['rho'])} geo2 {f(r['geo2_regret'])} | prec {f(r['precision'])} rec {f(r['recall'])} maeTP {f(r['mae_tp'])} maeU {f(r['mae_unk'])} newP {f(r['new_precision'])} newR {f(r['new_recall'])} | pair {f(r['pair'])} iv {f(r['iv_cov'])}", end="")
            if d:
                x = d.get("off_regret")
                print(f" | d_regret {x['d']:+.3f} {[round(v, 3) for v in x.get('ci', [])]} changed {d.get('changed_picks')}", end="")
            print()


if __name__ == "__main__":
    main()
