"""Experiment F -- analysis of the state cache (CPU). configs/F_config.yaml.
  --stage explore  F0 (regression, GT-depth audit), F1, F2, F3, F4 on the exploration layer (395 states); applies the frozen selection rules
                   and writes results/selection.json ONCE (never overwritten)
  --stage confirm  requires results/selection.json; F0 regression, the frozen candidates vs D-v2 with the pre-registered gates, F3 equal-budget
                   control, F4 interaction / combinations on the confirmation layer (360 states)
Every decision score is the planner's form: two-step score(c1) = max over the planner's children of the path value (no children -> one-step),
pick = m_common.argmax_lowest (all-zero -> the planner's seeded fallback). Official truth = one-step official coverage gain (pp) of the picked
c1; geometric two-step truth = max over children of the true union (pp). CI: paired bootstrap of clusters (scene = pre-registered; trajectory
= secondary), 10000 draws, seed 0, ratio estimator (sum of differences / number of states)."""
import argparse
import csv
import glob
import json
import os
import sys

import numpy as np
import yaml

HERE = os.path.dirname(os.path.abspath(__file__))
FD = os.path.dirname(HERE)
R = os.path.dirname(os.path.dirname(os.path.dirname(os.path.dirname(FD))))
sys.path.insert(0, os.path.join(R, "experiments", "exp3b_macaronspp_transfer"))
import m_common as MC  # noqa: E402

W = os.environ.get("F_STATES") or os.path.join(R, "work_dirs", "exp6_v3", "ablation_study", "experiment_F_stochastic_coverage_decision", "states")
RES = os.path.join(FD, "results")
CFG = yaml.safe_load(open(os.path.join(FD, "configs", "F_config.yaml")))
GAMMAS = [float(g) for g in CFG["F1"]["gamma_horizon"]]
SOFT = [f"s{s:g}_{w}" for s in CFG["F2"]["sigma_over_tau_cov"] for w in ("w", "area")]
PHASES = (("early", 0, 20), ("middle", 20, 50), ("late", 50, 100))
TOP1_TOL = 1e-4
NB = 10000


# ------------------------------------------------------------------------------------------------ state + scoring
class St:
    def __init__(self, r):
        self.r = r
        self.scene, self.start, self.t, self.tag = r["scene"], int(r["start"]), int(r["t"]), r["tag"]
        self.phase = next(p for p, a, b in PHASES if a <= self.t < b)
        self.keys1 = [tuple(k) for k in r["keys1"]]
        self.n = len(self.keys1)
        idx = {k: i for i, k in enumerate(self.keys1)}
        self.p1 = np.asarray([idx[tuple(a)] for a, _ in r["paths"]], int)
        self.p2 = [tuple(b) for _, b in r["paths"]]
        self.ch = [np.flatnonzero(self.p1 == i) for i in range(self.n)]
        self.orc = np.asarray([np.nan if x is None else x for x in r["oracle"]], float)
        self.geo1 = np.asarray(r["geo1"], float)
        self.gp = np.asarray(r["geo_path"], float)
        self.geo2 = np.asarray([self.gp[c].max() if len(c) else self.geo1[i] for i, c in enumerate(self.ch)])
        h = r["hard"]
        self.H = {k: (np.asarray(v["V1"], float), np.asarray(v["V12"], float).reshape(len(self.p1), -1)) for k, v in h.items()}

    def two(self, V1, V12):
        """two-step score per c1 + chosen path index per c1 (-1 = none)"""
        s, cp = V1.astype(float).copy(), np.full(self.n, -1)
        for i, c in enumerate(self.ch):
            if len(c):
                j = c[np.flatnonzero(V12[c] == V12[c].max())[0]]
                s[i], cp[i] = V12[j], j
        return s, cp

    def pick(self, s):
        s = np.asarray(s, float)
        if s.max() == 0.0:
            return MC.fallback_choice(self.scene, self.start, self.t, len(s))
        return MC.argmax_lowest(s)

    def ev(self, s, cp=None):
        pk = self.pick(s)
        o = {"pick": pk}
        if np.isfinite(self.orc).any() and np.isfinite(self.orc[pk]):
            m = np.nanmax(self.orc)
            o.update(off_regret=float(m - self.orc[pk]), off_top1=bool(self.orc[pk] >= m - TOP1_TOL), off_gain=float(self.orc[pk]))
        o.update(geo1_regret=float(self.geo1.max() - self.geo1[pk]), geo2_regret=float(self.geo2.max() - self.geo2[pk]),
                 geo2_top1=bool(self.geo2[pk] >= self.geo2.max() - TOP1_TOL), geo2_gain=float(self.geo2[pk]))
        if cp is not None and cp[pk] >= 0:
            o.update(path_union=float(self.gp[cp[pk]]), path_regret=float(self.gp.max() - self.gp[cp[pk]]))
        return o

    # hard per-sample (K4 run = the planner; K8 run = same seeds + 4)
    def hard(self, run="K4", ks=None):
        V1k, V12k = self.H[run]
        ks = list(range(V1k.shape[1])) if ks is None else list(ks)
        return V1k[:, ks].mean(1), V12k[:, ks].mean(1)

    def soft(self, run, sub, var, rule="max"):
        d = self.r["soft"][f"{run}|{sub}|{var}|{rule}"]
        return np.asarray(d["V1"], float), np.asarray(d["V12"], float)


def f1(st, V1, V12, g):
    """J_gamma(c1) = V1 + gamma * max_c2 max(0, V12 - V1); gamma 1 -> max_c2 V12 (the D-v2 score); no children -> V1"""
    if g == 1.0:
        return st.two(V1, V12)
    s, cp = V1.astype(float).copy(), np.full(st.n, -1)
    for i, c in enumerate(st.ch):
        if len(c):
            d = np.maximum(0.0, V12[c] - V1[i])
            j = c[np.flatnonzero(d == d.max())[0]]
            s[i], cp[i] = V1[i] + g * max(0.0, V12[j] - V1[i]), j
    return s, cp


def load(stage):
    pats = ("e3_*.jsonl", "g2_*.jsonl") if stage == "explore" else ("conf_*.jsonl",)
    rows = []
    for p in pats:
        for f in sorted(glob.glob(os.path.join(W, p))):
            rows += [json.loads(x) for x in open(f)]
    return [St(r) for r in rows]


# ------------------------------------------------------------------------------------------------ statistics
def boot(diff, clus, n=NB, seed=0):
    """ratio-estimator cluster bootstrap of the mean of paired differences"""
    diff, clus = np.asarray(diff, float), np.asarray(clus)
    keys = sorted(set(clus))
    if len(diff) == 0:
        return None
    tot = np.asarray([diff[clus == k].sum() for k in keys])
    cnt = np.asarray([(clus == k).sum() for k in keys], float)
    out = {"d": float(tot.sum() / cnt.sum()), "n": int(cnt.sum()), "clusters": len(keys)}
    if len(keys) >= 2:
        idx = np.random.default_rng(seed).integers(0, len(keys), (n, len(keys)))
        bs = tot[idx].sum(1) / cnt[idx].sum(1)
        out["ci"] = [float(np.quantile(bs, 0.025)), float(np.quantile(bs, 0.975))]
    return out


def summarize(sts, E, metric):
    v = [e.get(metric) for e in E]
    v = [float(x) for x in v if x is not None]
    return float(np.mean(v)) if v else None


def table(sts, methods, base=None):
    """methods: name -> list of ev dicts (aligned with sts). Means overall / by phase, paired diffs vs base with scene + trajectory CIs."""
    out = {}
    for nm, E in methods.items():
        row = {"n": len(E)}
        for m in ("off_regret", "off_top1", "geo1_regret", "geo2_regret", "geo2_top1", "path_union", "path_regret", "off_gain", "geo2_gain"):
            row[m] = summarize(sts, E, m)
            for ph, _, _ in PHASES:
                row[f"{m}@{ph}"] = summarize([s for s in sts if s.phase == ph], [e for s, e in zip(sts, E) if s.phase == ph], m)
        if base is not None and nm != base:
            B = methods[base]
            row["changed_picks"] = int(sum(e["pick"] != b["pick"] for e, b in zip(E, B)))
            for m in ("off_regret", "geo2_regret", "path_union", "off_top1"):
                pr = [(float(e[m]) - float(b[m]), s.scene, s.tag, s.phase) for s, e, b in zip(sts, E, B) if e.get(m) is not None and b.get(m) is not None]
                if not pr:
                    continue
                d, sc, tg, ph = map(np.asarray, zip(*pr))
                row[f"d_{m}"] = {"scene": boot(d, sc), "traj": boot(d, tg), **{ph_: boot(d[ph == ph_], sc[ph == ph_]) for ph_, _, _ in PHASES}}
                per = {}
                for s_ in sorted(set(sc)):
                    per[s_] = float(d[sc == s_].mean())
                row[f"d_{m}"]["per_scene"] = per
        out[nm] = row
    return out


def gate(row, extra=None):
    """pre-registered confirmation gate (configs gates_confirmation)"""
    a = row.get("d_off_regret", {}).get("scene")
    g = row.get("d_geo2_regret", {}).get("scene")
    if not a or not g or "ci" not in a:
        return {"verdict": "UNKNOWN"}
    off_ok = a["d"] < 0 and a["ci"][1] < 0
    geo_ok = g["d"] <= 0.05 and g["ci"][0] <= 0
    v = "PASS" if off_ok and geo_ok else ("TREND" if a["d"] < 0 and geo_ok else "FAIL")
    if extra is not None and v == "PASS" and not (extra["d"] < 0 and extra.get("ci", [0, 0])[1] < 0):
        v = "FAIL (equal-budget control)"
    return {"verdict": v, "off": a, "geo2": g, "extra": extra}


# ------------------------------------------------------------------------------------------------ F0
def f0(sts):
    reg = [s.r["regression"] for s in sts]
    ref_key = "replay_vs_multiview_bitwise" if "replay_vs_multiview_bitwise" in reg[0] else "replay_vs_closed_loop_log_bitwise"
    out = {"states": len(sts), "reference": reg[0]["reference"],
           "recorded_vs_planner_K4_max": max(r["recorded_vs_planner_K4"] for r in reg),
           "recorded_vs_planner_K8_max": max(r["recorded_vs_planner_K8"] for r in reg),
           "gamma1_bitwise_all": all(r["gamma1_bitwise"] for r in reg),
           "avg0_onestep_vs_hard_max_abs": max(r["avg0_onestep_vs_hard_max_abs"] for r in reg),
           "p1_unchanged_all": all(r["p1_unchanged_by_scoring"] for r in reg),
           "replay_reference_bitwise": int(sum(bool(r.get(ref_key)) for r in reg)), "replay_reference_missing": int(sum(r.get(ref_key) is None for r in reg)),
           "oracle_complete_states": int(sum(np.isfinite(s.orc).all() for s in sts)),
           "oracle_partial_states": int(sum(np.isfinite(s.orc).any() and not np.isfinite(s.orc).all() for s in sts))}
    # K = 8 run, first 4 samples vs the planner's K = 4 run (batch-size numerics)
    rel, chg = [], 0
    for s in sts:
        a, _ = s.two(*s.hard("K4"))
        b, _ = s.two(*s.hard("K8", range(4)))
        rel.append(float(np.max(np.abs(a - b) / np.maximum(1.0, np.abs(a)))))
        chg += int(s.pick(a) != s.pick(b))
    out["K8_first4_vs_K4"] = {"max_rel": max(rel), "median_rel": float(np.median(rel)), "pick_changes": chg}
    out["PASS"] = bool(out["recorded_vs_planner_K4_max"] == 0 and out["recorded_vs_planner_K8_max"] == 0 and out["gamma1_bitwise_all"]
                       and out["avg0_onestep_vs_hard_max_abs"] < 1e-9 and out["p1_unchanged_all"]
                       and out["replay_reference_bitwise"] + out["replay_reference_missing"] == len(sts) and out["replay_reference_missing"] < len(sts))
    # GT-depth audit (one-step ranking against the official truth; first-layer views)
    M = {}
    M["pred_hard_K4 (D-v2 one-step)"] = [s.ev(s.hard("K4")[0]) for s in sts]
    for k in ("true_vc0", "true_dv2", "gated_vc0", "gated_dv2", "true_vc0_cat_HIT"):
        M[f"value_on_{k}"] = [s.ev(np.asarray([a.get(k, 0.0) for a in s.r["audit"]], float)) for s in sts]
    M["geo1 (true coverage, upper reference)"] = [s.ev(s.geo1) for s in sts]
    out["gt_depth_audit"] = table(sts, M, base="value_on_true_vc0")
    out["gt_depth_audit_vs_pred"] = table(sts, M, base="pred_hard_K4 (D-v2 one-step)")
    cats = ["HIT", "KNOWN", "NOHIT", "INF_BAND", "INF_EMPTY", "INF_OTHER", "INF_OOD", "BLOCKED", "NEAR", "OUT"]
    sh = {c: [] for c in cats}
    ratio = {"gated_vc0/true_vc0": [], "gated_dv2/true_vc0": [], "true_dv2/true_vc0": [], "pred/true_vc0": []}
    for s in sts:
        pv = s.hard("K4")[0]
        for i, a in enumerate(s.r["audit"]):
            if a["true_vc0"] <= 0:
                continue
            for c in cats:
                sh[c].append(a.get(f"true_vc0_cat_{c}", 0.0) / a["true_vc0"])
            ratio["gated_vc0/true_vc0"].append(a["gated_vc0"] / a["true_vc0"])
            ratio["gated_dv2/true_vc0"].append(a["gated_dv2"] / a["true_vc0"])
            ratio["true_dv2/true_vc0"].append(a["true_dv2"] / a["true_vc0"])
            ratio["pred/true_vc0"].append(pv[i] / a["true_vc0"])
    out["gt_depth_value_share_by_category"] = {c: float(np.mean(v)) for c, v in sh.items() if v}
    out["gt_depth_value_ratios_median"] = {k: float(np.median(v)) for k, v in ratio.items() if v}
    return out


# ------------------------------------------------------------------------------------------------ F1
def f1_all(sts):
    M = {f"gamma_{g:g}": [s.ev(*f1(s, *s.hard("K4"), g)) for s in sts] for g in GAMMAS}
    tab = table(sts, M, base="gamma_1")
    mech = {"ratio_dV2_over_V1_pick": {}, "second_step_dominates_share": {}, "shared_best_c2_top2_share": {}, "rel_margin_top2_gamma1": {}, "rel_margin_top2_gamma0": {}}
    for ph in ("all",) + tuple(p for p, _, _ in PHASES):
        S_ = [s for s in sts if ph == "all" or s.phase == ph]
        rat, dom, shr, m1, m0 = [], [], [], [], []
        for s in S_:
            V1, V12 = s.hard("K4")
            sc, cp = s.two(V1, V12)
            pk = s.pick(sc)
            if cp[pk] >= 0 and V1[pk] > 0:
                d2 = max(0.0, V12[cp[pk]] - V1[pk])
                rat.append(d2 / V1[pk])
                dom.append(d2 > V1[pk])
            o = np.argsort(-sc, kind="stable")
            if s.n >= 2:
                a, b = o[0], o[1]
                if cp[a] >= 0 and cp[b] >= 0:
                    shr.append(s.p2[cp[a]] == s.p2[cp[b]])
                m1.append((sc[a] - sc[b]) / max(sc[a], 1e-9))
                s0 = np.sort(V1)[::-1]
                m0.append((s0[0] - s0[1]) / max(s0[0], 1e-9))
        mech["ratio_dV2_over_V1_pick"][ph] = {"median": float(np.median(rat)) if rat else None, "mean": float(np.mean(rat)) if rat else None}
        mech["second_step_dominates_share"][ph] = float(np.mean(dom)) if dom else None
        mech["shared_best_c2_top2_share"][ph] = float(np.mean(shr)) if shr else None
        mech["rel_margin_top2_gamma1"][ph] = float(np.median(m1)) if m1 else None
        mech["rel_margin_top2_gamma0"][ph] = float(np.median(m0)) if m0 else None
    # states where the second step dominates (dV2 > V1 at the D-v2 pick): regret by gamma
    dom_rows = []
    for s, *E in zip(sts, *[M[f"gamma_{g:g}"] for g in GAMMAS]):
        V1, V12 = s.hard("K4")
        sc, cp = s.two(V1, V12)
        pk = s.pick(sc)
        if cp[pk] >= 0 and V12[cp[pk]] - V1[pk] > V1[pk]:
            dom_rows.append([e.get("off_regret") for e in E])
    mech["second_step_dominates_off_regret_by_gamma"] = {f"gamma_{g:g}": (float(np.mean([r[i] for r in dom_rows if r[i] is not None])) if dom_rows else None)
                                                         for i, g in enumerate(GAMMAS)}
    mech["second_step_dominates_states"] = len(dom_rows)
    # cases: Barts early error + the largest changes of the best gamma < 1 (by exploration mean)
    cases = []
    for s in sts:
        if s.tag.endswith("barts_s0") and s.t == 15 and "B2" in s.tag:
            cases.append({"case": "barts_B2_s0_t15", **case_row(s)})
    return tab, mech, cases


def case_row(s):
    o = {"tag": s.tag, "t": s.t, "official": [None if not np.isfinite(x) else round(float(x), 3) for x in s.orc]}
    for g in GAMMAS:
        sc, cp = f1(s, *s.hard("K4"), g)
        e = s.ev(sc, cp)
        o[f"gamma_{g:g}"] = {"pick": e["pick"], "c2": None if cp[e["pick"]] < 0 else list(s.p2[cp[e["pick"]]]), "score": round(float(sc[e["pick"]]), 1),
                            "off_regret": e.get("off_regret"), "geo2_regret": e["geo2_regret"]}
    return o


def f1_cases(sts, g_star, n=6):
    rows = []
    for s in sts:
        a = s.ev(*f1(s, *s.hard("K4"), 1.0))
        b = s.ev(*f1(s, *s.hard("K4"), g_star))
        if a["pick"] != b["pick"] and a.get("off_regret") is not None and b.get("off_regret") is not None:
            rows.append((b["off_regret"] - a["off_regret"], s))
    rows.sort(key=lambda x: x[0])
    pick = rows[:n] + rows[-n:]
    return [{"d_off_regret": round(d, 3), **case_row(s)} for d, s in pick]


# ------------------------------------------------------------------------------------------------ F2
def f2_methods(sts, run="K4", sub="all4", variants=None, rule="max"):
    variants = variants or (["avg0_w", "avg0_area"] + SOFT)
    M = {"hard (D-v2)": [s.ev(*s.two(*s.hard("K4"))) for s in sts],
         "hard_area": [s.ev(*s.two(*s.hard("K4_area"))) for s in sts]}
    for v in variants:
        M[v] = [s.ev(*s.two(*s.soft(run, sub, v, rule))) for s in sts]
    return M


def f2_all(sts):
    M = f2_methods(sts)
    tab = table(sts, M, base="hard (D-v2)")
    one = {"hard (D-v2) one-step": [s.ev(s.hard("K4")[0]) for s in sts]}
    for v in ["avg0_w"] + SOFT:
        one[f"{v} one-step"] = [s.ev(s.soft("K4", "all4", v)[0]) for s in sts]
    tab1 = table(sts, one, base="hard (D-v2) one-step")
    dif = {}
    for v in ["avg0_w", "avg0_area"] + SOFT:
        x = [s.r["diffusion"][v] for s in sts if s.r["diffusion"][v]["mass"] > 0]
        dif[v] = {"near_gt_share": float(np.mean([a["near_gt_share"] for a in x])), "hard_cell_share": float(np.mean([a["hard_cell_share"] for a in x])),
                  "mass_ratio_to_avg0": float(np.median([a["mass"] / max(b["mass"], 1e-9) for a, b in
                                                         zip(x, [s.r["diffusion"]["avg0_" + v.split("_")[-1]] for s in sts if s.r["diffusion"][v]["mass"] > 0])]))}
    cost = {"soft_all_variants_s_per_state_mean": float(np.mean([s.r["timing"]["soft_build_s"] + s.r["timing"]["soft_score_s"] for s in sts])),
            "planner_K4_s_mean": float(np.mean([s.r["timing"]["planner_K4_s"] for s in sts])),
            "hard_rescoring_s_mean": float(np.mean([s.r["timing"]["hard_s"] for s in sts])),
            "peak_mem_gb_max": float(np.max([s.r["peak_mem_gb"] for s in sts])),
            "note": "soft timing covers 5 kernels x 2 weights x (K4 + K8 runs) x all K-subsets and union rules in one pass; unoptimised offline code"}
    return tab, tab1, dif, cost


def f2_unions(sts, var):
    M = {"hard (D-v2)": [s.ev(*s.two(*s.hard("K4"))) for s in sts]}
    for sub, run in (("all4", "K4"), ("first1", "K8"), ("first4", "K8")):
        for rule in ("max", "clip_sum", "noisy_or"):
            M[f"{var} {run}|{sub} {rule}"] = [s.ev(*s.two(*s.soft(run, sub, var, rule))) for s in sts]
    return table(sts, M, base="hard (D-v2)")


# ------------------------------------------------------------------------------------------------ F3
def adaptive_flags(s, rng_seed=0, z_thr=1.0, flip_thr=0.25, nboot=200):
    V1k, V12k = s.H["K8"]
    ks = [0, 1, 2, 3]
    sc, cp = s.two(V1k[:, ks].mean(1), V12k[:, ks].mean(1))
    o = np.argsort(-sc, kind="stable")
    if s.n < 2:
        return False, None, None
    a, b = o[0], o[1]
    xa = V12k[cp[a], ks] if cp[a] >= 0 else V1k[a, ks]
    xb = V12k[cp[b], ks] if cp[b] >= 0 else V1k[b, ks]
    se = np.sqrt(xa.var(ddof=1) / 4 + xb.var(ddof=1) / 4)
    z = (sc[a] - sc[b]) / se if se > 0 else np.inf
    rng = np.random.default_rng(rng_seed)
    p0 = s.pick(sc)
    flips = 0
    for _ in range(nboot):
        kk = rng.integers(0, 4, 4)
        s2, _ = s.two(V1k[:, kk].mean(1), V12k[:, kk].mean(1))
        flips += int(s.pick(s2) != p0)
    fr = flips / nboot
    return bool(z < z_thr or fr > flip_thr), float(z), float(fr)


def f3_all(sts, seeds=50):
    M = {f"K8_first{K}": [s.ev(*s.two(*s.hard("K8", range(K)))) for s in sts] for K in (1, 2, 4, 6, 8)}
    M["K4 (planner run)"] = [s.ev(*s.two(*s.hard("K4"))) for s in sts]
    fl = [adaptive_flags(s) for s in sts]
    up = np.asarray([f[0] for f in fl])
    u = float(up.mean())
    M["adaptive 4->8"] = [s.ev(*s.two(*s.hard("K8", range(8 if f else 4)))) for s, f in zip(sts, up)]
    Kfix = int(round(4 * (1 + u)))
    M[f"fixed K{Kfix} (equal budget)"] = [s.ev(*s.two(*s.hard("K8", range(Kfix)))) for s in sts]
    rnd = f3_rnd(sts, up, seeds)
    M["random upgrade (equal budget, 50 seeds)"] = rnd
    tab = table(sts, M, base="K8_first4")
    # subset averages and flip rates
    subs = {1: [[k] for k in range(8)], 2: [[0, 1], [2, 3], [4, 5], [6, 7]], 4: [[0, 1, 2, 3], [4, 5, 6, 7]]}
    sub = {}
    for K, ss in subs.items():
        reg, geo, flip = [], [], []
        for s in sts:
            E = [s.ev(*s.two(*s.hard("K8", x))) for x in ss]
            rr = [e["off_regret"] for e in E if e.get("off_regret") is not None]
            if rr:
                reg.append(np.mean(rr))
            geo.append(np.mean([e["geo2_regret"] for e in E]))
            pk = [e["pick"] for e in E]
            flip.append(np.mean([pk[a] != pk[b] for a in range(len(pk)) for b in range(a + 1, len(pk))]))
        sub[f"K{K}"] = {"off_regret_subset_mean": float(np.mean(reg)), "geo2_regret_subset_mean": float(np.mean(geo)), "flip_rate": float(np.mean(flip))}
    # variance decomposition by phase (K = 8 per-sample two-step path values)
    var = {}
    for ph in ("all",) + tuple(p for p, _, _ in PHASES):
        w_, b_, z4, z8 = [], [], [], []
        for s, f in zip(sts, fl):
            if ph != "all" and s.phase != ph:
                continue
            V1k, V12k = s.H["K8"]
            sc, cp = s.two(V1k.mean(1), V12k.mean(1))
            X = np.stack([V12k[cp[i]] if cp[i] >= 0 else V1k[i] for i in range(s.n)])
            if s.n >= 2:
                w_.append(X.var(1, ddof=1).mean())
                b_.append(X.mean(1).var(ddof=1))
                o = np.argsort(-X.mean(1), kind="stable")
                se = np.sqrt(X[o[0]].var(ddof=1) / 8 + X[o[1]].var(ddof=1) / 8)
                z8.append((X[o[0]].mean() - X[o[1]].mean()) / se if se > 0 else np.inf)
            if f[1] is not None:
                z4.append(f[1])
        var[ph] = {"within_sample_var_mean": float(np.mean(w_)), "between_camera_var_mean": float(np.mean(b_)),
                   "ratio_within_over_between": float(np.mean(w_) / np.mean(b_)), "top2_z_median_K4": float(np.median(z4)), "top2_z_median_K8": float(np.median(z8)),
                   "top2_z_lt1_share_K4": float(np.mean(np.asarray(z4) < 1.0))}
    ad = {"upgrade_share": u, "budget_samples_per_state": 4 * (1 + u), "fixed_equal_budget_K": Kfix,
          "upgrade_share_by_phase": {ph: float(np.mean([f for s, f in zip(sts, up) if s.phase == ph])) for ph, _, _ in PHASES}}
    return tab, sub, var, ad, up


# ------------------------------------------------------------------------------------------------ F4
def f4_interaction(sts, var):
    """[Gain(Soft,K4) - Gain(Soft,K1)] - [Gain(Hard,K4) - Gain(Hard,K1)], Gain = -regret; K1 = mean over the 8 single samples,
    K4 = mean over first4 and quadB ('quads') or first4 alone"""
    out = {}
    for m in ("off_regret", "geo2_regret"):
        rows = {"quads": [], "first4": []}
        for s in sts:
            def reg(V1, V12):
                return s.ev(*s.two(V1, V12)).get(m)
            h1 = [reg(*s.hard("K8", [k])) for k in range(8)]
            s1 = [reg(*s.soft("K8", f"single{k}", var)) for k in range(8)]
            h4 = [reg(*s.hard("K8", range(4))), reg(*s.hard("K8", range(4, 8)))]
            s4 = [reg(*s.soft("K8", "first4", var)), reg(*s.soft("K8", "quadB", var))]
            if any(x is None for x in h1 + s1 + h4 + s4):
                continue
            H1, S1 = np.mean(h1), np.mean(s1)
            rows["quads"].append(((S1 - np.mean(s4)) - (H1 - np.mean(h4)), s.scene, S1 - np.mean(s4), H1 - np.mean(h4), s.tag))
            rows["first4"].append(((S1 - s4[0]) - (H1 - h4[0]), s.scene, S1 - s4[0], H1 - h4[0], s.tag))
        for k, rr in rows.items():
            if not rr:
                continue
            d, sc, gs, gh, tg = map(np.asarray, zip(*rr))
            out[f"{m}|{k}"] = {"interaction": boot(d, sc), "interaction_traj": boot(d, tg), "gain_K4_vs_K1_soft": boot(gs, sc), "gain_K4_vs_K1_hard": boot(gh, sc)}
    return out


def combos(sts, g_star, var, up):
    """single families and their combinations (scored on the planner's K = 4 samples; adaptive = K8 first 4 / first 8 by the frozen rule)"""
    def soft_k(s, f):
        return s.soft("K8", "first8" if f else "first4", var) if f is not None else s.soft("K4", "all4", var)
    M = {"D-v2": [s.ev(*s.two(*s.hard("K4"))) for s in sts],
         "F1 gamma*": [s.ev(*f1(s, *s.hard("K4"), g_star)) for s in sts],
         "F2 soft*": [s.ev(*s.two(*s.soft("K4", "all4", var))) for s in sts],
         "F3 adaptive": [s.ev(*s.two(*s.hard("K8", range(8 if f else 4)))) for s, f in zip(sts, up)],
         "F1+F2": [s.ev(*f1(s, *s.soft("K4", "all4", var), g_star)) for s in sts],
         "F1+F3": [s.ev(*f1(s, *s.hard("K8", range(8 if f else 4)), g_star)) for s, f in zip(sts, up)],
         "F2+F3": [s.ev(*s.two(*soft_k(s, f))) for s, f in zip(sts, up)],
         "F1+F2+F3": [s.ev(*f1(s, *soft_k(s, f), g_star)) for s, f in zip(sts, up)]}
    return M


# ------------------------------------------------------------------------------------------------ main
def dump(path, obj):
    os.makedirs(os.path.dirname(path), exist_ok=True)
    json.dump(obj, open(path, "w"), indent=1, default=float)


def states_csv(path, sts, methods):
    with open(path, "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["scene", "start", "t", "tag", "phase", "n_legal", "method", "pick", "off_regret", "off_top1", "geo2_regret", "path_union"])
        for nm, E in methods.items():
            for s, e in zip(sts, E):
                w.writerow([s.scene, s.start, s.t, s.tag, s.phase, s.n, nm, e["pick"], e.get("off_regret"), e.get("off_top1"), e.get("geo2_regret"), e.get("path_union")])


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--stage", required=True, choices=("explore", "confirm"))
    ap.add_argument("--res", default=None, help="smoke / partial runs only: write elsewhere; selection.json is never written then")
    a = ap.parse_args()
    global RES
    official = a.res is None
    RES = a.res or RES
    sts = load(a.stage)
    L = a.stage
    print(f"{L}: {len(sts)} states, {len(set(s.tag for s in sts))} trajectories, scenes {sorted(set(s.scene for s in sts))}")
    F0 = f0(sts)
    dump(os.path.join(RES, "F0_regression", f"F0_{L}.json"), F0)
    print("F0", {k: v for k, v in F0.items() if not isinstance(v, dict)})
    sel_path = os.path.join(RES, "selection.json")
    if L == "explore":
        t1, mech, cases = f1_all(sts)
        g_star = min([g for g in GAMMAS if g < 1.0], key=lambda g: (round(t1[f"gamma_{g:g}"]["off_regret"], 12), -g))
        cases += f1_cases(sts, g_star)
        dump(os.path.join(RES, "F1_horizon_discount", "F1_explore.json"), {"table": t1, "mechanism": mech, "cases": cases, "gamma_star": g_star})
        t2, t2one, dif, cost = f2_all(sts)
        hard_geo = t2["hard (D-v2)"]["geo2_regret"]
        ok = [v for v in SOFT if t2[v]["geo2_regret"] <= hard_geo + 0.02]
        v_star = min(ok, key=lambda v: t2[v]["off_regret"]) if ok else None
        t2u = f2_unions(sts, v_star) if v_star else None
        dump(os.path.join(RES, "F2_soft_coverage", "F2_explore.json"), {"table": t2, "one_step": t2one, "diffusion": dif, "cost": cost, "unions": t2u, "variant_star": v_star,
                                                                      "eligible": ok})
        t3, sub, var, ad, up = f3_all(sts)
        dump(os.path.join(RES, "F3_adaptive_K", "F3_explore.json"), {"table": t3, "subsets": sub, "variance": var, "adaptive": ad})
        sel = {"frozen_from": "exploration layer (395 states), rules of configs/F_config.yaml selection_on_exploration", "F1_gamma_star": g_star,
               "F2_variant_star": v_star, "F3": "adaptive 4->8 (rule fixed in configs)",
               "exploration_point_estimates": {"F1": t1[f"gamma_{g_star:g}"].get("d_off_regret", {}).get("scene"),
                                               "F2": t2[v_star].get("d_off_regret", {}).get("scene") if v_star else None,
                                               "F3_vs_K4": t3["adaptive 4->8"].get("d_off_regret", {}).get("scene")}}
        if not official or len(sts) != 395:
            print("partial / non-official run: selection.json NOT written", sel)
        elif os.path.exists(sel_path):
            old = json.load(open(sel_path))
            assert old["F1_gamma_star"] == g_star and old["F2_variant_star"] == v_star, "selection.json differs from a recomputation -- investigate, never overwrite"
            print("selection.json exists and matches")
        else:
            dump(sel_path, sel)
            print("selection.json written", sel)
        if v_star:
            dump(os.path.join(RES, "F4_interaction", "F4_explore.json"), {"interaction": f4_interaction(sts, v_star),
                                                                           "combos": table(sts, combos(sts, g_star, v_star, up), base="D-v2")})
        states_csv(os.path.join(RES, "F0_regression", "states_explore.csv"), sts,
                   {"D-v2": [s.ev(*s.two(*s.hard("K4"))) for s in sts], f"F1 gamma {g_star:g}": [s.ev(*f1(s, *s.hard("K4"), g_star)) for s in sts],
                    **({f"F2 {v_star}": [s.ev(*s.two(*s.soft("K4", "all4", v_star))) for s in sts]} if v_star else {})})
        for nm in ("gamma_0", "gamma_0.5", "gamma_1"):
            print(nm, {k: round(t1[nm][k], 4) for k in ("off_regret", "off_top1", "geo2_regret", "path_union")})
        print("gamma*", g_star, "F2 variant*", v_star, "eligible", ok)
        return
    # ---------------- confirmation
    sel_path = os.environ.get("F_SELECTION_TEST") or os.path.join(FD, "results", "selection.json")
    assert os.path.exists(sel_path), "selection.json missing: run --stage explore first"
    sel = json.load(open(sel_path))
    g_star, v_star = sel["F1_gamma_star"], sel["F2_variant_star"]
    t1, mech, _ = f1_all(sts)
    cases = f1_cases(sts, g_star)
    t2, t2one, dif, cost = f2_all(sts)
    t2u = f2_unions(sts, v_star) if v_star else None
    t3, sub, var, ad, up = f3_all(sts)
    base = {"D-v2": [s.ev(*s.two(*s.hard("K4"))) for s in sts]}
    fams = {"F1 gamma*": [s.ev(*f1(s, *s.hard("K4"), g_star)) for s in sts]}
    if v_star:
        fams["F2 soft*"] = [s.ev(*s.two(*s.soft("K4", "all4", v_star))) for s in sts]
    fams["F3 adaptive"] = t3_ad = [s.ev(*s.two(*s.hard("K8", range(8 if f else 4)))) for s, f in zip(sts, up)]
    tg = table(sts, {**base, **fams}, base="D-v2")
    d_ad_rnd = t3["adaptive 4->8"].get("d_off_regret", {}).get("scene")   # adaptive - K8 first4
    # adaptive vs random equal budget (paired per state)
    tr = table(sts, {"random": [{**e, "pick": -1} for e in f3_rnd(sts, up)], "adaptive": t3_ad}, base="random")
    gates = {"F1 gamma*": gate(tg["F1 gamma*"]), "F3 adaptive": gate(tg["F3 adaptive"], extra=tr["adaptive"].get("d_off_regret", {}).get("scene"))}
    if v_star:
        gates["F2 soft*"] = gate(tg["F2 soft*"])
    passed = [k for k, v in gates.items() if v["verdict"] == "PASS"]
    out = {"selection": sel, "gate_table": tg, "gates": gates, "passed": passed, "adaptive_vs_random": tr, "adaptive_vs_K8first4": d_ad_rnd}
    if v_star:
        out["F4_interaction"] = f4_interaction(sts, v_star)
        M = combos(sts, g_star, v_star, up)
        keep = ["D-v2"] + [k for k in M if k != "D-v2" and all(any(p.startswith(f) for p in passed) for f in k.split(" ")[0].split("+") if f.startswith("F"))]
        out["combos_passed_only"] = table(sts, {k: M[k] for k in keep}, base="D-v2") if len(keep) > 1 else None
        out["combos_all_diagnostic"] = table(sts, M, base="D-v2")
    out["F5_entry"] = bool(passed)
    dump(os.path.join(RES, "F4_interaction", "F_confirm_gates.json"), out)
    dump(os.path.join(RES, "F1_horizon_discount", "F1_confirm.json"), {"table": t1, "mechanism": mech, "cases": cases})
    dump(os.path.join(RES, "F2_soft_coverage", "F2_confirm.json"), {"table": t2, "one_step": t2one, "diffusion": dif, "cost": cost, "unions": t2u})
    dump(os.path.join(RES, "F3_adaptive_K", "F3_confirm.json"), {"table": t3, "subsets": sub, "variance": var, "adaptive": ad})
    states_csv(os.path.join(RES, "F0_regression", "states_confirm.csv"), sts, {**base, **fams})
    for k, v in gates.items():
        print(k, v["verdict"], v.get("off"), v.get("geo2"))
    print("F5 entry", out["F5_entry"])


def f3_rnd(sts, up, seeds=50):
    nup = int(up.sum())
    rng = np.random.default_rng(0)
    E4 = [s.ev(*s.two(*s.hard("K8", range(4)))) for s in sts]
    E8 = [s.ev(*s.two(*s.hard("K8", range(8)))) for s in sts]
    acc = [dict() for _ in sts]
    for _ in range(seeds):
        sel = np.zeros(len(sts), bool)
        sel[rng.choice(len(sts), nup, replace=False)] = True
        for i in range(len(sts)):
            e = E8[i] if sel[i] else E4[i]
            for m in ("off_regret", "geo2_regret", "path_union", "off_top1"):
                if e.get(m) is not None:
                    acc[i].setdefault(m, []).append(float(e[m]))
    return [{"pick": -1, **{m: float(np.mean(v)) for m, v in a.items()}} for a in acc]


if __name__ == "__main__":
    main()
