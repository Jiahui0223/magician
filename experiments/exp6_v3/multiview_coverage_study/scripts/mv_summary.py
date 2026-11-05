"""Multi-View Coverage Reasoning Study -- aggregation (CPU, read-only) of work_dirs/.../multiview_coverage_study/states/*.jsonl
-> results/summary.json, results/*.csv and printed tables. Groups: all states, t < 20, t >= 20, E3 / G2."""
import csv
import glob
import json
import os

import numpy as np
from scipy.stats import spearmanr

HERE = os.path.dirname(os.path.abspath(__file__))
MV = os.path.dirname(HERE)
R = os.path.dirname(os.path.dirname(os.path.dirname(MV)))
ST = os.path.join(R, "work_dirs", "exp6_v3", "multiview_coverage_study", "states")
RES = os.path.join(MV, "results")


def load():
    rows = []
    for f in sorted(glob.glob(os.path.join(ST, "*.jsonl"))):
        rows += [json.loads(l) for l in open(f)]
    return rows


def sp(a, b):
    a, b = np.asarray(a, float), np.asarray(b, float)
    ok = np.isfinite(a) & np.isfinite(b)
    if ok.sum() < 3 or np.ptp(a[ok]) == 0 or np.ptp(b[ok]) == 0:
        return None
    return float(spearmanr(a[ok], b[ok]).correlation)


def m(x):
    x = [v for v in x if v is not None and np.isfinite(v)]
    return float(np.mean(x)) if x else None


def groups(rows):
    return {"all": rows, "t<20": [r for r in rows if r["t"] < 20], "t>=20": [r for r in rows if r["t"] >= 20],
            "E3": [r for r in rows if r["set"] == "e3"], "G2": [r for r in rows if r["set"] == "g2"]}


def rank_table(rows, names, oracles=("official", "geo2")):
    out = {}
    for g, rr in groups(rows).items():
        out[g] = {"states": len(rr)}
        for n in names:
            for o in oracles:
                x = [r["rank"][n][o] for r in rr if n in r["rank"]]
                if x and isinstance(x[0], list):              # subset averages
                    x = [{"top1": m([y["top1"] for y in xs]), "regret": m([y["regret"] for y in xs]), "gain": m([y["gain"] for y in xs]), "rho": m([y["rho"] for y in xs])} for xs in x]
                out[g][f"{n}|{o}"] = {"top1": m([y["top1"] for y in x]), "regret": m([y["regret"] for y in x]), "gain": m([y["gain"] for y in x]), "rho": m([y["rho"] for y in x])}
    return out


def gt_depth_rank(rows):
    """The same value function on TRUE depth (a 'perfect prediction' planner), one- and two-step, against the oracles."""
    import importlib.util  # noqa: F401
    out = {}
    for g, rr in groups(rows).items():
        o = {}
        for sname, key in (("gtvalue_one_step", "gtvalue_one_step"), ("gtvalue_two_step", "gtvalue_two_step"), ("pred_one_step_K4first", None)):
            for orc in ("official", "geo2"):
                t1, rg, rh = [], [], []
                for r in rr:
                    s = np.asarray(r[key] if key else r["K_study"]["scores"]["one_step_first4"], float)
                    g_ = np.asarray(r["oracle_one_step_pp"] if orc == "official" else r["geo_two_step_pp"], float)
                    ok = np.isfinite(g_)
                    if ok.sum() < 2:
                        continue
                    s_ = np.where(ok, s, -np.inf)
                    pk = int(np.argmax(s_))
                    t1.append(float(g_[pk] >= np.nanmax(g_) - 1e-4))
                    rg.append(float(np.nanmax(g_) - g_[pk]))
                    rh.append(sp(s[ok], g_[ok]))
                o[f"{sname}|{orc}"] = {"top1": m(t1), "regret": m(rg), "rho": m(rh)}
        out[g] = o
    return out


def task_a(rows):
    P = [(r, p) for r in rows for p in r["pairs"]]
    res = {}
    for kind in ("first", "path"):
        pp = [(r, p) for r, p in P if p["kind"] == kind]
        pred_ov, true_ov, gtv_ov, true_jac, fail = [], [], [], [], []
        dup_w = marg_w = 0.0
        for r, p in pp:
            pr, gv, cv = p["pred"], p["gtv"], p["cov"]
            mn = min(pr["VA"], pr["VB"])
            pred_ov.append(pr["I"] / mn if mn > 1e-9 else np.nan)
            mt = min(gv["VA"], gv["VB"])
            gtv_ov.append(gv["I"] / mt if mt > 1e-9 else np.nan)
            mc = min(cv["nA"], cv["nB"])
            true_ov.append(cv["nI"] / mc if mc > 0 else np.nan)
            true_jac.append(cv["nI"] / cv["nU"] if cv["nU"] > 0 else np.nan)
            if mc >= 20 and mn > 1e-9 and pr["I"] / mn < 0.2 and cv["nI"] / mc > 0.5:
                fail.append({"set": r["set"], "tag": r["tag"], "t": r["t"], "a": p["a"], "b": p["b"], "pred_overlap": pr["I"] / mn, "true_overlap": cv["nI"] / mc,
                             "pred_U": pr["U"], "true_U_pp": cv["pp_U"], "nA": cv["nA"], "nB": cv["nB"]})
            dup_w += p["marg_B_on_true_duplicate"]
            marg_w += p["marg_B_pred"]
        po, to, go = np.asarray(pred_ov), np.asarray(true_ov), np.asarray(gtv_ov)
        ok = np.isfinite(po) & np.isfinite(to)
        both = [(r, p) for r, p in pp if min(p["cov"]["nA"], p["cov"]["nB"]) >= 20]
        res[kind] = {"pairs": len(pp), "pairs_both_cover_20pts": len(both),
                     "true_overlap_median": float(np.nanmedian(to)) if np.isfinite(to).any() else None,
                     "true_jaccard_median": float(np.nanmedian(true_jac)) if np.isfinite(np.asarray(true_jac)).any() else None,
                     "pred_overlap_median": float(np.nanmedian(po)) if np.isfinite(po).any() else None,
                     "spearman_pred_vs_true_overlap": sp(po[ok], to[ok]), "spearman_gtvalue_vs_true_overlap": sp(go, to),
                     "pred_minus_true_overlap_mean": float(np.mean(po[ok] - to[ok])) if ok.any() else None,
                     "failure_cases": len(fail), "failure_share_of_pairs_with_20pts": len(fail) / max(1, len(both)), "failure_examples": fail[:12],
                     "double_count_share_of_pred_marginal": dup_w / max(marg_w, 1e-9)}
        if kind == "path":
            rho_within, rho_union = [], []
            by = {}
            for r, p in pp:
                by.setdefault((r["set"], r["tag"], r["t"], tuple(p["a"])), []).append(p)
            for k, ps in by.items():
                if len(ps) >= 3:
                    rho_within.append(sp([q["pred"]["U"] - q["pred"]["VA"] for q in ps], [q["cov"]["pp_marg_B"] for q in ps]))
            bys = {}
            for r, p in pp:
                bys.setdefault((r["set"], r["tag"], r["t"]), []).append(p)
            for k, ps in bys.items():
                rho_union.append(sp([q["pred"]["U"] for q in ps], [q["cov"]["pp_U"] for q in ps]))
            res[kind].update({"spearman_c2_marginal_within_c1_mean": m(rho_within), "n_c1_with_3plus_children": len(rho_within),
                              "spearman_path_union_within_state_mean": m(rho_union)})
    # Q1: how different are the candidates' true new coverages
    spread, uniq = [], []
    for r in rows:
        g = np.asarray(r["geo_one_step_pp"], float)
        if len(g) > 1:
            spread.append(float(g.max() - g.min()))
    res["candidate_true_gain_spread_pp_median"] = float(np.median(spread)) if spread else None
    res["candidate_true_gain_spread_pp_mean"] = float(np.mean(spread)) if spread else None
    return res


def task_b(rows):
    out = {}
    for g, rr in groups(rows).items():
        ks = [r["K_study"] for r in rr]
        out[g] = {"within_cam_jaccard_mean": m([m(k["within_cam_jaccard"]) for k in ks]),
                  "between_cam_jaccard_same_k": m([k["between_cam_jaccard_same_k"] for k in ks]),
                  "between_cam_jaccard_diff_k": m([k["between_cam_jaccard_diff_k"] for k in ks]),
                  "value_var_within_over_between": m([k["value_var_within"] / k["value_var_between"] for k in ks if k["value_var_between"]]),
                  "K_eff_8": m([k["K_eff_8"] for k in ks]),
                  "hit_disagreement": m([m([d["hit_disagreement"] for d in k["depth"]]) for k in ks]),
                  "rel_depth_sd_median": m([m([d["rel_depth_sd_median"] for d in k["depth"]]) for k in ks]),
                  "seconds_K4": m([r["seconds"]["K4"] for r in rr]), "seconds_K8": m([r["seconds"]["K8"] for r in rr]),
                  "K8_first4_vs_K4_rel_max": max(r["regression"]["K8_first4_vs_K4"] for r in rr) if rr else None,
                  "K8_first4_vs_K4_pick_changed": m([float(np.argmax(r["K_study"]["scores"]["two_step_first4"]) != np.argmax(r["planner_scores_K4"])) for r in rr])}
    # how often does the pick depend on which samples are drawn (noise sensitivity of the decision)
    flip = {}
    for K_ in (1, 2, 4):
        x = []
        for r in rows:
            picks = [int(np.argmax(s)) for s in r["K_study"]["scores"][f"two_step_subsets{K_}"]]
            x.append(float(len(set(picks)) > 1))
        flip[f"two_step_K{K_}_subsets_disagree"] = m(x)
    out["decision_noise_sensitivity"] = flip
    return out


def task_c(rows):
    def agg(ps, lv):
        a = sum(p[f"cons_{d}"][f"{lv}_agree"] for p in ps for d in ("AB", "BA"))
        t = sum(p[f"cons_{d}"][f"{lv}_through"] for p in ps for d in ("AB", "BA"))
        f = sum(p[f"cons_{d}"][f"{lv}_front"] for p in ps for d in ("AB", "BA"))
        return {"agree": a, "through": t, "front": f, "inconsistency": t / max(a + t, 1e-9)}
    out = {}
    for g, rr in groups(rows).items():
        o = {}
        for kind in ("first", "path"):
            ps = [p for r in rr for p in r["pairs"] if p["kind"] == kind]
            o[kind] = {lv: agg(ps, lv) for lv in ("paired", "unpaired", "dist", "dist_novel")}
            th = {k: sum(p[f"cons_{d}"][k] for p in ps for d in ("AB", "BA")) for k in ("gt_through_Awrong", "gt_through_Bwrong", "gt_through_bothok", "gt_agree_Aok", "gt_agree_Awrong")}
            tt = th["gt_through_Awrong"] + th["gt_through_Bwrong"] + th["gt_through_bothok"]
            ta = th["gt_agree_Aok"] + th["gt_agree_Awrong"]
            o[kind]["gt_attribution_of_conflicts"] = {"A_point_is_phantom": th["gt_through_Awrong"] / max(tt, 1e-9), "B_misses_a_true_surface": th["gt_through_Bwrong"] / max(tt, 1e-9),
                                                       "both_right_geometrically": th["gt_through_bothok"] / max(tt, 1e-9)}
            o[kind]["gt_of_agreements"] = {"agreed_point_is_true_surface": th["gt_agree_Aok"] / max(ta, 1e-9)}
            vg = sum(p[f"cons_{d}"]["valid"] for p in ps for d in ("AB", "BA"))
            gg = sum(p[f"cons_{d}"]["gen"] for p in ps for d in ("AB", "BA"))
            o[kind]["covisible_share_of_generated"] = vg / max(gg, 1e-9)
        out[g] = o
    # correlations
    def inc_pair(p, lv="dist"):
        a = p["cons_AB"][f"{lv}_agree"] + p["cons_BA"][f"{lv}_agree"]
        t = p["cons_AB"][f"{lv}_through"] + p["cons_BA"][f"{lv}_through"]
        return t / (a + t) if a + t > 1.0 else np.nan
    cor = {}
    # (1) per camera: mean inconsistency with the other first-layer cameras vs gain error
    xi, ye, ye2, within = [], [], [], []
    for r in rows:
        keys = [tuple(k) for k in r["keys1"]]
        pred = np.asarray([np.mean(v[:4]) for v in r["pred_one_step_k8"]])
        gtv = np.asarray(r["gtvalue_one_step"])
        orc = np.asarray(r["oracle_one_step_pp"] if r["oracle_one_step_pp"] else [np.nan] * len(keys), float)
        inc = []
        for kk in keys:
            ps = [p for p in r["pairs"] if p["kind"] == "first" and (tuple(p["a"]) == kk or tuple(p["b"]) == kk)]
            a = sum(p["cons_AB" if tuple(p["a"]) == kk else "cons_BA"]["dist_agree"] for p in ps)
            t = sum(p["cons_AB" if tuple(p["a"]) == kk else "cons_BA"]["dist_through"] for p in ps)
            inc.append(t / (a + t) if a + t > 1.0 else np.nan)
        err = np.abs(np.log((pred + 1) / (gtv + 1)))
        # rank error vs the official gain: |rank(pred) - rank(oracle)| / n
        if np.isfinite(orc).sum() > 2:
            rp = np.argsort(np.argsort(-pred)); ro = np.argsort(np.argsort(-np.nan_to_num(orc, nan=-1e9)))
            rerr = np.abs(rp - ro) / len(keys)
        else:
            rerr = np.full(len(keys), np.nan)
        xi += inc; ye += list(err); ye2 += list(rerr)
        within.append(sp(inc, err))
    cor["camera_inconsistency_vs_abs_log_gain_error"] = {"pooled": sp(xi, ye), "within_state_mean": m(within), "n": int(np.isfinite(np.asarray(xi)).sum())}
    cor["camera_inconsistency_vs_rank_error_official"] = {"pooled": sp(xi, ye2)}
    # (2) per pair: inconsistency vs union error and vs double counting
    for kind in ("first", "path"):
        xs, ue, dshare = [], [], []
        for r in rows:
            for p in r["pairs"]:
                if p["kind"] != kind:
                    continue
                xs.append(inc_pair(p))
                ue.append(abs(np.log((p["pred"]["U"] + 1) / (p["gtv"]["U"] + 1))))
                dshare.append(p["marg_B_on_true_duplicate"] / p["marg_B_pred"] if p["marg_B_pred"] > 1e-9 else np.nan)
        cor[f"{kind}_pair_inconsistency_vs_abs_log_union_error"] = sp(xs, ue)
        cor[f"{kind}_pair_inconsistency_vs_double_count_share"] = sp(xs, dshare)
    # (3) per state: mean inconsistency vs regret of the D-v2 decision
    si, sr, sr2 = [], [], []
    for r in rows:
        v = [inc_pair(p) for p in r["pairs"] if p["kind"] == "first"]
        si.append(np.nanmean(v) if np.isfinite(np.asarray(v, float)).any() else np.nan)
        sr.append(r["rank"]["dv2"]["official"].get("regret", np.nan) if r["rank"]["dv2"]["official"].get("regret") is not None else np.nan)
        sr2.append(r["rank"]["dv2"]["geo2"].get("regret", np.nan) if r["rank"]["dv2"]["geo2"].get("regret") is not None else np.nan)
    cor["state_inconsistency_vs_dv2_regret_official"] = sp(si, sr)
    cor["state_inconsistency_vs_dv2_regret_geo2"] = sp(si, sr2)
    wrong = [i for i, r in enumerate(rows) if r["rank"]["dv2"]["official"].get("top1") is False]
    right = [i for i, r in enumerate(rows) if r["rank"]["dv2"]["official"].get("top1") is True]
    cor["state_inconsistency_wrong_vs_right"] = {"wrong": m([si[i] for i in wrong]), "right": m([si[i] for i in right]), "n_wrong": len(wrong), "n_right": len(right)}
    out["correlations"] = cor
    return out


def attribution(rows):
    """Wrong D-v2 decisions (official Top-1 false): which factor would have fixed them."""
    w = [r for r in rows if r["rank"]["dv2"]["official"].get("top1") is False]
    o = {"wrong_states": len(w), "states": len(rows)}
    def fixed(r, name, orc="official"):
        x = r["rank"].get(name, {}).get(orc)
        return x.get("top1") if x else None
    o["fixed_by_perfect_depth_one_step"] = m([float(np.nanargmax(np.where(np.isfinite(np.asarray(r["oracle_one_step_pp"], float)), r["gtvalue_one_step"], -np.inf)) ==
                                               np.nanargmax(np.asarray(r["oracle_one_step_pp"], float))) for r in w])
    o["fixed_by_perfect_depth_two_step"] = m([float(np.nanargmax(np.where(np.isfinite(np.asarray(r["oracle_one_step_pp"], float)), r["gtvalue_two_step"], -np.inf)) ==
                                               np.nanargmax(np.asarray(r["oracle_one_step_pp"], float))) for r in w])
    o["fixed_by_any_other_K4_or_K1_subset"] = m([float(any(x.get("top1") for x in r["rank"]["two_step_subsets1"]["official"] + r["rank"]["two_step_subsets4"]["official"] if x.get("top1") is not None)) for r in w])
    o["fixed_by_K8"] = m([fixed(r, "two_step_first8") for r in w])
    o["fixed_by_unpaired"] = m([fixed(r, "unpaired") for r in w])
    o["fixed_by_cvar"] = m([fixed(r, "cvar") for r in w])
    o["fixed_by_one_step"] = m([fixed(r, "one_step") for r in w])
    o["regret_of_wrong_official_mean"] = m([r["rank"]["dv2"]["official"]["regret"] for r in w])
    o["small_regret_share_lt_0.1pp"] = m([float(r["rank"]["dv2"]["official"]["regret"] < 0.1) for r in w])
    return o


def cluster_boot(rows, a, b, orc, met, n=10000, seed=0):
    """Paired difference a - b of a ranking metric, mean over states, 95 % CI by resampling trajectories (clusters)."""
    by = {}
    for r in rows:
        x, y = r["rank"].get(a, {}).get(orc), r["rank"].get(b, {}).get(orc)
        if not x or not y or x.get(met) is None or y.get(met) is None:
            continue
        by.setdefault((r["set"], r["tag"]), []).append(float(x[met]) - float(y[met]))
    keys = list(by)
    if len(keys) < 2:
        return None
    tot = np.asarray([sum(by[k]) for k in keys]); cnt = np.asarray([len(by[k]) for k in keys])
    rng = np.random.default_rng(seed)
    idx = rng.integers(0, len(keys), (n, len(keys)))
    bs = tot[idx].sum(1) / cnt[idx].sum(1)
    return {"d": float(tot.sum() / cnt.sum()), "ci": [float(np.quantile(bs, 0.025)), float(np.quantile(bs, 0.975))], "states": int(cnt.sum()), "clusters": len(keys),
            "changed_states": int(sum(1 for r in rows if r["rank"].get(a, {}).get(orc, {}).get("pick") != r["rank"].get(b, {}).get(orc, {}).get("pick")))}


def value_diag():
    import glob as _g
    rows = []
    for f in sorted(_g.glob(os.path.join(R, "work_dirs", "exp6_v3", "multiview_coverage_study", "value_diag", "*.jsonl"))):
        rows += [json.loads(l) for l in open(f)]
    if not rows:
        return None
    out = {"states": len(rows)}
    for g, rr in {"all": rows, "t<20": [r for r in rows if r["t"] < 20], "t>=20": [r for r in rows if r["t"] >= 20]}.items():
        o = {}
        for k in ("pred_w", "pred_u", "pred_w_d1", "pred_occ", "true_w", "true_u", "true_occ", "true_area", "true_gt"):
            t1, rg, rh = [], [], []
            for r in rr:
                gg = np.asarray([np.nan if x is None else x for x in r["oracle"]], float)
                ok = np.isfinite(gg)
                if ok.sum() < 2:
                    continue
                if k not in r:
                    continue
                s_ = np.where(ok, np.asarray(r[k], float), -np.inf)
                pk = int(np.argmax(s_))
                t1.append(float(gg[pk] >= np.nanmax(gg) - 1e-4)); rg.append(float(np.nanmax(gg) - gg[pk])); rh.append(sp(np.asarray(r[k], float)[ok], gg[ok]))
            o[k] = {"top1": m(t1), "regret": m(rg), "rho": m(rh), "n": len(t1)}
        out[g] = o
    return out


def main():
    rows = load()
    os.makedirs(RES, exist_ok=True)
    reg = {"states": len(rows), "recorded_vs_planner_K4_max": max(r["regression"]["recorded_vs_planner_K4"] for r in rows),
           "recorded_vs_planner_K8_max": max(r["regression"]["recorded_vs_planner_K8"] for r in rows),
           "K8_first4_vs_K4_median": float(np.median([r["regression"]["K8_first4_vs_K4"] for r in rows])),
           "projection_self_test_min": min(r["regression"]["projection_self_test"] for r in rows),
           "views_per_state_mean": m([r["n_views"] for r in rows]), "legal_per_state_mean": m([r["n_legal"] for r in rows]),
           "peak_mem_gb_max": max(r.get("peak_mem_gb", 0) for r in rows), "seconds_total_mean": m([r["seconds_total"] for r in rows]),
           "by_set": {s: sum(1 for r in rows if r["set"] == s) for s in ("e3", "g2")}}
    # truth alignment
    ta = {}
    for g, rr in groups(rows).items():
        ta[g] = {"rho_geo1_vs_official": m([sp(r["geo_one_step_pp"], r["oracle_one_step_pp"]) for r in rr]),
                 "rho_gtvalue1_vs_official": m([sp(r["gtvalue_one_step"], r["oracle_one_step_pp"]) for r in rr]),
                 "rho_pred1_vs_official": m([sp(r["K_study"]["scores"]["one_step_first4"], r["oracle_one_step_pp"]) for r in rr]),
                 "rho_pred1_vs_gtvalue1": m([sp(r["K_study"]["scores"]["one_step_first4"], r["gtvalue_one_step"]) for r in rr]),
                 "rho_dv2_vs_geo2": m([sp(r["planner_scores_K4"], r["geo_two_step_pp"]) for r in rr])}
    names = ["dv2", "unpaired", "cvar", "cvar_occ", "cvar_free", "one_step"] + [f"two_step_first{k}" for k in (1, 2, 4, 8)] + [f"one_step_first{k}" for k in (1, 2, 4, 8)] + \
            [f"two_step_subsets{k}" for k in (1, 2, 4)] + [f"one_step_subsets{k}" for k in (1, 2, 4)]
    out = {"regression": reg, "truth_alignment": ta, "rank": rank_table(rows, names), "perfect_depth": gt_depth_rank(rows),
           "task_a": task_a(rows), "task_b": task_b(rows), "task_c": task_c(rows), "attribution": attribution(rows)}
    out["bootstrap"] = {f"{a} - {b} | {o} {mt}": cluster_boot(rows, a, b, o, mt) for a, b in (("unpaired", "dv2"), ("cvar", "dv2"), ("cvar_free", "dv2"), ("cvar_occ", "dv2"),
                                                                                         ("two_step_first8", "dv2"), ("two_step_first1", "dv2"), ("one_step", "dv2"))
                        for o in ("official", "geo2") for mt in ("regret", "top1")}
    out["value_diag"] = value_diag()
    # D gate (configs/mv_v1.yaml)
    A, E = out["rank"]["all"], out["rank"]["t<20"]
    gate = {"official_regret_down": A["cvar|official"]["regret"] < A["dv2|official"]["regret"], "geo2_regret_down": A["cvar|geo2"]["regret"] < A["dv2|geo2"]["regret"],
            "official_top1_not_down": A["cvar|official"]["top1"] >= A["dv2|official"]["top1"],
            "early_official_regret_not_up_0.02": E["cvar|official"]["regret"] <= E["dv2|official"]["regret"] + 0.02}
    out["cvar_gate"] = {**gate, "pass": all(gate.values())}
    out["cvar_pick_changed_share"] = m([float(r["rank"]["cvar"]["official"]["pick"] != r["rank"]["dv2"]["official"]["pick"]) for r in rows])
    json.dump(out, open(os.path.join(RES, "summary.json"), "w"), indent=1, default=float)
    with open(os.path.join(RES, "states.csv"), "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["set", "tag", "scene", "start", "t", "n_legal", "n_views", "dv2_pick", "dv2_top1_official", "dv2_regret_official", "dv2_regret_geo2", "cvar_pick", "cvar_regret_official",
                    "cvar_regret_geo2", "K8_regret_official", "K1_regret_official", "seconds_K4", "seconds_K8"])
        for r in rows:
            k = r["rank"]
            w.writerow([r["set"], r["tag"], r["scene"], r["start"], r["t"], r["n_legal"], r["n_views"], k["dv2"]["official"]["pick"], k["dv2"]["official"].get("top1"),
                        k["dv2"]["official"].get("regret"), k["dv2"]["geo2"].get("regret"), k["cvar"]["official"]["pick"], k["cvar"]["official"].get("regret"), k["cvar"]["geo2"].get("regret"),
                        k["two_step_first8"]["official"].get("regret"), k["two_step_first1"]["official"].get("regret"), r["seconds"]["K4"], r["seconds"]["K8"]])
    print(json.dumps(reg, indent=0))
    print("truth alignment", json.dumps(ta, indent=0))
    print("| group | policy | Top-1 off | regret off (pp) | Top-1 geo2 | regret geo2 (pp) | rho off |")
    for g in ("all", "t<20", "t>=20", "E3", "G2"):
        for n in names:
            x, y = out["rank"][g][f"{n}|official"], out["rank"][g][f"{n}|geo2"]
            f_ = lambda v: "—" if v is None else f"{v:.3f}"  # noqa: E731
            print(f"| {g} ({out['rank'][g]['states']}) | {n} | {f_(x['top1'])} | {f_(x['regret'])} | {f_(y['top1'])} | {f_(y['regret'])} | {f_(x['rho'])} |")
    print("perfect depth", json.dumps(out["perfect_depth"]["all"], indent=0))
    print("task A", json.dumps({k: v for k, v in out["task_a"].items() if k not in ("first", "path")}, indent=0))
    for kind in ("first", "path"):
        print(kind, json.dumps({k: v for k, v in out["task_a"][kind].items() if k != "failure_examples"}, indent=0))
    print("task B", json.dumps(out["task_b"], indent=0)[:3000])
    print("task C", json.dumps(out["task_c"]["all"], indent=0)[:3000])
    print("correlations", json.dumps(out["task_c"]["correlations"], indent=0))
    print("attribution", json.dumps(out["attribution"], indent=0))
    print("cvar gate", out["cvar_gate"], "pick changed", out["cvar_pick_changed_share"])
    for k, v in out["bootstrap"].items():
        if v:
            print(f"boot {k}: {v['d']:+.4f} [{v['ci'][0]:+.4f}, {v['ci'][1]:+.4f}] changed {v['changed_states']}/{v['states']} clusters {v['clusters']}")
    print("value diag", json.dumps(out["value_diag"], indent=0))


if __name__ == "__main__":
    main()
