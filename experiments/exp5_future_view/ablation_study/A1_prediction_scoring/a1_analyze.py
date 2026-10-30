"""EXP5 Ablation A1 -- prediction-scoring alignment: tables and per-state / per-candidate records (seen2scene env, CPU).

Scorers on the same fixed state and candidate set: V2O / V2M predicted DistP1 (K = 4), GT-depth DistP1 (GTD), plus the
random-choice expectation and each source policy's executed move. Metrics (protocol §3.3-§3.5): Spearman vs oracle
gain, tie-aware / strict Top-1, pairwise accuracy, regret (percentage points), normalised regret, prediction fidelity
Spearman(pred, GTD), prediction quality (fair ES, Brier, false / missed surface, depth relative error).
Aggregation: state -> episode -> scene; mean of scene means, bootstrap CI over scenes.
Output A1_prediction_scoring/{results,tables}/
"""
import csv
import json
import os
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.join(os.path.dirname(HERE), "common"))
import ab_common as A  # noqa: E402
import ab_load as AL  # noqa: E402
import t13_analyze as T13  # noqa: E402

import numpy as np  # noqa: E402

TAU = A.TIE_TOL * 100.0                     # percentage points
RES, TAB = os.path.join(HERE, "results"), os.path.join(HERE, "tables")
MAIN_SCORERS = ("V2O", "V2M", "GTD", "GTDn")


def rank_metrics(s, g, pk_local):
    """s, g over the candidate set; pk_local = index of the pick in the candidate set (None = hover)."""
    gm = g.max()
    out = {"spearman": T13.spearman(s, g) if s is not None else None}
    if pk_local is None:
        return out
    gp = g[pk_local]
    out.update({"top1": float(gp >= gm - TAU), "top1_strict": float(gp == gm), "regret": float(gm - gp),
                "nregret": float((gm - gp) / gm) if gm > TAU else None})
    if s is not None:
        num = den = 0.0
        for i in range(len(g)):
            for j in range(i + 1, len(g)):
                if abs(g[i] - g[j]) > TAU:
                    den += 1
                    ds = s[i] - s[j]
                    num += 0.5 if ds == 0 else float(np.sign(ds) == np.sign(g[i] - g[j]))
        out["pairwise"] = num / den if den else None
    return out


def best_set(x, tol):
    return set(np.flatnonzero(x >= x.max() - tol).tolist())


def state_rows(r, dataset):
    C = r["C"]
    loc = {int(c): i for i, c in enumerate(C)}
    g = r["gain"][C]
    rows = []
    base = {"dataset": dataset, "sid": r["sid"], "scene": r["scene"], "split": r["split"], "start": r["start"], "t": r["t"],
            "primary_t": r["primary_t"], "stale": r["stale"], "src": r["src"], "n_cand": len(C), "cov_t": r["cov_t"],
            "max_gain": float(g.max()), "t_bin": AL.t_bin(dataset, r["t"])}
    ob = best_set(g, TAU)
    picks = {}
    for m in MAIN_SCORERS:
        s = r["scores"][m][C]
        pk = r["picks"][m]
        pl = loc[pk] if pk is not None else None
        picks[m] = pl
        row = dict(base, scorer=m, fallback=r["fallback"][m], pick=pk, **rank_metrics(s, g, pl))
        if m in ("V2O", "V2M"):
            row["fidelity"] = T13.spearman(s, r["scores"]["GTD"][C])
            cm = [r["cand"][m][int(c)] for c in C]
            for k in ("fair_es", "brier", "false_surface", "missed_surface", "spread_vox", "hit_disagree_rays"):
                v = [x.get(k) for x in cm if x.get(k) is not None]
                row[k] = float(np.mean(v)) if v else None
            dr = [p["depth_rel_err"] for x in cm for p in x["per_sample"] if p["depth_rel_err"] is not None]
            row["depth_rel_err"] = float(np.mean(dr)) if dr else None
            row["regret_minus_gtd"] = None
        if r["gain_official"] is not None and pl is not None:
            go = r["gain_official"][C]
            row["regret_official"] = float(go.max() - go[pl])
            row["top1_official"] = float(go[pl] >= go.max() - TAU)
        rows.append(row)
    for row in rows:
        if row["scorer"] in ("V2O", "V2M") and "regret" in row:
            row["regret_minus_gtd"] = row["regret"] - next(x for x in rows if x["scorer"] == "GTD")["regret"]
            sb = best_set(r["scores"][row["scorer"]][C], 0.0)
            gb = best_set(r["scores"]["GTD"][C], 0.0)
            row["agree_pred_gtd"] = float(picks[row["scorer"]] in gb)
            row["agree_pred_oracle"] = float(picks[row["scorer"]] in ob)
            row["pred_best_ties"] = len(sb)
    gtd_row = next(x for x in rows if x["scorer"] == "GTD")
    gtd_row["agree_gtd_oracle"] = float(picks["GTD"] in ob)
    rows.append(dict(base, scorer="Random", top1=len(ob) / len(C), top1_strict=float(np.mean(g == g.max())),
                     regret=float(g.max() - g.mean()), nregret=float((g.max() - g.mean()) / g.max()) if g.max() > TAU else None,
                     pairwise=0.5, spearman=None))
    for e in r["sources"]:
        pk = e["executed_pick"]
        if pk is None or pk not in loc:
            continue
        rows.append(dict(base, scorer=f"Exec:{e['source']}", pick=pk, **rank_metrics(None, g, loc[pk])))
    return rows


def cand_rows(r, dataset):
    out = []
    Cs = set(r["C"].tolist())
    hist = {tuple(int(x) for x in p) for p in r["rec"]["executed_pose_idx_history"][:-1]}
    for j, idx in enumerate(r["legal_idx"]):
        row = {"dataset": dataset, "scene": r["scene"], "split": r["split"], "episode": f"{r['scene']}_s{r['start']}",
               "sources": "+".join(e["source"] for e in r["sources"]), "t": r["t"], "sid": r["sid"], "cand": j,
               "pose_idx": " ".join(map(str, idx)), "in_candidate_set": j in Cs, "revisit": tuple(idx) in hist,
               "V2O_pred": r["scores"]["V2O"][j], "V2M_pred": r["scores"]["V2M"][j], "GTD": r["scores"]["GTD"][j],
               "oracle_gain_pp": r["gain"][j]}
        if r["gain_official"] is not None:
            row["oracle_gain_official_pp"] = r["gain_official"][j]
        for m in MAIN_SCORERS:
            row[f"selected_{m}"] = r["picks"][m] == j
        g = r["gain"][r["C"]]
        row["oracle_best"] = j in Cs and r["gain"][j] >= g.max() - TAU
        row["oracle_regret_if_selected_pp"] = float(g.max() - r["gain"][j]) if j in Cs else None
        for m in ("V2O", "V2M"):
            c = r["cand"][m][j]
            for k in ("fair_es", "brier", "false_surface", "missed_surface"):
                row[f"{m}_{k}"] = c.get(k)
        out.append(row)
    return out


def agg(rows, scorer, key, group=AL.scene_of):
    sel = [x for x in rows if x["scorer"] == scorer]
    return AL.boot(AL.hier_mean(sel, key, group))


def paired(rows, key, a="V2M", b="V2O"):
    ma = AL.hier_mean([x for x in rows if x["scorer"] == a], key)
    mb = AL.hier_mean([x for x in rows if x["scorer"] == b], key)
    return AL.boot({s: ma[s] - mb[s] for s in ma if s in mb}), {s: ma[s] - mb[s] for s in ma if s in mb}


def fmt(b, nd=3):
    if b["mean"] is None:
        return "—"
    if b["ci95"][0] is None or b["scenes"] < 4:
        return f"{b['mean']:.{nd}f}"
    return f"{b['mean']:.{nd}f} [{b['ci95'][0]:.{nd}f}, {b['ci95'][1]:.{nd}f}]"


def write_csv(p, rows):
    keys = []
    for r in rows:
        for k in r:
            if k not in keys:
                keys.append(k)
    with open(p, "w", newline="") as f:
        w = csv.DictWriter(f, keys)
        w.writeheader()
        for r in rows:
            w.writerow(r)


STRATA = {"t_bin": lambda x: x["t_bin"], "stale": lambda x: "refresh" if x["stale"] == 0 else "stale5",
          "source": lambda x: x["src"], "split": lambda x: x["split"], "primary_t": lambda x: "brief_t" if x["primary_t"] else "t+5"}


def main():
    os.makedirs(RES, exist_ok=True)
    os.makedirs(TAB, exist_ok=True)
    summary = {}
    for dataset in ("indoor", "outdoor"):
        sts = AL.load_states(dataset)
        rows, crow = [], []
        for r in sts:
            rows += state_rows(r, dataset)
            crow += cand_rows(r, dataset)
        write_csv(os.path.join(RES, f"a1_states_{dataset}.csv"), rows)
        write_csv(os.path.join(RES, f"a1_candidates_{dataset}.csv"), crow)
        # bins fixed per dataset (terciles of the state-level values, computed on the GTD rows)
        g_rows = [x for x in rows if x["scorer"] == "GTD"]
        qg = np.quantile([x["max_gain"] for x in g_rows], [1 / 3, 2 / 3])
        qc = np.quantile([x["cov_t"] for x in g_rows], [1 / 3, 2 / 3])
        strata = dict(STRATA)
        strata["max_gain"] = lambda x: "low" if x["max_gain"] <= qg[0] else ("mid" if x["max_gain"] <= qg[1] else "high")
        strata["coverage"] = lambda x: "low" if x["cov_t"] <= qc[0] else ("mid" if x["cov_t"] <= qc[1] else "high")
        strata["n_cand"] = lambda x: "<=6" if x["n_cand"] <= 6 else ("7" if x["n_cand"] == 7 else ">=8")
        S = {"n_states": len(sts), "n_states_by_source": {s: sum(r["src"] == s for r in sts) for s in A.SOURCES},
             "repro_max_abs_diff": max([v for r in sts for v in r["repro"].values()] or [None]),
             "repro_states": sum(len(r["repro"]) for r in sts),
             "bins": {"max_gain_pp": qg.tolist(), "cov_t": qc.tolist()},
             "spearman_undefined": {m: sum(1 for x in rows if x["scorer"] == m and x.get("spearman") is None) for m in MAIN_SCORERS},
             "fallback": {m: sum(1 for x in rows if x["scorer"] == m and x.get("fallback")) for m in MAIN_SCORERS},
             "scorers": {}, "paired_V2M_minus_V2O": {}, "per_scene": {}, "strata": {}}
        keys = ("spearman", "top1", "top1_strict", "pairwise", "regret", "nregret", "fidelity", "regret_minus_gtd",
                "agree_pred_gtd", "agree_pred_oracle", "agree_gtd_oracle", "fair_es", "brier", "false_surface",
                "missed_surface", "depth_rel_err", "regret_official", "top1_official")
        scorers = list(MAIN_SCORERS) + ["Random"] + [f"Exec:{s}" for s in A.SOURCES]
        by = {}
        for x in rows:
            by.setdefault(x["sid"], {})[x["scorer"]] = x
        dd = [(v["V2M"]["fair_es"] - v["V2O"]["fair_es"], v["V2M"]["regret"] - v["V2O"]["regret"]) for v in by.values()
              if v["V2M"].get("fair_es") is not None and v["V2O"].get("fair_es") is not None]
        S["state_rho_dES_dregret"] = T13.spearman(*zip(*dd))
        S["state_frac_regret_lower_V2M"] = float(np.mean([b < 0 for _, b in dd]))
        S["state_frac_regret_higher_V2M"] = float(np.mean([b > 0 for _, b in dd]))
        for m in scorers:
            S["scorers"][m] = {k: agg(rows, m, k) for k in keys}
            if dataset == "outdoor":
                for sp in ("test", "dev"):
                    sub = [x for x in rows if x["split"] == sp]
                    S["scorers"][m].update({f"{k}@{sp}": agg(sub, m, k) for k in ("spearman", "top1", "regret", "pairwise", "fair_es", "brier")})
        for k in ("fair_es", "brier", "false_surface", "spearman", "top1", "pairwise", "regret", "nregret", "fidelity"):
            b, per = paired(rows, k)
            S["paired_V2M_minus_V2O"][k] = dict(b, per_scene=per)
        for m in scorers:
            S["per_scene"][m] = {k: AL.hier_mean([x for x in rows if x["scorer"] == m], k) for k in
                                 ("spearman", "top1", "regret", "pairwise", "fair_es", "brier", "fidelity", "nregret")}
        for nm, f in strata.items():
            d = {}
            for x in rows:
                if x["scorer"] in ("V2O", "V2M", "GTD", "Random"):
                    d.setdefault(f(x), {}).setdefault(x["scorer"], []).append(x)
            S["strata"][nm] = {lb: {m: {"states": len(v), **{k: (float(np.mean([y[k] for y in v if y.get(k) is not None]))
                                                                 if any(y.get(k) is not None for y in v) else None)
                                                             for k in ("spearman", "top1", "regret", "nregret", "pairwise", "fair_es")}}
                                    for m, v in dd.items()} for lb, dd in sorted(d.items())}
        summary[dataset] = S
        print(dataset, "states", len(sts), "repro max", S["repro_max_abs_diff"])
        for m in scorers:
            s = S["scorers"][m]
            print(f"  {m:10s} rho {fmt(s['spearman'])} top1 {fmt(s['top1'])} regret {fmt(s['regret'])} pair {fmt(s['pairwise'])} "
                  f"ES {fmt(s['fair_es'], 2)} brier {fmt(s['brier'])}")
    json.dump(summary, open(os.path.join(RES, "a1_summary.json"), "w"), indent=1, default=float)
    write_tables(summary)


def write_tables(S):
    L = []
    for dataset, nm in (("indoor", "MP3D（5 个 val 楼层，场景 bootstrap）"), ("outdoor", "Macarons++（Test-3 + Dev-3，6 个场景 bootstrap）")):
        s = S[dataset]["scorers"]
        L += [f"### Table A1 — {nm}，{S[dataset]['n_states']} 个固定状态", "",
              "| 评分 | Fair ES ↓ | Brier ↓ | Spearman ↑ | Top-1 ↑ | Pairwise ↑ | Regret (pp) ↓ | 归一化 Regret ↓ |",
              "|---|---|---|---|---|---|---|---|"]
        for m, lab in (("V2O", "V2 Original"), ("V2M", "V2-Mix"), ("GTD", "GT Depth + DistP1"),
                       ("GTDn", "GT Depth + 新体素计数（无距离权重，诊断）"), ("Random", "随机（期望）")):
            x = s[m]
            L.append(f"| {lab} | {fmt(x['fair_es'], 2)} | {fmt(x['brier'])} | {fmt(x['spearman'])} | {fmt(x['top1'])} | "
                     f"{fmt(x['pairwise'])} | {fmt(x['regret'])} | {fmt(x['nregret'])} |")
        L += ["", f"### Table A2 — 评分误差分解（{nm}）", "",
              "| 评分 | 预测保真度 ρ(pred, GTD) ↑ | Pairwise ↑ | Regret (pp) ↓ | Regret − GTD Regret | 与 GTD 选点一致 | 与 oracle 最优一致 |",
              "|---|---|---|---|---|---|---|"]
        for m, lab in (("V2O", "V2 Original"), ("V2M", "V2-Mix")):
            x = s[m]
            L.append(f"| {lab} | {fmt(x['fidelity'])} | {fmt(x['pairwise'])} | {fmt(x['regret'])} | {fmt(x['regret_minus_gtd'])} | "
                     f"{fmt(x['agree_pred_gtd'])} | {fmt(x['agree_pred_oracle'])} |")
        x = s["GTD"]
        L.append(f"| GT Depth + DistP1 | — | {fmt(x['pairwise'])} | {fmt(x['regret'])} | 0 | — | {fmt(x['agree_gtd_oracle'])} |")
        L += ["", "| 配对差 V2M − V2O（按场景） | 均值 [95% CI] |", "|---|---|"]
        for k, v in S[dataset]["paired_V2M_minus_V2O"].items():
            L.append(f"| {k} | {fmt(v)} |")
        L.append("")
    open(os.path.join(TAB, "A1_tables.md"), "w").write("\n".join(L) + "\n")


if __name__ == "__main__":
    main()
