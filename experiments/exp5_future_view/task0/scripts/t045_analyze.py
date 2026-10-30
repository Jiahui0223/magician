"""EXP5 Task 0.4 / 0.5 analysis (CPU, evaluation side). Per protocol root (M = fixed_state, R = fixed_state_R,
O = fixed_state_out): per-state rows of t04_metrics.state_rows (shared metrics) and eval3d (world-model 3D metrics);
aggregates per coverage stratum with cluster bootstrap over houses / scenes (10,000, seed 0); paired contrasts
(V2 vs world models, K4 vs K1, frozen vs C, U vs DistP1) on the same states; M vs R on the same states; indoor (M)
vs outdoor (O) per stratum (unpaired, cluster bootstrap of each side); C - frozen on the 3D metrics; per t bin
(same number of frames, descriptive); O split into the Macarons++ indoor scenes and the outdoor ones (v1.2 §5).
Output W/analysis/t045.json + figures.
"""
import glob
import json
import os
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
import t04_metrics as TM  # noqa: E402

import numpy as np  # noqa: E402

W = TM.W
ROOTS = {"M": ("fixed_state", "indoor_states.json"), "R": ("fixed_state_R", "indoor_states.json"),
         "O": ("fixed_state_out", "outdoor_states.json")}
STRATA = ["c01-05", "c05-10", "c10-20", "c20-40"]
DEPTH = ("crps", "es_view", "mae", "brier", "false_conf", "spread", "hit_rate_pred", "hit_rate_label")
PLAN = ("spearman", "regret", "top1")
E3D = ("P@1", "R@1", "F@1", "P@2", "R@2", "F@2", "band_abs", "band_sign_agree", "empty_conflict")


def cboot(vals, groups, n=10000, seed=0):
    """Mean of cluster means and 95 % bootstrap CI over clusters."""
    by = {}
    for v, g in zip(vals, groups):
        if v is not None and np.isfinite(v):
            by.setdefault(g, []).append(v)
    if not by:
        return None
    cm = np.asarray([np.mean(v) for v in by.values()])
    rng = np.random.default_rng(seed)
    bs = cm[rng.integers(0, len(cm), (n, len(cm)))].mean(1)
    return {"mean": float(cm.mean()), "ci95": [float(np.quantile(bs, 0.025)), float(np.quantile(bs, 0.975))],
            "clusters": int(len(cm)), "n": int(sum(len(v) for v in by.values()))}


def load(proto):
    root, sf = ROOTS[proto]
    rd = os.path.join(W, root)
    sts = json.load(open(os.path.join(W, "manifests", sf)))["states"]
    rows = []
    for st in sts:
        if not (os.path.exists(os.path.join(rd, "labels", f"{st['id']}.npz")) and os.path.exists(os.path.join(rd, "V2", f"{st['id']}.npz"))):
            continue
        d, p, gmax = TM.state_rows(st["id"], rd)
        e3 = {}
        for m in ("frozen", "C"):
            q = os.path.join(rd, "eval3d", m, f"{st['id']}.json")
            if os.path.exists(q):
                e3[m] = json.load(open(q))
        rows.append({"state": st, "depth": d, "plan": p, "gain_max": gmax, "e3d": e3})
    return rows


def aggregate(rows):
    out = {}
    for grp in ["all"] + STRATA:
        sel = [r for r in rows if grp == "all" or r["state"]["stratum"] == grp]
        if not sel:
            continue
        g = [r["state"]["house"] for r in sel]
        methods = sorted({m for r in sel for m in r["depth"]} | {m for r in sel for m in r["plan"]})
        out[grp] = {"n_states": len(sel), "clusters": len(set(g))}
        for m in methods:
            res = {}
            for k in DEPTH:
                res[k] = cboot([r["depth"].get(m, {}).get(k) for r in sel], g)
            for k in PLAN:
                res[f"plan_{k}"] = cboot([r["plan"].get(m, {}).get(k) for r in sel], g)
            out[grp][m] = {k: v for k, v in res.items() if v}
        for m in ("frozen", "C"):
            for kk in ("K1", "K4"):
                res = {k: cboot([(r["e3d"].get(m, {}).get(kk) or {}).get(k) for r in sel], g) for k in E3D}
                if any(res.values()):
                    out[grp][f"{m}_{kk}_3d"] = {k: v for k, v in res.items() if v}
            res = {"union_R@2": cboot([r["e3d"].get(m, {}).get("K4_union_R@2") for r in sel], g),
                   "diversity": cboot([r["e3d"].get(m, {}).get("diversity") for r in sel], g)}
            if any(res.values()):
                out[grp][f"{m}_K4_3d_extra"] = {k: v for k, v in res.items() if v}
    return out


PAIRS = [("V2_K4", "C_K4"), ("V2_K4", "frozen_K4"), ("C_K4", "frozen_K4"), ("V2_K4", "V2_K1"), ("C_K4", "C_K1"),
         ("frozen_K4", "frozen_K1"), ("V2_K1", "C_K1"), ("V2_K1", "frozen_K1"), ("V2_DistU", "V2_K4"), ("V2_Uraw", "V2_K4")]


def paired(rows):
    out = {}
    for grp in ["all"] + STRATA:
        sel = [r for r in rows if grp == "all" or r["state"]["stratum"] == grp]
        g = [r["state"]["house"] for r in sel]
        res = {}
        for a, b in PAIRS:
            d = {}
            for k in ("crps", "es_view", "brier"):
                d[k] = cboot([(r["depth"][a][k] - r["depth"][b][k]) if a in r["depth"] and b in r["depth"]
                              and r["depth"][a].get(k) is not None and r["depth"][b].get(k) is not None else None for r in sel], g)
            for k in ("spearman", "regret"):
                d[f"plan_{k}"] = cboot([(r["plan"][a][k] - r["plan"][b][k]) if a in r["plan"] and b in r["plan"]
                                        and r["plan"][a][k] is not None and r["plan"][b][k] is not None else None for r in sel], g)
            d = {k: v for k, v in d.items() if v}
            if d:
                res[f"{a}-{b}"] = d
        out[grp] = res
    return out


def m_vs_r(rm, rr):
    byid = {r["state"]["id"]: r for r in rr}
    out = {}
    for grp in ["all"] + STRATA:
        sel = [(r, byid[r["state"]["id"]]) for r in rm if r["state"]["id"] in byid and (grp == "all" or r["state"]["stratum"] == grp)]
        g = [a["state"]["house"] for a, _ in sel]
        res = {}
        for m in ("V2_K4", "V2_K1", "C_K4", "C_K1", "frozen_K4", "frozen_K1"):
            d = {}
            for k in ("crps", "es_view", "brier"):
                d[k] = cboot([(b["depth"][m][k] - a["depth"][m][k]) if m in a["depth"] and m in b["depth"]
                              and a["depth"][m].get(k) is not None and b["depth"][m].get(k) is not None else None for a, b in sel], g)
            for k in ("spearman", "regret"):
                d[f"plan_{k}"] = cboot([(b["plan"][m][k] - a["plan"][m][k]) if m in a["plan"] and m in b["plan"]
                                        and a["plan"][m][k] is not None and b["plan"][m][k] is not None else None for a, b in sel], g)
            d = {k: v for k, v in d.items() if v}
            if d:
                res[f"{m} (R - M)"] = d
        out[grp] = res
    return out


MPP_INDOOR = ("OUT_barts", "OUT_sestino_museum")    # Macarons++ indoor photogrammetry scenes (v1.2 §5.4)
CMP = ("frozen_K4", "frozen_K1", "C_K4", "C_K1", "V2_K4", "V2_K1", "V2_DistU", "V2_Uraw", "Frontier")
E3D_P = ("F@1", "F@2", "P@2", "R@2", "band_abs", "band_sign_agree", "empty_conflict")
T_BINS = ((0, 0), (1, 2), (3, 5), (6, 10), (11, 20), (21, 100))


def cboot_diff(va, ga, vb, gb, n=10000, seed=0):
    """Unpaired difference of cluster means (a - b); each side resampled over its own clusters."""
    def cm(v, g):
        by = {}
        for x, c in zip(v, g):
            if x is not None and np.isfinite(x):
                by.setdefault(c, []).append(x)
        return np.asarray([np.mean(x) for x in by.values()])
    a, b = cm(va, ga), cm(vb, gb)
    if not len(a) or not len(b):
        return None
    rng = np.random.default_rng(seed)
    bs = a[rng.integers(0, len(a), (n, len(a)))].mean(1) - b[rng.integers(0, len(b), (n, len(b)))].mean(1)
    return {"mean": float(a.mean() - b.mean()), "ci95": [float(np.quantile(bs, 0.025)), float(np.quantile(bs, 0.975))],
            "a": float(a.mean()), "b": float(b.mean()), "clusters": [int(len(a)), int(len(b))]}


def _val(r, m, k):
    if k.startswith("plan_"):
        return (r["plan"].get(m) or {}).get(k[5:])
    return (r["depth"].get(m) or {}).get(k)


def m_vs_o(rm, ro):
    """Indoor (M) - outdoor (O) per coverage stratum (main comparison, v1.2 §5.1), unpaired, cluster bootstrap."""
    out = {}
    for grp in ["all"] + STRATA:
        a = [r for r in rm if grp == "all" or r["state"]["stratum"] == grp]
        b = [r for r in ro if grp == "all" or r["state"]["stratum"] == grp]
        ga, gb = [r["state"]["house"] for r in a], [r["state"]["house"] for r in b]
        res = {}
        for m in CMP:
            d = {k: cboot_diff([_val(r, m, k) for r in a], ga, [_val(r, m, k) for r in b], gb)
                 for k in ("crps", "es_view", "brier", "false_conf", "hit_rate_pred", "hit_rate_label", "plan_spearman", "plan_regret", "plan_top1")}
            d = {k: v for k, v in d.items() if v}
            if d:
                res[m] = d
        for m in ("frozen", "C"):
            d = {k: cboot_diff([((r["e3d"].get(m) or {}).get("K4") or {}).get(k) for r in a], ga,
                               [((r["e3d"].get(m) or {}).get("K4") or {}).get(k) for r in b], gb) for k in E3D_P}
            d = {k: v for k, v in d.items() if v}
            if d:
                res[f"{m}_K4_3d"] = d
        out[grp] = res
    return out


def paired3d(rows):
    """C_K4 - frozen_K4 (and K1) on the 3D metrics, same states, cluster bootstrap."""
    out = {}
    for grp in ["all"] + STRATA:
        sel = [r for r in rows if (grp == "all" or r["state"]["stratum"] == grp) and "frozen" in r["e3d"] and "C" in r["e3d"]]
        g = [r["state"]["house"] for r in sel]
        res = {}
        for kk in ("K4", "K1"):
            res[f"C_{kk}-frozen_{kk}"] = {k: cboot([(r["e3d"]["C"].get(kk) or {}).get(k) - (r["e3d"]["frozen"].get(kk) or {}).get(k)
                                                     if (r["e3d"]["C"].get(kk) or {}).get(k) is not None and (r["e3d"]["frozen"].get(kk) or {}).get(k) is not None
                                                     else None for r in sel], g) for k in E3D_P}
        out[grp] = {"n_states": len(sel), **res}
    return out


def by_t(rows):
    """Same number of observed frames (v1.2 §5.2): descriptive, per t bin, not paired."""
    out = {}
    for lo, hi in T_BINS:
        sel = [r for r in rows if lo <= r["state"]["t"] <= hi]
        if not sel:
            continue
        g = [r["state"]["house"] for r in sel]
        res = {"n_states": len(sel), "clusters": len(set(g)), "cov_mean": float(np.mean([r["state"]["cov"] for r in sel]))}
        for m in CMP:
            d = {k: cboot([_val(r, m, k) for r in sel], g) for k in ("crps", "brier", "plan_spearman", "plan_regret")}
            d = {k: v for k, v in d.items() if v}
            if d:
                res[m] = d
        for m in ("frozen", "C"):
            v = cboot([((r["e3d"].get(m) or {}).get("K4") or {}).get("F@2") for r in sel], g)
            if v:
                res[f"{m}_K4_F@2"] = v
        out[f"t{lo}-{hi}"] = res
    return out


def main():
    res, rows = {"doc": __doc__}, {}
    for proto in ROOTS:
        rows[proto] = load(proto)
        res[proto] = {"n_states": len(rows[proto]), "aggregate": aggregate(rows[proto]), "paired": paired(rows[proto])}
        print(proto, len(rows[proto]), "states", flush=True)
    res["M_vs_R"] = m_vs_r(rows["M"], rows["R"])
    res["M_vs_O"] = m_vs_o(rows["M"], rows["O"])
    res["paired_3d"] = {p: paired3d(rows[p]) for p in rows}
    res["by_t"] = {p: by_t(rows[p]) for p in ("M", "O")}
    res["O_split"] = {k: aggregate([r for r in rows["O"] if (r["state"]["house"] in MPP_INDOOR) == (k == "mpp_indoor")])
                      for k in ("mpp_indoor", "outdoor")}
    res["per_state"] = {p: rows[p] for p in rows}
    od = os.path.join(W, "analysis")
    os.makedirs(od, exist_ok=True)
    json.dump(res, open(os.path.join(od, "t045.json"), "w"), indent=1, default=float)
    figures(res)


def figures(res):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    fig, ax = plt.subplots(1, 4, figsize=(20, 4.2))
    for j, (key, lab) in enumerate((("crps", "fair CRPS (vox), M_unknown rays"), ("brier", "hit Brier"),
                                    ("plan_spearman", "DistP1 Spearman vs true gain"), ("plan_regret", "one-step regret (pp)"))):
        for proto, ls in (("M", "-"), ("O", "--")):
            ag = res[proto]["aggregate"]
            for m, col in (("frozen_K4", "C0"), ("C_K4", "C1"), ("V2_K4", "C2"), ("V2_K1", "C3")):
                ys = [ag.get(s, {}).get(m, {}).get(key, {}).get("mean") if ag.get(s, {}).get(m, {}).get(key) else np.nan for s in STRATA]
                ax[j].plot(range(4), ys, ls, color=col, marker="o", label=f"{'indoor' if proto == 'M' else 'outdoor'} {m}")
        ax[j].set_xticks(range(4))
        ax[j].set_xticklabels(STRATA)
        ax[j].set_title(lab)
    ax[0].legend(fontsize=7)
    fig.tight_layout()
    fig.savefig(os.path.join(W, "analysis", "t045_strata.png"), dpi=110)
    plt.close(fig)


if __name__ == "__main__":
    main()
