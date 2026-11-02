"""EXP6 V5.2-M -- post-hoc read-only diagnostic of high-confidence no-hit (NOT part of the frozen gate or of any choice).

Question: where are the rays that V5.2-M calls "no hit" with p_nohit >= 0.9 but that truly hit (oracle t*)? Split by
ray class (UNKNOWN / known) and by where the true first hit lies relative to the model's planning grid: inside the grid
(t* < t_max = min(grid exit, tcap)) vs beyond the grid exit (t_max <= t* < tcap; the planner's region crop). Same
states / candidates / maps as eval_fixed_v52m.py (its episode replay is reused; only the first-hit statistic differs).
Output <out>/<set>/<stratum>/<episode>.jsonl rows with "firsthit" = the split counts; summary with --summarise.
Usage: python diag_nohit_v52m.py --set valsel|dev3 [--stratum B0|B2] --ckpt CK --score-cfg CFG --out DIR [--part i/n]
       python diag_nohit_v52m.py --summarise DIR"""
import glob
import json
import os
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)


def fh_diag(out, view, tstar, G):
    import numpy as np
    import torch
    ts = torch.as_tensor(np.asarray(tstar, np.float64), device=out["edges"].device).float()
    e = out["edges"]
    tmax = view["pr"]["tmax"].float().to(e.device)
    valid = ~(ts < e[:, 0])
    hit = torch.isfinite(ts) & (ts < e[:, -1])
    inside = hit & (ts < tmax)
    beyond = hit & (ts >= tmax)
    pn = out["p_nohit"].float()
    unk = view["pr"]["cls"] == G.CLS_UNKNOWN
    acc = {}
    for grp, m0 in (("all", valid), ("unknown", valid & unk), ("known", valid & ~unk)):
        hc = (pn >= 0.9) & m0
        acc[grp] = {"n": int(m0.sum()), "n_hit": int((hit & m0).sum()), "n_hit_beyond": int((beyond & m0).sum()),
                    "hc": int(hc.sum()), "hc_true_nohit": int((hc & ~hit).sum()), "hc_hit_inside": int((hc & inside).sum()), "hc_hit_beyond": int((hc & beyond).sum()),
                    "pn_sum_hit_inside": float(pn[inside & m0].sum()), "pn_sum_hit_beyond": float(pn[beyond & m0].sum()), "pn_sum_nohit": float(pn[m0 & ~hit].sum()),
                    "n_hit_inside": int((inside & m0).sum()), "n_nohit": int((m0 & ~hit).sum())}
    return acc


def summarise(d):
    import numpy as np
    out = {}
    for sset in ("valsel", "dev3"):
        acc = {}
        for f in glob.glob(os.path.join(d, sset, "*", "*.jsonl")):
            for x in open(f):
                r = json.loads(x)
                for g, v in r["firsthit"].items():
                    a = acc.setdefault(g, {})
                    for k, z in v.items():
                        a[k] = a.get(k, 0) + z
        if not acc:
            continue
        out[sset] = {}
        for g, a in acc.items():
            out[sset][g] = {"rays": a["n"], "hit_rate": a["n_hit"] / max(a["n"], 1), "hit_beyond_grid_share_of_hits": a["n_hit_beyond"] / max(a["n_hit"], 1),
                            "hc_rays": a["hc"], "hc_share": a["hc"] / max(a["n"], 1), "hc_true_nohit": a["hc_true_nohit"] / max(a["hc"], 1),
                            "hc_hit_inside_grid": a["hc_hit_inside"] / max(a["hc"], 1), "hc_hit_beyond_grid": a["hc_hit_beyond"] / max(a["hc"], 1),
                            "mean_pnohit_hit_inside": a["pn_sum_hit_inside"] / max(a["n_hit_inside"], 1), "mean_pnohit_hit_beyond": a["pn_sum_hit_beyond"] / max(a["n_hit_beyond"], 1),
                            "mean_pnohit_true_nohit": a["pn_sum_nohit"] / max(a["n_nohit"], 1)}
    json.dump(out, open(os.path.join(d, "summary.json"), "w"), indent=1)
    L = ["| 数据 | 射线组 | 射线 | 命中率 | 命中中越出规划网格的份额 | p_nohit≥0.9 射线 | 其中真 no-hit | 网格内命中 | 网格外命中 | 平均 p_nohit：网格内命中 / 网格外命中 / 真 no-hit |", "|---|---|---|---|---|---|---|---|---|---|"]
    for s, v in out.items():
        for g in ("all", "unknown", "known"):
            x = v.get(g)
            if x:
                L.append(f"| {s} | {g} | {x['rays']:,} | {x['hit_rate']:.3f} | {x['hit_beyond_grid_share_of_hits']:.3f} | {x['hc_rays']:,}（{x['hc_share']:.3f}） | {x['hc_true_nohit']:.3f} | "
                         f"{x['hc_hit_inside_grid']:.3f} | {x['hc_hit_beyond_grid']:.3f} | {x['mean_pnohit_hit_inside']:.3f} / {x['mean_pnohit_hit_beyond']:.3f} / {x['mean_pnohit_true_nohit']:.3f} |")
    open(os.path.join(d, "summary.md"), "w").write("\n".join(L) + "\n")
    print("\n".join(L))


if __name__ == "__main__":
    if "--summarise" in sys.argv:
        summarise(sys.argv[sys.argv.index("--summarise") + 1])
        sys.exit(0)
    import eval_fixed_v52m as EFM
    EFM.fh_metrics = fh_diag
    EFM.main()
