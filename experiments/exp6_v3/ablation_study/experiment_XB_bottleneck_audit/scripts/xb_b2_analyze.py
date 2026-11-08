"""Experiment XB / Task B2 -- oracle upper-bound ladder from the counterfactual rows (CPU). Reads states/xb_cf/<CKPT>/<tag>.jsonl (CT ct_cf format: cf[variant]["one"|"two"] = planner
scores after replacing one error type by the truth). Truth: indoor oracle gains of the candidates (CT g1 rows carry gain_pp per state), outdoor Experiment F official one-step gains of keys1.
Per model, layer and variant: official Top-1 and regret, paired difference to `pred` with a cluster bootstrap (floors indoor, scenes outdoor). Output results/B2_oracle/ladder.json."""
import glob
import json
import os

import numpy as np

R = "/bsuscratch/jiahuizhang/projects/occupancy/seen2scene"
X = f"{R}/experiments/exp6_v3/ablation_study/experiment_XB_bottleneck_audit"
W = f"{R}/work_dirs/exp6_v3/ablation_study/experiment_XB_bottleneck_audit/states/xb_cf"
FW = f"{R}/work_dirs/exp6_v3/ablation_study/experiment_F_stochastic_coverage_decision/states"
G1 = f"{R}/work_dirs/exp6_v3/ablation_study/experiment_CT_continued_training/states/g1"
OUT = f"{X}/results/B2_oracle"
VARS = ("pred", "no_fp", "no_fn", "hit_oracle", "depth_oracle", "perfect", "perfect_all")
DEV5 = {"colosseum", "barts", "pantheon", "eiffel", "sestino_museum"}
TOL = 1e-4


def boot(v, cl, n=10000, seed=0):
    v, cl = np.asarray(v, float), np.asarray(cl)
    cs = sorted(set(cl)); per = np.array([v[cl == c].mean() for c in cs])
    o = {"d": float(v.mean()), "n": len(v), "clusters": len(cs)}
    if len(cs) >= 2:
        b = per[np.random.default_rng(seed).integers(0, len(cs), (n, len(cs)))].mean(1)
        o["ci"] = [float(np.quantile(b, .025)), float(np.quantile(b, .975))]
    return o


def pick(s, cand=None):
    s = np.asarray(s, float)
    if cand is not None:
        m = np.full_like(s, -np.inf); m[cand] = s[cand]; s = m
    return int(np.flatnonzero(s == s.max())[0])


def main():
    frows = {(r["tag"], r["t"]): r for f in glob.glob(FW + "/*.jsonl") for r in map(json.loads, open(f))}
    g1 = {(r["tag"], r["t"]): r for f in glob.glob(G1 + "/*.jsonl") for r in map(json.loads, open(f)) if r["ckpt"] == "A_40K"}
    out = {}
    for ck in sorted(os.listdir(W)):
        rows = [r for f in glob.glob(f"{W}/{ck}/*.jsonl") for r in map(json.loads, open(f))]
        data = {}
        for r in rows:
            key = (r["tag"], r["t"])
            if r["domain"] == "indoor":
                g = g1.get(key)
                if g is None:
                    continue
                orc = np.asarray([np.nan if x is None else x for x in g["gain_pp"]], float); cand = r["cand"]; hs = ("one",); layer = "indoor"; cl = r["scene"]
            else:
                f = frows.get(key)
                if f is None:
                    continue
                orc = np.asarray([np.nan if x is None else x for x in f["oracle"]], float); cand = None; hs = ("one", "two"); layer = "outdoor_dev" if r["scene"] in DEV5 else "outdoor_train"; cl = r["scene"]
            m = np.nanmax(orc)
            for h in hs:
                for v in VARS:
                    s = r["cf"][v][h]
                    pk = pick(s, cand)
                    if not np.isfinite(orc[pk]):
                        continue
                    data.setdefault((layer, h), {}).setdefault(v, []).append((float(m - orc[pk]), float(orc[pk] >= m - TOL), cl, key))
        res = {}
        for (layer, h), dd in data.items():
            base = {k: (rg, tp) for rg, tp, _, k in dd.get("pred", [])}
            o = {}
            for v in VARS:
                it = dd.get(v, [])
                if not it:
                    continue
                x = {"n": len(it), "regret": float(np.mean([a for a, _, _, _ in it])), "top1": float(np.mean([b for _, b, _, _ in it]))}
                if v != "pred":
                    pairs = [(a - base[k][0], c) for a, _, c, k in it if k in base]
                    if pairs:
                        x["d_regret_vs_pred"] = boot([p for p, _ in pairs], [c for _, c in pairs])
                        x["d_top1_vs_pred"] = float(np.mean([b - base[k][1] for _, b, _, k in it if k in base]))
                o[v] = x
            res[f"{layer}/{h}"] = o
        out[ck] = res
    os.makedirs(OUT, exist_ok=True)
    json.dump(out, open(f"{OUT}/ladder.json", "w"), indent=1)
    for ck, res in out.items():
        for lh, o in res.items():
            print(f"== {ck} {lh}")
            for v, x in o.items():
                d = x.get("d_regret_vs_pred", {})
                print("   %-13s n %3d top1 %.3f regret %.3f %s" % (v, x["n"], x["top1"], x["regret"], ("d %+.3f [%+.3f, %+.3f]" % (d["d"], *d["ci"])) if d.get("ci") else ""))


if __name__ == "__main__":
    main()
