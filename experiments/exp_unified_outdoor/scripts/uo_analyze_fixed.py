"""EXP Unified Outdoor -- fixed-state G1 / G2 analysis of trained models next to the historical references (CPU).
Merges, state by state, the uo_fixed rows of the given models with the Density A/B rows (B2, E2S, E2D; same states, same
replay / noise / candidates), then runs exp_density_ab/scripts/da_analyze_fixed.py (unchanged metric code: historical
choice rule, hierarchical means, scene-level bootstrap) with the model list extended. Pairs: every new model vs B2 and vs
E2D at each (inference, scoring), plus the new model's own inference / scoring contrasts.
Output results/fixed_<names>.json  Usage: python uo_analyze_fixed.py NAME [NAME ...]
"""
import glob
import json
import os
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
EXP = os.path.dirname(HERE)
REPO = os.path.dirname(os.path.dirname(EXP))
sys.path.insert(0, os.path.join(REPO, "experiments", "exp_density_ab", "scripts"))
import da_analyze_fixed as DF  # noqa: E402

DA_FIX = os.path.join(REPO, "work_dirs", "exp_density_ab", "fixed")
UO_FIX = os.path.join(REPO, "work_dirs", "exp_unified_outdoor", "eval_fixed")
MERGED = os.path.join(REPO, "work_dirs", "exp_unified_outdoor", "analysis_merged")


def merge(names):
    n = 0
    for f in sorted(glob.glob(os.path.join(DA_FIX, "*", "*", "*.jsonl"))):
        rel = os.path.relpath(f, DA_FIX)
        base = {json.loads(x)["sid"]: json.loads(x) for x in open(f)}
        for nm in names:
            g = os.path.join(UO_FIX, nm, rel)
            if not os.path.exists(g):
                raise SystemExit(f"missing {g}")
            for x in open(g):
                r = json.loads(x)
                b = base[r["sid"]]
                assert b["t"] == r["t"] and b["oracle_gain_pp"] == r["oracle_gain_pp"]
                for k in ("scores", "g1", "pred_e2", "masks"):
                    b[k].update(r[k])
        out = os.path.join(MERGED, "_".join(names), rel)
        os.makedirs(os.path.dirname(out), exist_ok=True)
        with open(out, "w") as fo:
            for r in base.values():
                fo.write(json.dumps(r) + "\n")
        n += 1
    return n


def main():
    names = sys.argv[1:]
    n = merge(names)
    DF.FIX = os.path.join(MERGED, "_".join(names))
    DF.CK = ("B2", "E2S", "E2D") + tuple(names)
    DF.CFGS = [f"{m}|{i}|{s}" for m in DF.CK for i in DF.DN for s in DF.DN]

    def pairs():
        out = []
        for nm in names:
            for i in DF.DN:
                for s in DF.DN:
                    out += [(f"{nm}|{i}|{s}", f"B2|{i}|{s}", f"{nm} - B2"), (f"{nm}|{i}|{s}", f"E2D|{i}|{s}", f"{nm} - E2D")]
            for s in DF.DN:
                out.append((f"{nm}|D100|{s}", f"{nm}|S10|{s}", f"{nm} inference D100-S10"))
            for i in DF.DN:
                out.append((f"{nm}|{i}|D100", f"{nm}|{i}|S10", f"{nm} scoring D100-S10"))
        return out
    DF.pairs_k = pairs
    recs = DF.load_rows()
    res = {}
    for d in ("indoor", "outdoor"):
        rr = [r for r in recs if r["dataset"] == d]
        S = DF.summarize(rr)
        res[d] = {"n_states": S["n_states"], "means": {k: v for k, v in S["means"].items() if k.split("#")[0] in DF.CFGS},
                  "paired": {k: {kk: vv for kk, vv in v.items() if kk != "per_scene"} for k, v in S["paired"].items() if not k.startswith("G1:")}}
        M = S["means"]
        print(d, "states", S["n_states"], "episode files merged", n)
        for c in DF.CFGS:
            if c.split("|")[0] in names or c in ("B2|D100|S10", "E2D|D100|S10", "B2|S10|S10"):
                print(f"  {c:24s} top1 {M[c + '#top1']['mean']:.3f} sel {M[c + '#sel_gain']['mean']:.3f} rho {M[c + '#rho']['mean']:.3f}")
        for nm in names:
            for n_ in ("S10", "D100"):
                k = f"{nm}|{n_}#unk_common#mae"
                print(f"  G1 {nm}|{n_} unk_common mae {M[k]['mean']:.4f} acc_first {M[f'{nm}|{n_}#unk_common#acc_first']['mean']:.3f}")
    os.makedirs(os.path.join(EXP, "results"), exist_ok=True)
    json.dump(res, open(os.path.join(EXP, "results", f"fixed_{'_'.join(names)}.json"), "w"), indent=1, default=float)


if __name__ == "__main__":
    main()
