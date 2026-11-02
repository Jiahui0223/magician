"""EXP6 V5 -- G1 / G2 analysis on the development fixed states (seen2scene env, CPU): exp6_v4/scripts/analyze_g12_v4.py
(readers, per-state records, summaries, tables) with the V5 and V4C rows of work_dirs/exp6_v5/eval_fixed/<set>_<tag>
merged by sid into the historical rows (B2 / E2D from E2, T0 / T1 from V3, V31 / V4 from exp6_v4); the B2 replay of the
V5 rows must equal the E2 B2 rows bitwise. Main fixed-state metrics: true one-step coverage gain of the pick (pp) and
Top-1 (regret only in the appendix); plus the share of states whose pick differs from V4's. Output
experiments/exp6_v5/reports/g12_v5_<tag>.{json,md}. Usage: python analyze_g12_v5.py [--tags s_final_v5 s_final_v4c] [--tag s_final]"""
import argparse
import glob
import json
import os
import sys

import numpy as np

from seen2scene import PROJECT_ROOT

REPO = str(PROJECT_ROOT)
sys.path.insert(0, os.path.join(REPO, "experiments", "exp6_v4", "scripts"))
import analyze_g12_v4 as A4  # noqa: E402

EA = A4.EA
W5 = os.path.join(REPO, "work_dirs", "exp6_v5", "eval_fixed")
OUT = os.path.join(REPO, "experiments", "exp6_v5", "reports")
MODELS = ("B2", "E2D", "T1", "V31", "V4", "V4C", "V5")
LAB = dict(A4.LAB, V4="V4@40k", V4C="V4-Continue12k", V5="V5-Warm12k")
COL = dict(A4.COL, V4C="#495057", V5="#7048e8")
PAIRS = (("V5", "V4C"), ("V5", "V4"), ("V4C", "V4"), ("V5", "B2"), ("V4C", "B2"), ("V5", "E2D"))


def load(sset, tags):
    """tags = row directories of the V5 round (e.g. s_final_v5, s_final_v4c: separate runs, same states / candidates / noise)."""
    base = {r["sid"]: r for r in A4.load(sset, "s_final_v4", "s_final_v31")}
    news = [A4.read_rows(os.path.join(W5, f"{sset}_{t}")) for t in tags]
    rows = []
    for sid, r in base.items():
        if not all(sid in nw for nw in news):
            continue
        for nw in news:
            n = nw[sid]
            assert n["scores"]["B2"] == r["scores"]["B2"], ("B2 replay differs", sid)
            for m in ("V5", "V4C"):
                if m in n["scores"]:
                    r["scores"][m], r["pred"][m] = n["scores"][m], n["pred"][m]
                    r.setdefault("timing", {})[m] = n.get("timing", {}).get(m)
        rows.append(r)
    return rows


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--tags", nargs="+", default=["s_final_v5", "s_final_v4c"])
    ap.add_argument("--tag", default="s_final", help="output name")
    a = ap.parse_args()
    loaded = {d: load(d, a.tags) for d in ("valsel", "dev3")}
    avail = set(loaded["valsel"][0]["scores"]) & set(loaded["dev3"][0]["scores"])
    models = tuple(m for m in MODELS if m in avail)
    pairs = tuple((x, y) for x, y in PAIRS if x in models and y in models)
    A4.MODELS, A4.PAIRS, A4.LAB = models, pairs, LAB
    EA.MODELS, EA.LAB, EA.COL, EA.PAIRS = models, LAB, COL, pairs
    EA.OUT, EA.FIG = OUT, os.path.join(REPO, "experiments", "exp6_v5", "figures")
    os.makedirs(EA.FIG, exist_ok=True)
    EA.ST.setup()
    S, extra = {}, {}
    for d in ("valsel", "dev3"):
        recs = A4.rows_to_recs(d, loaded[d])
        S[d] = {sub: EA.summ(recs, sub) for sub in ("early", "middle", "late", "all")}
        S[d]["n"] = len(recs)
        extra[d] = {m: {"pick_differs_from_V4": float(np.mean([r[f"{m}_pick"] != r["V4_pick"] for r in recs]))} for m in models if m != "V4"}
        for m in models:
            extra[d].setdefault(m, {})["n_states"] = len(recs)
    json.dump({"tag": a.tag, "models": models, "summary": S, "pick_vs_V4": extra}, open(os.path.join(OUT, f"g12_v5_{a.tag}.json"), "w"), indent=1, default=float)
    txt = A4.tables(S, {}, a.tag).replace("# V4 / V3.1 G1 / G2", "# V5 G1 / G2").replace("G2 → G3 放行（V3 协议 §4，仅供参考）：", "")
    txt += "\n\n## 与 V4@40k 选择不同的状态比例\n\n| 集 | " + " | ".join(LAB[m] for m in models if m != "V4") + " |\n|---|" + "---|" * (len(models) - 1) + "\n"
    for d in ("valsel", "dev3"):
        txt += f"| {A4.NAME[d]} | " + " | ".join(f"{extra[d][m]['pick_differs_from_V4']:.3f}" for m in models if m != "V4") + " |\n"
    open(os.path.join(OUT, f"g12_v5_{a.tag}.md"), "w").write(txt)
    try:
        EA.fig_g12(S, f"v5_{a.tag}")
    except Exception as e:                                   # noqa: BLE001
        print("figure skipped:", e)
    print(json.dumps({d: {m: {k: S[d]["all"]["means"][f"{m}_{k}"]["mean"] for k in ("gain", "top1", "mae_unknown")} for m in models} for d in S}, indent=1, default=float))


if __name__ == "__main__":
    main()
