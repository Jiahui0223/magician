"""EXP6 V5.2-M -- G3 smoke analysis (seen2scene env, CPU): the V5.2 smoke analysis (exp6_v5_2/scripts/eval_g3_smoke.py) on the
same pre-registered episodes -- Macarons++ Test-3 alhambra s0 / s1, bridge s0 / s1 (100 moves) and MP3D VALSEL 8194nk5LbLH
L0 S10, EU6Fwq7SyZv L0 S67 (150 moves) -- with the V5.2-M groups V52ML0 / V52ML01 (work_dirs/exp6_v5_2_scratch70k/eval/g3)
added to the old V5.2 pilot and the references. Smoke validity (FROZEN_PROTOCOL §5): all 12 V5.2-M episodes present with
the full number of logged steps and zero guard calls. Exploratory, not a formal result; evaluation scenes are not blind.
Output experiments/exp6_v5_2_scratch70k/reports/g3_smoke_v52m.json + G3_SMOKE.md. Usage: python eval_g3_smoke_v52m.py"""
import argparse
import json
import os
import sys

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
EM = os.path.dirname(HERE)
sys.path.insert(0, os.path.join(os.path.dirname(os.path.dirname(EM)), 'experiments', 'exp6_v5_2', 'scripts'))
import v52_common as VC  # noqa: E402

REPO = VC.REPO
sys.path.insert(0, os.path.join(REPO, "experiments", "exp6_v5", "scripts"))
import eval_g3 as G5  # noqa: E402  (registers the V5 / V4C / V4 / A1 / historical roots)

AG = G5.AG
W52 = os.path.join(VC.W52, "eval")
AG.ROOTS.update({g: (os.path.join(W52, "closed_loop"), os.path.join(W52, "metrics", "outdoor"), "V2M") for g in ("V52L0", "V52L01")})
WM = os.path.join(REPO, "work_dirs", "exp6_v5_2_scratch70k", "eval", "g3")
AG.ROOTS.update({g: (os.path.join(WM, "closed_loop"), os.path.join(WM, "metrics", "outdoor"), "V2M") for g in ("V52ML0", "V52ML01")})
NEW = ("V52ML0", "V52ML01")
OUT_SMOKE = [("alhambra", 0), ("alhambra", 1), ("bridge", 0), ("bridge", 1)]
IN_SMOKE = [("8194nk5LbLH_L0", 10), ("EU6Fwq7SyZv_L0", 67)]
OUT_REF = ("MAG-G", "MAG-B", "B2", "E2D", "V4", "V4C", "V5", "V52L0", "V52L01", "V52ML0", "V52ML01")
IN_REF = ("V4C", "V5", "V52L0", "V52L01", "V52ML0", "V52ML01")


def stall(c, w=30, eps=0.005):
    c = np.asarray(c)
    return int(max([0] + [w for i in range(len(c) - w) if c[i + w] - c[i] < eps]))


def outdoor(g, s, k):
    base, tg, mp = AG._outdoor(g, s, k)
    m = json.load(open(mp))
    c = np.asarray(m["cov_mono"])
    r = {"auc": float(m["auc_mono"]), "final": float(m["cov_mono_final"]), "cov20": float(c[20]), "curve": c.tolist()}
    sl = os.path.join(base, "v52_steps", f"{tg}.jsonl")
    if os.path.exists(sl):
        r.update(steplog(sl))
    return r


def indoor(g, tag):
    d = os.path.join(WM if g in NEW else W52, "closed_loop", "indoor", g, "V2M", tag)
    e = json.load(open(os.path.join(d, "episode.json")))
    c = np.asarray(e["coverage"])
    r = {"auc": float(np.trapz(c, dx=1) / e["moves"]), "final": float(c[-1]), "cov20": float(c[20]), "curve": c.tolist(), "stall30": stall(c)}
    pos = [tuple(p[:3]) for p in e["executed"]]
    r["revisit_position"] = float(np.mean([pos[i] in set(pos[:i]) for i in range(1, len(pos))]))
    sl = os.path.join(d, "v52_steps.jsonl")
    if os.path.exists(sl):
        r.update(steplog(sl))
    return r


def steplog(p):
    S = [json.loads(x) for x in open(p)]
    g = [sum(s["guard_counts"].values()) for s in S]
    pick = [int(np.argmax(s["distp1"])) if s["distp1"] else None for s in S]
    pick0 = [int(np.argmax(s["v52_score_l0"])) if s["v52_score_l0"] else None for s in S]
    share = [s["v52_parts"]["pot_share"][j] for s, j in zip(S, pick) if j is not None]
    return {"steps_logged": len(S), "guard_calls_total": int(sum(g)), "lambda_flip_rate": float(np.mean([a != b for a, b in zip(pick, pick0) if a is not None])),
            "pot_share_of_argmax_mean": float(np.mean(share)) if share else None, "score_s_mean": float(np.mean([s["timing"]["v52_score_s"] for s in S])),
            "zero_visible_steps": int(sum(max(s["v52_s_vis"]) == 0 for s in S if s["v52_s_vis"]))}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--full", action="store_true")
    a = ap.parse_args()
    res = {"outdoor": {}, "indoor": {}}
    for s, k in OUT_SMOKE:
        res["outdoor"][f"{s}_s{k}"] = {}
        for g in OUT_REF:
            try:
                res["outdoor"][f"{s}_s{k}"][g] = outdoor(g, s, k)
            except FileNotFoundError:
                pass
    for tag, st in IN_SMOKE:
        t = f"{tag}_S{st}"
        res["indoor"][t] = {}
        for g in IN_REF:
            try:
                res["indoor"][t][g] = indoor(g, t)
            except FileNotFoundError:
                pass
    mean = {}
    for d, refs in (("outdoor", OUT_REF), ("indoor", IN_REF)):
        mean[d] = {}
        for g in refs:
            v = [e[g] for e in res[d].values() if g in e]
            if len(v) == len(res[d]):
                mean[d][g] = {k: float(np.mean([x[k] for x in v])) for k in ("auc", "final", "cov20")}
    res["mean"] = mean
    rule = {}
    for d in ("outdoor", "indoor"):
        rule[d] = {lam: {ref: (mean[d][lam]["auc"] - mean[d][ref]["auc"]) for ref in ("V5", "V52L0", "B2", "E2D") if ref in mean[d]} for lam in NEW if lam in mean[d]}
    res["auc_minus_ref"] = rule
    chk = []
    for d, n_mv in (("outdoor", 100), ("indoor", 150)):
        for ep, v in res[d].items():
            for g in NEW:
                x = v.get(g)
                chk.append({"domain": d, "episode": ep, "group": g, "present": x is not None, "steps_logged": (x or {}).get("steps_logged"),
                            "guard_calls": (x or {}).get("guard_calls_total"),
                            "ok": x is not None and x.get("steps_logged") == n_mv and x.get("guard_calls_total") == 0})
    res["validity"] = chk
    res["valid"] = bool(chk) and all(c["ok"] for c in chk)
    os.makedirs(os.path.join(EM, "reports"), exist_ok=True)
    json.dump(res, open(os.path.join(EM, "reports", "g3_smoke_v52m.json"), "w"), indent=1, default=float)
    L = ["# V5.2-M G3 smoke（预先冻结的 6 个 episode × λ 0 / 0.1；与旧 V5.2 smoke 同一组 episode；探索性诊断，不是正式结论；评估场景非盲）", ""]
    for d, nm in (("outdoor", "Macarons++ Test-3（100 步）"), ("indoor", "MP3D VALSEL（150 步）")):
        refs = OUT_REF if d == "outdoor" else IN_REF
        L += [f"## {nm}", "", "| episode | " + " | ".join(refs) + " |", "|---|" + "---|" * len(refs)]
        for ep, v in res[d].items():
            L.append(f"| {ep} | " + " | ".join((f"{v[g]['auc']:.3f} / {v[g]['final']:.3f} / {v[g]['cov20']:.3f}" if g in v else "—") for g in refs) + " |")
        L.append(f"| **均值** | " + " | ".join((f"**{mean[d][g]['auc']:.3f} / {mean[d][g]['final']:.3f} / {mean[d][g]['cov20']:.3f}**" if g in mean[d] else "—") for g in refs) + " |")
        L += ["", "AUC / Final / Cov@20。V5.2 轨迹诊断：", ""]
        for ep, v in res[d].items():
            for g in NEW + ("V52L0", "V52L01"):
                if g in v and "steps_logged" in v[g]:
                    x = v[g]
                    L.append(f"- {ep} {g}: 守卫调用 {x['guard_calls_total']}，λ 翻转率 {x['lambda_flip_rate']:.3f}，被选候选潜力占比 {x['pot_share_of_argmax_mean']:.3f}，可见项全 0 的步 {x['zero_visible_steps']}，"
                             f"打分 {x['score_s_mean']:.2f} s/步" + (f"，停滞窗 {x['stall30']}，位置回访率 {x['revisit_position']:.3f}" if "stall30" in x else ""))
        L.append("")
    L += ["V5.2-M 平均 AUC 减参照（同 episode）：" + json.dumps({d: {g: {k: round(x, 3) for k, x in v.items()} for g, v in r.items()} for d, r in rule.items()}, ensure_ascii=False), "",
          f"**smoke 有效性（FROZEN_PROTOCOL §5：12 个 episode 都在、步数完整、守卫 0）：{'有效' if res['valid'] else '无效'}**" + ("" if res["valid"] else "；问题：" + "，".join(f"{c['episode']} {c['group']}" for c in chk if not c["ok"]))]
    open(os.path.join(EM, "G3_SMOKE.md"), "w").write("\n".join(L) + "\n")
    print(json.dumps(mean, indent=1), res['valid'])


if __name__ == "__main__":
    main()
