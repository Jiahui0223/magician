"""EXP6 V3 / H -- G2 fixed-state diagnostics (seen2scene env, 1 GPU per process; diagnostic only, never a G3 gate).
States: the Macarons++ Dev-3 fixed states of E1.2 (strata B0 / B2; e2_fixed.states('dev3', stratum)), with their legal
candidates and evaluation-side oracle one-step coverage gains (oracle.npz, read only AFTER scoring, for the metrics).
Replay = the closed-loop planner state of V3-T1: P1 fusion of the b2 mode (D100 model input + sparse S10 twin for
band_dil), records 0..t integrated, R10 refresh (V3 encode of the D100 partial TSDF grid), K = 4 noise maps
sha_seed(rp.key(t_r, k)) (the fixed-state keys of E2 / V3 G2), then ONE frozen score_state(keep_hits=True) per state:
the same V3 predictions feed every policy:
  H0 = frozen DistP1 (K mean; checked == the h0 per-sample recomputation), H1 = coverage-aligned value (K mean),
  H2 = mean - 0.25 std, H3 = two-step score of every first-step candidate (second-layer views predicted on the same
  cached features / noise; h_policy.h3_scores), extra: lambda 0 and 0.5 variants of H2 (fixed-state sensitivity only).
Rows: work_dirs/exp6_v3/H/eval_fixed/dev3/<stratum>/<tag>.jsonl (resumable). Analysis: --analyze -> results/
fixed_state_scores.csv, reports/G2_FIXED_STATE.md.
Usage: python eval_fixed.py --stratum B0|B2 [--part i/n] | python eval_fixed.py --analyze"""
import argparse
import csv
import glob
import json
import os
import sys
import time

HERE = os.path.dirname(os.path.abspath(__file__))
H = os.path.dirname(HERE)
R = os.path.dirname(os.path.dirname(os.path.dirname(H)))
X_ = os.path.join(R, "experiments")
sys.path.insert(0, os.path.join(X_, "exp5_future_view", "task3_full_depth_p1", "e2_pilot", "scripts"))
import e2_fixed as E2F  # noqa: E402

W = os.path.join(R, "work_dirs", "exp6_v3", "H", "eval_fixed", "dev3")
POL = ("H0", "H1", "H2", "H3", "H2_l0", "H2_l05")


def run(a):
    X = E2F.X
    X.setup(a.stratum)
    sys.path.insert(0, os.path.join(X_, "exp5_future_view", "task2", "scripts"))
    sys.path.insert(0, os.path.join(X_, "exp5_future_view", "task3_full_depth_p1", "scripts"))
    sys.path.insert(0, os.path.join(X_, "exp6_v3", "shared"))
    sys.path.insert(0, HERE)
    import p1_full as PF
    PF.patch("b2")
    import planner_v3 as PV3
    PC = PV3.install()
    import m_p1
    import h_policy as HP
    import yaml
    from seen2scene.futureview import predictor as PR
    from seen2scene.futureview import rays as RY
    from seen2scene.futureview import scoring as SCO
    np, torch, MC = X.np, X.torch, X.MC
    torch.use_deterministic_algorithms(True, warn_only=True)
    dev = torch.device("cuda")
    cfg = yaml.safe_load(open(os.path.join(H, "FROZEN_CONFIG.yaml")))
    core = PV3.V3Core(os.path.join(R, cfg["model"]["checkpoint"]), dev, expect_sha=cfg["model"]["sha256"])
    c1, c3 = cfg["policies"]["h1"], cfg["policies"]["h3"]
    lam = float(cfg["policies"]["h2"]["lambda_risk"])
    od = os.path.join(W, a.stratum)
    os.makedirs(od, exist_ok=True)
    pi, pn = map(int, a.part.split("/"))
    for tag, sts in E2F.states("dev3", a.stratum)[pi::pn]:
        po = os.path.join(od, f"{tag}.jsonl")
        if os.path.exists(po):
            continue
        tb = time.time()
        rp = E2F.Rep("dev3", sts[0], dev)
        fr, dims, dom = rp.frame
        byt = {s["t"]: s for s in sts}
        T = max(byt)
        recs = [MC.load_service_record(rp.svc(t)) for t in range(T + 1)]
        HP.STATE.update({"obs": [], "tree": None, "n_tree": -1, "rec": None, "visited": set()})
        z = np.load(os.path.join(cfg["pose_tables"], f"{sts[0]['scene']}.npz"))
        HP.STATE["pose"] = {"lut": {tuple(int(x) for x in r): j for j, r in enumerate(z["idx"])}, "R": z["R"], "T": z["T"], "C": z["C"], "grid": json.loads(str(z["grid"]))}
        fus = m_p1.P1Fusion(fr, dims, dom, device=dev)
        rows, h, eps = [], None, None
        for t in range(T + 1):
            HP.capture(recs[t])
            fus.integrate_record(recs[t])
            if t % 10 == 0:
                geo = PC.Geometry.from_fusion(fr, dims, dom, fus, dev)
                h = core.pred.encode(geo.coarse())
                eps = PR.noise(dev, 4, lambda kk, tr=t: RY.sha_seed(rp.key(tr, kk)))
            if t not in byt:
                continue
            st, rec = byt[t], recs[t]
            geo = PC.Geometry.from_fusion(fr, dims, dom, fus, dev)
            t0 = time.time()
            out, tm, views = PC.score_state(core, geo, h, rec["legal_R"], rec["legal_T"], eps, rp.z0, keep_hits=True)
            t_frozen = time.time() - t0
            K = eps.shape[0]
            s0 = [[HP.value_h0(SCO, out["hits"][m]["t"][k], v, geo, rp.z0) for k in range(K)] for m, v in enumerate(views)]
            s1 = [[HP.value_cov(out["hits"][m]["t"][k], v, geo, rp.z0, c1) for k in range(K)] for m, v in enumerate(views)]
            g0 = np.asarray([[float(x[1].sum()) for x in r] for r in s0])
            g1 = np.asarray([[float(x[1].sum()) for x in r] for r in s1])
            h0r = [float(np.mean(list(r))) for r in g0]                            # recomputation (numpy sums)
            h0 = list(out["distp1"])                                               # the frozen DistP1 itself (torch sums)
            rel = max(abs(x - y) / max(abs(y), 1e-12) for x, y in zip(h0r, h0))
            assert rel <= 1e-12 and int(np.argmax(h0r)) == int(np.argmax(h0)), ("h0 recomputation differs beyond float rounding", rel)
            mu, sd = np.asarray([float(np.mean(list(r))) for r in g1]), g1.std(1)
            t1 = time.time()
            s3, info3 = HP.h3_scores(core, geo, h, rec["legal_R"], rec["legal_T"], eps, rp.z0, s1, mu - lam * sd, lam, 2, PC.score_state, c3, {})
            t_h3 = time.time() - t1
            sc = {"H0": h0, "H1": mu.tolist(), "H2": (mu - lam * sd).tolist(), "H3": [float(x) for x in s3], "H2_l0": (mu - 0.0 * sd).tolist(), "H2_l05": (mu - 0.5 * sd).tolist()}
            rows.append({"sid": st["sid"], "scene": st["scene"], "start": st["start"], "t": t, "tag": tag, "stratum": a.stratum, "n_legal": len(views),
                         "scores": sc, "h1_std": sd.tolist(), "h0_recompute_max_rel": rel, "h1_value_k": g1.tolist(), "h0_value_k": g0.tolist(), "h3": info3,
                         "timing": {"frozen_score_state_s": t_frozen, "h3_extra_s": t_h3}})
            del geo, out, views
        with open(po + ".tmp", "w") as f:
            for r in rows:
                f.write(json.dumps(r, default=float) + "\n")
        os.replace(po + ".tmp", po)
        print("dev3", a.stratum, tag, len(rows), "%.0fs" % (time.time() - tb), flush=True)
        torch.cuda.empty_cache()


def analyze():
    import numpy as np
    from scipy.stats import spearmanr
    sys.path.insert(0, os.path.join(X_, "exp5_future_view", "ablation_study", "common"))
    X = E2F.X
    X.setup("B0")
    import ab_load as AL
    rows = []
    for st in ("B0", "B2"):
        for f in sorted(glob.glob(os.path.join(W, st, "*.jsonl"))):
            rows += [json.loads(l) for l in open(f)]
    m0 = {s["sid"]: s for s in json.load(open(os.path.join(X.ORIG_W, "manifests", "states_outdoor.json")))["states"]}
    m2 = {s["sid"]: s for s in json.load(open(os.path.join(X.P.ROOT_B2, "manifests", "states_outdoor.json")))["states"]}
    TAU = 1e-4 * 100
    recs = []
    for r in rows:
        st = m0[r["sid"]] if r["stratum"] == "B0" else m2[r["sid"]]
        root = X.ORIG_W if r["stratum"] == "B0" else X.P.ROOT_B2
        o = np.load(os.path.join(root, "states", "outdoor", r["sid"], "oracle.npz"))
        C = np.asarray(st["cand"], int)
        g = np.asarray(o["gain"], float)[C] * 100.0
        src = next(e for e in st["sources"] if e["source"] == "V2M")
        rec = {k: np.asarray(x) for k, x in np.load(os.path.join(R, src["svc"])).items() if k in ("legal_idx", "executed_pose_idx_history")}
        loc = {int(c): i for i, c in enumerate(C)}
        x = {"sid": r["sid"], "scene": r["scene"], "t": r["t"], "stratum": r["stratum"], "oracle": float(g.max())}
        for p in POL:
            s = np.asarray(r["scores"][p], float)
            pk, fb = AL.pick_of("outdoor", st, s, rec, "V2M")
            gp = g[loc[pk]] if pk in loc else np.nan
            rho = spearmanr(s[C], g).correlation if len(C) > 2 and np.std(s[C]) > 0 and np.std(g) > 0 else np.nan
            srt = np.sort(s[C])[::-1]
            x.update({f"{p}_pick": int(pk), f"{p}_fallback": bool(fb), f"{p}_gain": float(gp), f"{p}_top1": float(gp >= g.max() - TAU), f"{p}_regret": float(g.max() - gp),
                      f"{p}_rho": float(rho), f"{p}_margin": float((srt[0] - srt[1]) / max(abs(srt[0]), 1e-9)) if len(srt) > 1 else np.nan})
        sd = np.asarray(r["h1_std"], float)[C]
        mu = np.asarray(r["scores"]["H1"], float)[C]
        x["rho_std_gain"] = float(spearmanr(sd, g).correlation) if len(C) > 2 and np.std(sd) > 0 and np.std(g) > 0 else np.nan
        cv = sd / np.maximum(mu, 1e-9)
        x["rho_cv_gain"] = float(spearmanr(cv, g).correlation) if len(C) > 2 and np.std(cv) > 0 and np.std(g) > 0 else np.nan
        x["h3_c2_unique"] = r["h3"].get("n_c2_unique")
        recs.append(x)
    with open(os.path.join(H, "results", "fixed_state_scores.csv"), "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(recs[0]))
        w.writeheader()
        w.writerows(recs)
    L = ["# G2_FIXED_STATE — Dev-3 固定状态单步诊断（自动生成，`scripts/eval_fixed.py --analyze`）", "",
         "**说明**：",
         "- 状态：E1.2 的 Dev-3 固定状态（B0 层与 B2 层）。",
         "- 所有策略都使用同一次冻结的 V3-T1 预测（D100 输入、R10、K = 4、固定状态的噪声键）。",
         "- oracle 真实一步覆盖增益只在打分之后用于评估。选择规则与闭环相同（`m_score.choose`）。",
         "- H3 列是它的**第一步**选择；H3 优化的是两步目标，单步增益本来就不是它的目标。",
         "- 本诊断**不是 G3 的门槛**。", ""]
    from collections import defaultdict
    for ph, sel in (("全部", lambda x: True), ("t ≤ 20", lambda x: x["t"] <= 20), ("t > 20", lambda x: x["t"] > 20)):
        sub = [x for x in recs if sel(x)]
        L += [f"## {ph}（{len(sub)} 个状态）", "", "| 策略 | 所选增益（pp） | Top-1 | Regret（pp） | Spearman ρ | 与 H0 选择不同的比例 | 分数 margin 中位数 | 回退比例 |", "|---|---|---|---|---|---|---|---|"]
        for p in POL:
            chg = np.mean([x[f"{p}_pick"] != x["H0_pick"] for x in sub])
            L.append(f"| {p} | {np.nanmean([x[f'{p}_gain'] for x in sub]):.3f} | {np.nanmean([x[f'{p}_top1'] for x in sub]):.3f} | {np.nanmean([x[f'{p}_regret'] for x in sub]):.3f} | "
                     f"{np.nanmean([x[f'{p}_rho'] for x in sub]):.3f} | {chg:.3f} | {np.nanmedian([x[f'{p}_margin'] for x in sub]):.3f} | {np.mean([x[f'{p}_fallback'] for x in sub]):.3f} |")
        # paired by scene (scene-level mean of per-state differences) for the nested steps
        L += ["", "| 配对（同一状态） | Δ增益（pp），场景均值 | 胜 / 负场景 | Δ增益（pp），状态均值 |", "|---|---|---|---|"]
        for a_, b_ in (("H1", "H0"), ("H2", "H1"), ("H3", "H2"), ("H3", "H0")):
            per = defaultdict(list)
            for x in sub:
                per[x["scene"]].append(x[f"{a_}_gain"] - x[f"{b_}_gain"])
            scm = {s: float(np.nanmean(v)) for s, v in per.items()}
            L.append(f"| {a_} − {b_} | {np.mean(list(scm.values())):+.3f} | {sum(v > 0 for v in scm.values())}/{sum(v < 0 for v in scm.values())} | "
                     f"{np.nanmean([x[f'{a_}_gain'] - x[f'{b_}_gain'] for x in sub]):+.3f} |")
        L.append("")
    sw = [x for x in recs if x["H2_pick"] != x["H1_pick"]]
    L += ["## H2 的风险项：什么时候改变选择", "",
          f"- 风险项（λ = 0.25）改变 H1 选择的状态：{len(sw)} / {len(recs)}；在这些状态上，所选真实增益 H2 − H1 的均值为 {np.nanmean([x['H2_gain'] - x['H1_gain'] for x in sw]) if sw else float('nan'):+.3f} pp。",
          f"- λ = 0.5 改变选择的状态：{sum(x['H2_l05_pick'] != x['H1_pick'] for x in recs)}；λ = 0 与 H1 选择不同的状态：{sum(x['H2_l0_pick'] != x['H1_pick'] for x in recs)}（应为 0）。",
          f"- 候选间 Spearman(std_k, 真实增益) 的均值为 {np.nanmean([x['rho_std_gain'] for x in recs]):+.3f}，Spearman(std / mean, 真实增益) 的均值为 {np.nanmean([x['rho_cv_gain'] for x in recs]):+.3f}。",
          "  - 第一项为正，说明样本分歧大的候选，真实增益往往也大：惩罚分歧有可能把真实的高增益区域一起压低；",
          "  - 第二项为负，说明相对分歧大的候选，真实增益偏低。",
          f"- H3 每个状态的第二层唯一视角数：均值 {np.nanmean([x['h3_c2_unique'] for x in recs]):.1f}。"]
    open(os.path.join(H, "reports", "G2_FIXED_STATE.md"), "w").write("\n".join(L) + "\n")
    print("states", len(recs))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--stratum", choices=["B0", "B2"])
    ap.add_argument("--part", default="0/1")
    ap.add_argument("--analyze", action="store_true")
    a = ap.parse_args()
    if a.analyze:
        analyze()
    else:
        run(a)


if __name__ == "__main__":
    main()
