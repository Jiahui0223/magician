"""Experiment C -- G0 judgement (CPU, seen2scene env) over the smoke runs of tests/g0_run.sh (protocol §5):
1 single-view GT, 2 GT-DistP1 regression vs Experiment A, 3 oracle gain vs EXP5 oracle.npz + executed check, 4 no
leakage, 5 legality, 6 monotonicity, 7 determinism, 8 baseline replay vs stored B2, 9 smoke / online mono == replay.
Output report/C_G0_VALIDATION.md and results/g0_check.json. Usage: python g0_check.py [--g0 DIR]"""
import argparse
import hashlib
import json
import os
import sys

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
EXPC = os.path.dirname(HERE)
sys.path.insert(0, os.path.join(EXPC, "scripts"))
import c_common as CC  # noqa: E402

sys.path.insert(0, os.path.join(CC.EXP5, "ablation_study", "common"))
import ab_common as A  # noqa: E402

W3 = os.path.join(CC.REPO, "work_dirs", "exp5_future_view", "task3_full_depth_p1")
R = {}          # name -> {"pass": bool, "detail": str}


def rec_(name, ok, detail):
    R[name] = {"pass": bool(ok), "detail": detail}
    print(("PASS " if ok else "FAIL ") + name + ": " + detail, flush=True)


def jl(p):
    return [json.loads(x) for x in open(p)]


def pool_state(dataset, scene, start, t=0):
    man = A.load_manifest(dataset)
    for st in man["states"]:
        if st["scene"] == scene and int(st["start"]) == int(start) and int(st["t"]) == t:
            return st
    return None


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--g0", default=CC.WG0)
    a = ap.parse_args()
    G0 = a.g0
    tag_in = "Z6MFQCViBuw_L0_S6"
    scene_in = "Z6MFQCViBuw_L0"
    din = {g: os.path.join(G0, "indoor_a", g, tag_in) for g in ("C1", "C2", "B2replay")}
    din_b = os.path.join(G0, "indoor_b", "C1", tag_in)
    ep = {g: json.load(open(os.path.join(d, "episode.json"))) for g, d in din.items()}
    pl = {g: jl(os.path.join(d, "planner.jsonl")) for g, d in din.items()}
    st_in = pool_state("indoor", scene_in, 6)
    cache_in = json.load(open(os.path.join(CC.WA, "cache", "cl", "indoor", st_in["sid"] + ".json")))
    orc_in = np.load(os.path.join(A.state_dir("indoor", st_in["sid"]), "oracle.npz"))
    rec0 = np.load(os.path.join(din["C1"], "svc", "svc_t0.npz"))
    cq0 = np.load(os.path.join(din["C1"], "cq", "cq_t0.npz"))
    # ---- outdoor files
    dout = {g: os.path.join(G0, "outdoor_a", g) for g in ("C1", "C2")}
    epo = {g: json.load(open(os.path.join(dout[g], "episodes", f"ep_{CC.tag_out(g, 'barts', 0)}.json"))) for g in dout}
    plo = {g: json.load(open(os.path.join(dout[g], "planner_logs", f"pl_{CC.tag_out(g, 'barts', 0)}.json"))) for g in dout}
    st_out = pool_state("outdoor", "barts", 0)
    cache_out = json.load(open(os.path.join(CC.WA, "cache", "cl", "outdoor", st_out["sid"] + ".json")))
    orc_out = np.load(os.path.join(A.state_dir("outdoor", st_out["sid"]), "oracle.npz"))
    cqo0 = np.load(os.path.join(dout["C1"], "cq", CC.tag_out("C1", "barts", 0), "cq_t0.npz"))
    reco0 = np.load(os.path.join(dout["C1"], "svc", CC.tag_out("C1", "barts", 0), "svc_barts_s0_t0.npz"))

    # 1. single-view GT
    ts = cq0["tstar"]
    ok1 = ts.shape == (len(rec0["legal_idx"]), 7296) and ts.dtype == np.float32 and np.all(ts[np.isfinite(ts)] > 0) and 0 < np.isfinite(ts).mean() < 1
    same_legal = np.array_equal(np.asarray(st_in["legal_idx"]), rec0["legal_idx"])
    eq_ts = same_legal and np.array_equal(ts, np.asarray(orc_in["tstar"], np.float32))
    tso = cqo0["tstar"]
    same_legal_o = np.array_equal(np.asarray(st_out["legal_idx"]), reco0["legal_idx"])
    eq_tso = same_legal_o and np.array_equal(tso, np.asarray(orc_out["tstar"], np.float32))
    rec_("1_single_view_gt", ok1 and eq_ts and eq_tso,
         f"indoor t* shape {ts.shape} hit {np.isfinite(ts).mean():.3f}, legal == Pool CL t=0 {same_legal}, t* == EXP5 oracle.npz bitwise {eq_ts}; "
         f"outdoor shape {tso.shape} hit {np.isfinite(tso).mean():.3f}, legal == {same_legal_o}, t* bitwise {eq_tso}")

    # 2. GT-DistP1 regression vs Experiment A (t = 0 state = Pool CL state)
    def reg(s_new, s_ref):
        s_new, s_ref = np.asarray(s_new, float), np.asarray(s_ref, float)
        bit = np.array_equal(s_new, s_ref)
        rel = float(np.max(np.abs(s_new - s_ref) / np.maximum(1e-300, np.abs(s_ref)))) if len(s_ref) else 0.0
        return bit, rel, int(np.argmax(s_new)) == int(np.argmax(s_ref))
    b_in, rel_in, am_in = reg(pl["C1"][0]["scores"], cache_in["gtd_distp1"])
    b_out, rel_out, am_out = reg(plo["C1"]["steps"][0]["scores"], cache_out["gtd_distp1"])
    rec_("2_gtd_regression_vs_A", (b_in or (rel_in <= 1e-9 and am_in)) and (b_out or (rel_out <= 1e-9 and am_out)),
         f"indoor bitwise {b_in} (max rel {rel_in:.1e}, argmax same {am_in}); outdoor bitwise {b_out} (max rel {rel_out:.1e}, argmax same {am_out})")

    # 3. oracle gain vs EXP5 oracle.npz (t = 0) + executed checks
    g_in = np.load(os.path.join(din["C2"], "cq", "cq_t0.npz"))["gain"]
    g_out = np.load(os.path.join(dout["C2"], "cq", CC.tag_out("C2", "barts", 0), "cq_t0.npz"))["gain"]
    eq_gin = np.array_equal(g_in, np.asarray(orc_in["gain"]))
    eq_gout = np.array_equal(g_out, np.asarray(orc_out["gain"]))
    eq_goff = np.allclose(np.asarray(epo["C2"]["steps"][0]["planner"]["c_diag"]["gain_official"]), np.asarray(orc_out["gain_official"]) * 1.0, atol=0, rtol=0)
    ck_in = {g: ep[g]["checks"] for g in ("C1", "C2")}
    ck_out = {g: epo[g]["planner_bye"]["checks"] for g in ("C1", "C2")}
    ex_ok = all(c["executed_sample_equal"] == c["executed_gain_equal"] == c["n"] for c in ck_in.values()) and \
        all(c["executed_sample_equal"] == c["executed_official_equal"] == c["executed_mono_equal"] == c["n"] for c in ck_out.values())
    rec_("3_oracle_gain", eq_gin and eq_gout and eq_goff and ex_ok,
         f"indoor t=0 gains == oracle.npz {eq_gin}; outdoor mono {eq_gout}, official {eq_goff}; executed-candidate checks indoor {ck_in} outdoor {ck_out}")

    # 4. no leakage
    hash_ok = all(ep[g]["checks"]["state_hash_equal"] == ep[g]["checks"]["n"] for g in ("C1", "C2")) and \
        all(ck_out[g]["state_hash_equal"] == ck_out[g]["n"] for g in ("C1", "C2"))
    keys_ok = True
    for g in ("C1", "C2"):
        for f in os.listdir(os.path.join(din[g], "cq")):
            keys_ok &= set(np.load(os.path.join(din[g], "cq", f)).files) == set(CC.CQ_KEYS[g])
        for f in os.listdir(os.path.join(dout[g], "cq", CC.tag_out(g, "barts", 0))):
            keys_ok &= set(np.load(os.path.join(dout[g], "cq", CC.tag_out(g, "barts", 0), f)).files) == set(CC.CQ_KEYS[g])
    denied = sum(len(x["guard_denied"]) for g in ("C1", "C2") for x in pl[g]) + sum(len(plo[g]["guard"]["denied"]) for g in ("C1", "C2"))
    cq_reads_in = [x["cq_reads"] for x in pl["C1"]]
    cq_reads_out = plo["C1"]["cq_reads"]
    # repeated query of the same state: diag gains (computed after the answer) == cq gains (computed before) for C2
    rep_ok = all(np.array_equal(np.asarray(s["diag"]["gain"]), np.load(os.path.join(din["C2"], "cq", CC.cq_name(s["t"])))["gain"]) for s in ep["C2"]["steps"])
    rec_("4_no_leakage", hash_ok and keys_ok and denied == 0 and rep_ok,
         f"state hash unchanged at every query {hash_ok}; cq keys minimal {keys_ok}; guard denials {denied}; cq reads indoor cumulative per step {cq_reads_in}, outdoor total {cq_reads_out}; repeated-query identical {rep_ok}")

    # 5. legality
    leg_ok = all(ep[g]["checks"]["pick_in_candidates"] == ep[g]["checks"]["n"] for g in ("C1", "C2"))
    leg_out = all(all(s["pick"] is None or s["pick"] < s["n_legal"] for s in epo[g]["steps"][:-1]) for g in ("C1", "C2"))
    rec_("5_legality", leg_ok and leg_out, f"indoor pick in prefer-unvisited set every step {leg_ok}; outdoor pick < n_legal {leg_out}")

    # 6. monotonicity
    mono_in = all(np.all(np.diff(ep[g]["coverage"]) >= 0) for g in ("C1", "C2", "B2replay"))
    mets = {g: json.load(open(os.path.join(G0, "outdoor_a", "metrics", g, CC.tag_out(g, "barts", 0) + ".json"))) for g in ("C1", "C2")}
    mono_out = all(np.all(np.diff(m["cov_mono"]) >= 0) for m in mets.values())
    rec_("6_monotonicity", mono_in and mono_out, f"C_obs non-decreasing {mono_in}; outdoor cov_mono non-decreasing {mono_out}; official max drop " +
         ", ".join(f"{g} {m['official_max_drop']:.4f}" for g, m in mets.items()))

    # 7. determinism
    epb = json.load(open(os.path.join(din_b, "episode.json")))
    plb = jl(os.path.join(din_b, "planner.jsonl"))
    det_in = epb["coverage"] == ep["C1"]["coverage"] and epb["executed"] == ep["C1"]["executed"] and [x["scores"] for x in plb] == [x["scores"] for x in pl["C1"]]
    epo_b = json.load(open(os.path.join(G0, "outdoor_b", "C2", "episodes", f"ep_{CC.tag_out('C2', 'barts', 0)}.json")))
    det_out = epo_b["coverage"] == epo["C2"]["coverage"] and epo_b["executed_keys"] == epo["C2"]["executed_keys"] and \
        [s["planner"]["scores"] for s in epo_b["steps"][:-1]] == [s["planner"]["scores"] for s in epo["C2"]["steps"][:-1]]
    rec_("7_determinism", det_in and det_out, f"indoor C1 twice identical {det_in}; outdoor C2 twice identical {det_out}")

    # 8. baseline replay vs stored B2
    ref_in = json.load(open(os.path.join(W3, "closed_loop", "indoor", "B2", "V2M", tag_in, "episode.json")))
    ref_pl = jl(os.path.join(W3, "closed_loop", "indoor", "B2", "V2M", tag_in, "planner.jsonl"))
    n = ep["B2replay"]["moves"]
    rep_in = ref_in["coverage"][:n + 1] == ep["B2replay"]["coverage"] and ref_in["executed"][:n + 1] == ep["B2replay"]["executed"]
    sc_in = [x["scores"] for x in ref_pl[:n]] == [x["scores"] for x in pl["B2replay"]]
    ref_out = json.load(open(os.path.join(W3, "closed_loop", "outdoor", "B2", "V2M", "episodes", "ep_V2M_barts_s0_r0.json")))
    ref_plo = json.load(open(os.path.join(W3, "closed_loop", "outdoor", "B2", "V2M", "planner_logs", "pl_V2M_barts_s0_r0.json")))
    epr = json.load(open(os.path.join(G0, "outdoor_rep", "V2M", "episodes", "ep_V2M_barts_s0_r0.json")))
    plr = json.load(open(os.path.join(G0, "outdoor_rep", "V2M", "planner_logs", "pl_V2M_barts_s0_r0.json")))
    m = epr["moves"]
    rep_out = ref_out["coverage"][:m + 1] == epr["coverage"] and ref_out["executed_keys"][:m + 1] == epr["executed_keys"]
    sc_out = [s["scores"] for s in ref_plo["steps"][:m]] == [s["scores"] for s in plr["steps"]]
    rec_("8_baseline_replay", rep_in and sc_in and rep_out and sc_out,
         f"indoor B2 {n} moves: coverage+executed {rep_in}, scores bitwise {sc_in}; outdoor B2 {m} moves: coverage+keys {rep_out}, scores bitwise {sc_out}")

    # 9. smoke + online mono == replay
    fin = all(np.all(np.isfinite(ep[g]["coverage"])) and all(np.all(np.isfinite(x["scores"])) for x in pl[g]) for g in ("C1", "C2"))
    fin_o = all(np.all(np.isfinite(epo[g]["coverage"])) and all(np.all(np.isfinite(s["scores"])) for s in plo[g]["steps"]) for g in ("C1", "C2"))
    online = {g: [s["planner"]["c_diag"]["mono_cov_t"] for s in epo[g]["steps"][:-1]] for g in ("C1", "C2")}
    mono_eq = all(online[g] == mets[g]["cov_mono"][:len(online[g])] for g in ("C1", "C2"))
    off_eq = all(mets[g]["official_equal_stored"] for g in ("C1", "C2"))
    rec_("9_smoke", fin and fin_o and mono_eq and off_eq, f"finite indoor {fin} outdoor {fin_o}; online mono == t25 replay {mono_eq}; official replay == stored {off_eq}; "
         f"walls indoor C1 {ep['C1']['wall_s']:.0f}s C2 {ep['C2']['wall_s']:.0f}s, outdoor C1 {epo['C1']['seconds_this_process']:.0f}s C2 {epo['C2']['seconds_this_process']:.0f}s")

    # ---- timing summary for the resource estimate
    tim = {"indoor": {g: {"decision_s": float(np.mean([s["decision_s"] for s in ep[g]["steps"]])), "gt_query_s": float(np.mean([s["gt_query_s"] for s in ep[g]["steps"]])),
                          "diag_s": float(np.mean([s["diag"]["diag_s"] for s in ep[g]["steps"] if "diag" in s])), "roundtrip_s": float(np.mean([s["roundtrip_s"] for s in ep[g]["steps"]])),
                          "peak_gb_child": float(max(x["peak_mem_gb"] for x in pl[g]))} for g in ("C1", "C2")},
           "outdoor": {g: {"decision_s": float(np.mean([s["decision_s"] for s in plo[g]["steps"]])), "gt_query_s": float(np.mean([s["planner"]["c_diag"]["gt_query_s"] for s in epo[g]["steps"][:-1]])),
                           "diag_s": float(np.mean([s["planner"]["c_diag"].get("diag_s", 0) for s in epo[g]["steps"][:-1]])), "peak_gb_child": plo[g]["peak_mem_gb_this_process"],
                           "sim_peak_gb": epo[g]["sim_peak_mem_gb_this_process"]} for g in ("C1", "C2")}}
    all_pass = all(v["pass"] for v in R.values())
    os.makedirs(os.path.join(EXPC, "results"), exist_ok=True)
    json.dump({"checks": R, "timing": tim, "all_pass": all_pass, "g0_dir": G0}, open(os.path.join(EXPC, "results", "g0_check.json"), "w"), indent=1)
    L = ["# Experiment C — G0 校验（自动生成 `tests/g0_check.py`；冒烟运行 `tests/g0_run.sh`）", "",
         f"运行目录 `{os.path.relpath(G0, CC.REPO)}`；室内冒烟 = Z6MFQCViBuw_L0 起点 6，5 步；室外 = barts 起点 0，3 步。**总判定：{'PASS' if all_pass else 'FAIL'}**", "",
         "| # | 检查 | 结果 | 细节 |", "|---|---|---|---|"]
    for k, v in R.items():
        L.append(f"| {k.split('_')[0]} | {k[2:]} | {'✅ PASS' if v['pass'] else '❌ FAIL'} | {v['detail']} |")
    L += ["", "## 冒烟计时（每步均值；共享 GPU，非独占）", "", "| 数据集 | 组 | 子进程决策 s | GT 查询 s | 诊断 s | 子进程峰值显存 GB |", "|---|---|---|---|---|---|"]
    for d in ("indoor", "outdoor"):
        for g in ("C1", "C2"):
            t = tim[d][g]
            L.append(f"| {d} | {g} | {t['decision_s']:.2f} | {t['gt_query_s']:.2f} | {t['diag_s']:.2f} | {t['peak_gb_child']:.1f} |")
    L += ["", "- 检查 2 的参照 = Experiment A 缓存 `work_dirs/exp6_v3/ablation_study/cache/cl/<dataset>/<sid>.json: gtd_distp1`（t=0 状态与 Pool CL 同 sid）；检查 1 / 3 的参照 = EXP5 `states/<dataset>/<sid>/oracle.npz`。",
          "- 室外官方覆盖允许下降（MAGICIAN 单元格上限规则），`official_max_drop` 仅记录。"]
    open(os.path.join(EXPC, "report", "C_G0_VALIDATION.md"), "w").write("\n".join(L) + "\n")
    print("ALL PASS" if all_pass else "SOME FAIL")
    return 0 if all_pass else 1


if __name__ == "__main__":
    sys.exit(main())
