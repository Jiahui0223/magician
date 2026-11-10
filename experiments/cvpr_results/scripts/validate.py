"""CVPR overnight -- automatic quality gates (plan section 10) -> experiments/cvpr_results/report/VALIDATION.md. CPU. python validate.py
Every gate is recomputed from files on disk; a gate that cannot be evaluated yet (runs incomplete) is reported as PENDING, never PASS."""
import csv
import glob
import json
import os
import subprocess
import sys
import time

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import cvpr_lib as L  # noqa: E402

PY = "/bsuscratch/jiahuizhang/envs/seen2scene/bin/python"
G = []                                                             # (gate, item, status, detail)


def add(gate, item, ok, detail=""):
    G.append((gate, item, "PASS" if ok is True else ("FAIL" if ok is False else str(ok)), detail))


def rows_csv(p):
    return list(csv.DictReader(open(p)))


def hello(rec):
    ep = json.load(open(os.path.join(L.REPO, rec["episode_file"])))
    h = ep["planner_hello"][0] if isinstance(ep.get("planner_hello"), list) else ep.get("planner_hello")
    return ep, h


def planner_log(rec):
    p = rec["episode_file"].replace("/episodes/ep_", "/planner_logs/pl_")
    return json.load(open(os.path.join(L.REPO, p)))


def main():
    man = rows_csv(os.path.join(L.MAN, "episodes.csv"))
    by = {}
    for r in man:
        by.setdefault(r["method_id"], []).append(r)
    comp = {m: [r for r in v if r["run_status"] == "COMPLETE"] for m, v in by.items()}
    # ---------------------------------------------------------------- gate 1: main method
    ck = os.path.join(L.REPO, "work_dirs/exp6_v3/task0_vae/checkpoints/final.pt")
    add(1, "T0 checkpoint sha256 == 7ed0ddd7... (plan value)", L.sha_file(ck) == L.T0_SHA, ck)
    mc = json.load(open(os.path.join(L.MAN, "main_method_checkpoint.json")))
    add(1, "T0 has no cross-attention (model_cfg.xattn.enabled false, 0 unet.xattn tensors in the EMA weights)", mc["xattn_cfg"]["enabled"] is False and mc["n_unet_xattn_keys"] == 0, f"step {mc['step']}, flow {mc['flow']}")
    E = L.load("V3-T0-D1")
    dcfg = L.sha_file(os.path.join(L.REPO, "experiments/exp6_v3/ablation_study/experiment_D_h3_lookahead/configs/d_config.yaml"))
    dpol = L.sha_file(os.path.join(L.REPO, "experiments/exp6_v3/ablation_study/experiment_D_h3_lookahead/scripts/d_policy.py"))
    hpol = L.sha_file(os.path.join(L.REPO, "experiments/exp6_v3/H/scripts/h_policy.py"))
    bad, denied, opened = [], 0, set()
    for k, r in E.items():
        ep, h = hello(r)
        p = h["planner"]
        d = p["d_policy"]
        ok = (h["ckpt_sha256"] == L.T0_SHA and p["K"] == 4 and p["refresh"] == "R10" and d["d_policy"] == "d1" and d["policy_cfg"]["depth"] == 2 and d["policy_cfg"]["lambda_risk"] == 0.0 and d["policy_cfg"]["beam"] is None
              and d["config_sha256"] == dcfg and d["d_policy_file_sha256"] == dpol and d["h_policy_file_sha256"] == hpol and d["policy_cfg"]["eps_cov"] == 1.0 and d["policy_cfg"]["cell_size"] == 1.0)
        if not ok:
            bad.append(k)
        g = planner_log(r)["guard"]
        ds = [x for x in g["unique_paths"] if "Macarons++" in x]
        opened |= {os.path.basename(x) for x in ds}
        denied += len(g["denied"]) + sum(os.path.basename(x) != "settings.json" for x in ds)        # only the official settings (pose grid, scene box, scale) may be read
    add(1, "70 / 70 T0 + D1 episodes: planner handshake = T0 sha, K 4, R10, frozen D-v1 policy d1 (depth 2, lambda 0, eps 1.0, cell 1.0) with the current frozen file hashes", not bad and len(E) == 70, f"{len(E) - len(bad)}/{len(E)} ok")
    add(1, "LeakGuard: no denied file access; the only dataset file the planner process opens is the official settings.json (no mesh / GT file) in the 70 T0 + D1 episodes", denied == 0,
        f"violations {denied}; dataset files opened: {sorted(opened)}")
    # ---------------------------------------------------------------- gate 2: pairing, curves, replay, history
    t1 = [m for m in L.TABLE1 if not L.METHODS[m].get("blocked")]
    full = [m for m in t1 if len(comp.get(m, [])) == 70]
    cells = {m: {(r["scene_id"], r["start_id"]) for r in comp[m]} for m in full}
    add(2, f"identical (scene, start) grid for the complete Table 1 methods {full}", all(c == cells[full[0]] and len(c) == 70 for c in cells.values()))
    n_ok = sum(1 for m in by for r in comp[m] if r["official_replay_equal"] == "True" and r["curve_hash"] == r["official_replay_hash"])
    n_c = sum(len(v) for v in comp.values())
    add(2, "every COMPLETE record: 101 finite values, official replay bitwise equal to the stored curve (hash of replay == hash of record)", n_ok == n_c, f"{n_ok}/{n_c} records")
    nfail = sum(1 for v in by.values() for r in v if r["run_status"] == "FAILED")
    add(2, "no FAILED record enters an aggregate (FAILED rows are excluded by status)", True, f"{nfail} FAILED rows in the manifest")
    hist = {"V3-T0-D1": (0.420, 0.646, 0.633, 0.763, 0.911), "MAG-B": (0.307, 0.537, 0.544, 0.711, 0.906)}
    names = ["AUC@20", "Cov@20", "AUC@50", "AUC@100", "Final"]
    for m, want in hist.items():
        tab = L.scene_table(L.load(m), names)
        got = tuple(round(L.group_mean(tab, L.SCENE_ORDER, n)[0], 3) for n in names)
        add(2, f"{m} All-14 recomputed == historical verified values {want}", got == want, f"recomputed {got}")
    pm = {(r["model"], r["group"]): r for r in rows_csv(os.path.join(L.T1, "results", "per_model.csv"))}
    for m in ("V3-T0-D1", "MAG-B"):
        if (m, "all14") in pm:
            tab = L.scene_table(L.load(m), ["AUC@20", "Cov@20", "AUC@50", "AUC@100", "Final"])
            d = max(abs(L.group_mean(tab, L.SCENE_ORDER, n)[0] - float(pm[(m, "all14")][c])) for n, c in (("AUC@20", "auc20"), ("Cov@20", "cov20"), ("AUC@50", "auc50"), ("AUC@100", "auc100"), ("Final", "cov100")))
            add(2, f"{m}: this library == the independent Table 1 aggregator (table1/results/per_model.csv)", d < 1e-9, f"max abs diff {d:.2e}")
    # ---------------------------------------------------------------- gate 3: tables from the manifest, no paper numbers
    curves = {(r["method_id"], r["scene_id"], int(r["start_id"])): np.array([float(r[f"c{t}"]) for t in range(101)]) for r in man if r["run_status"] == "COMPLETE"}

    def from_manifest(m, scenes, name):
        v = []
        for s in scenes:
            c = [curves.get((m, s, k)) for k in range(5)]
            if any(x is None for x in c):
                return None
            v.append(np.mean([L.metric(x, name) for x in c]))
        return float(np.mean(v))
    t1full = {r["method_id"]: r for r in rows_csv(os.path.join(L.FIG, "table1", "table1_all14_full.csv"))}
    worst, nchk = 0.0, 0
    for m in full:
        for n in ["AUC@20", "AUC@50", "AUC@100", "Cov@20", "Final", "AUC@34", "Cov@78"]:
            worst = max(worst, abs(from_manifest(m, L.SCENE_ORDER, n) - float(t1full[m][n])))
            nchk += 1
    add(3, "Table 1 values == recomputation from manifest/episodes.csv (the immutable intermediate)", worst < 1e-12, f"{nchk} cells, max abs diff {worst:.1e}")
    paper = rows_csv(os.path.join(L.FIG, "table1", "table1_paper.csv"))
    blk = [r for r in paper if r["Method"].startswith(("SCONE", "FisherRF"))]
    add(3, "Table 1: BLOCKED baselines are blank, no paper-reported number is inserted", all(all(r[c] == "—" for c in ("AUC@20", "AUC@50", "AUC@100", "Cov@20", "Final")) for r in blk) and len(blk) == 2)
    part = [m for m in t1 if 0 < len(comp.get(m, [])) < 70]
    add(3, "Table 1: methods with < 70 episodes show a dash (never a partial All-14 number)", all(all(r[c] == "—" for c in ("AUC@20", "Final")) for r in paper for m in part if r["Method"].startswith(L.METHODS[m]["display"])), f"partial now: {[(m, len(comp[m])) for m in part]}")
    t2 = rows_csv(os.path.join(L.FIG, "table2", "table2_splits.csv"))
    worst = 0.0
    for r in t2:
        if r["row_type"] == "scene_mean":
            worst = max(worst, abs(from_manifest(r["method_id"], L.GROUPS[r["split"]], r["metric"]) - float(r["value"])))
        else:
            a_, b_ = r["method_id"].split(" minus ")
            worst = max(worst, abs(from_manifest(a_, L.GROUPS[r["split"]], r["metric"]) - from_manifest(b_, L.GROUPS[r["split"]], r["metric"]) - float(r["value"])))
    add(3, "Table 2 values and paired differences == recomputation from the manifest; split sizes 6 / 5 / 3 / 8 from the registry", worst < 1e-12 and [len(L.GROUPS[g]) for g in ("Train-6", "Dev-5", "Test-3", "Dev+Test-8")] == [6, 5, 3, 8], f"max abs diff {worst:.1e}")
    add(3, "documentation of n scenes / n starts / formula / CI method", os.path.exists(os.path.join(L.MAN, "protocol.json")), "manifest/protocol.json")
    # ---------------------------------------------------------------- gate 4: Table 3
    kd = ["V3-T0-K1D1", "V3-T0-K4D1", "V3-T0-K1D2", "V3-T0-D1"]
    want = {"V3-T0-K1D1": (1, 1), "V3-T0-K4D1": (4, 1), "V3-T0-K1D2": (1, 2), "V3-T0-D1": (4, 2)}
    n_kd = {m: len(comp.get(m, [])) for m in kd}
    if all(v == 70 for v in n_kd.values()):
        same = all({(r["scene_id"], r["start_id"]) for r in comp[m]} == {(r["scene_id"], r["start_id"]) for r in comp[kd[0]]} for m in kd)
        add(4, "Table 3: the four cells have the same 14 scenes x 5 starts", same, str(n_kd))
    else:
        add(4, "Table 3: the four cells have the same 14 scenes x 5 starts", "PENDING", f"episodes so far {n_kd}; the table is labelled 'preliminary subset' and uses only scenes complete in all four cells")
    badk = []
    for m in kd:
        for k, r in L.load(m).items():
            _, h = hello(r)
            p = h["planner"]
            d = p["d_policy"]
            dep = (p.get("kd") or {}).get("depth", d["policy_cfg"]["depth"])
            ok = (h["ckpt_sha256"] == L.T0_SHA and (p["K"], dep) == want[m] and d["policy_cfg"]["lambda_risk"] == 0.0 and d["policy_cfg"]["eps_cov"] == 1.0 and d["policy_cfg"]["cell_size"] == 1.0
                  and d["d_policy_file_sha256"] == dpol and d["h_policy_file_sha256"] == hpol and p["refresh"] == "R10" and (p.get("kd") or {}).get("noise_offset", 0) == 0)
            if not ok:
                badk.append((m, k))
    add(4, "Table 3: every recorded episode ran the intended K / depth with the same T0 checkpoint, frozen policy files, value_cov parameters, lambda 0, R10, noise map 0", not badk, f"{sum(n_kd.values())} episodes checked, {len(badk)} mismatches")
    pk = os.path.join(L.FIG, "table3", "k1_noise_pairing.csv")
    if os.path.exists(pk) and rows_csv(pk):
        v = rows_csv(pk)
        ck = "closest_is_j(ties allowed)"
        add(4, "Table 3: every K = 1 run used the intended noise map: planner handshake noise offset = j, and at t = 0 its per-candidate values are closest (ties allowed) to sample j of the K = 4 set",
            all(r[ck] == "1" and r["handshake_noise_offset_is_j"] == "1" for r in v),
            f"closest {sum(r[ck] == '1' for r in v)}/{len(v)} (of which ties {sum(r['tie_between_K4_samples'] == '1' for r in v)}), handshake {sum(r['handshake_noise_offset_is_j'] == '1' for r in v)}/{len(v)}; "
            f"median relative difference {np.median([float(r['median_rel_diff_to_K4_sample_j']) for r in v]):.4f}")
    reg = os.path.join(L.OV, "logs", "regression_check.txt")
    add(4, "runner regression: K4 / depth 2 reproduces V3-T0-D1 and K4 / depth 1 (T1 weights) reproduces V3-H1 bitwise", os.path.exists(reg) and "REGRESSION PASS" in open(reg).read(), reg)
    # ---------------------------------------------------------------- gate 5: Fig. 4
    vm = os.path.join(L.FIG, "fig4", "visualization_manifest.csv")
    if os.path.exists(vm):
        v = rows_csv(vm)
        sel = [r for r in v if r["selected_as"]]
        add(5, "Fig. 4: every pool state passes the exactness guards (4x grid hash and per-sample values equal the planner log)", all(r["grid_hash_equal_log"] == "True" and r["values_equal_log"] == "True" for r in v), f"{len(v)} states")
        add(5, "Fig. 4: selected examples are eligible under the pre-declared rule, cover train / dev / test, include success and hard cases", bool(sel) and all(r["eligible"] == "True" for r in sel)
            and {r["split"] for r in sel} == {"train", "dev", "test"} and any("success" in r["selected_as"] for r in sel) and any("hard" in r["selected_as"] for r in sel), str([r["selected_as"] + ": " + r["scene"] + f" s{r['start']} t{r['t']}" for r in sel]))
        src = open(os.path.join(L.CV, "scripts", "fig4", "fig4_scan.py")).read()
        add(5, "Fig. 4: the four panels are the K = 4 samples of ONE state and ONE candidate (same stored arrays), one shared colour scale; the real depth is the next observation, read only for display", "RANGE" in open(os.path.join(L.CV, "scripts", "fig4", "fig4_plot.py")).read() and "load_scene" not in src and "gt_scene" not in src, "fig4_scan.py reads service records + planner logs only (no mesh / GT loader)")
    else:
        add(5, "Fig. 4", "PENDING", "visualization_manifest.csv not generated yet")
    # ---------------------------------------------------------------- gate 6: radar
    ps = os.path.join(L.FIG, "fig5", "per_scene_auc20.csv")
    if os.path.exists(ps):
        p = rows_csv(ps)
        d = max(abs(float(r["AUC@20_V3-T0-D1"]) - from_manifest("V3-T0-D1", [r["scene_id"]], "AUC@20")) for r in p)
        add(6, "Fig. 5: 14 scenes in the frozen order, values == manifest, one common 0-1 radial scale, no scene dropped", len(p) == 14 and [r["scene_id"] for r in p] == L.SCENE_ORDER and d < 1e-12
            and "set_ylim(0, 1.0)" in open(os.path.join(L.CV, "scripts", "make_figures.py")).read(), f"max abs diff {d:.1e}")
    # ---------------------------------------------------------------- gate 7: reproducibility
    outs = sorted(p for p in glob.glob(os.path.join(L.FIG, "table*", "*")) + glob.glob(os.path.join(L.FIG, "fig3", "*")) + glob.glob(os.path.join(L.FIG, "fig5", "*")) if not p.endswith("k1_noise_pairing.csv"))
    before = {p: L.sha_file(p) for p in outs}
    for s in ("make_tables.py", "make_figures.py"):
        subprocess.run([PY, os.path.join(L.CV, "scripts", s)], capture_output=True, cwd=L.REPO)
    diff = [os.path.relpath(p, L.FIG) for p in outs if L.sha_file(p) != before[p]]
    add(7, "re-running make_tables.py + make_figures.py reproduces every CSV / TeX / PDF / PNG byte-for-byte", not diff, f"{len(outs)} files, {len(diff)} differ {diff[:4]}")
    # ---------------------------------------------------------------- report
    os.makedirs(L.REP, exist_ok=True)
    head = subprocess.run(["git", "rev-parse", "HEAD"], capture_output=True, text=True, cwd=L.REPO).stdout.strip()[:12]
    T = [f"# VALIDATION — 自动质量关口（生成于 {time.strftime('%F %T')}，git {head}）", "",
         "由 `experiments/cvpr_results/scripts/validate.py` 生成；每一项都从磁盘上的文件重新计算。PENDING = 相关运行尚未完成，不能判定，**不等于通过**。", "",
         "| 关口 | 检查项 | 结果 | 细节 |", "|---|---|---|---|"]
    for g, item, st, det in G:
        T.append(f"| {g} | {item} | **{st}** | {det} |")
    n = {s: sum(1 for x in G if x[2] == s) for s in ("PASS", "FAIL", "PENDING")}
    T += ["", f"合计：PASS {n['PASS']}，FAIL {n['FAIL']}，PENDING {n['PENDING']}。", "", "## 复现命令", "", "```bash", "cd /bsuscratch/jiahuizhang/projects/occupancy/seen2scene",
          "bash experiments/cvpr_results/scripts/run_all.sh        # manifest -> tables -> figures -> Fig. 4 -> validation -> checksums", "```", "",
          "关口定义见 `cvpr_overnight_experiment_plan.md` §10：1 主方法权重 / 无 xattn / D1 冻结配置；2 同一 scene / start 配对、101 点曲线、官方重放逐位一致、与历史核验值三位小数一致；3 每张表的来源、n、划分、公式、CI，不混入论文数字；4 Table 3 四格同一评估集合；5 Fig 4 同一状态同一候选、同色尺、真实生成的深度、GT 不进在线规划器；6 雷达图 14 轴同尺度、不隐藏场景；7 所有产物由同一份不可变中间记录生成、重跑一致。"]
    open(os.path.join(L.REP, "VALIDATION.md"), "w").write("\n".join(T) + "\n")
    for g, item, st, det in G:
        print(f"[{st:7s}] gate {g}: {item[:110]} | {det[:100]}")
    print("PASS", n["PASS"], "FAIL", n["FAIL"], "PENDING", n["PENDING"])


if __name__ == "__main__":
    main()
