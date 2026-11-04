"""EXP6 V3 / H -- reports/G0_REGRESSION.md (CPU; generated) from the barts_s0 regression runs in work_dirs/exp6_v3/H/smoke:
  h0     (H runner, policy h0)          == CVPR Table 1 V3 record (executed keys, official coverage), bitwise
  h1deg  (H1 code path, h0 value)       == Table 1 V3 record, bitwise; per-step scores == the frozen DistP1 of the h0 run
  h3d1   (H3 code path, depth 1)        == the h2 run, bitwise
  h2     score == mean - 0.25 std (ddof 0) of the logged per-sample values at every step; lambda-0 identity (code: mean - 0 * std)
plus the policy / config / checkpoint hashes logged by every planner (H_POLICY line) and the pose-table verification."""
import glob
import json
import os

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
H = os.path.dirname(HERE)
R = os.path.dirname(os.path.dirname(os.path.dirname(H)))
SM = os.path.join(R, "work_dirs", "exp6_v3", "H", "smoke")


def ep(p):
    return json.load(open(os.path.join(SM, p, "V2M", "episodes", "ep_V2M_barts_s0_r0.json")))


def steps(p):
    return [json.loads(l) for l in open(glob.glob(os.path.join(SM, p, "V2M", "planner_logs", "*.steps.jsonl"))[0])]


def main():
    ref = json.load(open(os.path.join(R, "work_dirs", "cvpr_results", "table1", "episodes", "V3", "barts_s0.json")))
    L = ["# G0_REGRESSION — H 回归（barts_s0；自动生成，`scripts/g0_regression.py`）", "", "| 检查 | 结果 |", "|---|---|"]
    for p in ("h0", "h1deg"):
        e = ep(p)
        L.append(f"| {p} 与 Table 1 V3 记录：执行动作 / 官方覆盖逐位一致 | {e['executed_keys'] == ref['executed_keys']} / {e['coverage'] == ref['coverage_official']} |")
    s0, s1 = steps("h0"), steps("h1deg")
    pairs = [(a, b) for a, b in zip(s0, s1) if a["scores"]]
    n_eq = sum(a["scores"] == b["scores"] for a, b in pairs)
    rel = max(max(abs(x - y) / max(abs(x), 1e-12) for x, y in zip(a["scores"], b["scores"])) for a, b in pairs)
    picks = all(a["choice"] == b["choice"] for a, b in zip(s0, s1))
    L.append(f"| h1deg 每步分数与 h0 冻结 DistP1：逐位相同的步数 / 最大相对差 / 每步选择相同 | {n_eq}/{len(pairs)} / {rel:.2e}（浮点求和顺序：numpy 与 torch） / {picks} |")
    a, b = ep("h2"), ep("h3d1")
    L.append(f"| h3d1（H3 代码、深度 1）与 h2：执行动作 / 官方覆盖逐位一致 | {a['executed_keys'] == b['executed_keys']} / {a['coverage'] == b['coverage']} |")
    st = steps("h2")
    d = max(max(abs(s - (m - 0.25 * sd)) for s, m, sd in zip(x["scores"], x["all_scores"]["h_mean"], x["all_scores"]["h_std"])) for x in st if x["scores"])
    sd_ok = all(np.allclose(np.asarray(x["all_scores"]["h_value_k"]).std(1), x["all_scores"]["h_std"], rtol=0, atol=0) for x in st if x["scores"])
    L.append(f"| H2 分数 == mean − 0.25·std（ddof 0），每步最大绝对差 | {d} |")
    L.append(f"| H2 记录的 std 与样本值重算（ddof 0）逐位相同 | {sd_ok} |")
    nf = [x for x in st if x["scores"] and x["choice"]["fallback"] is None]
    ch = sum(int(np.argmax(x["all_scores"]["h_mean"])) != x["choice"]["pick"] for x in nf)
    L.append(f"| H2：风险项改变同一状态的 argmax（相对 H1 均值）的步数 | {ch} / {len(nf)} 个非回退步 |")
    L.append("| λ = 0 恒等 | 代码中 H2 = mean − λ·std，λ = 0 时就是 H1 的 mean（同一个 np.mean 结果）；固定状态上的 λ = 0 变体与 H1 的选择见 G2_FIXED_STATE.md |")
    h3 = steps("h3")
    n2 = [x["all_scores"]["h3"]["n_c2_unique"] for x in h3 if x.get("all_scores", {}).get("h3")]
    L.append(f"| H3 冒烟：步数 / 每步第二层唯一视角数均值 / 每步决策中位数（s） | {len(h3)} / {np.mean(n2):.1f} / {np.median([x['decision_s'] for x in h3]):.2f} |")
    pv = json.load(open(os.path.join(R, "work_dirs", "exp6_v3", "H", "pose_tables", "verification.json")))
    L.append(f"| 位姿表与历史服务记录逐位核对（14 个场景） | {sum(v['verified_rows'] for v in pv.values())} 行，{sum(v['verified_records'] for v in pv.values())} 条记录，不符 {sum(v['mismatch_rows'] for v in pv.values())} |")
    hashes = set()
    for p in ("h0", "h1deg", "h2", "h3", "h3d1"):
        for f in glob.glob(os.path.join(SM, p, "V2M", "planner_logs", "*.stderr")):
            for line in open(f):
                if line.startswith("H_POLICY "):
                    j = json.loads(line[9:])
                    hashes.add((p, j["policy_file_sha256"][:12], j["config_sha256"][:12]))
    L += ["", "| 运行 | h_policy.py sha256 | FROZEN_CONFIG.yaml sha256 |", "|---|---|---|"] + [f"| {p} | {a_}… | {b_}… |" for p, a_, b_ in sorted(hashes)]
    L += ["", f"Table 1 V3 记录的 checkpoint sha256：`{ref.get('ckpt_sha256')}`；回归运行的规划器报告的 sha256：`{ep('h0')['planner_hello'][0]['ckpt_sha256']}`。"]
    open(os.path.join(H, "reports", "G0_REGRESSION.md"), "w").write("\n".join(L) + "\n")
    print("\n".join(L))


if __name__ == "__main__":
    main()
