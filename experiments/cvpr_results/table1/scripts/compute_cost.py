"""CVPR Table 1 -- reports/COMPUTE_COST.md + results/compute_cost.csv (CPU; generated).
Per model, from the episode files referenced by the canonical records (historic + Table 1 runs): planner decision time per
step (planner-reported `decision_s`, model forward + scoring; excludes the simulator), wall time per decision (sim side),
episode wall time of the simulator process, planner peak GPU memory. CAVEAT: runs shared GPUs with other workers (2-3 per GPU,
two nodes); times are upper bounds and only comparable in order of magnitude. Also counts the GPU time of this Table 1 run
(new episodes, successful + failed attempts from the queue logs)."""
import csv
import glob
import json
import os
import re

import numpy as np

import aggregate as AG

T1 = AG.T1
G3 = os.path.join(AG.REPO, "work_dirs", "cvpr_results", "table1", "g3")


def main():
    _, order, label, regs = AG.load_registries()
    rows = []
    for m in order:
        dec, wall, ep_s, mem, n, n_new = [], [], [], [], 0, 0
        for f in sorted(glob.glob(os.path.join(AG.EPI, m, "*.json"))):
            r = json.load(open(f))
            p = r["episode_file"] if os.path.isabs(r["episode_file"]) else os.path.join(AG.REPO, r["episode_file"])
            if not os.path.exists(p):
                continue
            e = json.load(open(p))
            st = [s for s in e["steps"] if s.get("planner")]
            dec += [s["planner"]["decision_s"] for s in st if s["planner"].get("decision_s") is not None]
            wall += [s["decide_wall_s"] for s in st if s.get("decide_wall_s") is not None]
            if e.get("seconds_this_process"):
                ep_s.append(e["seconds_this_process"])
            pk = (e.get("planner_bye") or {}).get("peak_mem_gb")
            if pk:
                mem.append(pk)
            n += 1
            n_new += r.get("source") == "table1_run"
        if not n:
            continue
        rows.append({"model": m, "label": label[m], "n_episodes": n, "n_new": n_new,
                     "decision_s_median": float(np.median(dec)), "decision_s_p90": float(np.quantile(dec, 0.9)),
                     "decide_wall_s_median": float(np.median(wall)), "episode_min_median": float(np.median(ep_s)) / 60 if ep_s else None,
                     "planner_peak_mem_gb_median": float(np.median(mem)) if mem else None, "planner_peak_mem_gb_max": float(max(mem)) if mem else None})
    # GPU time of this run from queue logs (exit lines: "exit RC MODEL:scene:k host gpuG SECONDSs date time")
    used = {}
    for f in glob.glob(os.path.join(G3, "logs", "queue_*.log")) + glob.glob(os.path.join(G3, "logs", "exclusive.log")):
        for line in open(f):
            mm = re.match(r"^exit (\d+) (?:EXCL )?([^:]+):\S+ \S+ \S+ (\d+)s", line)
            if mm:
                u = used.setdefault(mm.group(2), {"ok": 0, "fail": 0, "s_ok": 0, "s_fail": 0})
                k = "ok" if mm.group(1) == "0" else "fail"
                u[k] += 1
                u["s_" + k] += int(mm.group(3))
    os.makedirs(os.path.join(T1, "results"), exist_ok=True)
    with open(os.path.join(T1, "results", "compute_cost.csv"), "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0]))
        w.writeheader()
        w.writerows(rows)
    fm = lambda x, d=2: "—" if x is None else f"{x:.{d}f}"  # noqa: E731
    out = ["# COMPUTE_COST — 推理成本（自动生成，`scripts/compute_cost.py`）", "",
           "**说明**：",
           "- 所有 episode 都在共享 GPU 上运行（每卡 2–3 个 worker、两台 L40 节点），时间只能作数量级比较，是上界。",
           "- 规划器决策时间由规划器自己报告，包括模型前向与评分，不含模拟器。",
           "- episode 时间是模拟器进程的总时间，包括渲染、观测与等待规划器。", "",
           "| 模型 | episode 数（新跑） | 每步决策 中位数 / P90（s） | 每步墙钟 中位数（s） | 每 episode 中位数（min） | 规划器峰值显存 中位数 / 最大（GB） |",
           "|---|---|---|---|---|---|"]
    for r in rows:
        out.append(f"| {r['label']} | {r['n_episodes']} ({r['n_new']}) | {fm(r['decision_s_median'])} / {fm(r['decision_s_p90'])} | {fm(r['decide_wall_s_median'])} | "
                   f"{fm(r['episode_min_median'], 1)} | {fm(r['planner_peak_mem_gb_median'], 1)} / {fm(r['planner_peak_mem_gb_max'], 1)} |")
    if used:
        tot = sum(u["s_ok"] + u["s_fail"] for u in used.values()) / 3600
        out += ["", "## 本次 Table 1 补跑的 GPU 占用（队列日志，进程墙钟，GPU 是共享的）", "",
                "| 模型 | 成功 / 失败 | 成功（h） | 失败（h） |", "|---|---|---|---|"]
        for m in order:
            if m in used:
                u = used[m]
                out.append(f"| {m} | {u['ok']} / {u['fail']} | {u['s_ok'] / 3600:.1f} | {u['s_fail'] / 3600:.1f} |")
        out.append(f"| 合计 | {sum(u['ok'] for u in used.values())} / {sum(u['fail'] for u in used.values())} | | {tot:.1f} h（总进程墙钟） |")
    open(os.path.join(T1, "reports", "COMPUTE_COST.md"), "w").write("\n".join(out) + "\n")
    print("\n".join(out))


if __name__ == "__main__":
    main()
