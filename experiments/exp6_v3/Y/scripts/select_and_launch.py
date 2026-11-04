"""EXP6 V3 overnight -- pre-registered Dev-5 selections and the All-14 job files (seen2scene env, CPU; run once when the
Dev-5 runs are complete). Rules (Y FROZEN_PROTOCOL §4, D FROZEN_PROTOCOL §3), official coverage, scene-equal Dev-5 means:
  Y : between V3-Y45K-H0 and V3-Y50K-H0 the higher AUC@100; |diff| < 0.002 -> the earlier checkpoint
  D : among D1 / D2 / D3 the higher AUC@100; |diff| < 0.002 -> the cheaper (D1 < D2 < D3)
Writes work_dirs/exp6_v3/overnight/SELECTION.json (all Dev-5 numbers of every candidate), registers V3-YBEST-H3 in the Table 1
registry (READY, sha256 of the selected Y EMA; launch = H-v1 h3 policy, unchanged) and its t1_queue line, and writes
  jobs_main_all14.txt : D-main + Y-best missing Train-6 / Test-3 episodes (dunnottar excluded) + YBEST-H3 Dev-5, interleaved
  jobs_d1_all14.txt   : D1 missing Train / Test episodes (dunnottar excluded) if D1 is not the D main version (anchor, lower priority)
  jobs_dunnottar.txt  : dunnottar x 5 for D-main, Y-best (and D1) -- run with one worker per GPU
Refuses to run if a Dev-5 candidate is incomplete (prints what is missing). Usage: python select_and_launch.py [--dry]"""
import argparse
import hashlib
import json
import os
import sys

import numpy as np

R = "/bsuscratch/jiahuizhang/projects/occupancy/seen2scene"
E = os.path.join(R, "work_dirs", "cvpr_results", "table1", "episodes")
C = os.path.join(R, "experiments", "cvpr_results", "table1", "configs")
DEV = ["barts", "colosseum", "eiffel", "pantheon", "sestino_museum"]
TRAIN = ["bannerman", "dunnottar", "fushimi", "liberty", "pisa", "redeemer"]
TEST = ["alhambra", "bridge", "neuschwanstein"]


def dev5(mid):
    vals = {"auc20": [], "auc100": [], "final": []}
    for s in DEV:
        eps = []
        for k in range(5):
            p = os.path.join(E, mid, f"{s}_s{k}.json")
            if not os.path.exists(p):
                return None, f"{mid}:{s}:{k}"
            c = np.asarray(json.load(open(p))["coverage_official"])
            eps.append((np.trapz(c[:21]) / 20, np.trapz(c) / 100, c[100]))
        m = np.mean(eps, 0)
        vals["auc20"].append(m[0]); vals["auc100"].append(m[1]); vals["final"].append(m[2])
    return {k: float(np.mean(v)) for k, v in vals.items()}, None


def missing(mid, scenes):
    return [f"{mid}:{s}:{k}" for k in range(5) for s in scenes if not os.path.exists(os.path.join(E, mid, f"{s}_s{k}.json"))]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--dry", action="store_true")
    a = ap.parse_args()
    cand_y = ["V3-Y45K-H0", "V3-Y50K-H0"]
    cand_d = ["V3-D1-H3-NORISK", "V3-D2-DEPTH3-B3", "V3-D3-DEPTH3-B5"]
    res, miss = {}, []
    for m in cand_y + cand_d:
        r, ms = dev5(m)
        res[m] = r
        if ms:
            miss.append(ms)
    if miss:
        print("NOT READY, first missing:", miss)
        sys.exit(2)
    ys = sorted(cand_y, key=lambda m: -res[m]["auc100"])
    ybest = ys[0] if res[ys[0]]["auc100"] - res[ys[1]]["auc100"] >= 0.002 else "V3-Y45K-H0"
    best = max(cand_d, key=lambda m: res[m]["auc100"])
    dmain = next(m for m in cand_d if res[best]["auc100"] - res[m]["auc100"] < 0.002)       # cheapest within 0.002 of the best
    step = 45000 if ybest == "V3-Y45K-H0" else 50000
    ema = os.path.join(R, "work_dirs", "exp6_v3", "Y", "runs", "main", f"ema_{step:06d}.pt")
    sha = hashlib.sha256(open(ema, "rb").read()).hexdigest()
    sel = {"rules": {"Y": "higher Dev-5 AUC@100 (official); |d| < 0.002 -> earlier", "D": "higher Dev-5 AUC@100; |d| < 0.002 -> cheaper (D1 < D2 < D3)"},
           "dev5": res, "Y_selected": ybest, "Y_selected_ema": os.path.relpath(ema, R), "Y_selected_sha256": sha, "D_main": dmain}
    os.makedirs(os.path.join(R, "work_dirs", "exp6_v3", "overnight"), exist_ok=True)
    print(json.dumps(sel, indent=1))
    if a.dry:
        return
    json.dump(sel, open(os.path.join(R, "work_dirs", "exp6_v3", "overnight", "SELECTION.json"), "w"), indent=1)
    reg = os.path.join(R, "experiments", "cvpr_results", "table1", "MODEL_REGISTRY.yaml")
    s = open(reg).read()
    if "  - id: V3-YBEST-H3\n" not in s:
        ent = f'''  - id: V3-YBEST-H3
    label: "V3-YBEST-H3: Y {step // 1000}K（Dev-5 选出）+ 原 H3 策略（H-v1 h3，未改）"
    role: y
    status: READY
    status_reason: EXP6 V3 / Y 的 Dev-5 预注册选择（work_dirs/exp6_v3/overnight/SELECTION.json）；同一 H3 规划器（H-v1），只换权重；用于测同一规划器下的微调收益，不是 D 的结果
    sha256: {sha}
    checkpoint: {{real_path: {os.path.relpath(ema, R)}, parent: "V3-T1 40K", added_steps: {step - 40000}}}
    inference: {{p1_mode: b2, inference_density: D100, scoring_density: S10, refresh: R10, planner: "Greedy-NV + h3 (H-v1)", K: 4, euler_steps: 10, planning_region: 整个场景}}
    launch:
      command: |
        CUDA_VISIBLE_DEVICES=$G $MG experiments/exp6_v3/H/scripts/sim_out_h.py --v3-ckpt $R/{os.path.relpath(ema, R)} --policy h3 --group V2M --scene $S --start $K --root $T1W/closed_loop/outdoor/V3-YBEST-H3/V2M
    historical_outdoor: {{reusable_episodes: 0}}
'''
        c = "# 不进入 models 列表的参考数据（aggregate.py 只读 models）\n"
        i = s.index(c)
        s = s[:i].rstrip("\n") + "\n" + ent + "\n" + c + s[i + len(c):]
        open(reg, "w").write(s)
        q = os.path.join(R, "experiments", "cvpr_results", "table1", "scripts", "t1_queue.sh")
        t = open(q).read()
        anchor = '    *) echo "unknown model $m"; return 2 ;;'
        t = t.replace(anchor, f'    V3-YBEST-H3) $MG $X/exp6_v3/H/scripts/sim_out_h.py --v3-ckpt $R/{os.path.relpath(ema, R)} --policy h3 --group V2M --scene $s --start $k --root $root ;;\n' + anchor)
        open(q, "w").write(t)
    nodun = [x for x in TRAIN if x != "dunnottar"] + TEST
    main_jobs = []
    a1, a2, a3 = missing(dmain, nodun), missing(ybest, nodun), missing("V3-YBEST-H3", DEV)
    for i in range(max(len(a1), len(a2), len(a3))):
        for lst in (a1, a3, a2):
            if i < len(lst):
                main_jobs.append(lst[i])
    open(os.path.join(C, "jobs_main_all14.txt"), "w").write("\n".join(main_jobs) + "\n")
    d1 = missing("V3-D1-H3-NORISK", nodun) if dmain != "V3-D1-H3-NORISK" else []
    open(os.path.join(C, "jobs_d1_all14.txt"), "w").write("\n".join(d1) + ("\n" if d1 else ""))
    dun = missing(dmain, ["dunnottar"]) + missing(ybest, ["dunnottar"]) + (missing("V3-D1-H3-NORISK", ["dunnottar"]) if d1 else [])
    open(os.path.join(C, "jobs_dunnottar.txt"), "w").write("\n".join(dun) + "\n")
    print("jobs", len(main_jobs), len(d1), len(dun))


if __name__ == "__main__":
    main()
