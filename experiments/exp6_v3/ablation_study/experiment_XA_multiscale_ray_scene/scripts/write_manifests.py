"""Experiment XA -- manifests (CPU): sha256 of the code and configs the runs depend on, the evaluation splits, the baseline records. Written before the chains start; re-run at the end
(a changed hash is reported, the first version is kept as *.frozen.json). Usage: python write_manifests.py"""
import glob
import hashlib
import json
import os
import subprocess
import time

R = "/bsuscratch/jiahuizhang/projects/occupancy/seen2scene"
X = os.path.join(R, "experiments", "exp6_v3", "ablation_study", "experiment_XA_multiscale_ray_scene")
M = os.path.join(X, "manifests")


def sha(p):
    return hashlib.sha256(open(p, "rb").read()).hexdigest()


def rel(p):
    return os.path.relpath(p, R)


def dump(name, obj):
    p = os.path.join(M, name)
    fz = p.replace(".json", ".frozen.json")
    if not os.path.exists(fz):
        json.dump(obj, open(fz, "w"), indent=1)
    else:
        old = json.load(open(fz))
        ch = [k for k in obj.get("files", {}) if old.get("files", {}).get(k) != obj["files"][k]]
        if ch:
            print(name, "CHANGED since frozen:", ch)
    json.dump(obj, open(p, "w"), indent=1)


os.makedirs(M, exist_ok=True)
E5 = "experiments/exp5_future_view"
files = sorted(glob.glob(f"{R}/seen2scene/v3/xa/*.py") + glob.glob(f"{R}/seen2scene/v3/*.py") + glob.glob(f"{R}/seen2scene/futureview/*.py") + glob.glob(f"{X}/scripts/*") + glob.glob(f"{X}/configs/*.json")
               + [f"{R}/experiments/exp6_v3/ablation_study/experiment_D_h3_lookahead/scripts/d_policy.py", f"{R}/experiments/exp6_v3/ablation_study/experiment_D_h3_lookahead/configs/d_config.yaml",
                  f"{R}/experiments/exp6_v3/H/scripts/h_policy.py", f"{R}/experiments/exp6_v3/shared/planner_v3.py", f"{R}/{E5}/task1/scripts/v2_plan_core.py", f"{R}/{E5}/task1/scripts/v2_planner.py",
                  f"{R}/{E5}/task1/scripts/v2_sim.py", f"{R}/{E5}/task2/scripts/t24_out_sim.py", f"{R}/{E5}/task2/scripts/t24_v2_out_planner.py", f"{R}/{E5}/task2/scripts/t23_in_sim.py",
                  f"{R}/{E5}/task2/scripts/t23_v2_in_planner.py", f"{R}/{E5}/task2/scripts/t25_out_cov.py", f"{R}/{E5}/task3_full_depth_p1/scripts/p1_full.py",
                  f"{R}/experiments/exp3b_macaronspp_transfer/m_p1.py", f"{R}/experiments/exp3b_macaronspp_transfer/m_t6_sim.py", f"{R}/experiments/exp6_v3/configs/train_v3.json",
                  f"{R}/experiments/exp6_v3/configs/model_v3.json", f"{X}/FROZEN_PROTOCOL.md", f"{R}/experiments/exp6_v3/XA_Experiment_Protocol.md"])
git = {"head": subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=R).decode().strip(), "status": subprocess.check_output(["git", "status", "--short"], cwd=R).decode().splitlines()[:40]}
dump("code_hashes.json", {"time": time.strftime("%F %T"), "git": git, "files": {rel(p): sha(p) for p in files if os.path.isfile(p)}})
import yaml  # noqa: E402
reg = yaml.safe_load(open(f"{R}/experiments/cvpr_results/table1/SCENE_REGISTRY.yaml"))["scenes"]
items = reg.items() if isinstance(reg, dict) else [(s["local_id"], s) for s in reg]
split = {k: str(v["split"]).lower() for k, v in items if (v.get("runnable") or {}).get("overall")}
g = {"Train-6": sorted(s for s in split if split[s].startswith("train")), "Dev-5": sorted(s for s in split if split[s].startswith("dev")), "Test-3": sorted(s for s in split if split[s].startswith("test"))}
g["Unseen-8"] = sorted(g["Dev-5"] + g["Test-3"])
g["All-14"] = sorted(split)
dump("splits.json", {"outdoor": {"groups": g, "starts": [0, 1, 2, 3, 4], "steps": 100, "episodes": 70, "planner": "D1 (d_config d1)", "coverage": "official MAGICIAN (replay-checked)",
                                 "note": "Test-3 is not a blind test; Train-6 scenes are training scenes"},
                     "indoor": {"episodes": [["Z6MFQCViBuw", 0, 6], ["Z6MFQCViBuw", 0, 17], ["oLBMNvg9in8", 1, 36], ["oLBMNvg9in8", 1, 78], ["QUCTc6BB5sX", 1, 34], ["QUCTc6BB5sX", 1, 135]],
                                "rule": "first three floors of experiments/exp5_future_view/task2/configs/eval_indoor_v1.json, both starts", "steps": 150, "planner": "H0 (frozen indoor path)", "coverage": "C_obs"},
                     "training": {"data_manifest_sha256_T1": "9f432c3f943a891ab1d0090745f2bbc39534e49a7858577fa611bace6c25d3aa", "indoor_trajectories": 228, "outdoor_trajectories": 138, "seed": 0, "world": 4}})
tb = f"{R}/work_dirs/cvpr_results/table1/episodes/V3-D1-H3-NORISK"
h = hashlib.sha256()
eps = sorted(glob.glob(tb + "/*.json"))
for p in eps:
    h.update(json.dumps(json.load(open(p))["coverage_official"]).encode())
t1in = f"{R}/work_dirs/exp6_v3/eval/closed_loop/indoor/T1/V2M"
dump("checkpoints.json", {"T1_40K": {"path": "work_dirs/exp6_v3/task1_xattn/checkpoints/final.pt", "sha256": sha(f"{R}/work_dirs/exp6_v3/task1_xattn/checkpoints/final.pt")},
                          "baseline_T1_D1_outdoor": {"records": rel(tb), "episodes": len(eps), "sha256_of_official_curves": h.hexdigest()},
                          "table1_csv_sha256": sha(f"{R}/experiments/cvpr_results/table1/TABLE1_FULL.csv"),
                          "baseline_T1_H0_indoor": {x: sha(f"{t1in}/{x}/episode.json") for x in ("Z6MFQCViBuw_L0_S6", "Z6MFQCViBuw_L0_S17", "oLBMNvg9in8_L1_S36", "oLBMNvg9in8_L1_S78", "QUCTc6BB5sX_L1_S34", "QUCTc6BB5sX_L1_S135")},
                          "T1_common_init_sha256": "6a9d6071a22564c362177bcd2fac0ea8652fe83f3e60de655f299eed0363b513"})
print("manifests written", sorted(os.listdir(M)))
