"""Geometry / ROI audit H (CPU, read-only): planner-side information boundary of the 70 Table 1 D-v2 episodes (V3-DV2-KSV).
From every planner log: the LeakGuard summary (paths opened by the planner process while the guard was installed; denied opens),
classified; any path that looks like GT / evaluation data is listed. From every service record of 14 sample episodes (start 0):
the key set (whitelist). Output results/h_leak_check.json."""
import collections
import glob
import json
import os

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
AUD = os.path.dirname(HERE)
REPO = os.path.dirname(os.path.dirname(os.path.dirname(AUD)))
ROOT = os.path.join(REPO, "work_dirs", "cvpr_results", "table1", "g3", "closed_loop", "outdoor", "V3-DV2-KSV", "V2M")
SUSPECT = (".obj", "occupied_pose", "/episodes/", "/metrics/", "coverage", "magician_lmdb", "/evalside/", "gt_", "oracle", "/frames/")
WHITELIST = {"scene_key", "start", "t", "zbuf", "mask", "R", "T", "cam_center", "pose_idx", "sample_pixel_flat", "sample_points",
             "executed_pose_idx_history", "legal_idx", "legal_R", "legal_T", "legal_center", "frozen_params"}


def main():
    logs = sorted(glob.glob(os.path.join(ROOT, "planner_logs", "pl_V2M_*_r0.json")))
    cat, denied, suspects, n_open = collections.Counter(), 0, set(), 0
    for f in logs:
        g = json.load(open(f)).get("guard") or {}
        denied += len(g.get("denied", []))
        n_open += g.get("n_opens", 0)
        for p in g.get("unique_paths", []):
            if "/svc/" in p:
                cat["this episode's service records"] += 1
            elif "planner_logs" in p:
                cat["this episode's planner logs"] += 1
            elif p.endswith("settings.json"):
                cat["settings.json (scene / camera boxes)"] += 1
            elif "site-packages" in p or "/envs/" in p or p.startswith(("/usr", "/proc", "/sys", "/dev", "/etc", "/tmp")):
                cat["python / system files"] += 1
            elif p.endswith((".py", ".pyc", ".yaml", ".json", ".so")):
                cat["code / config"] += 1
            else:
                cat["other"] += 1
                suspects.add(p) if len(suspects) < 40 else None
            if any(x in p for x in SUSPECT) and "/svc/" not in p and "planner_logs" not in p:
                suspects.add(p)
    keys, extra = collections.Counter(), set()
    for d in sorted(glob.glob(os.path.join(ROOT, "svc", "V2M_*_s0_r0"))):
        for f in sorted(glob.glob(os.path.join(d, "*.npz")))[::10]:
            ks = set(np.load(f, allow_pickle=False).files)
            keys.update(ks)
            extra |= ks - WHITELIST
    out = {"planner_logs": len(logs), "opens_total": n_open, "denied_opens": denied, "unique_path_categories": dict(cat), "suspect_or_unclassified_paths": sorted(suspects),
           "service_record_keys": sorted(keys), "service_record_keys_outside_whitelist": sorted(extra)}
    json.dump(out, open(os.path.join(AUD, "results", "h_leak_check.json"), "w"), indent=1)
    print(json.dumps({k: v for k, v in out.items() if k != "service_record_keys"}, indent=1)[:1800])
    print("keys", out["service_record_keys"])


if __name__ == "__main__":
    main()
