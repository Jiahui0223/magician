"""EXP5 Task 0 / T0.12-C -- offline diagnosis "why V2 does not go far" (descriptive; seen2scene env, one GPU).
Replays the FROZEN V2K4 planner (scripts/t06_planner.Planner, unchanged) along a stored grid trajectory (any group) and
records its DistP1 scores of every legal candidate at every state. Nothing is executed: the trajectory is fixed.
  --variant region : the frozen planner as run (region = n_win x n_win windows around the camera at each refresh,
                     t % 10 == 0); on a V2K4 trajectory the scores must reproduce the stored ones
  --variant level  : PROBE -- identical except that the region is the whole level box (mp3d_frame.region_box_m
                     patched in this process only); same model, seeds and refresh rule
Output MC/diag_c/v2_<variant>/<group>/<tag>.json
"""
import argparse
import json
import os
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
T0S = os.path.join(HERE, "..", "..", "scripts")
sys.path.insert(0, T0S)
import mp3d_frame as F  # noqa: E402
import t06_planner as P  # noqa: E402

import numpy as np  # noqa: E402
import torch  # noqa: E402

MC_DIR = os.path.join(F.W, "magician_comparison")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--group", required=True)
    ap.add_argument("--tag", required=True)
    ap.add_argument("--variant", choices=["region", "level"], default="region")
    a = ap.parse_args()
    if a.variant == "level":
        F.region_box_m = lambda card, center_rt: np.asarray(card["box_m"], np.float64)
    ep_dir = os.path.join(MC_DIR, "grid_protocol", "closed_loop", a.group, a.tag)
    ep = json.load(open(os.path.join(ep_dir, "episode.json")))
    stored = [json.loads(x) for x in open(os.path.join(ep_dir, "planner.jsonl"))] if a.group == "G-V2K4" else None
    dev = torch.device("cuda")
    pl = P.Planner("V2K4", ep["level_key"], a.tag, dev)
    rows = []
    for t in range(ep["moves"]):
        rec = P.MC.load_service_record(os.path.join(ep_dir, "svc", f"svc_t{t}.npz"))
        ans = pl.step(t, rec)
        r = {"t": t, "scores": ans["scores"], "argmax": ans["pick"], "fallback": ans["fallback"],
             "region_box_m": np.asarray(pl.region[4]).tolist(), "decision_s": ans["timing"]["decision_s"]}
        if stored is not None:
            r["max_abs_diff_vs_stored"] = float(np.max(np.abs(np.asarray(ans["scores"]) - np.asarray(stored[t]["scores"])))) if ans["scores"] else 0.0
        rows.append(r)
        if t % 25 == 0:
            print(a.variant, a.group, a.tag, t, "max %.1f" % max(ans["scores"]), "%.2f s" % ans["timing"]["decision_s"],
                  "peak %.1f GB" % (torch.cuda.max_memory_allocated(dev) / 2 ** 30), flush=True)
    od = os.path.join(MC_DIR, "diag_c", f"v2_{a.variant}", a.group)
    os.makedirs(od, exist_ok=True)
    json.dump({"group": a.group, "tag": a.tag, "variant": a.variant, "rows": rows,
               "peak_mem_gb": torch.cuda.max_memory_allocated(dev) / 2 ** 30}, open(os.path.join(od, f"{a.tag}.json"), "w"))
    if stored is not None:
        print("reproduction max |d| =", max(r["max_abs_diff_vs_stored"] for r in rows), flush=True)


if __name__ == "__main__":
    main()
