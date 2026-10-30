"""EXP5 Task 0 / T0.12-C -- offline diagnosis (descriptive; magician env, one GPU). Replays the T0.12 MAGICIAN Greedy
planner child (adapters/magician_planner_grid.Planner "MAG-G", unchanged: same model, seeds, settings, prefer-unvisited
validity) along a stored grid trajectory of any group and records its layer-0 gains of every legal candidate at every
state (None = not valid: already visited). Nothing is executed. On a G-MAG-G / G-MAG-B trajectory the gains must
reproduce the stored ones (layer-0 gains do not depend on the beam).
Output MC/diag_c/mag/<group>/<tag>.json
"""
import argparse
import json
import os
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.join(HERE, "..", "adapters"))
import magician_planner_grid as MPG  # noqa: E402

import numpy as np  # noqa: E402
import torch  # noqa: E402

MPL = MPG.MPL
MC_DIR = os.path.join(MPL.W, "magician_comparison")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--group", required=True)
    ap.add_argument("--tag", required=True)
    a = ap.parse_args()
    ep_dir = os.path.join(MC_DIR, "grid_protocol", "closed_loop", a.group, a.tag)
    ep = json.load(open(os.path.join(ep_dir, "episode.json")))
    stored = [json.loads(x) for x in open(os.path.join(ep_dir, "planner.jsonl"))] if a.group.startswith("G-MAG") else None
    dev = torch.device("cuda:0")
    torch.cuda.set_device(dev)
    pl = MPG.Planner("MAG-G", ep["level_key"], a.tag, dev)
    rows = []
    for t in range(ep["moves"]):
        rec = MPL.MC.load_service_record(os.path.join(ep_dir, "svc", f"svc_t{t}.npz"))
        ans = pl.step(t, rec)
        r = {"t": t, "scores": ans["scores"], "argmax": ans["pick"], "fallback": ans["fallback"], "n_gaussians": ans["diag"]["n_gaussians"]}
        if stored is not None:
            x = np.asarray([np.nan if v is None else v for v in ans["scores"]], float)
            y = np.asarray([np.nan if v is None else v for v in stored[t]["scores"]], float)
            r["max_abs_diff_vs_stored"] = float(np.nanmax(np.abs(x - y))) if np.isfinite(x - y).any() else 0.0
            r["nan_pattern_equal"] = bool(np.array_equal(np.isnan(x), np.isnan(y)))
        rows.append(r)
        if t % 25 == 0:
            print(a.group, a.tag, t, "n_gauss", ans["diag"]["n_gaussians"], "%.2f s" % ans["timing"]["decision_s"], flush=True)
    od = os.path.join(MC_DIR, "diag_c", "mag", a.group)
    os.makedirs(od, exist_ok=True)
    json.dump({"group": a.group, "tag": a.tag, "rows": rows}, open(os.path.join(od, f"{a.tag}.json"), "w"))
    if stored is not None:
        print("reproduction max |d| =", max(r["max_abs_diff_vs_stored"] for r in rows), "nan pattern equal:",
              all(r["nan_pattern_equal"] for r in rows), flush=True)


if __name__ == "__main__":
    main()
