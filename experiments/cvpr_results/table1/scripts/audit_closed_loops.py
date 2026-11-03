"""Table-1 dataset audit: read-only inventory of historical Macarons++ closed-loop episodes per scene.

Scans the given roots for finished episode files ep_*.json (written read-only at the end of an episode) and for
unfinished per-step checkpoints ckpt/*.jsonl, and counts them per (root group, scene, start). Episode names follow
<group>_<scene>_s<start>_r<rep>.json; the scene is matched against the 15 local Macarons++ ids.
Usage: python audit_closed_loops.py OUT.json
"""
import json
import os
import re
import sys

REPO = "/bsuscratch/jiahuizhang/projects/occupancy/seen2scene"
W = os.path.join(REPO, "work_dirs")
SCENES = ["alhambra", "bannerman", "barts", "bridge", "colosseum", "dunnottar", "eiffel", "fushimi", "liberty",
          "neuschwanstein", "pantheon", "pisa", "redeemer", "sestino_museum", "sofia_church"]
ROOTS = [os.path.join(W, "exp5_future_view", "task2", "eval", "outdoor", "closed_loop"),
         os.path.join(W, "exp5_future_view", "task3_full_depth_p1", "closed_loop", "outdoor"),
         os.path.join(W, "exp5_future_view", "task1", "closed_loop"),
         os.path.join(W, "exp4_multiworld_belief", "ablation_study"),
         os.path.join(W, "exp4_multiworld_belief", "task3"),
         os.path.join(W, "exp3b_macaronspp_transfer"),
         os.path.join(W, "exp_density_ab", "g3"),
         os.path.join(W, "exp_temporal_refresh", "g3"),
         os.path.join(W, "exp_camera_pose"),
         os.path.join(W, "exp_unified_outdoor")] + \
        sorted(os.path.join(W, d) for d in os.listdir(W) if d.startswith("exp6_"))
PAT = re.compile(r"^ep_(?P<pre>.*?)_?(?P<scene>" + "|".join(sorted(SCENES, key=len, reverse=True)) +
                 r")_s(?P<start>\d+)(?:_r(?P<rep>\d+))?.*\.json$")


def main():
    out = {"roots": ROOTS, "episodes": {}}
    for root in ROOTS:
        if not os.path.isdir(root):
            continue
        for dp, dn, fn in os.walk(root):
            dn[:] = [d for d in dn if d not in ("frames", "svc", "replay_frames", "imgs", "planner_logs", "cov_tmp")]
            if os.path.basename(dp) != "episodes":
                continue
            for f in fn:
                m = PAT.match(f)
                if not m:
                    continue
                rel = os.path.relpath(dp, W)
                try:
                    ep = json.load(open(os.path.join(dp, f)))
                    moves = ep.get("moves")
                    ncov = len(ep.get("coverage", []))
                except Exception as e:  # noqa: BLE001
                    moves, ncov = None, repr(e)[:80]
                d = out["episodes"].setdefault(rel, {}).setdefault(m["scene"], {})
                d.setdefault("starts", set()).add(int(m["start"]))
                d.setdefault("n", 0)
                d["n"] += 1
                d.setdefault("moves", set()).add(moves)
                d.setdefault("n_coverage", set()).add(ncov)
    for rel, sc in out["episodes"].items():
        for s, d in sc.items():
            for k in ("starts", "moves", "n_coverage"):
                d[k] = sorted(d[k], key=lambda x: (x is None, str(x)))
    json.dump(out, open(sys.argv[1], "w"), indent=1)
    by_scene = {}
    for rel, sc in sorted(out["episodes"].items()):
        print(rel, {s: (d["n"], d["starts"], d["moves"]) for s, d in sorted(sc.items())})
        for s, d in sc.items():
            by_scene.setdefault(s, []).append(rel)
    print("\nscenes with any closed-loop episode:", sorted(by_scene))
    print("scenes without:", [s for s in SCENES if s not in by_scene])


if __name__ == "__main__":
    main()
