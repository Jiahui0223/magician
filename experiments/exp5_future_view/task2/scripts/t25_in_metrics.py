"""EXP5 Task 2 -- indoor metrics of the final comparison, evaluation side (magician env; protocol §5).

Runs the T0.8 / T0.12 unified metric code (task0/magician_comparison/scripts/t08_metrics.episode, unchanged) on the Task
2 indoor episodes: C_obs reproduction from the stored 10 % samples (|d| <= 1e-12, else STOP), per frame t = 0..150
Comp@5cm and Comp distance (mean / median, cm) of the face-referenced metric GT vertices of the level box against the
full-depth back-projection of all frames so far, Comp@5cm over station-visible vertices, behaviour metrics (grid
branch). Only the paths are redirected (Task 2 groups V2O / V2M / MAG-G / MAG-B -> W2/eval/indoor/closed_loop; output
W2/eval/indoor/metrics/episodes/<group>/<tag>.json); the vertex-visibility files are the frozen T0.8 ones (read-only).
Usage: python t25_in_metrics.py <worker index> <n workers>
"""
import json
import os
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
import t2_common as C  # noqa: E402

sys.path.insert(0, os.path.join(C.EXP5, "task0", "magician_comparison", "scripts"))
import t08_metrics as TM  # noqa: E402

M = TM.M
GROUPS = ("V2O", "V2M", "MAG-G", "MAG-B")
ROOT = os.path.join(C.W2, "eval", "indoor", "closed_loop")
OUT = os.path.join(C.W2, "eval", "indoor", "metrics")
M.GRID_GROUPS = tuple(M.GRID_GROUPS) + GROUPS
_svc0, _cur0 = M.svc_path, M.stored_curve


def svc_path(group, tag, t):
    return os.path.join(ROOT, group, tag, "svc", f"svc_t{t}.npz") if group in GROUPS else _svc0(group, tag, t)


def stored_curve(group, tag):
    if group not in GROUPS:
        return _cur0(group, tag)
    e = json.load(open(os.path.join(ROOT, group, tag, "episode.json")))
    return {"coverage": e["coverage"], "executed": e["executed"], "decision_s": [s["decision_s"] for s in e["steps"]], "wall_s": e["wall_s"]}


M.svc_path, M.stored_curve = svc_path, stored_curve
TM.OUT = OUT


def main():
    import torch
    wi, wn = int(sys.argv[1]), int(sys.argv[2])
    vis = os.path.join(OUT, "visible")
    if not os.path.exists(vis):
        os.makedirs(OUT, exist_ok=True)
        os.symlink(os.path.join(M.MC, "metric_replay", "visible"), vis)
    bp = M.BackProjector()
    jobs = [(g, f"{s}_L{lv}_S{st}", s, lv) for s, lv, st in C.EVAL_LEVELS for g in GROUPS]
    for g, tag, s, lv in jobs[wi::wn]:
        if os.path.exists(os.path.join(ROOT, g, tag, "episode.json")):
            TM.episode(g, tag, s, lv, bp)
        torch.cuda.empty_cache()


if __name__ == "__main__":
    main()
