"""FisherRF audit A4 (CPU, read-only): behaviour statistics of the recorded closed loops, per method, on the same 70 starts.

Methods: FisherRF-full f1c4 / v4 / v3 (where recorded), FisherRF-score-1step D3a, MAG-B, RANDOM (Table 1 canonical records).
Per episode, from the executed pose keys (5D settings-grid index, t = 0..100):
  n_trans / n_rot / n_hover  steps that change the position / only the angle / nothing
  n_pos / n_pose             distinct positions / poses visited
  revisit_pos                translation steps that land on an already visited position
  span                       max L1 grid distance from the start position
  no_depth                   frames of the service records without any valid depth pixel (mask & 0 < zbuf < 70)
FisherRF-full only (planner logs): fraction of moves that follow a planned path vs fallback turns / seeded random, number of
replans, distance (L1 grid) from the current position to the selected goal, and whether the executed path reached the goal.
Output: work_dirs/.../audit/a4_behaviour.json + printed summary.
"""
import glob
import json
import os
import sys

import numpy as np

REPO = "/bsuscratch/jiahuizhang/projects/occupancy/seen2scene"
FRF = os.path.join(REPO, "work_dirs/cvpr_results/fisherrf_reproduction")
EPI = os.path.join(REPO, "work_dirs/cvpr_results/table1/episodes")
OUT = os.path.join(FRF, "audit")
SCENES = ["alhambra", "bannerman", "barts", "bridge", "colosseum", "dunnottar", "eiffel", "fushimi", "liberty", "neuschwanstein",
          "pantheon", "pisa", "redeemer", "sestino_museum"]


def keys_stats(keys):
    P = [tuple(json.loads(k)) if isinstance(k, str) else tuple(k) for k in keys]
    seen = {P[0][:3]}
    nt = nr = nh = rv = 0
    for a, b in zip(P[:-1], P[1:]):
        if a == b:
            nh += 1
        elif a[:3] != b[:3]:
            nt += 1
            if b[:3] in seen:
                rv += 1
            seen.add(b[:3])
        else:
            nr += 1
    s0 = np.array(P[0][:3])
    span = max(int(np.abs(np.array(p[:3]) - s0).sum()) for p in P)
    return {"n_trans": nt, "n_rot": nr, "n_hover": nh, "n_pos": len({p[:3] for p in P}), "n_pose": len(set(P)),
            "revisit_pos": rv, "span": span}


def no_depth(svc_dir):
    n = 0
    for f in glob.glob(os.path.join(svc_dir, "svc_*_t*.npz")):
        if f.endswith("_rgb.npz"):
            continue
        z = np.load(f)
        zb, mk = z["zbuf"], z["mask"].astype(bool)
        n += int(not (mk & (zb > 0) & (zb < 70)).any())
    return n


def frf_episode(root, s, k):
    d = os.path.join(root, f"{s}_s{k}")
    ep = glob.glob(os.path.join(d, "episodes", "ep_*.json"))
    if not ep:
        return None
    e = json.load(open(ep[0]))
    keys = e["executed_keys"]
    r = keys_stats(keys)
    svc = glob.glob(os.path.join(d, "svc", "*"))
    r["no_depth"] = no_depth(svc[0]) if svc else None
    pl = glob.glob(os.path.join(d, "planner_logs", "*.steps.jsonl"))
    if pl:
        rows = [json.loads(x) for x in open(pl[0])]
    if pl and rows and "pose" in rows[0]:
        fb = [r_["fallback"] for r_ in rows]
        r["n_fallback_turn"] = sum(1 for x in fb if x and "turn" in x)
        r["n_nomap_random"] = sum(1 for x in fb if x == "no_map_seeded_random")
        r["n_path_moves"] = sum(1 for x in fb if x is None)
        reps = [rp for r_ in rows for rp in r_.get("replans", [])]
        r["n_replans"] = len(reps)
        gd, ok = [], 0
        for r_ in rows:
            for rp in r_.get("replans", []):
                g = rp["info"].get("best_goal")
                if g:
                    gd.append(int(np.abs(np.array(g[:3]) - np.array(r_["pose"][:3])).sum()))
        r["goal_l1_median"] = float(np.median(gd)) if gd else None
        r["goal_l1_mean"] = float(np.mean(gd)) if gd else None
        # goal reached: after a replan at t with best_goal g and best_len L, is pose(t + L) == g ?
        P = {r_["t"]: tuple(r_["pose"]) for r_ in rows}
        reached = tried = 0
        for r_ in rows:
            for rp in r_.get("replans", []):
                g, L = rp["info"].get("best_goal"), rp["info"].get("best_len")
                if g and L and r_["t"] + L in P:
                    tried += 1
                    reached += int(P[r_["t"] + L] == tuple(g))
        r["goal_reached_frac"] = reached / tried if tried else None
    return r


def table_episode(method, s, k):
    p = os.path.join(EPI, method, f"{s}_s{k}.json")
    if not os.path.exists(p):
        return None
    rec = json.load(open(p))
    r = keys_stats(rec["executed_keys"])
    ep = rec["episode_file"]
    base = os.path.dirname(os.path.dirname(os.path.join(REPO, ep)))
    tag = os.path.basename(ep)[3:-5]
    svc = os.path.join(base, "svc", tag)
    r["no_depth"] = no_depth(svc) if os.path.isdir(svc) else None
    return r


def main():
    roots = {"f1c4": os.path.join(FRF, "full/outdoor/closed_loop_f1c4"), "v4": os.path.join(FRF, "full/outdoor/closed_loop_v4"),
             "v3": os.path.join(FRF, "full/outdoor/closed_loop_v3"), "D3a": os.path.join(FRF, "outdoor/closed_loop")}
    res = {}
    for m, root in roots.items():
        for s in SCENES:
            for k in range(5):
                r = frf_episode(root, s, k)
                if r is not None:
                    res.setdefault(m, {})[f"{s}_s{k}"] = r
    for m in ("MAG-B", "RANDOM"):
        for s in SCENES:
            for k in range(5):
                r = table_episode(m, s, k)
                if r is not None:
                    res.setdefault(m, {})[f"{s}_s{k}"] = r
    os.makedirs(OUT, exist_ok=True)
    json.dump(res, open(os.path.join(OUT, "a4_behaviour.json"), "w"), indent=1)
    cols = ["n_trans", "n_rot", "n_hover", "n_pos", "n_pose", "revisit_pos", "span", "no_depth", "n_replans", "n_path_moves",
            "n_fallback_turn", "n_nomap_random", "goal_l1_median", "goal_reached_frac"]
    print("method n_ep " + " ".join(cols))
    for m, E in res.items():
        vals = []
        for c in cols:
            v = [e[c] for e in E.values() if e.get(c) is not None]
            vals.append("%.2f" % np.mean(v) if v else "-")
        print(m, len(E), " ".join(vals))
    # per scene, f1c4 vs MAG-B
    print("\nper scene (f1c4 | MAG-B): n_pos, span, no_depth")
    for s in SCENES:
        a = [res["f1c4"][f"{s}_s{k}"] for k in range(5) if f"{s}_s{k}" in res.get("f1c4", {})]
        b = [res["MAG-B"][f"{s}_s{k}"] for k in range(5) if f"{s}_s{k}" in res.get("MAG-B", {})]
        f = lambda L, c: np.mean([x[c] for x in L if x.get(c) is not None]) if L else float("nan")  # noqa: E731
        print("%-15s %5.1f %5.1f %5.1f | %5.1f %5.1f %5.1f" % (s, f(a, "n_pos"), f(a, "span"), f(a, "no_depth"), f(b, "n_pos"), f(b, "span"), f(b, "no_depth")))


if __name__ == "__main__":
    main()
