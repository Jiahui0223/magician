"""EXP5 Task 0 / T0.12 -- post-hoc DESCRIPTIVE diagnostic of the grid action space (evaluation side; uses the GT mesh;
not pre-registered, changes no result). Magician env, CPU.
Per T0.12 episode: the grid positions reachable from the start by legal translations (official single-segment rule,
the same Blocker as t12_sim), and how much of the level they reach:
  stations_reached  share of the level's panorama stations with a reachable grid position within 1.0 m horizontally
                    and 1.5 m vertically (proxy for "rooms the grid can enter")
  regions_reached   share of the level's .house regions (region of those panoramas)
  no_floor          reachable positions with no GT surface within (height above floor + 0.5 m) straight below
                    (over a void: outside the building, stairwell, missing floor)
and per group the share of executed positions that are no_floor positions.
Output W/magician_comparison/analysis/t12_reach.json
"""
import json
import os
import sys
from collections import deque

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
sys.path.insert(0, os.path.join(HERE, "..", "adapters"))
import mc_common as M  # noqa: E402
import grid_env as GE  # noqa: E402
import t12_sim as T12  # noqa: E402

S = M.S


def first_hits(ri, P, D):
    """Distance to the first hit along each ray (inf if none)."""
    out = np.full(len(P), np.inf)
    for i in range(0, len(P), 4000):
        loc, idx, _ = ri.intersects_location(P[i:i + 4000], D[i:i + 4000], multiple_hits=False)
        if len(idx):
            d = np.linalg.norm(loc - P[i:i + 4000][idx], axis=1)
            np.minimum.at(out, i + idx, d)
    return out


def main():
    res = {"doc": __doc__, "levels": {}, "episodes": {}, "groups": {}}
    by_level = {}
    for tag, scan, lv, st in M.grid_episodes():
        by_level.setdefault((scan, lv), []).append((tag, st))
    for (scan, lv), eps in by_level.items():
        L = S.Level(scan, lv)
        crd = M.card(scan, lv)
        g = GE.grid_def(crd)
        v, f = S.mesh_zup(scan)
        blk = T12.Blocker(v, f)
        n = g["n"]
        idx = [tuple(int(c) for c in i) for i in np.ndindex(*n)]
        pos = {i: S.rt_to_zup_m(GE.position(g, i)) for i in idx}
        # all axis-adjacent position pairs, blocked or not (batched single-segment test)
        pairs = [(i, i[:a] + (i[a] + 1,) + i[a + 1:]) for i in idx for a in range(3) if i[a] + 1 < n[a]]
        P = np.stack([pos[a] for a, _ in pairs])
        Q = np.stack([pos[b] for _, b in pairs])
        D = Q - P
        ln = np.linalg.norm(D, axis=1)
        hit = first_hits(blk.ri, P, D / ln[:, None])
        adj = {i: [] for i in idx}
        for (a, b), h, l in zip(pairs, hit, ln):
            if not h < l:
                adj[a].append(b)
                adj[b].append(a)
        X = np.stack([pos[i] for i in idx])
        down = first_hits(blk.ri, X, np.tile([0.0, 0.0, -1.0], (len(X), 1)))
        nofloor = {i: bool(d > X[k, 2] - crd["floor_m"] + 0.5) for k, (i, d) in enumerate(zip(idx, down))}
        Pp = S.rt_to_zup_m(np.asarray(crd["pano_rt"], np.float64))
        reg = M.regions(scan, crd)
        res["levels"][L.key] = {"grid_n": list(n), "n_positions": len(idx), "n_edges": len(pairs),
                                "edges_legal": int(sum(len(v) for v in adj.values()) // 2), "no_floor_positions": int(sum(nofloor.values())),
                                "n_stations": len(Pp), "n_regions": len(set(reg))}
        for tag, st in eps:
            ep = json.load(open(os.path.join(M.MC, "grid_protocol", "closed_loop", "G-MAG-B", tag, "episode.json")))
            s0 = tuple(ep["grid"]["start_pose"][:3])
            seen, q = {s0}, deque([s0])
            while q:
                a = q.popleft()
                for b in adj[a]:
                    if b not in seen:
                        seen.add(b)
                        q.append(b)
            R = np.stack([pos[i] for i in seen])
            ok = [bool(np.any((np.linalg.norm(R[:, :2] - p[:2], axis=1) <= 1.0) & (np.abs(R[:, 2] - p[2]) <= 1.5))) for p in Pp]
            e = {"reachable_positions": len(seen), "reachable_share": len(seen) / len(idx),
                 "reachable_no_floor": int(sum(nofloor[i] for i in seen)),
                 "stations_reached": float(np.mean(ok)), "regions_reached": len({reg[j] for j, o in enumerate(ok) if o}) / len(set(reg))}
            for grp in M.GRID_GROUPS:
                ex = json.load(open(os.path.join(M.MC, "grid_protocol", "closed_loop", grp, tag, "episode.json")))["executed"]
                ps = {tuple(x[:3]) for x in ex}
                e[grp] = {"distinct_positions": len(ps), "no_floor_positions": int(sum(nofloor[p] for p in ps)),
                          "frames_at_no_floor": float(np.mean([nofloor[tuple(x[:3])] for x in ex])),
                          "share_of_reachable_visited": len(ps) / len(seen)}
            res["episodes"][tag] = e
            print(tag, {k: (round(v, 3) if isinstance(v, float) else v) for k, v in e.items() if not k.startswith("G-")}, flush=True)
    for grp in M.GRID_GROUPS:
        res["groups"][grp] = {k: float(np.mean([e[grp][k] for e in res["episodes"].values()]))
                              for k in ("distinct_positions", "no_floor_positions", "frames_at_no_floor", "share_of_reachable_visited")}
        print(grp, {k: round(v, 3) for k, v in res["groups"][grp].items()}, flush=True)
    json.dump(res, open(os.path.join(M.MC, "analysis", "t12_reach.json"), "w"), indent=1)


if __name__ == "__main__":
    main()
