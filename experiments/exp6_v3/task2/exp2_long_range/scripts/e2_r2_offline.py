"""EXP6 V3 Task 2 / Experiment 2 -- R2 offline evaluation of the frontier-progress term on the MP3D fixed states (CPU;
GT only on the evaluation side; protocol §4 R2).

Per Pool CL MP3D state (368; source trajectories V2O / V2M / MAG-G / MAG-B): the frontier map is rebuilt from the
records 0..t (planner side), FP of every legal candidate; the frozen B2 K-mean DistP1 of the Experiment A cache is the
model gain. Evaluated (evaluation side, oracle.npz gains + the replayed covered state of the coverage GT points):
  * pick under S0 (B2), bonus (Gain + lambda FP), trigger (FP first when max Gain < theta) -> one-step oracle regret;
  * at the Experiment A stall states and the non-stall states separately;
  * frontier validity: for the nearest reachable frontier cell, the share of the remaining (uncovered, visible) GT points
    within 3 m of that cell (is there anything to gain in that direction?) and the distance;
  * how often the trigger fires; how often FP-driven moves differ from B2's pick.
Output results/r2_offline.json, results/r2_states.csv
"""
import csv
import json
import os
import sys

import numpy as np
from scipy.spatial import cKDTree

HERE = os.path.dirname(os.path.abspath(__file__))
EXP = os.path.dirname(HERE)
T2 = os.path.dirname(EXP)
E6 = os.path.dirname(T2)
REPO = os.path.dirname(os.path.dirname(E6))
sys.path.insert(0, os.path.join(T2, "common"))
sys.path.insert(0, os.path.join(E6, "ablation_study", "shared"))
sys.path.insert(0, os.path.join(REPO, "experiments", "exp5_future_view", "ablation_study", "common"))
sys.path.insert(0, os.path.join(REPO, "experiments", "exp5_future_view", "task2", "scripts"))
sys.path.insert(0, os.path.join(REPO, "experiments", "exp3b_macaronspp_transfer"))
import t2_frontier as TF  # noqa: E402
import ab_data as D  # noqa: E402
import ab_common as A  # noqa: E402
import t2_common as C2  # noqa: E402
import m_common as MC  # noqa: E402
import mp3d_frame as F  # noqa: E402

W0 = os.path.join(REPO, "work_dirs", "exp5_future_view", "task0")
WA = os.path.join(REPO, "work_dirs", "exp6_v3", "ablation_study", "cache", "cl", "indoor")
OUT = os.path.join(EXP, "results")
LAM, THETA = 119.1, 75.3            # protocol §4 R2 (VALSEL B2 DistP1: median non-zero; 20 % quantile of the per-state max)
NEAR_M = 3.0


def near(tree_pts, G, eps):
    if len(tree_pts) == 0:
        return np.zeros(len(G), bool)
    d, _ = cKDTree(np.asarray(tree_pts, np.float64)).query(G, k=1, distance_upper_bound=eps)
    return np.isfinite(d)


def main():
    os.makedirs(OUT, exist_ok=True)
    recs_cl, _ = D.load_pool("cl", "indoor")
    stall_of = {r["sid"]: bool(r.get("stall", False)) for r in recs_cl}
    man = A.load_manifest("indoor")
    by = {}
    for st in man["states"]:
        by.setdefault((st["sources"][0]["source"], st["tag"]), []).append(st)
    rows = []
    for (src, tag), sts in sorted(by.items()):
        scene = sts[0]["scene"]
        card = F.load_card(scene)
        cz = np.load(os.path.join(W0, "preprocessing", "coverage", f"{scene}.npz"))
        G, vis, eps = cz["pts"], cz["visible"].astype(bool), float(cz["eps_rt"])
        nvis = max(1, int(vis.sum()))
        fm = TF.FrontierMap(card)
        cov = np.zeros(len(G), bool)
        want = {int(s["t"]): s for s in sts}
        svc = lambda t: os.path.join(A.CL_IN, src, tag, "svc", f"svc_t{t}.npz")
        for t in range(max(want) + 1):
            rec = MC.load_service_record(svc(t))
            fm.add_record(rec)
            cov |= near(rec["sample_points"], G, eps)
            if t not in want:
                continue
            st = want[t]
            cj = json.load(open(os.path.join(WA, st["sid"] + ".json")))
            gain = np.asarray(cj["oracle_gain"], float)
            b2 = np.asarray(cj["models"]["B2"]["distp1"], float)
            cand = [int(x) for x in st["cand"]]
            fp, info = fm.progress(rec, rec["legal_R"], rec["legal_T"])
            fp = np.asarray(fp, float)
            picks = {}
            picks["S0"] = D.pick_indoor(b2, cand, "S0", tag, t)[0]
            picks["bonus"] = D.pick_indoor(b2 + LAM * fp, cand, "bonus", tag, t)[0]
            trig = bool(b2[cand].max() < THETA and info["n_reachable_frontier"] > 0)
            picks["trigger"] = D.pick_indoor(fp * 1e6 + b2, cand, "trigger", tag, t)[0] if trig else picks["S0"]
            g = gain[cand]
            # frontier validity: remaining GT points near the nearest reachable frontier cell
            free, occ, unk = fm.states()
            fr = fm.frontier(free, unk)
            rem = vis & ~cov
            Gl, Gh = fm.to_cell(G)
            valid = None
            if info["dist_to_frontier"] >= 0 and fr.any():
                passable = free.copy(); passable[info["cell"][0], info["cell"][1]] = True
                dist = TF.FrontierMap.bfs(passable | fr, [tuple(info["cell"])])
                cells = np.array(np.nonzero(fr)).T
                dd = np.array([dist[l, h] for l, h in cells])
                ok = dd >= 0
                if ok.any():
                    l, h = cells[ok][np.argmin(dd[ok])]
                    centre = fm.x_min + (np.array([l + 0.5, 0.5, h + 0.5])) * fm.cell
                    dist_rt = np.linalg.norm(G[:, [0, 2]] - centre[[0, 2]][None], axis=1)
                    near_mask = dist_rt < NEAR_M * 7.0
                    valid = {"frontier_cell": [int(l), int(h)], "path_dist": int(dd[ok].min()), "remaining_near_frontier_pp": float((rem & near_mask).sum() / nvis * 100),
                             "remaining_share_near_frontier": float((rem & near_mask).sum() / max(1, rem.sum()))}
            row = {"scene": scene, "start": st["start"], "t": t, "sid": st["sid"], "source": src, "tag": tag, "episode": f"{tag}|{src}", "stall": stall_of.get(st["sid"], False),
                   "n_cand": len(cand), "gstar_pp": float(g.max() * 100), "max_b2": float(b2[cand].max()), "triggered": trig, "fp": fp.tolist(),
                   "n_fp_pos": int((fp[cand] > 0).sum()), "remaining_pp": float(rem.sum() / nvis * 100), "frontier": {k: info[k] for k in ("n_free", "n_occ", "n_unknown", "n_frontier", "n_reachable_frontier", "dist_to_frontier")},
                   "validity": valid, "picks": picks, "regret": {m: float((g.max() - gain[p]) * 100) for m, p in picks.items()}, "gain_pick": {m: float(gain[p] * 100) for m, p in picks.items()},
                   "fp_pick": {m: float(fp[p]) for m, p in picks.items()}}
            rows.append(row)
            print(tag, t, "stall" if row["stall"] else "", "G*", round(row["gstar_pp"], 3), "maxB2", round(row["max_b2"], 1), "trig", trig, "regret", {m: round(v, 3) for m, v in row["regret"].items()}, "dF", info["dist_to_frontier"], flush=True)
    res = {"lambda": LAM, "theta": THETA, "n_states": len(rows), "n_stall": int(sum(r["stall"] for r in rows)), "n_triggered": int(sum(r["triggered"] for r in rows)), "tables": {}}
    for name, sub in (("all", rows), ("stall", [r for r in rows if r["stall"]]), ("non_stall", [r for r in rows if not r["stall"]]), ("triggered", [r for r in rows if r["triggered"]])):
        if not sub:
            continue
        res["tables"][name] = {"n": len(sub), "regret": {m: D.boot(D.hier_mean(sub, lambda r, m=m: r["regret"][m]).values()) for m in ("S0", "bonus", "trigger")},
                               "gain_pick": {m: D.boot(D.hier_mean(sub, lambda r, m=m: r["gain_pick"][m]).values()) for m in ("S0", "bonus", "trigger")},
                               "fp_pick_pos": {m: float(np.mean([r["fp_pick"][m] > 0 for r in sub])) for m in ("S0", "bonus", "trigger")},
                               "pick_differs_from_S0": {m: float(np.mean([r["picks"][m] != r["picks"]["S0"] for r in sub])) for m in ("bonus", "trigger")},
                               "has_reachable_frontier": float(np.mean([r["frontier"]["n_reachable_frontier"] > 0 for r in sub])),
                               "dist_to_frontier_median": float(np.median([r["frontier"]["dist_to_frontier"] for r in sub if r["frontier"]["dist_to_frontier"] >= 0] or [-1])),
                               "remaining_share_near_frontier": float(np.mean([r["validity"]["remaining_share_near_frontier"] for r in sub if r["validity"]] or [0])),
                               "remaining_near_frontier_pp": float(np.mean([r["validity"]["remaining_near_frontier_pp"] for r in sub if r["validity"]] or [0])),
                               "gstar_pp_median": float(np.median([r["gstar_pp"] for r in sub]))}
    json.dump({"summary": res, "states": rows}, open(os.path.join(OUT, "r2_offline.json"), "w"), indent=1, default=float)
    with open(os.path.join(OUT, "r2_states.csv"), "w", newline="") as fh:
        w = csv.writer(fh)
        w.writerow(["scene", "start", "t", "source", "stall", "n_cand", "gstar_pp", "max_b2", "triggered", "dist_to_frontier", "n_reachable_frontier", "remaining_pp", "remaining_share_near_frontier",
                    "regret_S0", "regret_bonus", "regret_trigger", "pick_S0", "pick_bonus", "pick_trigger", "fp_pick_bonus", "fp_pick_trigger"])
        for r in rows:
            w.writerow([r["scene"], r["start"], r["t"], r["source"], int(r["stall"]), r["n_cand"], r["gstar_pp"], r["max_b2"], int(r["triggered"]), r["frontier"]["dist_to_frontier"], r["frontier"]["n_reachable_frontier"],
                        r["remaining_pp"], r["validity"]["remaining_share_near_frontier"] if r["validity"] else "", r["regret"]["S0"], r["regret"]["bonus"], r["regret"]["trigger"],
                        r["picks"]["S0"], r["picks"]["bonus"], r["picks"]["trigger"], r["fp_pick"]["bonus"], r["fp_pick"]["trigger"]])
    print(json.dumps({k: {m: round(v["regret"][m]["mean"], 3) for m in v["regret"]} for k, v in res["tables"].items()}), "triggered", res["n_triggered"], "/", res["n_states"])


if __name__ == "__main__":
    main()
