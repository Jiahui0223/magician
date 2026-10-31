"""EXP6 V3 Task 2 / Experiment 2 -- R1 scoring-region counterfactual on the MP3D fixed states (CPU, numpy / scipy; GT only
offline; protocol §4 R1).

For every MP3D state of Pool CL (5 eval floors) and Pool FX VALSEL (dev), per legal candidate:
  * the oracle gain (frozen C_obs rule, recomputed from the cached GT points / visibility of the level, the replayed
    covered state of the records 0..t and the candidate's stored frame-(t+1) sample points of oracle.npz; must equal
    oracle.npz:gain within 1e-12) and its part inside the Task 2 planning region for n_win in {5, 7, 9} around the
    refresh camera (t_r = 10 floor(t/10)) and in the whole level;
  * GT-depth scores with the region cap at n_win in {5, 7, 9, inf}: the true depth t* (whole mesh) capped at the box
    exit, weights min(1,(z/z0)^2) (S0 cap) with the point-based novelty of Task 2 (N_hard at eps, N_lin) against the
    historical 10 % sample points, dedup by voxel -> regret / Top-1 vs the oracle gain of the whole level.
GT is used only here (evaluation side). Output results/r1_region.json, results/r1_states.csv
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
sys.path.insert(0, os.path.join(E6, "ablation_study", "shared"))
sys.path.insert(0, os.path.join(REPO, "experiments", "exp5_future_view", "ablation_study", "common"))
sys.path.insert(0, os.path.join(REPO, "experiments", "exp5_future_view", "task2", "scripts"))
sys.path.insert(0, os.path.join(REPO, "experiments", "exp5_future_view", "task3_full_depth_p1", "e2_pilot", "scripts"))
import ab_data as D  # noqa: E402
import ab_common as A  # noqa: E402
import t2_common as C2  # noqa: E402

W0 = os.path.join(REPO, "work_dirs", "exp5_future_view", "task0")
E2W = os.path.join(REPO, "work_dirs", "exp5_future_view", "task3_full_depth_p1", "e2")
OUT = os.path.join(EXP, "results")
NWINS = (5, 7, 9, 0)                 # 0 = no region cap (whole level, sensor range only)
ALPHA = 4 * 2 * np.tan(np.radians(30.0)) / 255.0
VS = 0.011


def near(tree_pts, G, eps):
    if len(tree_pts) == 0:
        return np.zeros(len(G), bool)
    d, _ = cKDTree(np.asarray(tree_pts, np.float64)).query(G, k=1, distance_upper_bound=eps)
    return np.isfinite(d)


def rays_for(R, T):
    """Scoring rays (runtime y-up): directions d_y [R,3], dz [R], centre c (m_common.pixel_rays on the 64 x 114 grid)."""
    import m_common as MC
    from seen2scene.futureview import rays as RY
    u, v = RY.grid_uv((0.0, 0.0))
    d_y, dz, c = MC.pixel_rays(np.asarray(R, np.float64), np.asarray(T, np.float64), u, v)
    return d_y, dz, c


def box_exit_t(c, d, lo, hi):
    """Ray length (runtime) at which the ray leaves the axis-aligned box [lo, hi] (y-up runtime coords); inf if outside."""
    inv = 1.0 / np.where(np.abs(d) < 1e-12, 1e-12, d)
    ta, tb = (lo - c) * inv, (hi - c) * inv
    t_out = np.minimum(ta, tb).max(1)
    t_exit = np.maximum(ta, tb).min(1)
    inside = np.all((c >= lo) & (c <= hi))
    return np.where(inside, np.maximum(t_exit, 0.0), np.where(t_exit > t_out, t_exit, 0.0))


def region_box_rt(card, center_rt, n_win):
    """Task 2 region (n_win windows) as a runtime y-up box: xy from region_box_m, z = level box."""
    box_m = C2.region_box_m(card, center_rt, n_win) if n_win else np.asarray(card["box_m"], np.float64)
    c = C2.zup_m_to_rt(box_m)
    return c.min(0), c.max(0)


def score_candidate(tstar_model, R, T, k_M, z0, hist_tree, eps_rt, lo, hi, cap_region):
    """GT-depth scores of one candidate under one region cap: returns dict of variants."""
    d_y, dz, c = rays_for(R, T)
    t_rt = np.asarray(tstar_model, np.float64) * k_M                      # runtime ray length
    t_rt = np.where(t_rt * dz <= 70.0, t_rt, np.inf)                       # sensor range on the z-depth
    if cap_region:
        t_rt = np.where(t_rt <= box_exit_t(c, d_y, lo, hi), t_rt, np.inf)
    fin = np.isfinite(t_rt)
    q = c[None] + d_y * np.where(fin, t_rt, 0.0)[:, None]
    z = t_rt * dz
    cap = np.where(fin, np.minimum(1.0, (z / z0) ** 2), 0.0)
    area = np.where(fin, (z / z0) ** 2, 0.0)                               # S1c-like (no cos; cos needs the grid; kept simple here)
    dobs = np.full(len(t_rt), np.inf)
    if fin.any():
        dobs[fin] = hist_tree.query(q[fin], k=1, workers=8)[0] if hist_tree is not None else np.inf
    n_lin = np.clip(dobs / (2 * eps_rt), 0, 1)
    n_hard = (dobs > eps_rt).astype(float)
    vox = np.floor(q / (VS * k_M)).astype(np.int64)
    key = (vox[:, 0] * 1000003 + vox[:, 1]) * 1000003 + vox[:, 2]
    out = {}
    for name, w in (("cap_hard", cap * n_hard), ("cap_lin", cap * n_lin), ("area_lin", area * n_lin)):
        if not fin.any():
            out[name] = 0.0
            continue
        ids, inv = np.unique(key[fin], return_inverse=True)
        wv = np.zeros(len(ids)); np.maximum.at(wv, inv, w[fin])
        out[name] = float(wv.sum())
    return out


def main():
    os.makedirs(OUT, exist_ok=True)
    rows = []
    # ---- Pool CL (eval floors) and Pool FX VALSEL (dev)
    jobs = []
    man = A.load_manifest("indoor")
    by = {}
    for st in man["states"]:
        by.setdefault((st["sources"][0]["source"], st["tag"]), []).append(st)
    for (src, tag), sts in sorted(by.items()):
        jobs.append(("cl", src, tag, sts, lambda t, src=src, tag=tag: os.path.join(A.CL_IN, src, tag, "svc", f"svc_t{t}.npz"),
                     lambda sid: os.path.join(A.state_dir("indoor", sid), "oracle.npz"), sts[0]["scene"]))
    # Pool FX VALSEL is NOT included: its oracle (e2_valsel_oracle.py) uses the 100k t21_traj GT points, not the frozen
    # C_obs point set of the evaluation floors, so the gain cannot be reproduced here (Task 2 deviation T2-1).
    for pool, src, tag, sts, svc, orc_of, scene in jobs:
        cz = np.load(os.path.join(W0, "preprocessing", "coverage", f"{scene}.npz"))
        G, vis, eps = cz["pts"], cz["visible"].astype(bool), float(cz["eps_rt"])
        card = C2.load_card(scene)
        k_M, z0 = float(card["k_I"]), float(card["z0_rt"])
        nvis = max(1, int(vis.sum()))
        Gxy = C2.rt_to_zup_m(G)[:, :2]
        cov = np.zeros(len(G), bool)
        hist = []
        want = {int(s["t"]): s for s in sts}
        for t in range(max(want) + 1):
            z = np.load(svc(t))
            pts = z["sample_points"]
            cov |= near(pts, G, eps)
            hist.append(np.asarray(pts, np.float64))
            if t not in want:
                continue
            st = want[t]
            tr = 10 * (t // 10)
            cam_r = np.load(svc(tr))["cam_center"]
            o = np.load(orc_of(st["sid"]))
            off = o["cand_off"]
            rem = vis & ~cov
            tree = cKDTree(np.concatenate(hist))
            legal_R, legal_T = z["legal_R"], z["legal_T"]
            cand = [int(x) for x in st["cand"]]
            g_all, g_in = [], {n: [] for n in NWINS if n}
            boxes = {n: region_box_rt(card, cam_r, n) for n in NWINS}
            boxes_m = {n: C2.region_box_m(card, cam_r, n) for n in NWINS if n}
            inside = {n: np.all((Gxy >= b[0, :2]) & (Gxy <= b[1, :2]), 1) for n, b in boxes_m.items()}
            for j in range(len(off) - 1):
                new = near(o["cand_pts"][off[j]:off[j + 1]], G, eps) & rem
                g_all.append(float(new.sum() / nvis))
                for n in g_in:
                    g_in[n].append(float((new & inside[n]).sum() / nvis))
            assert np.max(np.abs(np.asarray(g_all) - np.asarray(o["gain"]))) <= 1e-12, (tag, t, "oracle gain reproduction")
            scores = {n: {} for n in NWINS}
            for j in range(len(legal_R)):
                for n in NWINS:
                    lo, hi = boxes[n]
                    sc = score_candidate(o["tstar"][j], legal_R[j], legal_T[j], k_M, z0, tree, eps, lo, hi, cap_region=bool(n))
                    for k, v in sc.items():
                        scores[n].setdefault(k, []).append(v)
            g = np.asarray(g_all)[cand]
            row = {"pool": pool, "split": "dev" if pool == "fx" else "eval", "scene": scene, "start": st["start"], "t": t, "sid": st["sid"], "source": src, "tag": tag,
                   "n_cand": len(cand), "gstar_pp": float(g.max() * 100), "remaining_pp": float(rem.sum() / nvis * 100),
                   "remaining_in_region": {str(n): float((rem & inside[n]).sum() / max(1, rem.sum())) for n in inside},
                   "best_gain_in_region_share": {str(n): float(np.asarray(g_in[n])[cand][int(np.argmax(g))] / max(1e-12, g.max())) for n in g_in},
                   "metrics": {}}
            for n in NWINS:
                for k, s in scores[n].items():
                    s = np.asarray(s)
                    pk, fb = D.pick_indoor(s, cand, f"R1|{n}|{k}", tag, t)
                    row["metrics"][f"n{n}|{k}"] = D.rank_metrics(s[cand], g, cand.index(pk))
            rows.append(row)
            print(pool, tag, t, "G*", round(row["gstar_pp"], 3), {f"n{n}": round(row["metrics"][f"n{n}|cap_lin"]["regret"], 3) for n in NWINS}, flush=True)
    # ---- aggregate (scene-level bootstrap; episode = tag|source)
    for r in rows:
        r["episode"] = f"{r['tag']}|{r['source']}"
    res = {"n_states": len(rows), "tables": {}}
    for split in ("dev", "eval"):
        sub = [r for r in rows if r["split"] == split]
        if not sub:
            continue
        res["tables"][split] = {}
        for key in sub[0]["metrics"]:
            res["tables"][split][key] = {m: D.boot(D.hier_mean(sub, lambda r, key=key, m=m: (None if r["metrics"][key].get(m) is None else float(r["metrics"][key][m]))).values()) for m in ("regret", "top1", "rho")}
        res["tables"][split]["remaining_in_region"] = {n: D.boot(D.hier_mean(sub, lambda r, n=n: r["remaining_in_region"][n]).values()) for n in sub[0]["remaining_in_region"]}
        res["tables"][split]["best_gain_in_region_share"] = {n: D.boot(D.hier_mean(sub, lambda r, n=n: r["best_gain_in_region_share"][n]).values()) for n in sub[0]["best_gain_in_region_share"]}
    json.dump({"summary": res, "states": rows}, open(os.path.join(OUT, "r1_region.json"), "w"), indent=1, default=float)
    with open(os.path.join(OUT, "r1_states.csv"), "w", newline="") as fh:
        w = csv.writer(fh)
        keys = list(rows[0]["metrics"])
        w.writerow(["pool", "split", "scene", "start", "t", "sid", "n_cand", "gstar_pp", "remaining_pp"] + [f"rem_in_{n}" for n in rows[0]["remaining_in_region"]] + [f"regret:{k}" for k in keys] + [f"top1:{k}" for k in keys])
        for r in rows:
            w.writerow([r["pool"], r["split"], r["scene"], r["start"], r["t"], r["sid"], r["n_cand"], r["gstar_pp"], r["remaining_pp"]] + list(r["remaining_in_region"].values()) +
                       [r["metrics"][k]["regret"] for k in keys] + [int(r["metrics"][k]["top1"]) for k in keys])
    for split, t in res["tables"].items():
        print(split, {k: round(v["regret"]["mean"], 3) for k, v in t.items() if "regret" in v}, "rem_in_region", {n: round(v["mean"], 2) for n, v in t["remaining_in_region"].items()})


if __name__ == "__main__":
    main()
