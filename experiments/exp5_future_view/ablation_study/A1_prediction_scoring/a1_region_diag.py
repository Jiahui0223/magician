"""EXP5 Ablation A1 -- additional diagnostic (DEVIATIONS #7; evaluation side, CPU, numpy only): how much of the MP3D
oracle gain of each candidate lies OUTSIDE V2's planning region (n_win = 5 region of the last refresh t_r, xy box as
task2 t2_common.region_box_m; the T0.12-C split of t12c_oracle.py), and does GT-depth DistP1 rank the in-region gain
better than the full gain?
Independent implementation of the C_obs gain (cached GT points / visibility of the frozen Coverage, cKDTree); the total
gain must reproduce S1 (|d| <= 1e-12).  Output results/a1_region_diag.json
"""
import json
import os
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.join(os.path.dirname(HERE), "common"))
import ab_common as A  # noqa: E402

A.add_paths()
import t2_common as C2  # noqa: E402
import t13_analyze as T13  # noqa: E402

import numpy as np  # noqa: E402
from scipy.spatial import cKDTree  # noqa: E402

TAU = A.TIE_TOL * 100


def main():
    man = A.load_manifest("indoor")
    by = {}
    for st in man["states"]:
        by.setdefault((st["sources"][0]["source"], st["tag"]), []).append(st)
    out = {}
    for (src, tag), sts in sorted(by.items()):
        key = sts[0]["scene"]
        cz = np.load(os.path.join(A.REPO, "work_dirs", "exp5_future_view", "task0", "preprocessing", "coverage", f"{key}.npz"))
        G, vis, eps = cz["pts"], cz["visible"].astype(bool), float(cz["eps_rt"])
        Gxy = C2.rt_to_zup_m(G)[:, :2]
        card = C2.load_card(key)
        nvis = vis.sum()
        cov = np.zeros(len(G), bool)
        want = {s["t"]: s for s in sts}

        def near(P):
            if len(P) == 0:
                return np.zeros(len(G), bool)
            d, _ = cKDTree(np.asarray(P, np.float64)).query(G, k=1, distance_upper_bound=eps)
            return np.isfinite(d)
        for t in range(max(want) + 1):
            cov |= near(np.load(os.path.join(A.CL_IN, src, tag, "svc", f"svc_t{t}.npz"))["sample_points"])
            if t not in want:
                continue
            st = want[t]
            tr = 10 * (t // 10)
            box = C2.region_box_m(card, np.load(os.path.join(A.CL_IN, src, tag, "svc", f"svc_t{tr}.npz"))["cam_center"], C2.N_WIN)
            inside = np.all((Gxy >= box[0, :2]) & (Gxy <= box[1, :2]), 1)
            o = np.load(os.path.join(A.state_dir("indoor", st["sid"]), "oracle.npz"))
            off = o["cand_off"]
            g_all, g_in = [], []
            for j in range(len(off) - 1):
                new = near(o["cand_pts"][off[j]:off[j + 1]]) & vis & ~cov
                g_all.append(float(new.sum() / nvis))
                g_in.append(float((new & inside).sum() / nvis))
            assert np.allclose(g_all, o["gain"], rtol=0, atol=1e-12), ("total gain differs from S1", st["sid"])
            rem = vis & ~cov
            out[st["sid"]] = {"gain_in": g_in, "gain_all": g_all, "remaining_in_region_share": float((rem & inside).sum() / max(1, rem.sum()))}
        print(src, tag, "done", flush=True)
    # ranking of GTD / V2M against the in-region gain
    rows = []
    for st in man["states"]:
        d = out[st["sid"]]
        v = json.load(open(os.path.join(A.state_dir("indoor", st["sid"]), "v2.json")))
        C = np.asarray(st["cand"])
        ga, gi = np.asarray(d["gain_all"])[C] * 100, np.asarray(d["gain_in"])[C] * 100
        for m, s in (("GTD", np.asarray(v["gtd_distp1"])[C]), ("V2M", np.asarray(v["models"]["V2M"]["distp1"])[C]),
                     ("V2O", np.asarray(v["models"]["V2O"]["distp1"])[C])):
            pk = int(np.argmax(s))
            rows.append({"sid": st["sid"], "scene": st["scene"], "t": st["t"], "scorer": m,
                         "rho_all": T13.spearman(s, ga), "rho_in": T13.spearman(s, gi),
                         "top1_all": float(ga[pk] >= ga.max() - TAU), "top1_in": float(gi[pk] >= gi.max() - TAU),
                         "best_out_share": float(1 - gi[np.argmax(ga)] / ga.max()) if ga.max() > TAU else None,
                         "remaining_in_region_share": d["remaining_in_region_share"]})
    summ = {}
    for m in ("GTD", "V2M", "V2O"):
        r = [x for x in rows if x["scorer"] == m]
        f = lambda k, rr=r: float(np.nanmean([x[k] for x in rr if x[k] is not None]))   # noqa: E731
        fail = [x for x in r if x["top1_all"] == 0]
        ok = [x for x in r if x["top1_all"] == 1]
        summ[m] = {"rho_all": f("rho_all"), "rho_in": f("rho_in"), "top1_all": f("top1_all"), "top1_in": f("top1_in"),
                   "best_out_share_when_wrong": f("best_out_share", fail), "best_out_share_when_right": f("best_out_share", ok),
                   "states_wrong": len(fail), "states": len(r)}
    summ["remaining_in_region_share_mean"] = float(np.mean([x["remaining_in_region_share"] for x in rows if x["scorer"] == "GTD"]))
    json.dump({"summary": summ, "states": out}, open(os.path.join(HERE, "results", "a1_region_diag.json"), "w"))
    print(json.dumps(summ, indent=1))


if __name__ == "__main__":
    main()
