"""EXP5 Task 3 / E1.2-C -- how much of the metric-new surface was already SEEN (CPU, evaluation only).

For every S_B0 state and candidate of C: the GT point of each scoring ray (t*) is
  metric-new  : farther than the coverage eps from every past 10 % sample point (records 0..t; the coverage metrics'
                own observation set; = the pixel truth of e12_fixed)
  seen-full   : within eps of a past FULL-depth point (all valid pixels of records 0..t, the BAND points of D100)
Share of metric-new rays that are seen-full = surface the full-depth map knows but the coverage metric still counts as
new (definition gap between dense scoring and the 10 %-sample coverage evaluation).
Output analysis/e12_truth_audit.json. Usage: python e12_truth_audit.py [n_proc]
"""
import json
import os
import sys
from multiprocessing import Pool

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
import e12_fixed as X  # noqa: E402

X.setup("B0")
np, MC, A, D = X.np, X.MC, X.A, X.D
from scipy.spatial import cKDTree  # noqa: E402


def one(args):
    d, tag, sts = args
    ep = X.S2.episode_of(d, sts[0])
    T = max(s["t"] for s in sts)
    recs = [MC.load_service_record(A.svc_path(d, "V2M", ep, t)) for t in range(T + 1)]
    uu, vv = MC.scoring_pixel_grid()
    out = []
    for st in sorted(sts, key=lambda s: s["t"]):
        t = st["t"]
        orc = np.load(os.path.join(A.state_dir(d, st["sid"]), "oracle.npz"))
        k_M = float(orc["k_M"])
        eps = 0.08 * k_M if d == "indoor" else float(orc["eps"])
        s10 = cKDTree(np.concatenate([np.asarray(r["sample_points"], np.float64) for r in recs[:t + 1]]))
        full = cKDTree(np.concatenate([D.density_points(r, 100)[0].astype(np.float64) for r in recs[:t + 1]]))
        c = {"gt": 0, "new": 0, "new_seen_full": 0, "old": 0}
        for j in st["cand"]:
            ts = orc["tstar"][j].astype(np.float64)
            d_y, dz, cc = MC.pixel_rays(np.asarray(recs[t]["legal_R"][j], np.float64), np.asarray(recs[t]["legal_T"][j], np.float64), uu, vv)
            ok = np.isfinite(ts) & (ts * k_M * dz < MC.SENSOR_RANGE)
            p = cc[None] + d_y[ok] * (ts[ok] * k_M)[:, None]
            new = ~np.isfinite(s10.query(p, k=1, distance_upper_bound=eps)[0])
            seen = np.isfinite(full.query(p, k=1, distance_upper_bound=eps)[0])
            c["gt"] += int(ok.sum())
            c["new"] += int(new.sum())
            c["old"] += int((~new).sum())
            c["new_seen_full"] += int((new & seen).sum())
        out.append({"sid": st["sid"], "scene": st["scene"], "start": st["start"], "t": t, "phase": X.phase_of(d, t), **c})
    return out


def main():
    n = int(sys.argv[1]) if len(sys.argv) > 1 else 16
    jobs = [(d, tag, sts) for d in ("indoor", "outdoor") for tag, sts in X.states_of(d, "B0")]
    with Pool(n) as pool:
        res = pool.map(one, jobs, chunksize=1)
    rows = {"indoor": [], "outdoor": []}
    for (d, _, _), r in zip(jobs, res):
        rows[d] += r
    summ = {}
    for d, rr in rows.items():
        sc = {}
        for r in rr:
            sc.setdefault(r["scene"], []).append(r)
        per = {s: sum(x["new_seen_full"] for x in v) / max(1, sum(x["new"] for x in v)) for s, v in sc.items()}
        ph = {p: sum(x["new_seen_full"] for x in rr if x["phase"] == p) / max(1, sum(x["new"] for x in rr if x["phase"] == p)) for p in ("early", "middle", "late")}
        summ[d] = {"states": len(rr), "metric_new_share": sum(x["new"] for x in rr) / sum(x["gt"] for x in rr),
                   "new_seen_full_share": sum(x["new_seen_full"] for x in rr) / max(1, sum(x["new"] for x in rr)),
                   "per_scene": per, "per_phase": ph}
    json.dump({"summary": summ, "rows": rows}, open(os.path.join(os.path.dirname(HERE), "analysis", "e12_truth_audit.json"), "w"), indent=1)
    print(json.dumps(summ, indent=1))


if __name__ == "__main__":
    main()
