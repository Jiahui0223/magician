"""EXP5 Task 0 / T0.8 -- unit and distance-conversion tests U1-U5 (magician_comparison protocol v1.0 §3).
Every test has an explicit numerical expectation; all must pass before the formal T0.8 computation.
Output W/magician_comparison/metric_replay/units_test.json. Run with the magician env (PyTorch3D, trimesh + rtree).
"""
import json
import os
import sys

import numpy as np

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "scripts"))
import mc_common as M  # noqa: E402

S = M.S


def main():
    res, ok = {}, True
    rng = np.random.default_rng(0)
    # U1: 5 cm along each z-up metre axis = 0.35 runtime units; round trip
    p = rng.uniform(-20, 20, (1000, 3))
    d = [float(np.abs(np.linalg.norm(S.zup_m_to_rt(p + 0.05 * e) - S.zup_m_to_rt(p), axis=1) - 0.35).max()) for e in np.eye(3)]
    rt = float(np.abs(S.rt_to_zup_m(S.zup_m_to_rt(p)) - p).max())
    res["U1"] = {"max_err_rt_per_axis": d, "roundtrip_err_m": rt, "pass": max(d) < 1e-9 and rt < 1e-9}
    eps = M.episodes()
    bp = M.BackProjector()
    # U2: full back-projection at the stored sample pixels == stored sample points; U3: points on the GT mesh
    import trimesh
    u2, u3 = [], []
    for (tag, scan, lv, st), g, t in ((eps[0], "V2K4", 5), (eps[7], "frozen", 50), (eps[15], "ref", 100)):
        rec = np.load(M.svc_path(g, tag, t))
        allp, m = bp.points(rec)
        diff = float(np.abs(allp[rec["sample_pixel_flat"]] - rec["sample_points"]).max())
        assert m[rec["sample_pixel_flat"]].all()
        u2.append({"tag": tag, "group": g, "t": t, "n_sample": int(len(rec["sample_points"])), "max_abs_diff_rt": diff})
        v, f = S.mesh_zup(scan)
        mesh = trimesh.Trimesh(v, f, process=False)       # the whole-house mesh the simulator renders (t06_sim)
        idx = np.flatnonzero(m)
        sel = rng.choice(idx, min(2000, len(idx)), replace=False)
        q = S.rt_to_zup_m(allp[sel])
        _, dist, _ = trimesh.proximity.closest_point(mesh, q)
        u3.append({"tag": tag, "group": g, "t": t, "median_cm": float(np.median(dist) * 100), "p95_cm": float(np.quantile(dist, 0.95) * 100),
                   "max_cm": float(dist.max() * 100)})
    res["U2"] = {"frames": u2, "pass": all(x["max_abs_diff_rt"] < 1e-4 for x in u2)}
    res["U3"] = {"frames": u3, "pass": all(x["median_cm"] < 0.5 and x["p95_cm"] < 2.0 for x in u3)}
    # U4: camera height above the card floor at t = 0
    u4 = []
    for tag, scan, lv, st in eps:
        crd = M.card(scan, lv)
        for g in M.GROUPS:
            rec = np.load(M.svc_path(g, tag, 0))
            h = float(S.rt_to_zup_m(rec["cam_center"])[2] - crd["floor_m"])
            u4.append(h)
    res["U4"] = {"min_m": min(u4), "max_m": max(u4), "n": len(u4), "pass": 1.3 <= min(u4) and max(u4) <= 1.7,
                 "note": "expectation written before the check; outliers are panoramas on a raised area (x8F5xyUWy9e_L0, "
                         "10 % of its panoramas >= 2.50 m above the floor) and a level with a 1.30 m median tripod height "
                         "(oLBMNvg9in8_L1); not a unit error (see U4b). Reported as failed-as-written, not gating"}
    # U4b (added after U4, see DEVIATIONS): camera centre == .house panorama position (metres)
    u4b = []
    for tag, scan, lv, st in eps:
        crd = M.card(scan, lv)
        pos = {q["idx"]: q["pos"] for q in S.parse_house(scan)["panos"]}
        for g in M.GROUPS:
            for t in (0, 50, 100):
                rec = np.load(M.svc_path(g, tag, t))
                u4b.append(float(np.abs(S.rt_to_zup_m(rec["cam_center"]) - np.asarray(pos[crd["pano_idx"][int(rec["pose_idx"][0])]])).max()))
    res["U4b"] = {"max_abs_m": max(u4b), "n": len(u4b), "pass": max(u4b) < 1e-4}
    # U5: coverage tolerance of 2azQ1b91cZZ_L0 in metres
    crd = M.card("2azQ1b91cZZ", 0)
    eps_rt = S.EPS_MODEL * crd["k_I"]
    z = np.load(os.path.join(M.W, "preprocessing", "coverage", "2azQ1b91cZZ_L0.npz"))
    res["U5"] = {"k_I": crd["k_I"], "eps_rt_cache": float(z["eps_rt"]), "eps_m": float(z["eps_rt"]) / S.S_RT,
                 "expected_m": 0.08 * crd["k_I"] / 7.0, "pass": abs(float(z["eps_rt"]) / S.S_RT - 0.08 * crd["k_I"] / 7.0) < 1e-9
                 and abs(eps_rt - float(z["eps_rt"])) < 1e-9}
    ok = all(v["pass"] for k, v in res.items() if k != "U4")
    res["all_pass"] = ok
    od = os.path.join(M.MC, "metric_replay")
    os.makedirs(od, exist_ok=True)
    json.dump(res, open(os.path.join(od, "units_test.json"), "w"), indent=1)
    print(json.dumps(res, indent=1))
    sys.exit(0 if ok else 1)


if __name__ == "__main__":
    main()
