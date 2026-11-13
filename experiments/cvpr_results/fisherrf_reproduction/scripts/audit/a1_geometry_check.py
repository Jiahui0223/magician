"""FisherRF audit A1 (CPU, fisherrf env, read-only): independent check of the planner-side geometry conventions.

For recorded service records of the f1c4 closed loops: un-project the record's zbuf at the 10 % sample pixels with the
planner's OWN conventions (fisherrf_planner.py: FX/CX/CY, F_FLIP, to_w2c, s = 7, depth = zbuf / s) and compare with the
record's sample_points (world, runtime units), which the frozen simulator computes with MAGICIAN's own back-projection
(m_common.unproject). A rotation / axis / scale / intrinsics error shows up as a large residual; the two intrinsics
conventions differ by ~0.4 % focal (PROTOCOL_ADAPTATION §3.2), so a residual of ~0.1-0.5 % of the depth is expected.
Also: per-frame depth statistics (valid fraction, out-of-range fraction, depth quantiles) over all recorded steps.
Output: work_dirs/.../audit/a1_geometry.json
"""
import glob
import json
import os
import sys

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.dirname(HERE))
import fisherrf_planner as FP  # noqa: E402

REPO = os.path.abspath(os.path.join(HERE, "..", "..", "..", "..", ".."))
CL = os.path.join(REPO, "work_dirs/cvpr_results/fisherrf_reproduction/full/outdoor/closed_loop_f1c4")
OUT = os.path.join(REPO, "work_dirs/cvpr_results/fisherrf_reproduction/audit")
S = 7.0


def to_w2c(R, T):
    Fm = np.diag(FP.F_FLIP)
    w = np.eye(4)
    w[:3, :3] = Fm @ np.asarray(R).T
    w[:3, 3] = Fm @ np.asarray(T) / S
    return w


def main():
    res = {"geometry": {}, "depth": {}}
    eps = sorted(os.listdir(CL))
    for ep in eps:
        svcd = glob.glob(os.path.join(CL, ep, "svc", "*"))
        if not svcd:
            continue
        files = sorted(glob.glob(os.path.join(svcd[0], "svc_*_t*.npz")))
        files = [f for f in files if not f.endswith("_rgb.npz")]
        vf, oor, med, p90, n_nodepth = [], [], [], [], 0
        geo = []
        for f in files:
            z = np.load(f, allow_pickle=False)
            zb, mk = z["zbuf"].astype(np.float64), z["mask"].astype(bool)
            valid = mk & (zb > 0) & (zb < FP.SENSOR_RANGE)
            vf.append(valid.mean())
            oor.append((mk & (zb >= FP.SENSOR_RANGE)).mean())
            if valid.any():
                med.append(float(np.median(zb[valid])))
                p90.append(float(np.quantile(zb[valid], 0.9)))
            else:
                n_nodepth += 1
            t = int(z["t"])
            if ep in ("barts_s0", "redeemer_s1", "colosseum_s0", "dunnottar_s3") and t % 10 == 0:
                flat = z["sample_pixel_flat"]
                v, u = flat // FP.W_IMG, flat % FP.W_IMG
                d = zb[v, u]
                ok = d > 0
                Pc = np.stack([(u - FP.CX) / FP.FX * d, (v - FP.CY) / FP.FX * d, d], 1) / S
                c2w = np.linalg.inv(to_w2c(z["R"], z["T"]))
                Pw = (Pc @ c2w[:3, :3].T + c2w[:3, 3]) * S
                ref = z["sample_points"].astype(np.float64)
                if len(ref) == len(Pw) and ok.any():
                    err = np.linalg.norm(Pw[ok] - ref[ok], axis=1)
                    rel = err / d[ok]
                    geo.append({"t": t, "n": int(ok.sum()), "err_median_rt": float(np.median(err)),
                                "rel_err_median": float(np.median(rel)), "rel_err_p99": float(np.quantile(rel, 0.99))})
                    # a deliberately wrong convention (no axis flip) for scale of comparison
                    Pc2 = Pc * np.array([-1, -1, 1])
                    Pw2 = (Pc2 @ c2w[:3, :3].T + c2w[:3, 3]) * S
                    geo[-1]["rel_err_median_if_no_flip"] = float(np.median(np.linalg.norm(Pw2[ok] - ref[ok], axis=1) / d[ok]))
                else:
                    geo.append({"t": t, "n_ref": int(len(ref)), "n": int(len(Pw)), "note": "sample size mismatch or no depth"})
        res["depth"][ep] = {"n_frames": len(files), "valid_frac_mean": float(np.mean(vf)), "valid_frac_min": float(np.min(vf)),
                            "out_of_range_frac_mean": float(np.mean(oor)), "frames_no_valid_depth": n_nodepth,
                            "median_depth_rt_mean": float(np.mean(med)) if med else None,
                            "p90_depth_rt_mean": float(np.mean(p90)) if p90 else None}
        if geo:
            res["geometry"][ep] = geo
    os.makedirs(OUT, exist_ok=True)
    json.dump(res, open(os.path.join(OUT, "a1_geometry.json"), "w"), indent=1)
    for ep, g in res["geometry"].items():
        r = [x["rel_err_median"] for x in g if "rel_err_median" in x]
        r2 = [x["rel_err_median_if_no_flip"] for x in g if "rel_err_median" in x]
        print(ep, "rel err median over frames: max %.5f | wrong-convention control %.3f" % (max(r), np.median(r2)))
    sc = {}
    for ep, d in res["depth"].items():
        sc.setdefault(ep.rsplit("_s", 1)[0], []).append(d)
    for s, L in sc.items():
        print("%-16s valid %.3f  out-of-range %.3f  no-depth frames %3d  median depth %.1f rt" % (
            s, np.mean([x["valid_frac_mean"] for x in L]), np.mean([x["out_of_range_frac_mean"] for x in L]),
            sum(x["frames_no_valid_depth"] for x in L), np.mean([x["median_depth_rt_mean"] for x in L if x["median_depth_rt_mean"]])))


if __name__ == "__main__":
    main()
