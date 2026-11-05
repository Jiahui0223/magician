"""Geometry / ROI audit A2 (seen2scene env, CPU, read-only): the planner map P1 against the official coverage target, per scene.
Inputs: a1 outputs (official GT target points, area-weighted mesh sample), the frozen P1 frame (exp3b t5_diagnostic.scene_geometry:
SceneFrame, grid dims, 1-D domain masks), the action-space pose tables (work_dirs/exp6_v3/H/pose_tables), and the stored service
records / episodes of the Table 1 D-v2 runs (V3-DV2-KSV).
Per scene -> results/a2_p1_roi.json (+ a2_p1_roi.csv):
  P1 grid bounds / valid-domain bounds in runtime coordinates, voxel size, dims, domain share of the grid, domain regularity
  ROI box (settings x 10) vs P1 domain: per-face offsets
  GT target points: share inside the P1 grid bounds / inside the valid domain; location of the missing points
  area-weighted target surface (points on target faces): share inside the valid domain
  cameras: share of action-space positions, of executed positions and of legal candidate positions inside the ROI box / P1 grid
  real observations: share of valid depth pixels (D100) and of the 10 % sample points (S10) outside the ROI box; outside the D-v2 grid
  checks: frame round trip, stored sample points == back-projection of the stored depth at the stored pixel indices"""
import csv
import json
import os
import sys

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
AUD = os.path.dirname(HERE)
REPO = os.path.dirname(os.path.dirname(os.path.dirname(AUD)))
X = os.path.join(REPO, "experiments")
sys.path.insert(0, os.path.join(X, "exp3b_macaronspp_transfer"))
sys.path.insert(0, os.path.join(X, "exp5_future_view", "task3_full_depth_p1", "scripts"))
import m_common as MC  # noqa: E402
import t5_diagnostic as TD  # noqa: E402
import p1_full as PF  # noqa: E402

A1 = os.path.join(REPO, "work_dirs", "cvpr_results", "geometry_roi_audit", "a1")
EP = os.path.join(REPO, "work_dirs", "cvpr_results", "table1", "episodes")
POSE = os.path.join(REPO, "work_dirs", "exp6_v3", "H", "pose_tables")
ALL14 = ["bannerman", "dunnottar", "fushimi", "liberty", "pisa", "redeemer", "barts", "colosseum", "eiffel", "pantheon", "sestino_museum",
         "alhambra", "bridge", "neuschwanstein"]
DV2_PAD = {"xz": 80.0, "y_lo": 30.0, "y_hi": 80.0}            # E3 configs/dv2_v2.yaml


def in_box(p, lo, hi):
    return ((p >= lo) & (p <= hi)).all(-1)


def main():
    out = {}
    for s in ALL14:
        z = np.load(os.path.join(A1, f"{s}.npz"))
        lo, hi = z["box_lo"].astype(np.float64), z["box_hi"].astype(np.float64)
        fr, dims, dom, win = TD.scene_geometry(s)
        dims = [int(d) for d in dims]
        VS = MC.VS
        # P1 grid / domain bounds in runtime coordinates (the frame is an axis permutation + sign + scale -> boxes stay axis-aligned)
        corners_m = np.array([[a, b, c] for a in (0.0, dims[0] * VS) for b in (0.0, dims[1] * VS) for c in (0.0, dims[2] * VS)])
        g_rt = fr.model_to_runtime(corners_m)
        g_lo, g_hi = g_rt.min(0), g_rt.max(0)
        d_idx = [(int(np.flatnonzero(m)[0]), int(np.flatnonzero(m)[-1])) for m in dom]
        contiguous = all(bool(m[i0:i1 + 1].all()) for m, (i0, i1) in zip(dom, d_idx))
        dc = np.array([[a, b, c] for a in (d_idx[0][0] * VS, (d_idx[0][1] + 1) * VS) for b in (d_idx[1][0] * VS, (d_idx[1][1] + 1) * VS)
                       for c in (d_idx[2][0] * VS, (d_idx[2][1] + 1) * VS)])
        d_rt = fr.model_to_runtime(dc)
        d_lo, d_hi = d_rt.min(0), d_rt.max(0)

        def idx_of(p_rt):
            return np.floor(fr.runtime_to_model(np.asarray(p_rt, np.float64)) / VS).astype(np.int64)

        def in_grid(ijk):
            return ((ijk >= 0) & (ijk < np.asarray(dims)[None])).all(-1)

        def in_dom(ijk):
            ok = in_grid(ijk)
            q = np.clip(ijk, 0, np.asarray(dims)[None] - 1)
            return ok & dom[0][q[:, 0]] & dom[1][q[:, 1]] & dom[2][q[:, 2]]

        gt = z["gt_points"].astype(np.float64)
        ijk = idx_of(gt)
        gin, gdm = in_grid(ijk), in_dom(ijk)
        miss = gt[~gdm]
        miss_face = None
        if len(miss):
            dist = np.minimum(miss - lo[None], hi[None] - miss)            # distance to the box faces (>= 0 inside the box)
            miss_face = {"n": int(len(miss)), "max_distance_to_nearest_box_face": float(dist.min(1).max()),
                         "by_axis_high_side": [int(((hi[None] - miss)[:, a] <= dist.min(1) + 1e-9).sum()) for a in range(3)]}
        ms, tf = z["mesh_sample"].astype(np.float64), z["mesh_sample_target_face"]
        tdm = in_dom(idx_of(ms[tf]))
        bdm = in_dom(idx_of(ms[z["mesh_sample_in_box"]]))
        odm = in_dom(idx_of(ms[~z["mesh_sample_in_box"]]))
        # cameras
        pt = np.load(os.path.join(POSE, f"{s}.npz"))
        C = np.unique(pt["C"], axis=0)
        lut = {tuple(int(x) for x in r): j for j, r in enumerate(pt["idx"])}
        cam_in = in_box(C, lo, hi)
        cam_in_dv2 = in_box(C, lo - [DV2_PAD["xz"], DV2_PAD["y_lo"], DV2_PAD["xz"]], hi + [DV2_PAD["xz"], DV2_PAD["y_hi"], DV2_PAD["xz"]])
        where = {"above_box_top": float((C[:, 1] > hi[1]).mean()), "below_box_bottom": float((C[:, 1] < lo[1]).mean()),
                 "lateral_outside_within_height": float((((C[:, 0] < lo[0]) | (C[:, 0] > hi[0]) | (C[:, 2] < lo[2]) | (C[:, 2] > hi[2])) & (C[:, 1] <= hi[1]) & (C[:, 1] >= lo[1])).mean())}
        exe, leg, obs = [], [], {"d100": [0, 0, 0], "s10": [0, 0], "frames": 0}
        chk = {"sample_vs_backprojection_max_abs": 0.0, "roundtrip_max_abs": 0.0}
        rt = fr.model_to_runtime(fr.runtime_to_model(gt[:1000]))
        chk["roundtrip_max_abs"] = float(np.abs(rt - gt[:1000]).max())
        for k in range(5):
            p = os.path.join(EP, "V3-DV2-KSV", f"{s}_s{k}.json")
            if not os.path.exists(p):
                continue
            r = json.load(open(p))
            e = json.load(open(os.path.join(REPO, r["episode_file"])))
            root = os.path.dirname(os.path.dirname(os.path.join(REPO, r["episode_file"])))
            for st in e["steps"]:
                exe.append(pt["C"][lut[tuple(json.loads(st["pose_key"]))]])
                for lk in st.get("legal_keys") or []:
                    leg.append(pt["C"][lut[tuple(json.loads(lk))]])
            if k == 0:
                for t in range(0, 101, 10):
                    f = os.path.join(root, "svc", e["tag"], f"svc_{s}_s{k}_t{t}.npz")
                    if not os.path.exists(f):
                        continue
                    rec = MC.load_service_record(f)
                    pts, nv = PF.full_points(rec)
                    pts = pts.astype(np.float64)
                    obs["frames"] += 1
                    obs["d100"][0] += len(pts)
                    obs["d100"][1] += int((~in_box(pts, lo, hi)).sum())
                    obs["d100"][2] += int((~in_box(pts, lo - [DV2_PAD["xz"], DV2_PAD["y_lo"], DV2_PAD["xz"]], hi + [DV2_PAD["xz"], DV2_PAD["y_hi"], DV2_PAD["xz"]])).sum())
                    sp = np.asarray(rec["sample_points"], np.float64).reshape(-1, 3)
                    obs["s10"][0] += len(sp)
                    obs["s10"][1] += int((~in_box(sp, lo, hi)).sum())
                    sf = np.asarray(rec["sample_pixel_flat"], np.int64)
                    if len(sf):
                        depth = np.clip(np.asarray(rec["zbuf"], np.float64), 0.5, 750.0).reshape(-1)
                        d_y, dz, c = MC.pixel_rays(rec["R"], rec["T"], sf % MC.W, sf // MC.W)
                        bp = c[None] + d_y * (depth[sf] / dz)[:, None]
                        chk["sample_vs_backprojection_max_abs"] = max(chk["sample_vs_backprojection_max_abs"], float(np.abs(bp - sp).max()))
                        chk["cam_center_vs_record_max_abs"] = max(chk.get("cam_center_vs_record_max_abs", 0.0), float(np.abs(c - np.asarray(rec["cam_center"], np.float64).reshape(3)).max()))
        exe, leg = np.asarray(exe), np.asarray(leg)
        out[s] = {
            "k_M": float(fr.k_M), "voxel_rt": float(VS * fr.k_M), "dims": dims, "n_windows": int(win["n"]),
            "roi_box": {"lo": lo.tolist(), "hi": hi.tolist()},
            "p1_grid_rt": {"lo": g_lo.tolist(), "hi": g_hi.tolist()}, "p1_domain_rt": {"lo": d_lo.tolist(), "hi": d_hi.tolist()},
            "domain_minus_box": {"lo": (d_lo - lo).tolist(), "hi": (d_hi - hi).tolist()}, "grid_minus_box": {"lo": (g_lo - lo).tolist(), "hi": (g_hi - hi).tolist()},
            "domain_share_of_grid": float(np.prod([m.sum() for m in dom]) / np.prod(dims)), "domain_is_axis_aligned_box": contiguous,
            "domain_index_range": d_idx,
            "gt_points": {"n": int(len(gt)), "in_p1_grid": float(gin.mean()), "in_p1_domain": float(gdm.mean()), "n_outside_domain": int((~gdm).sum()), "missing": miss_face},
            "target_surface_area_weighted_in_domain": float(tdm.mean()), "in_box_surface_by_point_in_domain": float(bdm.mean()),
            "outside_box_surface_by_point_in_domain": float(odm.mean()) if len(odm) else None,
            "cameras": {"n_positions": int(len(C)), "action_space_in_roi_box": float(cam_in.mean()), "action_space_in_dv2_grid": float(cam_in_dv2.mean()), **where,
                        "camera_box_lo": C.min(0).tolist(), "camera_box_hi": C.max(0).tolist(),
                        "executed_in_roi_box": float(in_box(exe, lo, hi).mean()) if len(exe) else None, "n_executed": int(len(exe)),
                        "legal_candidates_in_roi_box": float(in_box(leg, lo, hi).mean()) if len(leg) else None, "n_legal": int(len(leg))},
            "observations": {"frames": obs["frames"], "d100_points": obs["d100"][0], "d100_outside_roi_box": obs["d100"][1] / max(1, obs["d100"][0]),
                             "d100_outside_dv2_grid": obs["d100"][2] / max(1, obs["d100"][0]), "s10_points": obs["s10"][0], "s10_outside_roi_box": obs["s10"][1] / max(1, obs["s10"][0])},
            "checks": chk}
        o = out[s]
        print(f"{s:15s} vox {o['voxel_rt']:.3f} dims {dims} dom/grid {o['domain_share_of_grid']:.3f} | GT in grid {o['gt_points']['in_p1_grid']:.5f} in domain {o['gt_points']['in_p1_domain']:.5f} "
              f"(miss {o['gt_points']['n_outside_domain']}) area-w {o['target_surface_area_weighted_in_domain']:.5f} | cams in box: space {o['cameras']['action_space_in_roi_box']:.2f} "
              f"exec {o['cameras']['executed_in_roi_box']:.2f} legal {o['cameras']['legal_candidates_in_roi_box']:.2f} | obs outside box D100 {o['observations']['d100_outside_roi_box']:.2f} "
              f"S10 {o['observations']['s10_outside_roi_box']:.2f} outside dv2 grid {o['observations']['d100_outside_dv2_grid']:.3f} | chk {chk}", flush=True)
    json.dump(out, open(os.path.join(AUD, "results", "a2_p1_roi.json"), "w"), indent=1)
    with open(os.path.join(AUD, "results", "a2_p1_roi.csv"), "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["scene", "voxel_rt", "dims", "domain_share_of_grid", "gt_points", "gt_in_p1_grid", "gt_in_p1_domain", "target_area_in_domain", "cams_in_box_action_space",
                    "cams_in_box_executed", "cams_in_box_legal", "d100_outside_box", "s10_outside_box", "d100_outside_dv2_grid"])
        for s, o in out.items():
            w.writerow([s, round(o["voxel_rt"], 4), "x".join(map(str, o["dims"])), round(o["domain_share_of_grid"], 4), o["gt_points"]["n"], round(o["gt_points"]["in_p1_grid"], 6),
                        round(o["gt_points"]["in_p1_domain"], 6), round(o["target_surface_area_weighted_in_domain"], 6), round(o["cameras"]["action_space_in_roi_box"], 4),
                        round(o["cameras"]["executed_in_roi_box"], 4), round(o["cameras"]["legal_candidates_in_roi_box"], 4), round(o["observations"]["d100_outside_roi_box"], 4),
                        round(o["observations"]["s10_outside_roi_box"], 4), round(o["observations"]["d100_outside_dv2_grid"], 5)])


if __name__ == "__main__":
    main()
