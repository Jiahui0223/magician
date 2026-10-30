"""EXP5 Task 0.3 -- P1 / TSDF construction check on MP3D (seen2scene env; validation, evaluation side reads GT points).

For states of an indoor reference trajectory: region frame around the current camera (mp3d_frame.region_geometry,
card only), frozen P1Fusion over the service records 0..t, frozen structure(). Against the GT surface samples
(coverage cache, runtime y-up):
  band_near_gt   : BAND voxel centres within 2 voxels of a GT point (fraction)
  gt_in_empty    : GT points (in grid) whose voxel is EMPTY (should be ~0; carving through real surfaces)
  gt_in_band_dil : GT points whose voxel is in the 1-voxel dilation of BAND (observed surface proxy)
  frame chain    : camera centre -> model -> runtime round trip error
Output W/preprocessing/p1check/<traj>.json
"""
import argparse
import json
import os
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
import mp3d_frame as F  # noqa: E402

import numpy as np  # noqa: E402
import torch  # noqa: E402

import m_common as MC  # noqa: E402
import m_p1  # noqa: E402


def dense_gt(key, box_m, s_rt, n=3_000_000):
    """Dense area-weighted GT surface samples inside the region box (validation only; runtime y-up)."""
    import mp3d_sim as S
    scan = key.split("_L")[0]
    v, f = S.mesh_zup(scan)
    cen = v[f].mean(1)
    inb = np.all((cen >= box_m[0] - 0.05) & (cen <= box_m[1] + 0.05), axis=1)
    tri = v[f[inb]]
    area = 0.5 * np.linalg.norm(np.cross(tri[:, 1] - tri[:, 0], tri[:, 2] - tri[:, 0]), axis=1)
    rng = np.random.default_rng(0)
    fi = rng.choice(len(tri), n, p=area / area.sum())
    r1, r2 = rng.random(n), rng.random(n)
    s = np.sqrt(r1)
    p = (1 - s)[:, None] * tri[fi, 0] + (s * (1 - r2))[:, None] * tri[fi, 1] + (s * r2)[:, None] * tri[fi, 2]
    return F.zup_m_to_rt(p, s_rt)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--traj", required=True, help="trajectory dir name, e.g. zsNo4HB9uLZ_L0_S20")
    ap.add_argument("--times", default="0,5,10,30,60,100")
    a = ap.parse_args()
    dev = torch.device("cuda")
    key = a.traj.rsplit("_S", 1)[0]
    card = F.load_card(key)
    td = os.path.join(F.W, "preprocessing", "traj", a.traj)
    gt = np.load(os.path.join(F.W, "preprocessing", "coverage", f"{key}.npz"))["pts"]
    from scipy.spatial import cKDTree
    rows = []
    for t in [int(x) for x in a.times.split(",")]:
        recs = [MC.load_service_record(os.path.join(td, f"svc_t{i}.npz")) for i in range(t + 1)]
        c = recs[-1]["cam_center"]
        fr, dims, dom, win, box = F.region_geometry(card, c)
        fu = m_p1.P1Fusion(fr, dims, dom, device=dev)
        for r in recs:
            fu.integrate_record(r)
        st = fu.structure().cpu().numpy()
        rt = fr.model_to_runtime(fr.runtime_to_model(c))
        g = fr.runtime_to_model(dense_gt(key, box, card["s_rt"]))
        gi = np.floor(g / MC.VS).astype(np.int64)
        ing = np.all((gi >= 0) & (gi < np.asarray(dims)), axis=1)
        gi, gm = gi[ing], g[ing]
        band = st == MC.BAND
        vol = fu.input_volume().cpu().numpy()
        surf = band & (np.abs(vol) < MC.VS)          # zero-crossing shell (|TSDF| < 1 voxel)
        si = np.argwhere(surf)
        ds, _ = cKDTree(gm).query((si + 0.5) * MC.VS, k=1) if len(si) and len(gm) else (np.zeros(0), None)
        bi = np.argwhere(band)
        d, _ = cKDTree(gm).query((bi + 0.5) * MC.VS, k=1) if len(bi) and len(gm) else (np.zeros(0), None)
        bd = band.copy()
        for ax in range(3):
            bd |= np.roll(band, 1, ax) | np.roll(band, -1, ax)
        sv = st[gi[:, 0], gi[:, 1], gi[:, 2]] if len(gi) else np.zeros(0)
        rows.append({"t": t, "dims": dims, "n_windows": win["n"], "k_rt": fr.k_M, "box_m": box.tolist(),
                     "frame_roundtrip_err": float(np.abs(rt - c).max()),
                     "n_band": int(band.sum()), "n_empty": int((st == MC.EMPTY).sum()), "n_unknown": int((st == MC.UNKNOWN).sum()),
                     "band_near_gt_2vox": float((d < 2 * MC.VS).mean()) if len(d) else None,
                     "band_dist_median_vox": float(np.median(d) / MC.VS) if len(d) else None,
                     "surface_vox_near_gt_1vox": float((ds < MC.VS).mean()) if len(ds) else None,
                     "surface_vox_dist_median_vox": float(np.median(ds) / MC.VS) if len(ds) else None,
                     "gt_points_in_grid": int(len(gi)),
                     "gt_in_empty": float((sv == MC.EMPTY).mean()) if len(sv) else None,
                     "gt_in_band_dil": float(bd[gi[:, 0], gi[:, 1], gi[:, 2]].mean()) if len(gi) else None})
        print(rows[-1], flush=True)
    od = os.path.join(F.W, "preprocessing", "p1check")
    os.makedirs(od, exist_ok=True)
    json.dump({"traj": a.traj, "rows": rows}, open(os.path.join(od, f"{a.traj}.json"), "w"), indent=1, default=float)


if __name__ == "__main__":
    main()
