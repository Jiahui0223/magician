"""Experiment XB / Task B1 -- observed-geometry fidelity on fixed outdoor states (seen2scene env, 1 GPU; history only, no GT, no future records).
For a trajectory of the Task A pool, the observation records 0..t are replayed into (a) the S10 P1 (BAND from the 10 % sample points: the scoring map) and
(b) the D100 P1 (BAND from all valid pixels: the encoder input), and the OBSERVED surface points themselves (all valid depth pixels of every record so far,
back-projected with the frozen camera model) are looked up at every representation level:
  S10 voxel BAND | D100 voxel BAND | 4x cell band > 0 (encoder input) | 4x band ratio (fraction of fine voxels of the cell that are BAND) | 8x cell observed
  (max-pool of band / free over 2 x 2 x 2 fine 4x cells: the condition for a scene-memory token / a non-zero latent)
stratified by the point's distance to its camera (runtime units) and by in-domain. Also map-side counts: 4x BAND cells by band ratio (thin structures), and the
share of 4x UNKNOWN cells adjacent to observed cells whose 8x parent is observed (UNKNOWN-boundary representation in the latent).
Output results/B1_fidelity/<tag>.json. Usage: python xb_b1_fidelity.py --set explore|confirm --tag TAG"""
import argparse
import json
import os
import sys

R = "/bsuscratch/jiahuizhang/projects/occupancy/seen2scene"
for p in ("experiments/exp6_v3/ablation_study/experiment_CT_continued_training/scripts", "experiments/exp5_future_view/task3_full_depth_p1/scripts"):
    sys.path.insert(0, os.path.join(R, p))
import ct_fixed as CT  # noqa: E402
import p1_full as PF  # noqa: E402

torch, np, time = CT.torch, CT.np, CT.time
import torch.nn.functional as F  # noqa: E402
from seen2scene.v3 import partial_tsdf as PT  # noqa: E402

OUT = os.path.join(R, "experiments/exp6_v3/ablation_study/experiment_XB_bottleneck_audit/results/B1_fidelity")
EDGES = (0.0, 10.0, 25.0, 45.0, 70.0)


def lookup(fu, grids, pts_model):
    """pts_model [N, 3] float64 model units -> dict of per-point booleans / values (numpy)."""
    ijk = torch.floor(pts_model / PT.VS).long()
    dims = torch.as_tensor(fu.dims, device=ijk.device)
    inside = ((ijk >= 0) & (ijk < dims)).all(1)
    ijk_c = torch.minimum(ijk.clamp(min=0), dims - 1)
    dom = fu._in_domain(ijk_c) & inside
    flat = fu._flat(ijk_c)
    band_vox = (fu.cnt[flat] > 0) & dom
    c4 = ijk_c // PT.FACTOR
    b4 = grids["band"][c4[:, 0], c4[:, 1], c4[:, 2]].float() / 255.0
    f4 = grids["free"][c4[:, 0], c4[:, 1], c4[:, 2]].float() / 255.0
    band8 = F.max_pool3d((grids["band"][None, None] > 0).float(), 2)[0, 0] > 0
    free8 = F.max_pool3d((grids["free"][None, None] > 0).float(), 2)[0, 0] > 0
    c8 = c4 // 2
    obs8 = (band8 | free8)[c8[:, 0], c8[:, 1], c8[:, 2]]
    return {"dom": dom, "band_vox": band_vox, "band4": (b4 > 0) & dom, "ratio4": b4, "free4": f4, "obs8": obs8 & dom}


def strata(dist, dom, masks):
    out = {}
    for i in range(len(EDGES) - 1):
        sel = (dist >= EDGES[i]) & (dist < EDGES[i + 1])
        row = {"n": int(sel.sum()), "n_dom": int((sel & dom).sum())}
        sd = sel & dom
        for k, m in masks.items():
            row[k] = float(m[sd].float().mean()) if sd.any() else None
        out[f"{EDGES[i]:.0f}-{EDGES[i + 1]:.0f}"] = row
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--set", required=True, choices=("explore", "confirm"))
    ap.add_argument("--tag", required=True)
    a = ap.parse_args()
    dev = torch.device("cuda")
    J = dict(CT.FR.jobs(argparse.Namespace(set=a.set)))
    kind, spec = J[a.tag]
    P = CT.FR.spec_of(None, a.tag, kind, spec, dev)
    fr, dims, dom = P["frame"]
    CT.C.install(P["scene"], dev)
    fu_s10 = CT.FX.m_p1.P1Fusion(fr, dims, dom, device=dev)
    fu_d100 = PF.P1FusionFull(fr, dims, dom, device=dev)
    targets = sorted(P["targets"])
    pts, dists, t_obs = [], [], []
    out = {"tag": a.tag, "scene": P["scene"], "start": P["meta"]["start"], "set": a.set, "states": {}}
    for t in range(max(targets) + 1):
        rec = CT.C.MC.load_service_record(P["svc"](t))
        fu_s10.integrate_record(rec)
        fu_d100.integrate_record(rec)
        p_all, n_valid = PF.full_points(rec)
        c = np.asarray(rec["cam_center"], np.float64).reshape(1, 3)
        pts.append(np.asarray(p_all, np.float64)); dists.append(np.linalg.norm(p_all.astype(np.float64) - c, axis=1)); t_obs.append(np.full(len(p_all), t))
        if t not in targets:
            continue
        P_all = np.concatenate(pts); D_all = np.concatenate(dists)
        pm = fu_d100._to_model(P_all)
        dist = torch.as_tensor(D_all, device=dev)
        g_s10 = PT.aggregate_fusion(fu_s10)
        g_d100 = PT.aggregate_fusion(fu_d100)
        L_s = lookup(fu_s10, g_s10, pm)
        L_d = lookup(fu_d100, g_d100, pm)
        masks = {"s10_vox": L_s["band_vox"], "d100_vox": L_d["band_vox"], "s10_4x": L_s["band4"], "d100_4x": L_d["band4"], "d100_8x_obs": L_d["obs8"], "s10_8x_obs": L_s["obs8"],
                 "d100_4x_thin": L_d["band4"] & (L_d["ratio4"] <= 2.0 / 64.0)}
        st = {"n_points": int(len(P_all)), "n_records": t + 1, "by_distance": strata(dist, L_d["dom"], masks),
              "overall": {k: float(m[L_d["dom"]].float().mean()) for k, m in masks.items()}, "in_domain_share": float(L_d["dom"].float().mean())}
        # map side: 4x BAND cells by ratio; UNKNOWN boundary cells and their 8x parents
        b = g_d100["band"].float() / 255.0; fr_ = g_d100["free"].float() / 255.0; dm = g_d100["dom"] > 0
        band_cells = (b > 0) & dm
        st["map_4x"] = {"band_cells": int(band_cells.sum()), "band_cells_ratio_le_1_64": int((band_cells & (b <= 1.0 / 64.0)).sum()), "band_cells_ratio_le_2_64": int((band_cells & (b <= 2.0 / 64.0)).sum()),
                        "band_cells_ratio_ge_8_64": int((band_cells & (b >= 8.0 / 64.0)).sum()), "unknown_cells": int(((b <= 0) & (fr_ < 0.5) & dm).sum()), "free_cells": int(((b <= 0) & (fr_ >= 0.5) & dm).sum())}
        unk = ((b <= 0) & (fr_ < 0.5) & dm).float()
        obs4 = ((b > 0) | (fr_ > 0)).float()
        nb = F.max_pool3d(F.pad(obs4[None, None], (1, 1, 1, 1, 1, 1)), 3, stride=1)[0, 0]
        bnd = (unk > 0) & (nb > 0)                                                   # UNKNOWN cell with an observed 4x neighbour
        b8 = F.max_pool3d((b[None, None] > 0).float(), 2)[0, 0] > 0
        f8 = F.max_pool3d((fr_[None, None] > 0).float(), 2)[0, 0] > 0
        parent_obs = (b8 | f8).repeat_interleave(2, 0).repeat_interleave(2, 1).repeat_interleave(2, 2)[: b.shape[0], : b.shape[1], : b.shape[2]]
        st["unknown_boundary_4x"] = {"cells": int(bnd.sum()), "parent_8x_observed": int((bnd & parent_obs).sum())}
        out["states"][str(t)] = st
        print(f"{a.tag} t={t} pts {len(P_all)} dom {st['in_domain_share']:.3f} | S10 vox {st['overall']['s10_vox']:.3f} D100 vox {st['overall']['d100_vox']:.3f} 4x {st['overall']['d100_4x']:.3f} "
              f"8x-obs {st['overall']['d100_8x_obs']:.3f} thin {st['overall']['d100_4x_thin']:.3f} | band cells {st['map_4x']['band_cells']} (<=1/64: {st['map_4x']['band_cells_ratio_le_1_64']}) | "
              f"unk boundary {st['unknown_boundary_4x']['cells']} parent-obs {st['unknown_boundary_4x']['parent_8x_observed']}", flush=True)
    os.makedirs(OUT, exist_ok=True)
    json.dump(out, open(os.path.join(OUT, f"{a.tag}.json"), "w"), indent=1)


if __name__ == "__main__":
    main()
