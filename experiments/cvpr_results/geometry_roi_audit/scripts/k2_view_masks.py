"""Geometry / ROI audit K2 (seen2scene env, 1 GPU, read-only): per-ray masks of selected colosseum candidate views, for the real-image
figures. Same replay as c_ray_audit.py (Table 1 D1 episode, frozen first-layer scoring, D-v2 test of E3 dv2.py). For every requested
(start, step, candidate): the 114 x 64 scoring rays with their pixel coordinates, and per ray (mean over the K = 4 samples)
  novel_w   weight that D1 counts as new surface (predicted first hit in the valid domain, no S10 point within 1.0)
  cut_w     the part of it that D-v2 removes (predicted hit behind an already observed surface)
  hit_share share of the K samples with a predicted hit; t_pred (mean predicted depth, runtime units)
plus the candidate pose (R, T, centre). Output work_dirs/cvpr_results/geometry_roi_audit/k/masks_s<start>_t<t>_j<j>.npz."""
import json
import os
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
AUD = os.path.dirname(HERE)
REPO = os.path.dirname(os.path.dirname(os.path.dirname(AUD)))
E3 = os.path.join(REPO, "experiments", "exp6_v3", "ablation_study", "experiment_E_stability_plateau", "E3_dv2_closed_loop")
sys.path.insert(0, os.path.join(E3, "scripts"))
import dv2_fixed as FX  # noqa: E402
import numpy as np  # noqa: E402
import torch  # noqa: E402
import yaml  # noqa: E402

DV2, HP, PC, PF, MC = FX.DV2, FX.HP, FX.PC, FX.PF, FX.MC
VS = 0.011
OUT = os.path.join(REPO, "work_dirs", "cvpr_results", "geometry_roi_audit", "k")
SCENE = "colosseum"


def main():
    views = json.load(open(os.path.join(AUD, "results", "colosseum_real_views.json")))["views"]
    dev = torch.device("cuda")
    cfg = yaml.safe_load(open(os.path.join(E3, "configs", "dv2_v2.yaml")))["dv2"]
    DV2.CFG.update({kk: cfg[kk] for kk in DV2.CFG if kk in cfg})
    hcfg = yaml.safe_load(open(os.environ["H_CONFIG"]))
    core = FX.PV3.V3Core(os.path.join(REPO, hcfg["model"]["checkpoint"]), dev, expect_sha=hcfg["model"]["sha256"])
    fr, dims, dom, _ = FX.TD.scene_geometry(SCENE)
    dims_t = torch.as_tensor([int(x) for x in dims], device=dev)
    sb = json.load(open(os.path.join(FX.DATA, SCENE, "settings.json")))["scene"]
    z0 = sum((b - a_) * 10.0 for a_, b in zip(sb["x_min"], sb["x_max"])) / 3.0 / 2.0
    kM = float(fr.k_M)
    from seen2scene.futureview import rays as RY
    u, v_ = RY.grid_uv((0.0, 0.0))
    os.makedirs(OUT, exist_ok=True)
    for k in sorted({w["start"] for w in views}):
        want = {(w["t"], w["j"]) for w in views if w["start"] == k}
        rec1 = json.load(open(os.path.join(FX.EP, "V3-D1-H3-NORISK", f"{SCENE}_s{k}.json")))
        ep = json.load(open(os.path.join(REPO, rec1["episode_file"])))
        root = os.path.dirname(os.path.dirname(os.path.join(REPO, rec1["episode_file"])))
        svc = os.path.join(root, "svc", ep["tag"])
        fu = FX.m_p1.P1Fusion(fr, dims, dom, device=dev)
        grid = DV2.ObservedGrid(SCENE, dev)
        HP.STATE.update({"obs": [], "tree": None, "n_tree": -1, "rec": None, "visited": set()})
        K, h, eps = 4, None, None
        for t in range(max(tt for tt, _ in want) + 1):
            rec = MC.load_service_record(os.path.join(svc, f"svc_{SCENE}_s{k}_t{t}.npz"))
            HP.capture(rec)
            fu.integrate_record(rec)
            grid.add(PF.full_points(rec)[0])
            if t % 10 == 0:
                geo = PC.Geometry.from_fusion(fr, dims, dom, fu, dev)
                h = core.pred.encode(geo.coarse())
                eps = FX.PR.noise(dev, K, lambda kk, tr=t: FX.RY.sha_seed(FX.LC.seed_key(SCENE, k, 0, tr, kk)))
            js = [j for tt, j in want if tt == t]
            if not js:
                continue
            geo = PC.Geometry.from_fusion(fr, dims, dom, fu, dev)
            out, _, views_t = PC.score_state(core, geo, h, rec["legal_R"], rec["legal_T"], eps, z0, keep_hits=True)
            tree = HP.obs_tree()
            for j in js:
                v = views_t[j]
                tdec = out["hits"][j]["t"]
                vis = DV2.visible_mask(grid, fr, v, tdec, kM)
                o_rt = np.asarray(fr.model_to_runtime(v["o"].double().cpu().numpy()[None]))[0]
                d_rt = MC.to_y_up(v["d"].double().cpu().numpy())
                d_rt = d_rt / np.linalg.norm(d_rt, axis=1, keepdims=True)
                o_t, d_t = torch.as_tensor(o_rt, device=dev), torch.as_tensor(d_rt, device=dev)
                nov_w = torch.zeros(tdec.shape[1], dtype=torch.float64, device=dev)
                cut_w, hit, tsum = nov_w.clone(), nov_w.clone(), nov_w.clone()
                for kk in range(K):
                    tm_ = tdec[kk]
                    fin = torch.isfinite(tm_)
                    pm = v["o"][None] + v["d"] * torch.where(fin, tm_, torch.zeros_like(tm_))[:, None]
                    ij = torch.floor(pm / VS).long()
                    ing = ((ij >= 0) & (ij < dims_t)).all(-1)
                    q = torch.minimum(ij.clamp(min=0), dims_t - 1)
                    indom = ing & geo.domain[q[:, 0], q[:, 1], q[:, 2]]
                    tp = tm_.double() * kM
                    p_rt = o_t[None] + d_t * torch.where(fin, tp, torch.zeros_like(tp))[:, None]
                    dd, _ = tree.query(p_rt.cpu().numpy(), k=1, distance_upper_bound=1.0)
                    nov = fin & indom & torch.as_tensor(~np.isfinite(dd), device=dev)
                    w = torch.clamp((tp * v["dz"].double() / z0) ** 2, max=1.0)
                    nov_w += torch.where(nov, w, torch.zeros_like(w)) / K
                    cut_w += torch.where(nov & ~vis[kk], w, torch.zeros_like(w)) / K
                    hit += fin.double() / K
                    tsum += torch.where(fin, tp, torch.zeros_like(tp)) / K
                np.savez_compressed(os.path.join(OUT, f"masks_s{k}_t{t}_j{j}.npz"), u=np.asarray(u), v=np.asarray(v_), novel_w=nov_w.cpu().numpy(), cut_w=cut_w.cpu().numpy(),
                                    hit_share=hit.cpu().numpy(), t_pred=(tsum / hit.clamp(min=1e-9)).cpu().numpy(), R=np.asarray(rec["legal_R"][j], np.float64),
                                    T=np.asarray(rec["legal_T"][j], np.float64), center=o_rt, legal_idx=np.asarray(rec["legal_idx"]).reshape(-1, 5)[j])
                print(f"s{k} t={t} j={j} cam {np.round(o_rt, 1)} novel {float(nov_w.sum()):.0f} removed {float(cut_w.sum()):.0f}", flush=True)
            del geo, out, views_t
            torch.cuda.empty_cache()
        del fu, grid
        torch.cuda.empty_cache()


if __name__ == "__main__":
    main()
