"""Geometry / ROI audit C (seen2scene env, 1 GPU, read-only; GT mesh used OFFLINE only to score): ray-level re-check of the D-v2
filter and of the planner invariants on the states of the Table 1 D1 episodes (V3-D1-H3-NORISK, rep 0).

Replay (identical to the planner: same service records, b2 fusion, R10 encoder refresh, frozen noise keys): at every `state_every`
step the frozen first-layer score_state is run (decoded first hits of every legal candidate, K = 4) and the D-v2 visibility test
(E3 dv2.py, config dv2_v2.yaml: same functions, same grid) is applied to the same hits.

Per state -> results/rays/<scene>_s<k>.jsonl:
  invariants  : decoded hits outside the grid / outside the valid domain / behind t_known / hit_u not on an UNKNOWN voxel (counts);
                P1 and observed grid unchanged by score_state (checksums); BAND / EMPTY monotone; in-domain observed D100 points whose
                voxel is BAND in P1 (frame / scale / index consistency); observed points outside the ROI box
  cameras     : candidates inside the ROI box / inside the P1 grid
  filter      : D1-novel value (in domain, no S10 point within 1.0; w(z) weights; mean over K) cut by D-v2, split by
                camera in / out of box x blocking position in / out of box (centre-ray first observed cell and bundle position),
                GT-verified share (true first surface more than MARGIN in front of the predicted hit), GT surface in / out of box,
                strict observed-point share on a random subset (an actually observed depth point within TUBE of the ray, between
                min_t and t_pred - MARGIN)
  per view    : first-layer novelty value before / after, the split of its removed value, and removed rays drawn with probability
                proportional to their weight, with every coordinate needed to re-derive the decision (camera, direction, predicted
                hit, blocking position, first observed point, depths, in-box flags)
Section data for figures (observed cells near the vertical plane of selected rays) -> work_dirs/.../geometry_roi_audit/c/."""
import argparse
import hashlib
import json
import os
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
AUD = os.path.dirname(HERE)
REPO = os.path.dirname(os.path.dirname(os.path.dirname(AUD)))
E3 = os.path.join(REPO, "experiments", "exp6_v3", "ablation_study", "experiment_E_stability_plateau", "E3_dv2_closed_loop")
sys.path.insert(0, os.path.join(E3, "scripts"))
import dv2_fixed as FX  # noqa: E402  (module setup only: b2 P1 patch, V3 planner core; D policy NOT installed)
import numpy as np  # noqa: E402
import torch  # noqa: E402
import yaml  # noqa: E402

DV2, HP, PC, PF, MC = FX.DV2, FX.HP, FX.PC, FX.PF, FX.MC
VS = 0.011
TUBE, Q = 0.33, 0.125                         # strict observed-point check: tube radius (< one D-v2 cell), point lattice
WORK = os.path.join(REPO, "work_dirs", "cvpr_results", "geometry_roi_audit", "c")


def chk(*tensors):
    h = hashlib.sha256()
    for t in tensors:
        h.update(t.detach().cpu().numpy().tobytes())
    return h.hexdigest()[:16]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--scene", required=True)
    ap.add_argument("--start", type=int, required=True)
    ap.add_argument("--state-every", type=int, default=5)
    ap.add_argument("--tmax", type=int, default=None)
    a = ap.parse_args()
    s, k = a.scene, a.start
    dev = torch.device("cuda")
    torch.use_deterministic_algorithms(True, warn_only=True)
    cfg = yaml.safe_load(open(os.path.join(E3, "configs", "dv2_v2.yaml")))["dv2"]
    DV2.CFG.update({kk: cfg[kk] for kk in DV2.CFG if kk in cfg})
    margin = float(cfg["margin"])
    vcfg = {"eps_cov": 1.0, "cell_size": 1.0}                         # D-v1 d_config.yaml value block
    rec1 = json.load(open(os.path.join(FX.EP, "V3-D1-H3-NORISK", f"{s}_s{k}.json")))
    ep = json.load(open(os.path.join(REPO, rec1["episode_file"])))
    root = os.path.dirname(os.path.dirname(os.path.join(REPO, rec1["episode_file"])))
    svc = os.path.join(root, "svc", ep["tag"])
    logs = {json.loads(l)["t"]: json.loads(l) for l in open(os.path.join(root, "planner_logs", f"pl_{ep['tag']}.steps.jsonl"))}
    hcfg = yaml.safe_load(open(os.environ["H_CONFIG"]))
    core = FX.PV3.V3Core(os.path.join(REPO, hcfg["model"]["checkpoint"]), dev, expect_sha=hcfg["model"]["sha256"])
    fr, dims, dom, _ = FX.TD.scene_geometry(s)
    dims = [int(x) for x in dims]
    fu = FX.m_p1.P1Fusion(fr, dims, dom, device=dev)                # = P1FusionB2 (D100 + S10 twin) after the b2 patch
    sb = json.load(open(os.path.join(FX.DATA, s, "settings.json")))["scene"]
    lo = np.asarray(sb["x_min"], float) * 10.0
    hi = np.asarray(sb["x_max"], float) * 10.0
    lo_t, hi_t = torch.as_tensor(lo, device=dev), torch.as_tensor(hi, device=dev)
    z0 = sum((b - a_) * 10.0 for a_, b in zip(sb["x_min"], sb["x_max"])) / 3.0 / 2.0
    isect = FX.gt_mesh(s)
    kM = float(fr.k_M)
    grid = DV2.ObservedGrid(s, dev)
    HP.STATE.update({"obs": [], "tree": None, "n_tree": -1, "rec": None, "visited": set()})
    dense = torch.zeros(0, dtype=torch.int64, device=dev)            # unique observed D100 points on a Q lattice (strict check)
    qoff = 1 << 20
    dims_t = torch.as_tensor(dims, device=dev)
    dom3 = None
    rng = np.random.default_rng(0)

    def in_box(p):
        p = torch.as_tensor(p, device=dev, dtype=torch.float64)
        return ((p >= lo_t) & (p <= hi_t)).all(-1)

    def rt_of(m):                                                  # model [N, 3] torch -> runtime numpy
        return np.asarray(fr.model_to_runtime(m.double().cpu().numpy()))

    K, h, eps = 4, None, None
    T = len(ep["steps"]) - 1 if a.tmax is None else a.tmax + 1
    os.makedirs(os.path.join(AUD, "results", "rays"), exist_ok=True)
    os.makedirs(WORK, exist_ok=True)
    fo = open(os.path.join(AUD, "results", "rays", f"{s}_s{k}.jsonl.tmp"), "w")
    sections = {}
    prev_band = prev_empty = 0
    mono_ok = True
    for t in range(T):
        rec = MC.load_service_record(os.path.join(svc, f"svc_{s}_s{k}_t{t}.npz"))
        HP.capture(rec)                                            # S10 points of the executed frames (as d_policy.install does)
        fu.integrate_record(rec)
        pts, _ = PF.full_points(rec)
        grid.add(pts)
        pt = torch.as_tensor(np.asarray(pts, np.float64), device=dev)
        q = torch.floor(pt / Q).long() + qoff
        dense = torch.unique(torch.cat([dense, (q[:, 0] << 42) | (q[:, 1] << 21) | q[:, 2]]))
        nb, ne = int((fu.cnt > 0).sum()), int(fu.empty.sum())
        mono_ok &= (nb >= prev_band) and (ne >= prev_empty)
        prev_band, prev_empty = nb, ne
        if t % 10 == 0:
            geo = PC.Geometry.from_fusion(fr, dims, dom, fu, dev)
            h = core.pred.encode(geo.coarse())
            eps = FX.PR.noise(dev, K, lambda kk, tr=t: FX.RY.sha_seed(FX.LC.seed_key(s, k, 0, tr, kk)))
        if t % a.state_every or len(rec["legal_R"]) == 0:
            continue
        geo = PC.Geometry.from_fusion(fr, dims, dom, fu, dev)
        dom3 = geo.domain
        c0 = chk(fu.sum, fu.cnt, fu.empty, fu.sparse.sum, fu.sparse.cnt, fu.sparse.empty, grid.occ)
        out, _, views = PC.score_state(core, geo, h, rec["legal_R"], rec["legal_T"], eps, z0, keep_hits=True)
        c1 = chk(fu.sum, fu.cnt, fu.empty, fu.sparse.sum, fu.sparse.cnt, fu.sparse.empty, grid.occ)
        # frame / scale / index consistency: in-domain observed points of this frame are BAND voxels of the D100 P1
        pm = torch.as_tensor(fr.runtime_to_model(np.asarray(pts, np.float64)), device=dev)
        ijk = torch.floor(pm / VS).long()
        ing = ((ijk >= 0) & (ijk < dims_t)).all(-1)
        qq = ijk[ing]
        ind = dom3[qq[:, 0], qq[:, 1], qq[:, 2]]
        fl = (qq[ind][:, 0] * dims[1] + qq[ind][:, 1]) * dims[2] + qq[ind][:, 2]
        band_share = float((fu.cnt[fl] > 0).double().mean()) if fl.numel() else None
        ptb = in_box(pt)
        row = {"t": t, "n_legal": len(views), "p1_unchanged_by_scoring": c0 == c1, "band_empty_monotone_so_far": bool(mono_ok),
               "frame": {"valid_pixels": int(len(pts)), "outside_roi_box": float((~ptb).double().mean()), "in_grid": float(ing.double().mean()),
                         "in_domain_of_in_box": float((ind.double().sum() / max(1, int(ptb.sum())))), "in_domain_points_are_band": band_share},
               "viol": {"hit_outside_grid": 0, "hit_outside_domain": 0, "hit_behind_known": 0, "hit_u_not_unknown": 0, "finite_hits": 0},
               "cams": {"in_box": 0, "in_grid": 0}, "acc": {}, "views": []}
        acc = dict.fromkeys(("novel", "cut", "cut_cam_in_box", "cut_cam_out_box", "cut_occ_c_out", "cut_occ_c_in", "cut_occ_b_out", "cut_occ_b_in",
                             "cut_gt_ok", "cut_gt_ok_surface_out", "cut_gt_ok_surface_in", "cut_no_gt_surface", "cut_gt_surface_behind",
                             "cut_camout_occout", "cut_camout_occin", "cut_camin_occin", "cut_camin_occout",
                             "gt_ok_camout_occout", "gt_ok_camout_occin", "gt_ok_camin_occin", "gt_ok_camin_occout",
                             "inbox_no_band", "inbox_band_positive_only", "inbox_negative_no_crossing", "inbox_crossing",
                             "ingt", "ingt_no_band", "ingt_band_positive_only", "ingt_negative_no_crossing", "ingt_crossing",
                             "strict_n", "strict_ok", "strict_w", "strict_w_ok"), 0.0)
        log_h = (logs.get(t) or {}).get("all_scores", {}).get("h_mean") if isinstance((logs.get(t) or {}).get("all_scores"), dict) else None
        tree = HP.obs_tree()
        dq = torch.stack([(dense >> 42) & ((1 << 21) - 1), (dense >> 21) & ((1 << 21) - 1), dense & ((1 << 21) - 1)], 1)
        dpts = ((dq - qoff).double() + 0.5) * Q                        # lattice-cell centres of the observed points (runtime)
        dpts32 = dpts.float()
        best_views = []
        for j, v in enumerate(views):
            o_rt = np.asarray(fr.model_to_runtime(v["o"].double().cpu().numpy()[None]))[0]
            d_rt = MC.to_y_up(v["d"].double().cpu().numpy())
            d_rt = d_rt / np.linalg.norm(d_rt, axis=1, keepdims=True)
            cam_in = bool(np.all((o_rt >= lo) & (o_rt <= hi)))
            om = v["o"].double().cpu().numpy()
            cam_in_grid = bool(np.all((om >= 0) & (om <= np.asarray(dims) * VS)))
            row["cams"]["in_box"] += int(cam_in)
            row["cams"]["in_grid"] += int(cam_in_grid)
            t_true = torch.as_tensor(FX.gt_first_hit(isect, o_rt, d_rt, float(v["tcap"].max().item()) * kM), device=dev)
            tdec, hit_u = out["hits"][j]["t"], out["hits"][j]["hit_u"]
            vis = DV2.visible_mask(grid, fr, v, tdec, kM)                       # exactly the planner's test (bundle)
            t_b = v["_dv2_tobs_%d_%g" % (int(DV2.CFG["support"]), float(DV2.CFG["footprint"]))].to(dev)
            fp = DV2.CFG["footprint"]
            DV2.CFG["footprint"] = 0.0
            t_c = grid.t_obs(o_rt, d_rt, (v["tcap"].double() * kM).cpu().numpy())   # centre ray only: first observed cell
            DV2.CFG["footprint"] = fp
            o_t = torch.as_tensor(o_rt, device=dev)
            d_t = torch.as_tensor(d_rt, device=dev)
            tk = v["pr"]["t_known"]
            val_off = val_on = 0.0
            v_cut = v_cut_out = v_cut_gt = 0.0
            cand = []
            for kk in range(K):
                tm_ = tdec[kk]
                fin = torch.isfinite(tm_)
                pmod = v["o"][None] + v["d"] * torch.where(fin, tm_, torch.zeros_like(tm_))[:, None]
                ij = torch.floor(pmod / VS).long()
                ing_ = ((ij >= 0) & (ij < dims_t)).all(-1)
                qc = torch.minimum(ij.clamp(min=0), dims_t - 1)
                indom = ing_ & dom3[qc[:, 0], qc[:, 1], qc[:, 2]]
                row["viol"]["finite_hits"] += int(fin.sum())
                row["viol"]["hit_outside_grid"] += int((fin & ~ing_).sum())
                row["viol"]["hit_outside_domain"] += int((fin & ~indom).sum())
                row["viol"]["hit_behind_known"] += int((fin & (tm_ > tk + 1e-5)).sum())
                hu = hit_u[kk] & fin
                row["viol"]["hit_u_not_unknown"] += int((hu & ~(geo.struct[qc[:, 0], qc[:, 1], qc[:, 2]] == MC.UNKNOWN)).sum())
                tp = tm_.double() * kM
                p_rt = o_t[None] + d_t * torch.where(fin, tp, torch.zeros_like(tp))[:, None]
                dd, _ = tree.query(p_rt.cpu().numpy(), k=1, distance_upper_bound=1.0)
                nov = fin & indom & torch.as_tensor(~np.isfinite(dd), device=dev)       # D1's novelty rule (value_cov)
                w = torch.clamp((tp * v["dz"].double() / z0) ** 2, max=1.0)
                cut = nov & ~vis[kk]
                val_off += float(HP.value_cov(tm_, v, geo, z0, vcfg)[1].sum()) / K
                val_on += float(HP.value_cov(torch.where(vis[kk], tm_, torch.full_like(tm_, float("inf"))), v, geo, z0, vcfg)[1].sum()) / K
                acc["novel"] += float(w[nov].sum()) / K
                if not cut.any():
                    continue
                wc = w[cut] / K
                pc_ = o_t[None] + d_t[cut] * t_c[cut][:, None]                    # centre-ray first observed cell
                pb_ = o_t[None] + d_t[cut] * t_b[cut][:, None]                    # bundle blocking position (decision depth)
                occ_c_in, occ_b_in = in_box(pc_), in_box(pb_)
                tt = t_true[cut]
                gt_ok = torch.isfinite(tt) & (tt < tp[cut] - margin)
                pg = o_t[None] + d_t[cut] * torch.where(torch.isfinite(tt), tt, torch.zeros_like(tt))[:, None]
                g_in = in_box(pg)
                acc["cut"] += float(wc.sum())
                acc["cut_cam_in_box" if cam_in else "cut_cam_out_box"] += float(wc.sum())
                acc["cut_occ_c_in"] += float(wc[occ_c_in].sum()); acc["cut_occ_c_out"] += float(wc[~occ_c_in].sum())
                acc["cut_occ_b_in"] += float(wc[occ_b_in].sum()); acc["cut_occ_b_out"] += float(wc[~occ_b_in].sum())
                key = ("cut_camin_" if cam_in else "cut_camout_")
                acc[key + "occin"] += float(wc[occ_c_in].sum()); acc[key + "occout"] += float(wc[~occ_c_in].sum())
                gkey = ("gt_ok_camin_" if cam_in else "gt_ok_camout_")
                acc[gkey + "occin"] += float(wc[occ_c_in & gt_ok].sum()); acc[gkey + "occout"] += float(wc[~occ_c_in & gt_ok].sum())
                if bool(occ_c_in.any()):
                    # why did the D100 P1 not stop these rays? D100 cast volume along the ray within +-1.0 (runtime) of the first observed cell
                    ii = torch.nonzero(cut)[:, 0][occ_c_in]
                    n_s = int(np.ceil(2.0 / kM / (0.5 * VS))) + 1
                    tsm = (t_c[ii] / kM).float()[:, None] + torch.linspace(-1.0 / kM, 1.0 / kM, n_s, device=dev)[None]
                    pp = v["o"][None, None] + v["d"][ii][:, None] * tsm[..., None]
                    uvw = pp / VS - 0.5
                    gg = 2 * uvw / (dims_t.float() - 1) - 1
                    gr = torch.stack([gg[..., 2], gg[..., 1], gg[..., 0]], -1)[None, :, :, None]
                    val = torch.nn.functional.grid_sample(geo.vol_obs[None, None], gr.to(geo.vol_obs.dtype), mode="bilinear", padding_mode="border", align_corners=True)[0, 0, :, :, 0]
                    band = (val.abs() < MC.TRUNC * 0.999).any(1)
                    neg = (val <= 0).any(1)
                    cross = ((val[:, :-1] > 0) & (val[:, 1:] <= 0)).any(1)
                    wi = wc[occ_c_in]
                    acc["inbox_no_band"] += float(wi[~band].sum()); acc["inbox_band_positive_only"] += float(wi[band & ~neg].sum())
                    acc["inbox_negative_no_crossing"] += float(wi[neg & ~cross].sum()); acc["inbox_crossing"] += float(wi[cross].sum())
                acc["cut_gt_ok"] += float(wc[gt_ok].sum())
                acc["cut_gt_ok_surface_out"] += float(wc[gt_ok & ~g_in].sum()); acc["cut_gt_ok_surface_in"] += float(wc[gt_ok & g_in].sum())
                acc["cut_no_gt_surface"] += float(wc[~torch.isfinite(tt)].sum())
                acc["cut_gt_surface_behind"] += float(wc[torch.isfinite(tt) & ~gt_ok].sum())
                gsel_ = gt_ok & g_in
                if bool(gsel_.any()):
                    # GT-anchored (offline): the true first surface is inside the ROI and in front of the predicted hit, yet the D100 P1
                    # did not stop the ray there. State of the D100 cast volume within +-1.0 (runtime) of the TRUE surface along the ray.
                    ii = torch.nonzero(cut)[:, 0][gsel_]
                    n_s = int(np.ceil(2.0 / kM / (0.5 * VS))) + 1
                    tsm = (tt[gsel_] / kM).float()[:, None] + torch.linspace(-1.0 / kM, 1.0 / kM, n_s, device=dev)[None]
                    pp = v["o"][None, None] + v["d"][ii][:, None] * tsm[..., None]
                    gg = 2 * (pp / VS - 0.5) / (dims_t.float() - 1) - 1
                    gr = torch.stack([gg[..., 2], gg[..., 1], gg[..., 0]], -1)[None, :, :, None]
                    val = torch.nn.functional.grid_sample(geo.vol_obs[None, None], gr.to(geo.vol_obs.dtype), mode="bilinear", padding_mode="border", align_corners=True)[0, 0, :, :, 0]
                    band = (val.abs() < MC.TRUNC * 0.999).any(1)
                    neg = (val <= 0).any(1)
                    cross = ((val[:, :-1] > 0) & (val[:, 1:] <= 0)).any(1)
                    wi = wc[gsel_]
                    acc["ingt"] += float(wi.sum())
                    acc["ingt_no_band"] += float(wi[~band].sum()); acc["ingt_band_positive_only"] += float(wi[band & ~neg].sum())
                    acc["ingt_negative_no_crossing"] += float(wi[neg & ~cross].sum()); acc["ingt_crossing"] += float(wi[cross].sum())
                ci = torch.nonzero(cut)[:, 0]
                # strict observed-point check on a random subset of the cut rays
                sub = ci[torch.as_tensor(rng.permutation(len(ci))[:48], device=dev)]
                if len(sub):
                    vv = dpts32[None] - o_t.float()[None, None]                    # [1, N, 3]
                    sdep = (vv * d_t[sub].float()[:, None]).sum(-1)                # [n, N]
                    perp2 = (vv * vv).sum(-1) - sdep ** 2
                    okp = (perp2 <= TUBE ** 2) & (sdep >= float(cfg["min_t"])) & (sdep <= (tp[sub].float() - margin)[:, None])
                    has = okp.any(1)
                    acc["strict_n"] += len(sub); acc["strict_ok"] += int(has.sum())
                    acc["strict_w"] += float(w[sub].sum()); acc["strict_w_ok"] += float(w[sub][has].sum())
                    del vv, sdep, perp2, okp
                # rays of this view / sample drawn with probability proportional to their weight among the removed rays (seeded):
                # a value-representative sample (the weights have massive ties at 1, so "largest weight" is not informative)
                v_cut += float(wc.sum()); v_cut_out += float(wc[~occ_c_in].sum()); v_cut_gt += float(wc[gt_ok].sum())
                gsel = torch.Generator(device="cpu").manual_seed(1000003 * t + 1009 * j + kk)
                top = ci[torch.multinomial(w[ci].float().cpu().clamp(min=1e-12), min(3, len(ci)), replacement=False, generator=gsel).to(dev)]
                for r_ in top.tolist():
                    cand.append({"k": kk, "ray": r_, "w": float(w[r_]), "t_pred": float(tp[r_]), "t_block_bundle": float(t_b[r_]), "t_first_observed_centre": float(t_c[r_]),
                                 "t_true_gt": float(t_true[r_]) if bool(torch.isfinite(t_true[r_])) else None, "t_known_p1": float(tk[r_] * kM) if bool(torch.isfinite(tk[r_])) else None,
                                 "dir": d_rt[r_].tolist(), "pred_hit": p_rt[r_].tolist(), "block_bundle": (o_rt + d_rt[r_] * float(t_b[r_])).tolist(),
                                 "first_observed_centre": (o_rt + d_rt[r_] * float(t_c[r_])).tolist(),
                                 "pred_hit_in_box": bool(in_box(p_rt[r_][None])[0]), "pred_hit_in_domain": bool(indom[r_]),
                                 "block_bundle_in_box": bool(np.all((o_rt + d_rt[r_] * float(t_b[r_]) >= lo) & (o_rt + d_rt[r_] * float(t_b[r_]) <= hi))),
                                 "first_observed_in_box": bool(np.all((o_rt + d_rt[r_] * float(t_c[r_]) >= lo) & (o_rt + d_rt[r_] * float(t_c[r_]) <= hi)))})
            vrow = {"j": j, "cam": o_rt.tolist(), "cam_in_box": cam_in, "cam_in_p1_grid": cam_in_grid, "value_off": val_off, "value_on": val_on,
                    "log_h_mean": (log_h[j] if log_h else None), "removed_value": v_cut, "removed_value_occluder_outside_box": v_cut_out / max(v_cut, 1e-12),
                    "removed_value_gt_verified": v_cut_gt / max(v_cut, 1e-12), "rays": cand[:6]}
            row["views"].append(vrow)
            best_views.append((val_off - val_on, j, o_rt, d_rt, vrow))
            del t_true, vis
        # nearest actually observed point for the stored rays + section data of the view with the largest value reduction
        best_views.sort(key=lambda x: -x[0])
        for red, j, o_rt, d_rt, vrow in best_views:
            o_t = torch.as_tensor(o_rt, device=dev).float()
            for c in vrow["rays"]:
                dvec = torch.as_tensor(c["dir"], device=dev).float()
                vv = dpts32 - o_t[None]
                sdep = (vv * dvec[None]).sum(-1)
                perp = ((vv * vv).sum(-1) - sdep ** 2).clamp(min=0).sqrt()
                m = (sdep >= float(cfg["min_t"])) & (sdep <= c["t_pred"] - margin) & (perp <= 0.5)
                if bool(m.any()):
                    i = torch.nonzero(m)[:, 0]
                    b = i[torch.argmin(sdep[i])]                                   # first observed point along the ray inside the 0.5 tube
                    pb = dpts[b].cpu().numpy()
                    c["observed_point"] = pb.tolist()
                    c["observed_point_depth"] = float(sdep[b]); c["observed_point_perp"] = float(perp[b])
                    c["observed_point_in_box"] = bool(np.all((pb >= lo) & (pb <= hi)))
                    pmi = np.floor(fr.runtime_to_model(pb[None]) / VS).astype(np.int64)[0]
                    c["observed_point_in_domain"] = bool(np.all((pmi >= 0) & (pmi < np.asarray(dims))) and bool(dom3[min(max(pmi[0], 0), dims[0] - 1), min(max(pmi[1], 0), dims[1] - 1), min(max(pmi[2], 0), dims[2] - 1)]))
                else:
                    c["observed_point"] = None
        if best_views and best_views[0][4]["rays"]:
            red, j, o_rt, d_rt, vrow = best_views[0]
            c = vrow["rays"][0]
            dvec = np.asarray(c["dir"])
            hdir = np.array([dvec[0], 0.0, dvec[2]])
            hdir = hdir / max(np.linalg.norm(hdir), 1e-9)
            nrm = np.array([-hdir[2], 0.0, hdir[0]])                              # horizontal normal of the vertical plane through the ray
            vv = (dpts - torch.as_tensor(o_rt, device=dev)[None])
            off = (vv * torch.as_tensor(nrm, device=dev)[None]).sum(-1).abs()
            along = (vv * torch.as_tensor(hdir, device=dev)[None]).sum(-1)
            m = (off <= 0.5) & (along >= -10) & (along <= c["t_pred"] * float(np.hypot(dvec[0], dvec[2])) + 25)
            sections[f"t{t}_j{j}"] = dpts[m].cpu().numpy().astype(np.float32)
        row["acc"] = acc
        fo.write(json.dumps(row, default=float) + "\n")
        fo.flush()
        a_ = acc
        print(f"{s} s{k} t={t} cams in box {row['cams']['in_box']}/{len(views)} | cut/novel {a_['cut'] / max(a_['novel'], 1e-9):.2f} cam-out {a_['cut_cam_out_box'] / max(a_['cut'], 1e-9):.2f} "
              f"occ(centre) out {a_['cut_occ_c_out'] / max(a_['cut'], 1e-9):.2f} GT-ok {a_['cut_gt_ok'] / max(a_['cut'], 1e-9):.3f} strict {a_['strict_ok']:.0f}/{a_['strict_n']:.0f} "
              f"| viol {row['viol']} | unchanged {row['p1_unchanged_by_scoring']} band {band_share}", flush=True)
        del geo, out, views, dpts, dpts32, dq
        torch.cuda.empty_cache()
    fo.close()
    os.replace(os.path.join(AUD, "results", "rays", f"{s}_s{k}.jsonl.tmp"), os.path.join(AUD, "results", "rays", f"{s}_s{k}.jsonl"))
    np.savez_compressed(os.path.join(WORK, f"sections_{s}_s{k}.npz"), **sections)


if __name__ == "__main__":
    main()
