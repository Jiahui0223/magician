"""CVPR Fig. 4 -- real multi-hypothesis depth of the frozen V3-T0 at real decision states (seen2scene env, 1 GPU, read-only).
Run: bash work_dirs/exp6_v3/run_py.sh GPU experiments/cvpr_results/scripts/fig4/fig4_scan.py --scene S [--starts 0 1 2 3 4]

Single-state forwards only (no closed loop): for the stored Table 1 episodes of V3-T0-D1 (the main method) the planner inputs of a
decision step are rebuilt exactly as in the planner child -- service records 0..t fused with the D100 fusion, the latent of the
refresh step, the K = 4 noise maps of the frozen seed keys, the frozen score_state (same construction as the Fig. 2 replay
experiments/cvpr_results/figs/fig2/scripts/m3_planner_replay.py). Guards: the 4x grid hash equals the planner log and the
per-sample one-step values h_value_k equal the planner log (so the stored samples are the ones the planner really scored).

Candidate pool (declared before the scan, cvpr_overnight_experiment_plan.md section 7): the first two scenes of every split in the
frozen scene order without dunnottar (its planner state needs a whole GPU): Train bannerman, redeemer; Dev colosseum, pantheon;
Test alhambra, neuschwanstein; all 5 official starts; decision steps t in {0, 10, 20, 30, 40, 50} (latent-refresh steps, so the
latent is exactly the one of P_t). For each state only the EXECUTED view (the planner's pick c) is stored: the K = 4 samples
(b_hat, y_hat), the decoded first-hit distance, the ray classes (UNKNOWN mask) and known-surface distance of P_t along the same
rays, the per-sample novel rays / value (value_cov), and the REAL depth observed after executing the view (service record t + 1).
The real depth is used for visualisation / offline diagnosis only; nothing here feeds a planner.
Output: work_dirs/cvpr_results/overnight_20261110/intermediates/fig4/<scene>_s<k>.npz + <scene>.jsonl (metrics of every candidate)."""
import argparse
import hashlib
import json
import os
import sys

import numpy as np
import torch
from seen2scene import PROJECT_ROOT
from seen2scene.futureview import geometry as G
from seen2scene.futureview import predictor as PR
from seen2scene.futureview import rays as RY
from seen2scene.v3 import partial_tsdf as PT

REPO = str(PROJECT_ROOT)
X = os.path.join(REPO, "experiments")
for p in (os.path.join(X, "exp5_future_view", "task3_full_depth_p1", "scripts"), os.path.join(X, "exp5_future_view", "task1", "scripts"),
          os.path.join(X, "exp3b_macaronspp_transfer"), os.path.join(X, "exp6_v3", "shared"), os.path.join(X, "exp6_v3", "H", "scripts"),
          os.path.join(X, "exp4_multiworld_belief", "ablation_study", "abl_1", "common")):
    sys.path.insert(0, p)
import v2_plan_core as PC  # noqa: E402
import v2_common as LC  # noqa: E402
import p1_full as PF  # noqa: E402
import m_common as MC  # noqa: E402
import t5_diagnostic as TD  # noqa: E402
import planner_v3 as PV3  # noqa: E402
import h_policy as HP  # noqa: E402
import abl_common as AC  # noqa: E402

RUN = os.path.join(REPO, "work_dirs", "cvpr_results", "table1", "g3", "closed_loop", "outdoor", "V3-T0-D1", "V2M")
CKPT = os.path.join(REPO, "work_dirs", "exp6_v3", "task0_vae", "checkpoints", "final.pt")
SHA = "7ed0ddd72bb91d1e8c6563f3192201c24a49fe862b49d2387ea8770915145ab6"
OUT = os.path.join(REPO, "work_dirs", "cvpr_results", "overnight_20261110", "intermediates", "fig4")
CFG = {"eps_cov": 1.0, "cell_size": 1.0}
STEPS = (0, 10, 20, 30, 40, 50)
RANGE = 70.0                                                      # sensor range of the benchmark (runtime units)


def novel_rays(tt, v, geo):
    """Boolean [R]: rays whose predicted hit is counted as new coverage by h_policy.value_cov (in domain, > eps_cov from every observed point)."""
    hit = torch.isfinite(tt)
    p = v["o"][None] + v["d"] * torch.where(hit, tt, torch.zeros_like(tt))[:, None]
    ijk = torch.floor(p / 0.011).long()
    for ax in range(3):
        ijk[:, ax] = ijk[:, ax].clamp(0, geo.dims[ax] - 1)
    hit = hit & geo.domain[ijk[:, 0], ijk[:, 1], ijk[:, 2]]
    out = np.zeros(tt.shape[0], bool)
    sel = torch.nonzero(hit)[:, 0]
    if sel.numel():
        pr = geo.fr.model_to_runtime(p[sel].double().cpu().numpy())
        dist, _ = HP.obs_tree().query(pr, k=1, distance_upper_bound=CFG["eps_cov"])
        out[sel.cpu().numpy()[~np.isfinite(dist)]] = True
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--scene", required=True)
    ap.add_argument("--starts", type=int, nargs="+", default=[0, 1, 2, 3, 4])
    a = ap.parse_args()
    s = a.scene
    dev = torch.device("cuda")
    os.makedirs(OUT, exist_ok=True)
    core = PV3.V3Core(CKPT, dev, SHA)
    assert core.model.xa is None, "V3-T0 must have no cross-attention"
    fr, dims, dom, win = TD.scene_geometry(s)
    z0 = AC.z0_of(s)
    np_ = lambda x: x.detach().float().cpu().numpy()  # noqa: E731
    rows = []
    for k in a.starts:
        log = {r["t"]: r for r in map(json.loads, open(os.path.join(RUN, "planner_logs", f"pl_V2M_{s}_s{k}_r0.steps.jsonl")))}
        svc = lambda i: MC.load_service_record(os.path.join(RUN, "svc", f"V2M_{s}_s{k}_r0", f"svc_{s}_s{k}_t{i}.npz"))  # noqa: E731
        fu = PF.P1FusionFull(fr, dims, dom, device=dev)
        HP.STATE.update(obs=[], tree=None, n_tree=-1)
        save = {}
        for t in range(max(STEPS) + 1):
            rec = svc(t)
            HP.capture(rec)
            fu.integrate_record(rec)
            if t not in STEPS:
                continue
            step = log[t]
            assert int(step["t_r"]) == t
            packed = PT.pack_grid(PT.aggregate_fusion(fu))
            grid_ok = hashlib.sha256(packed.cpu().numpy().tobytes()).hexdigest() == step["refresh"]["coarse_sha256"]
            h = core.pred.encode(packed)
            eps = PR.noise(dev, 4, lambda kk: RY.sha_seed(LC.seed_key(s, k, 0, t, kk)))
            geo = PC.Geometry.from_fusion(fr, dims, dom, fu, dev)
            pick = step["choice"]["pick"]
            if pick is None or len(rec["legal_R"]) == 0:
                continue
            out, _, views = PC.score_state(core, geo, h, rec["legal_R"], rec["legal_T"], eps, z0, keep_hits=True)
            vals = np.asarray([[float(HP.value_cov(out["hits"][m]["t"][q], v, geo, z0, CFG)[1].sum()) for q in range(4)] for m, v in enumerate(views)])
            vals_ok = bool(np.array_equal(vals, np.asarray(step["all_scores"]["h_value_k"])))
            v, hit = views[pick], out["hits"][pick]
            nxt = svc(t + 1)
            next_is_pick = bool((np.asarray(nxt["pose_idx"]) == np.asarray(rec["legal_idx"])[pick]).all())
            td = np_(hit["t"])                                                     # [4, R] first-hit distance (model units), inf = no hit
            dz = np_(v["dz"])
            dep = td * float(fr.k_M) * dz[None]                                    # z-depth, runtime units
            fin = np.isfinite(dep)
            cls = np_(v["pr"]["cls"]).astype(np.int8)
            unk = cls == G.CLS_UNKNOWN
            zb, mk = np.asarray(nxt["zbuf"], np.float32), np.asarray(nxt["mask"], bool)
            real = zb[2::4, 2::4][:, :114].reshape(-1)                             # real depth at the 64 x 114 ray grid (4 x 4 block centres)
            rm = (mk & (zb < RANGE))[2::4, 2::4][:, :114].reshape(-1)
            nov = np.stack([novel_rays(hit["t"][q], v, geo) for q in range(4)])
            # metrics of the candidate state (definitions used by the pre-declared selection rule in fig4_plot.py)
            flip = float((fin.any(0) & ~fin.all(0))[unk].mean()) if unk.any() else 0.0
            hacc = float(np.mean([(fin[q][unk] == rm[unk]).mean() for q in range(4)])) if unk.any() else None
            errs = [float(np.median(np.abs(dep[q][unk & fin[q] & rm] - real[unk & fin[q] & rm]))) for q in range(4) if (unk & fin[q] & rm).any()]
            row = {"scene": s, "start": k, "t": t, "pick": int(pick), "n_legal": len(views), "grid_hash_equal_log": grid_ok, "values_equal_log": vals_ok, "next_is_pick": next_is_pick,
                   "unknown_share": float(unk.mean()), "n_unknown": int(unk.sum()), "real_valid_share": float(rm.mean()), "real_valid_share_on_unknown": float(rm[unk].mean()) if unk.any() else None,
                   "flip_share_on_unknown": flip, "hit_agreement_on_unknown": hacc, "median_abs_depth_err_on_unknown": float(np.mean(errs)) if errs else None,
                   "pred_hit_share_on_unknown": float(fin[:, unk].mean()) if unk.any() else None, "value_k": vals[pick].tolist(), "d1_score": float(step["scores"][pick]),
                   "coverage_before": float(nxt.get("cov", np.nan)) if "cov" in nxt else None}
            rows.append(row)
            key = f"t{t}_"
            save.update({key + "b": np_(hit["b"]), key + "y": np_(hit["y"]), key + "t_dec": td, key + "dz": dz, key + "cls": cls, key + "t_known": np_(v["pr"]["t_known"]),
                         key + "novel": nov, key + "real_depth": real, key + "real_valid": rm, key + "real_zbuf": zb, key + "real_mask": mk, key + "value_k": vals[pick],
                         key + "pick": np.int64(pick), key + "R": np.asarray(rec["legal_R"])[pick], key + "T": np.asarray(rec["legal_T"])[pick], key + "k_M": np.float64(fr.k_M),
                         key + "cur_zbuf": np.asarray(rec["zbuf"], np.float32), key + "cur_mask": np.asarray(rec["mask"], bool)})
            print(json.dumps(row), flush=True)
            del geo, out, views
        np.savez_compressed(os.path.join(OUT, f"{s}_s{k}.npz"), **save)
        del fu
        torch.cuda.empty_cache()
    with open(os.path.join(OUT, f"{s}.jsonl"), "a") as f:
        for r in rows:
            f.write(json.dumps(r) + "\n")


if __name__ == "__main__":
    main()
