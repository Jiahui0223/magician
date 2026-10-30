"""EXP5 Task 1.2 / 1.3 -- V2 on the S-c calibration states (Train-6 held-out starts x t in Sc_times; protocol §3.7,
§4.2-§4.5). Evaluation-side process: it reads the T1.1 label shards (GT-derived t*, categories) after the planner-side
scoring of each state is done; the scoring itself (v2_plan_core.score_state) sees only P1 + poses.

Per state: P1 replay from the whitelisted records, coarse grid (must equal the T1.1 shard bitwise), K = 4 samples with
seeds exp5-V2|sc|<scene>|<traj>|t<t>|k<k>, decoded hits, DistP1 (K4 mean, K1 = sample 0), U_raw / DistU for every
delta of the grid, U_flow, U_feat, Frontier-heuristic DistP1, true-depth DistP1 (labels), both-new pair deltas, and the
compact per-ray arrays of the valid M_unknown rays (for the stratified distribution-quality analysis).
--reliability: additionally K = 16 hit probabilities on a fixed subset of views. --smoke: single-camera checks only.
Outputs <out>/<name>/{states.jsonl, rays_<part>.npz, rel_<part>.npz, run_<part>.json}
"""
import argparse
import json
import os
import sys
import time

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
import v2_frozen as VF  # noqa: E402
import v2_plan_core as PC  # noqa: E402

import numpy as np  # noqa: E402
import torch  # noqa: E402

import m_common as MC  # noqa: E402
import m_p1  # noqa: E402
import t5_diagnostic as TD  # noqa: E402
import abl_common as AC  # noqa: E402
from seen2scene.futureview import geometry as G  # noqa: E402
from seen2scene.futureview import predictor as PR  # noqa: E402
from seen2scene.futureview import rays as RY  # noqa: E402
from seen2scene.futureview import scoring as SCO  # noqa: E402

SC = VF.load_cfg("scoring_v1.json")
DELTAS = tuple(SC["U"]["delta_grid_vox"])


def seed(key):
    return RY.sha_seed(key)


def sc_items(scenes=None):
    sp = VF.frozen_values()["split"]["scenes"]
    return [(s, tj) for s in sp if scenes is None or s in scenes for tj in sp[s]["heldout_trajs"]]


def records(scene, traj):
    d = os.path.join(VF.DATA_V1, "trajectories", scene, traj, "records")
    fs = {int(f.split("_t")[-1][:-4]): os.path.join(d, f) for f in os.listdir(d) if f.endswith(".npz")}
    return fs


def smoke_checks(core, geo, h, rec, z0, dev, out):
    eps = PR.noise(dev, 4, lambda k: seed(f"exp5-V2|smoke|k{k}"))
    a, _, views = PC.score_state(core, geo, h, rec["legal_R"][:2], rec["legal_T"][:2], eps, z0, keep_hits=True)
    b_, _, _ = PC.score_state(core, geo, h, rec["legal_R"][:2], rec["legal_T"][:2], eps, z0, keep_hits=True)
    det = all(torch.equal(x["b"], y["b"]) and torch.equal(x["y"], y["y"]) for x, y in zip(a["hits"], b_["hits"]))
    same_det, in_unk, n_hits = True, True, 0
    for hh, v in zip(a["hits"], views):
        det_rays = v["pr"]["cls"] != G.CLS_UNKNOWN
        same_det &= bool((hh["t"][:, det_rays] == hh["t"][0:1, det_rays]).all())
        hu = hh["hit_u"]
        p = v["o"][None, None] + v["d"][None] * torch.where(hu, hh["t"], torch.zeros_like(hh["t"]))[..., None]
        ijk = torch.floor(p / G.VS).long()[hu]
        n_hits += len(ijk)
        if len(ijk):
            st = geo.struct[ijk[:, 0], ijk[:, 1], ijk[:, 2]]
            in_unk &= bool((st == G.UNKNOWN).all()) and bool(geo.domain[ijk[:, 0], ijk[:, 1], ijk[:, 2]].all())
    out["smoke_single_camera"] = {"deterministic_resample": det, "determined_rays_identical_over_K": same_det,
                                  "hits_in_unknown_voxels": in_unk, "n_hits": n_hits,
                                  "pass": det and same_det and in_unk}
    print("smoke", out["smoke_single_camera"], flush=True)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--ckpt", required=True)
    ap.add_argument("--name", required=True)
    ap.add_argument("--out", default=os.path.join(VF.WORK, "eval_sc"))
    ap.add_argument("--part", default="0/1")
    ap.add_argument("--scenes", default="")
    ap.add_argument("--reliability", action="store_true")
    ap.add_argument("--smoke", action="store_true")
    a = ap.parse_args()
    VF.verify_frozen()
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    dev = torch.device("cuda")
    core = PC.Core(a.ckpt, dev)
    od = os.path.join(a.out, a.name)
    os.makedirs(od, exist_ok=True)
    pi, pn = (int(x) for x in a.part.split("/"))
    items = sc_items(a.scenes.split(",") if a.scenes else None)[pi::pn]
    times = SC["Sc_times"]
    run = {"ckpt": a.ckpt, "ckpt_sha256": core.ckpt_sha, "part": a.part, "items": items, "started": time.ctime(),
           "gpu": torch.cuda.get_device_name(0)}
    f_states = open(os.path.join(od, f"states_{pi}.jsonl"), "a" if not a.smoke else "w")
    comp = {k: [] for k in ("view_id", "lab", "tstar_vox", "tmax_vox", "lu_vox", "tmin_vox", "hit", "t_vox", "s", "b")}
    rel = {k: [] for k in ("view_id", "lab", "p16", "lu_vox")}
    rel_left = 512 // pn + 1
    for scene, traj in items:
        fr, dims, dom, win = TD.scene_geometry(scene)
        z0 = AC.z0_of(scene)
        shard = np.load(os.path.join(VF.WORK, "data", "views", scene, f"{traj}.npz"))
        assert bool(shard["heldout"]), (scene, traj)
        fs = records(scene, traj)
        fu = m_p1.P1Fusion(fr, dims, dom, device=dev)
        for t in range(max(times) + 1):
            rec = MC.load_service_record(fs[t])
            fu.integrate_record(rec)
            if t not in times:
                continue
            tb = time.time()
            geo = PC.Geometry.from_fusion(fr, dims, dom, fu, dev)
            coarse = geo.coarse()
            coarse_equal = bool(np.array_equal(coarse.cpu().numpy(), shard["coarse"][t]))
            if not coarse_equal:
                raise VF.InvalidRun(f"INVALID: coarse grid differs from the T1.1 shard {scene}/{traj} t={t}")
            h = core.pred.encode(coarse)
            if a.smoke:
                smoke_checks(core, geo, h, rec, z0, dev, run)
                json.dump(run, open(os.path.join(od, "smoke.json"), "w"), indent=1, default=float)
                return
            eps = PR.noise(dev, 4, lambda k: seed(f"exp5-V2|sc|{scene}|{traj}|t{t}|k{k}"))
            out, tm, views = PC.score_state(core, geo, h, rec["legal_R"], rec["legal_T"], eps, z0, deltas=DELTAS,
                                            want_flow=True, want_feat=True, keep_hits=True)
            rows = np.flatnonzero(shard["t"] == t)
            assert len(rows) == len(views)
            front, true_, deltas_both = [], [], []
            for j, (v, hh) in enumerate(zip(views, out["hits"])):
                r = rows[j]
                pr = v["pr"]
                if not torch.equal(pr["cls"].to(torch.uint8).cpu(), torch.as_tensor(shard["cls"][r])):
                    raise VF.InvalidRun(f"INVALID: projection class differs from the shard {scene}/{traj} t={t}")
                # Frontier heuristic: every M_unknown ray hits its first UNKNOWN sample
                tf = torch.where(pr["n_u"] > 0, pr["d_lo"], pr["t_known"])
                _, wf = SCO.new_set_from_depth(tf, v["o"], v["d"], v["dz"], fr.k_M, geo.dims, geo.domain, geo.caster.band_dil, z0)
                front.append(float(wf.sum()))
                ts = torch.as_tensor(shard["tstar"][r], device=dev, dtype=v["d"].dtype)
                tt = torch.where(ts <= pr["tmax"], ts, torch.full_like(ts, float("inf")))
                _, wt = SCO.new_set_from_depth(tt, v["o"], v["d"], v["dz"], fr.k_M, geo.dims, geo.domain, geo.caster.band_dil, z0)
                true_.append(float(wt.sum()))
                news = torch.stack([SCO.new_mask_from_depth(hh["t"][k], v["o"], v["d"], geo.dims, geo.domain,
                                                            geo.caster.band_dil)[0] for k in range(4)])
                deltas_both.append(SCO.both_new_deltas(news, hh["t"]).cpu().numpy())
                cat = torch.as_tensor(shard["cat"][r], device=dev).long()
                valid = (pr["cls"] == G.CLS_UNKNOWN) & ((cat == G.CAT["HIT"]) | (cat == G.CAT["NOHIT"]) | (cat == G.CAT["KNOWN"]))
                vid = f"{scene}|{traj}|{t}|{j}"
                n = int(valid.sum())
                if n:
                    tmax = pr["tmax"][valid]
                    comp["view_id"].append(np.full(n, len(comp["lab"])))
                    comp["lab"].append(cat[valid].cpu().numpy().astype(np.uint8))
                    comp["tstar_vox"].append((torch.minimum(ts[valid], tmax) / G.VS).cpu().numpy().astype(np.float32))
                    comp["tmax_vox"].append((tmax / G.VS).cpu().numpy().astype(np.float32))
                    comp["tmin_vox"].append((pr["tmin"][valid] / G.VS).cpu().numpy().astype(np.float32))
                    comp["lu_vox"].append((pr["n_u"][valid].float() * G.H_STEP / G.VS).cpu().numpy().astype(np.float32))
                    comp["hit"].append(hh["hit_u"][:, valid].T.cpu().numpy())
                    tk = torch.where(torch.isfinite(hh["t"][:, valid]), hh["t"][:, valid], tmax[None])
                    comp["t_vox"].append((tk / G.VS).T.cpu().numpy().astype(np.float32))
                    comp["s"].append(((hh["y"][:, valid] + 1) / 2).clamp(0, 1).T.cpu().numpy().astype(np.float16))
                    comp["b"].append(hh["b"][:, valid].T.cpu().numpy().astype(np.float16))
                    comp.setdefault("keys", []).append(vid)
            if a.reliability and rel_left > 0:
                eps16 = PR.noise(dev, 16, lambda k: seed(f"exp5-V2|sc16|{scene}|{traj}|t{t}|k{k}"))
                cond, m_unk = core.pred.cond(h, [(v["o"], v["d"], v["tcap"], v["pr"]) for v in views])
                b16, y16 = core.pred.sample(cond, m_unk, eps16)
                for j, v in enumerate(views[:rel_left]):
                    cat = torch.as_tensor(shard["cat"][rows[j]], device=dev).long()
                    pr = v["pr"]
                    valid = (pr["cls"] == G.CLS_UNKNOWN) & ((cat == G.CAT["HIT"]) | (cat == G.CAT["NOHIT"]) | (cat == G.CAT["KNOWN"]))
                    hit16 = ((b16[j] > 0) & (y16[j] < 1.0))[:, valid].float().mean(0)
                    rel["view_id"].append(np.full(int(valid.sum()), len(rel["lab"])))
                    rel["lab"].append((cat[valid] == G.CAT["HIT"]).cpu().numpy())
                    rel["p16"].append(hit16.cpu().numpy().astype(np.float32))
                    rel["lu_vox"].append((pr["n_u"][valid].float() * G.H_STEP / G.VS).cpu().numpy())
                rel_left -= len(views)
            row = {"scene": scene, "traj": traj, "t": t, "n": len(views), "keys": [str([int(v) for v in x]) for x in rec["legal_idx"]],
                   "coarse_equal": coarse_equal, "frontier_distp1": front, "truedepth_distp1": true_,
                   "deltas_both_new_vox": np.concatenate(deltas_both).tolist() if deltas_both else [],
                   "timing": tm, "seconds": time.time() - tb,
                   **{k: v for k, v in out.items() if k != "hits"}}
            row["deltas_both_new_vox"] = np.round(np.asarray(row["deltas_both_new_vox"], np.float32), 3).tolist()
            f_states.write(json.dumps(row, default=float) + "\n")
            f_states.flush()
            print(f"{scene}/{traj} t={t}: {len(views)} views, distp1 {np.round(out['distp1'], 1).tolist()}, "
                  f"{row['seconds']:.1f} s", flush=True)
            del geo, h
            torch.cuda.empty_cache()
        del fu
    np.savez_compressed(os.path.join(od, f"rays_{pi}.npz"), **{k: (np.concatenate(v) if k != "keys" else np.asarray(v))
                                                                 for k, v in comp.items() if len(v)})
    if a.reliability:
        np.savez_compressed(os.path.join(od, f"rel_{pi}.npz"), **{k: np.concatenate(v) for k, v in rel.items() if len(v)})
    run["finished"] = time.ctime()
    run["code_sha256"] = VF.code_fingerprint([a.ckpt])
    json.dump(run, open(os.path.join(od, f"run_{pi}.json"), "w"), indent=1, default=float)


if __name__ == "__main__":
    main()
