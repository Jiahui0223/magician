"""EXP6 V5.2 -- normalisation floors of the camera-utility terms from Train-6 S-c only (seen2scene env, 1 GPU).

FROZEN_PROTOCOL §2: on the S-c held-out trajectories of the six Train-6 scenes (the first S-c trajectory of every scene,
states t = 5, 15, ..., 95 = 60 states), replay the real service records with the planner's B2 maps (D100 fusion for the
model input / structure, S10 fusion for band_dil), refresh the encoder input every 10 steps (R10), score every legal
candidate with the frozen final checkpoint, take the per-state maximum of S_vis / P_surf / P_space over the candidates,
floor_k = 0.05 x the median of these maxima. Writes the frozen configs/score_v52.json (version score_v52_v1) and
reports/calibration_floors.json. No evaluation scene, no GT is read (scene frames from settings.json; z0 = MAGICIAN
scene_scale / 2 of the Train-6 scene).
Usage: python calibrate_floors.py --ckpt V52_FINAL"""
import argparse
import json
import os
import sys
import time

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
import v52_common as VC  # noqa: E402

REPO = VC.REPO
T1 = os.path.join(REPO, "experiments", "exp5_future_view", "task1", "scripts")
sys.path.insert(0, T1)
sys.path.insert(0, os.path.join(REPO, "experiments", "exp5_future_view", "task2", "scripts"))
sys.path.insert(0, os.path.join(REPO, "experiments", "exp5_future_view", "task3_full_depth_p1", "scripts"))
import v2_common as LC  # noqa: E402,F401  (planner path setup)
import v2_frozen as VF  # noqa: E402,F401
import v2_data as VD  # noqa: E402
import numpy as np  # noqa: E402
import torch  # noqa: E402


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--ckpt", required=True)
    ap.add_argument("--frac", type=float, default=0.05)
    ap.add_argument("--dry", action="store_true", help="smoke: 1 scene, nothing frozen (output under work_dirs)")
    a = ap.parse_args()
    import m_common as MC
    import p1_full as PF
    import t5_diagnostic as TD
    import t11_labels as L11
    import v2_plan_core as PC
    import abl_common as AC
    from seen2scene.futureview import model as M
    dev = torch.device("cuda")
    model, sha = VC.load_model(a.ckpt, dev)
    cfg = dict(VC.SC52.DEFAULT, **VC.score_cfg())
    _, held = VD.load_split("train6_minus_Sc")
    first = {}
    for s, tj in held:
        first.setdefault(s, tj)
    maxima = {"s_vis": [], "p_surf": [], "p_space": []}
    per_state = []
    t0 = time.time()
    with torch.no_grad():
        for s, tj in sorted(first.items())[:1 if a.dry else None]:
            fr, dims, dom, _ = TD.scene_geometry(s)
            z0 = AC.z0_of(s)
            recs = L11.records(s, tj)
            f100 = PF.P1FusionFull(fr, dims, dom, device=dev)
            f10 = PF._P1(fr, dims, dom, device=dev)
            h = None
            for t in range(96):
                rec = MC.load_service_record(recs[t])
                f100.integrate_record(rec); f10.integrate_record(rec)
                if t % 10 == 0:
                    geo_r = PC.Geometry.from_fusion(fr, dims, dom, f100, dev)
                    h = model.encode(M.encoder_input(geo_r.coarse().to(dev)))
                    del geo_r
                if t % 10 != 5 or len(rec["legal_R"]) == 0:
                    continue
                geo = VC.b2_geometry(PC, MC, (fr, dims, dom), f100, f10, dev)
                views = [geo.view(R, T) for R, T in zip(rec["legal_R"], rec["legal_T"])]
                raw = VC.SC52.score_state(model, h, geo, views, float(geo.fr.k_M), z0, cfg)
                row = {"scene": s, "traj": tj, "t": t, "n": len(views)}
                for k in maxima:
                    maxima[k].append(max(raw[k]))
                    row[k] = raw[k]
                per_state.append(row)
                del geo
            del f100, f10
            print(s, tj, len(per_state), "%.0fs" % (time.time() - t0), flush=True)
    floors = {nm: float(a.frac * np.median(maxima[k])) for nm, k in (("vis", "s_vis"), ("surf", "p_surf"), ("space", "p_space"))}
    out = {"ckpt": a.ckpt, "ckpt_sha256": sha, "rule": f"floor_k = {a.frac} x median over S-c states of the per-state candidate maximum",
           "states": len(per_state), "trajectories": first, "floors": floors,
           "max_quantiles": {k: np.quantile(v, [0.05, 0.25, 0.5, 0.75, 0.95]).tolist() for k, v in maxima.items()}, "per_state": per_state}
    if a.dry:
        json.dump(out, open(os.path.join(VC.W52, "logs", "calibration_dry.json"), "w"), indent=1, default=float)
        print(json.dumps({"floors": floors, "max_quantiles": out["max_quantiles"]}, indent=1))
        return
    json.dump(out, open(os.path.join(VC.E52, "reports", "calibration_floors.json"), "w"), indent=1, default=float)
    c = dict(VC.score_cfg())
    c.update({"version": "score_v52_v1", "floors": floors, "lambdas": [0.0, 0.1], "eta": 0.5,
              "doc": f"FROZEN {time.strftime('%F %T')}: floors from Train-6 S-c calibration ({len(per_state)} states, reports/calibration_floors.json) on {os.path.basename(a.ckpt)} sha {sha[:12]}"})
    json.dump(c, open(os.path.join(VC.E52, "configs", "score_v52.json"), "w"), indent=1)
    print(json.dumps({"floors": floors, "max_quantiles": out["max_quantiles"]}, indent=1))


if __name__ == "__main__":
    main()
