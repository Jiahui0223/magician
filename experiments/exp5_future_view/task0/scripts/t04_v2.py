"""EXP5 Task 0.4 -- EXP5 V2 arms (K = 4; K = 1 = sample 0) on the indoor fixed states (seen2scene env; planner side:
scene card + service records 0..t only). Frozen EXP5 main checkpoint (frozen_values.json "checkpoint", sha checked),
delta = frozen_values "scoring" delta (4); noise seeds exp5-V2|task0|<state id>|k<k> (shared by all candidates).
Per state: region frame + frozen P1 replay (identical to t04_gen), V2 Geometry, encoder, v2_plan_core.score_state over
the legal candidates of record t -> DistP1 (K4 mean / K1 sample 0), U_raw / DistU, Frontier DistP1, and per candidate
the decoded first-hit distance of every sample on the 64 x 114 scoring rays plus the P1 ray classes / t_known / t_max.
Outputs W/fixed_state/V2/<state>.npz + .json.
"""
import argparse
import json
import os
import sys
import time

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
import mp3d_frame as F  # noqa: E402
import t04_gen as TG  # noqa: E402

import numpy as np  # noqa: E402
import torch  # noqa: E402

sys.path.insert(0, os.path.join(F.REPO, "experiments", "exp5_future_view", "task1", "scripts"))
import v2_frozen as VF  # noqa: E402

OUT = os.path.join(F.W, "fixed_state", "V2")


def run_state(core, st, delta, dev):
    import v2_plan_core as PC
    from seen2scene.futureview import geometry as G
    from seen2scene.futureview import predictor as PR
    from seen2scene.futureview import rays as RY
    from seen2scene.futureview import scoring as SCO
    card = F.load_card(st["level_key"])
    t0 = time.time()
    rec, fr, dims, dom, win, box, fu = TG.replay(card, st["traj"], st["t"], dev)
    geo = PC.Geometry.from_fusion(fr, dims, dom, fu, dev)
    h = core.pred.encode(geo.coarse())
    eps = PR.noise(dev, 4, lambda k: RY.sha_seed(f"exp5-V2|task0|{st['id']}|k{k}"))
    z0 = F.z0_of_card(card)
    out, tm, views = PC.score_state(core, geo, h, rec["legal_R"], rec["legal_T"], eps, z0, deltas=(delta,), keep_hits=True)
    n = len(views)
    td = np.full((n, 4, 7296), np.inf, np.float32)
    hu = np.zeros((n, 4, 7296), bool)
    cls = np.zeros((n, 7296), np.uint8)
    tk = np.zeros((n, 7296), np.float32)
    tmax = np.zeros((n, 7296), np.float32)
    front = []
    for j, (v, hh) in enumerate(zip(views, out["hits"])):
        pr = v["pr"]
        td[j] = np.where(hh["hit_u"].cpu().numpy(), hh["t"].float().cpu().numpy(), np.inf)
        hu[j] = hh["hit_u"].cpu().numpy()
        cls[j] = pr["cls"].to(torch.uint8).cpu().numpy()
        tk[j] = pr["t_known"].float().cpu().numpy()
        tmax[j] = pr["tmax"].float().cpu().numpy()
        tf = torch.where(pr["n_u"] > 0, pr["d_lo"], pr["t_known"])
        _, wf = SCO.new_set_from_depth(tf, v["o"], v["d"], v["dz"], fr.k_M, geo.dims, geo.domain, geo.caster.band_dil, z0)
        front.append(float(wf.sum()))
    os.makedirs(OUT, exist_ok=True)
    np.savez_compressed(os.path.join(OUT, f"{st['id']}.npz"), t=td, hit_u=hu, cls=cls, t_known=tk, tmax=tmax,
                        legal_idx=rec["legal_idx"])
    res = {"state": st, "model": "V2", "delta": delta, "n_legal": n, "n_windows": win["n"], "dims": dims, "box_m": box.tolist(),
           "k_rt": fr.k_M, "z0_rt": z0, "distp1_K4": out["distp1"], "distp1_K1": [x[0] for x in out["distp1_k"]],
           "distp1_k": out["distp1_k"], "frontier_distp1": front, "timing": tm, "total_s": time.time() - t0,
           **{k: out[k] for k in out if k.startswith(("u_raw", "dist_u"))}, "ckpt_sha256": core.ckpt_sha}
    json.dump(res, open(os.path.join(OUT, f"{st['id']}.json"), "w"), indent=1, default=float)
    print(st["id"], "V2", f"{res['total_s']:.1f}s", flush=True)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--part", default="0/1")
    ap.add_argument("--states", default=os.path.join(F.W, "manifests", "indoor_states.json"))
    ap.add_argument("--out", default=None, help="output root (default: the formal fixed_state dir)")
    ap.add_argument("--protocol", choices=["M", "R", "O"], default="M")
    ap.add_argument("--only", default="")
    a = ap.parse_args()
    global OUT
    if a.protocol == "R":
        TG.TRAJ_DIR, OUT = "traj_R", os.path.join(F.W, "fixed_state_R", "V2")
    if a.protocol == "O":
        TG.TRAJ_DIR, OUT = "outdoor_traj", os.path.join(F.W, "fixed_state_out", "V2")
    OUT = os.path.join(a.out, "V2") if a.out else OUT
    import v2_plan_core as PC
    VF.verify_frozen()
    dev = torch.device("cuda")
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    fv = VF.frozen_values()
    core = PC.Core(os.path.join(VF.REPO, fv["checkpoint"]["path"]), dev, expect_sha=fv["checkpoint"]["sha256"])
    sts = json.load(open(a.states))["states"]
    if a.only:
        sts = [s for s in sts if s["id"] in a.only.split(",")]
    pi, pn = (int(x) for x in a.part.split("/"))
    for st in sts[pi::pn]:
        if os.path.exists(os.path.join(OUT, f"{st['id']}.json")):
            continue
        run_state(core, st, int(fv["scoring"]["delta_vox"]), dev)


if __name__ == "__main__":
    main()
