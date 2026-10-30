"""EXP5 Task 0.4 -- real-captured view protocol R (magician env; simulator / evaluation side).

traj  : for every M reference trajectory, the SAME poses, legal lists and seeds; the observation at each step is the
        real depth of the pose's panorama forward-projected into the frozen virtual sensor (mp3d_real.real_frame);
        10 % sample points with the frozen rule on that zbuf; level-scale C_obs (same GT universe / eps as M) from the
        R sample points. Writes W/preprocessing/traj_R/<traj>/{svc_t<t>.npz, traj.json}.
label : for every fixed state (same ids as M): per legal candidate, the real future depth on the 64 x 114 scoring
        rays (forward projection of the candidate panorama's real points into the T1.1 label cameras; NaN = no real
        sample -> the ray is not evaluated) and the true gain from the candidate's R observation (seed of step t + 1).
        Writes W/fixed_state_R/labels/<state>.npz + .json.
"""
import argparse
import json
import os
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
import mp3d_sim as S  # noqa: E402
import mp3d_real as RL  # noqa: E402

import numpy as np  # noqa: E402

sys.path.insert(0, os.path.join(S.REPO, "experiments", "exp5_future_view", "task1", "scripts"))
sys.path.insert(0, S.REPO)


def level_ctx(scan, lv, dev, ctx):
    if ctx.get("key") != (scan, lv):
        L = S.Level(scan, lv)
        L.build_graph()
        v, f = S.mesh_zup(scan)
        ctx.clear()
        ctx.update(key=(scan, lv), L=L, sen=S.Sensor(S.zup_m_to_rt(v), f, dev), real=RL.RealPanos(scan, dev))
    return ctx["L"], ctx["sen"], ctx["real"]


def mode_traj(traj, dev, ctx):
    import torch
    mdir = os.path.join(S.W, "preprocessing", "traj", traj)
    mt = json.load(open(os.path.join(mdir, "traj.json")))
    key, start = mt["scene"]["key"], mt["start_pano"]
    scan, lv = key.split("_L")[0], int(key.split("_L")[1])
    L, sen, real = level_ctx(scan, lv, dev, ctx)
    cov = S.Coverage(L, sen)
    od = os.path.join(S.W, "preprocessing", "traj_R", traj)
    os.makedirs(od, exist_ok=True)
    rows = []
    for t in range(len(mt["rows"])):
        m = S.MC.load_service_record(os.path.join(mdir, f"svc_t{t}.npz"))
        pose = tuple(int(x) for x in m["pose_idx"])
        R = torch.tensor(m["R"][None], dtype=torch.float32, device=dev)
        T = torch.tensor(m["T"][None], dtype=torch.float32, device=dev)
        fr = RL.real_frame(sen, real, L.panos[pose[0]]["name"], R, T)
        pts, flat, _, _, _ = sen.sample(fr, S.derived_seed(key, start, t, "sample"))
        c = cov.update(pts.cpu().numpy())
        rec = {k: m[k] for k in m if k not in ("zbuf", "mask", "sample_pixel_flat", "sample_points")}
        rec.update({"zbuf": fr["zbuf"][0, ..., 0].cpu().numpy().astype(np.float32),
                    "mask": fr["mask"][0, ..., 0].cpu().numpy().astype(bool),
                    "sample_pixel_flat": flat.numpy().astype(np.int64), "sample_points": pts.cpu().numpy().astype(np.float32)})
        S.MC.validate_service_record(rec)
        np.savez(os.path.join(od, f"svc_t{t}.npz"), **rec)
        rows.append({"t": t, "pose": list(pose), "cov": c, "cov_M": mt["rows"][t]["cov"], "fill": float(fr["mask"].float().mean())})
    json.dump({"scene": mt["scene"], "start_pano": start, "protocol": "R", "rows": rows}, open(os.path.join(od, "traj.json"), "w"),
              indent=1, default=float)
    print(traj, "R cov", [round(r["cov"], 3) for r in rows[::10]], "fill %.3f" % np.mean([r["fill"] for r in rows]), flush=True)


def mode_label(st, dev, ctx, out):
    import torch
    import t11_labels as TL
    from seen2scene.futureview import rays as RY
    scan, lv = st["house"], int(st["level_key"].split("_L")[1])
    L, sen, real = level_ctx(scan, lv, dev, ctx)
    td = os.path.join(S.W, "preprocessing", "traj_R", st["traj"])
    tr = json.load(open(os.path.join(td, "traj.json")))
    cov = S.Coverage(L, sen)
    for i in range(st["t"] + 1):
        c = cov.update(np.load(os.path.join(td, f"svc_t{i}.npz"))["sample_points"])
    assert abs(c - tr["rows"][st["t"]]["cov"]) < 1e-12
    rec = S.MC.load_service_record(os.path.join(td, f"svc_t{st['t']}.npz"))
    base = cov.covered.copy()
    u, v = RY.grid_uv((0.0, 0.0))
    gains, ts = [], []
    for j in range(len(rec["legal_R"])):
        name = L.panos[int(rec["legal_idx"][j][0])]["name"]
        R = torch.tensor(rec["legal_R"][j][None], dtype=torch.float32, device=dev)
        T = torch.tensor(rec["legal_T"][j][None], dtype=torch.float32, device=dev)
        fr = RL.real_frame(sen, real, name, R, T)
        pts, _, _, _, _ = sen.sample(fr, S.derived_seed(st["level_key"], st["start"], st["t"] + 1, "sample"))
        cov.covered = base.copy()
        gains.append(100.0 * (cov.update(pts.cpu().numpy()) - c))
        cam = TL.label_cameras(rec["legal_R"][j][None], rec["legal_T"][j][None], np.zeros((1, 2)), dev)
        zb = RL.zbuffer(cam, real.points(name), RY.GH, RY.GW).reshape(-1).double().cpu().numpy()
        _, dz, _ = S.MC.pixel_rays(rec["legal_R"][j], rec["legal_T"][j], u, v)
        ts.append(np.where(np.isfinite(zb), zb / dz / L.k, np.nan))
    cov.covered = base
    np.savez_compressed(os.path.join(out, f"{st['id']}.npz"), tstar=np.asarray(ts, np.float32), gain=np.asarray(gains),
                        legal_idx=rec["legal_idx"])
    json.dump({"state": st, "cov_t": c, "gain": gains, "k_rt": L.k, "label_valid_frac": float(np.isfinite(ts).mean())},
              open(os.path.join(out, f"{st['id']}.json"), "w"), indent=1, default=float)
    print(st["id"], "R gains", [round(g, 2) for g in gains], "valid %.3f" % float(np.isfinite(ts).mean()), flush=True)


def main():
    import torch
    ap = argparse.ArgumentParser()
    ap.add_argument("mode", choices=["traj", "label"])
    ap.add_argument("--part", default="0/1")
    a = ap.parse_args()
    dev = torch.device("cuda")
    pi, pn = (int(x) for x in a.part.split("/"))
    ctx = {}
    if a.mode == "traj":
        trajs = sorted(os.listdir(os.path.join(S.W, "preprocessing", "traj")))
        for tj in trajs[pi::pn]:
            if not os.path.exists(os.path.join(S.W, "preprocessing", "traj_R", tj, "traj.json")):
                mode_traj(tj, dev, ctx)
    else:
        out = os.path.join(S.W, "fixed_state_R", "labels")
        os.makedirs(out, exist_ok=True)
        sts = json.load(open(os.path.join(S.W, "manifests", "indoor_states.json")))["states"]
        sts = sorted(sts, key=lambda s: s["level_key"])[pi::pn]
        for st in sts:
            if not os.path.exists(os.path.join(out, f"{st['id']}.json")):
                mode_label(st, dev, ctx, out)


if __name__ == "__main__":
    main()
