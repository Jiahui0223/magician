"""EXP5 Task 0 / T0.12-C -- offline diagnosis, EVALUATION side (magician env, one GPU; uses the GT mesh and the frozen
C_obs evaluator; no planner sees anything of this). Along a stored grid trajectory, at every state t:
  * the true one-step C_obs gain of every legal candidate: render the candidate with the frozen sensor, the sim's 10 %
    sample with the seed the sim would use for frame t + 1, newly covered visible GT points / all visible points
  * the part of that gain inside V2's active region at t (n_win x n_win windows around the camera of the last refresh
    t_r = 10 * floor(t / 10); same formula as scripts/mp3d_frame.region_box_m)
  * the share of the still-uncovered visible GT points (remaining work) that lies inside that region
Output MC/diag_c/oracle/<group>/<tag>.json
"""
import argparse
import json
import os
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
import t12_sim as T12  # noqa: E402

import numpy as np  # noqa: E402

S = T12.S
MC_DIR = os.path.join(S.W, "magician_comparison")


def region_box_xy(card, cam_rt):
    c = S.rt_to_zup_m(np.asarray(cam_rt, np.float64))[:2]
    box = np.asarray(card["box_m"])
    half = 0.5 * (1 + (card["n_win"] - 1) * 0.8) * card["height_m"] * 0.99
    return np.maximum(c - half, box[0, :2]), np.minimum(c + half, box[1, :2])


def main():
    import torch
    ap = argparse.ArgumentParser()
    ap.add_argument("--group", required=True)
    ap.add_argument("--tag", required=True)
    a = ap.parse_args()
    ep_dir = os.path.join(MC_DIR, "grid_protocol", "closed_loop", a.group, a.tag)
    ep = json.load(open(os.path.join(ep_dir, "episode.json")))
    scan, lv = ep["level_key"].rsplit("_L", 1)
    L = S.Level(scan, int(lv))
    card = json.load(open(os.path.join(S.W, "preprocessing", "scenes", f"{L.key}.json")))
    v, f = S.mesh_zup(scan)
    dev = torch.device("cuda")
    sen = S.Sensor(S.zup_m_to_rt(v), f, dev)
    cov = S.Coverage(L, sen)
    P_xy = S.rt_to_zup_m(cov.pts)[:, :2]
    nvis = max(1, int(cov.visible.sum()))
    rows = []
    for t in range(ep["moves"]):
        rec = np.load(os.path.join(ep_dir, "svc", f"svc_t{t}.npz"))
        cov.update(rec["sample_points"])
        if abs(cov.value() - ep["coverage"][t]) > 1e-12:
            raise SystemExit(f"STOP: coverage replay differs at t={t}")
        tr = 10 * (t // 10)
        lo, hi = region_box_xy(card, np.load(os.path.join(ep_dir, "svc", f"svc_t{tr}.npz"))["cam_center"])
        inside = np.all((P_xy >= lo) & (P_xy <= hi), axis=1)
        rem = cov.visible & ~cov.covered
        gains, gains_in = [], []
        seed = S.derived_seed(L.key, ep["start"], t + 1, "sample")
        for R, T in zip(rec["legal_R"], rec["legal_T"]):
            fr = sen.render(torch.tensor(R[None], dtype=torch.float32, device=dev), torch.tensor(T[None], dtype=torch.float32, device=dev))
            pts = sen.sample(fr, seed)[0]
            new = cov._near(cov.tree, pts.cpu().numpy()) & rem
            gains.append(float(new.sum() / nvis))
            gains_in.append(float((new & inside).sum() / nvis))
        rows.append({"t": t, "cov": cov.value(), "gain": gains, "gain_in_region": gains_in, "region_xy": [lo.tolist(), hi.tolist()],
                     "remaining": float(rem.sum() / nvis), "remaining_in_region_share": float((rem & inside).sum() / max(1, rem.sum()))})
        if t % 25 == 0:
            print(a.group, a.tag, t, "best gain %.4f" % max(gains), "rem in region %.2f" % rows[-1]["remaining_in_region_share"], flush=True)
    od = os.path.join(MC_DIR, "diag_c", "oracle", a.group)
    os.makedirs(od, exist_ok=True)
    json.dump({"group": a.group, "tag": a.tag, "rows": rows}, open(os.path.join(od, f"{a.tag}.json"), "w"))


if __name__ == "__main__":
    main()
