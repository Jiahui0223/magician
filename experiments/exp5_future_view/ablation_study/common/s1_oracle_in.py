"""EXP5 Ablation S1 (indoor) -- EVALUATION side (magician env, 1 GPU; GT mesh + frozen C_obs evaluator; nothing here is
read by any planner-side code of S2 except t*, which S2 uses only for evaluation-side metrics / GT-depth DistP1).

Per stored source episode (manifest groups): replay the frozen C_obs Coverage over the stored 10 % samples of records
0..t (must equal the stored curve, |d| <= 1e-12, else STOP); at every manifest state t, for EVERY legal candidate
(T0.12-C t12c_oracle.py, unchanged rule):
  oracle gain g = |visible GT points newly within eps of the candidate's frame-(t+1) sample| / |visible GT points|
  (frozen sensor render, the sim's 10 % sample with the seed the sim uses for frame t + 1)
  t* = first GT intersection of the 64 x 114 V2 scoring rays (jitter 0; T1.1 / T2.1a Renderer), model units (k_I)
Check: the executed candidate's sample equals the stored record t + 1 bitwise.
Output W/states/indoor/<sid>/oracle.npz and W/oracle/indoor/<source>_<tag>.json.
Usage: python s1_oracle_in.py --part i/n
"""
import argparse
import json
import os
import sys
import time

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
import ab_common as A  # noqa: E402

sys.path.insert(0, os.path.join(A.EXP5, "task0", "magician_comparison", "scripts"))
sys.path.insert(0, os.path.join(A.EXP5, "task1", "scripts"))
sys.path.insert(0, os.path.join(A.EXP5, "task2", "scripts"))
import t12_sim as T12  # noqa: E402
import t2_common as C2  # noqa: E402

import numpy as np  # noqa: E402

S = T12.S


def groups(man):
    g = {}
    for st in man["states"]:
        src = st["sources"][0]
        g.setdefault((src["source"], st["tag"]), []).append(st)
    return sorted(g.items())


def run_group(src, tag, states, dev):
    import torch
    from pytorch3d.structures import Meshes
    import t11_labels as TL
    st0 = states[0]
    scan, lv, start = st0["scan"], st0["level"], st0["start"]
    C2.assert_mesh_allowed(scan, "eval")
    ep_dir = os.path.join(A.CL_IN, src, tag)
    ep = json.load(open(os.path.join(ep_dir, "episode.json")))
    L = S.Level(scan, lv)
    card = json.load(open(os.path.join(S.W, "preprocessing", "scenes", f"{L.key}.json")))
    v, f = S.mesh_zup(scan)
    sen = S.Sensor(S.zup_m_to_rt(v), f, dev)
    cov = S.Coverage(L, sen)
    mesh = Meshes(verts=[torch.as_tensor(C2.zup_m_to_rt(v), dtype=torch.float32, device=dev)],
                  faces=[torch.as_tensor(f, dtype=torch.int64, device=dev)])
    ren = TL.Renderer(mesh, dev, 0, None, 8)
    nvis = max(1, int(cov.visible.sum()))
    want = {s["t"]: s for s in states}
    summ = {"source": src, "tag": tag, "states": [], "replay_equal": True, "executed_sample_equal": []}
    for t in range(max(want) + 1):
        rec = np.load(os.path.join(ep_dir, "svc", f"svc_t{t}.npz"))
        cov.update(rec["sample_points"])
        if abs(cov.value() - ep["coverage"][t]) > 1e-12:
            raise SystemExit(f"STOP: coverage replay differs {src} {tag} t={t}")
        if t not in want:
            continue
        st = want[t]
        tb = time.time()
        rem = cov.visible & ~cov.covered
        seed = S.derived_seed(L.key, start, t + 1, "sample")
        gains, pts_all, off = [], [], [0]
        for R, T in zip(rec["legal_R"], rec["legal_T"]):
            fr = sen.render(torch.tensor(R[None], dtype=torch.float32, device=dev), torch.tensor(T[None], dtype=torch.float32, device=dev))
            pts = sen.sample(fr, seed)[0].cpu().numpy()
            new = cov._near(cov.tree, pts) & rem
            gains.append(float(new.sum() / nvis))
            pts_all.append(pts.astype(np.float32))
            off.append(off[-1] + len(pts))
        ex = st["sources"][0]["executed_pick"]
        nxt = np.load(os.path.join(ep_dir, "svc", f"svc_t{t + 1}.npz"))["sample_points"]
        eq = bool(ex is not None and np.array_equal(pts_all[ex], nxt))
        summ["executed_sample_equal"].append(eq)
        if not eq:
            raise SystemExit(f"STOP: executed candidate sample differs from record t+1 ({src} {tag} t={t})")
        tstar = ren.tstar(list(rec["legal_R"]), list(rec["legal_T"]), [(0.0, 0.0)] * len(rec["legal_R"]), card["k_I"]).astype(np.float32)
        od = A.state_dir("indoor", st["sid"])
        os.makedirs(od, exist_ok=True)
        np.savez_compressed(os.path.join(od, "oracle.tmp.npz"), gain=np.asarray(gains), cov_t=np.asarray(cov.value()),
                            remaining=np.asarray(rem.sum() / nvis), tstar=tstar, k_M=np.asarray(card["k_I"]),
                            cand_pts=np.concatenate(pts_all), cand_off=np.asarray(off), legal_idx=rec["legal_idx"])
        os.replace(os.path.join(od, "oracle.tmp.npz"), os.path.join(od, "oracle.npz"))
        summ["states"].append({"sid": st["sid"], "t": t, "cov_t": cov.value(), "max_gain": max(gains), "seconds": time.time() - tb})
        print(src, tag, t, "max gain %.4f" % max(gains), "%.1fs" % (time.time() - tb), flush=True)
    return summ


def main():
    import torch
    ap = argparse.ArgumentParser()
    ap.add_argument("--part", default="0/1")
    a = ap.parse_args()
    pi, pn = map(int, a.part.split("/"))
    dev = torch.device("cuda")
    man = A.load_manifest("indoor")
    od = os.path.join(A.W, "oracle", "indoor")
    os.makedirs(od, exist_ok=True)
    for (src, tag), sts in groups(man)[pi::pn]:
        p = os.path.join(od, f"{src}_{tag}.json")
        if os.path.exists(p):
            continue
        tb = time.time()
        summ = run_group(src, tag, sts, dev)
        summ["seconds"] = time.time() - tb
        json.dump(summ, open(p + ".tmp", "w"), indent=1)
        os.replace(p + ".tmp", p)
        torch.cuda.empty_cache()


if __name__ == "__main__":
    main()
