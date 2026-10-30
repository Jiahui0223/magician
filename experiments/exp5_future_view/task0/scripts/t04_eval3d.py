"""EXP5 Task 0.4 -- 3D metrics of the world-model arms on the indoor fixed states (seen2scene env, evaluation side:
reads the GT mesh; protocol §6). Per state and model (frozen / C): frozen P1 replay into the state's region (as t04_gen),
the int8 worlds (w = q / 127 * TRUNC), dense GT surface samples inside the region (3 M, area weighted, seed 0).
  U          : UNKNOWN voxels of the P1 structure inside the domain
  S_k        : predicted surface voxels of world k, |w_k| < 1 voxel
  P@tau      : fraction of S_k & U voxel centres within tau voxels of a GT sample
  R@tau      : fraction of GT samples lying in U voxels within tau voxels of a centre of S_k
  F@tau      : harmonic mean; tau in {1, 2}; K4 = mean over worlds, plus the union recall R@2 over the 4 worlds
  band_abs   : mean |w_k - P1 TSDF| on BAND voxels (voxels); band_sign_agree: sign agreement there
  empty_conflict : fraction of S_k voxels lying in observed EMPTY voxels
  diversity  : mean pairwise |S_i xor S_j| / |S_i or S_j| inside U
Output W/fixed_state/eval3d/<model>/<state>.json
"""
import argparse
import json
import os
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
import mp3d_frame as F  # noqa: E402
import t04_gen as TG  # noqa: E402
import t03_p1check as PC1  # noqa: E402

import numpy as np  # noqa: E402
import torch  # noqa: E402

import m_common as MC  # noqa: E402

OUT = os.path.join(F.W, "fixed_state", "eval3d")
Q_SURF = 42                     # |q| <= 42  <=>  |w| < 1 voxel (TRUNC = 3 voxels -> 127 / 3 = 42.3)


def run(model, st, dev, root):
    from scipy.spatial import cKDTree
    card = F.load_card(st["level_key"])
    rec, fr, dims, dom, win, box, fu = TG.replay(card, st["traj"], st["t"], dev)
    z = np.load(os.path.join(root, model, f"{st['id']}.npz"))
    j = json.load(open(os.path.join(root, model, f"{st['id']}.json")))
    out = {"state": st, "model": model, "availability": j["availability"]}
    if j["availability"] != "A1":
        return out
    s = fu.structure().cpu().numpy()
    assert np.array_equal(s, z["structure"]), "P1 replay differs from the generation run"
    p1 = fu.input_volume().cpu().numpy() / MC.VS
    D = np.ones(dims, bool)
    for a in range(3):
        sh = [1, 1, 1]
        sh[a] = -1
        D &= np.asarray(dom[a]).reshape(sh)
    U = (s == MC.UNKNOWN) & D
    B = (s == MC.BAND) & D
    E = (s == MC.EMPTY) & D
    g = fr.runtime_to_model(PC1.dense_gt(st["level_key"], box, card["s_rt"]))
    gi = np.floor(g / MC.VS).astype(np.int64)
    ing = np.all((gi >= 0) & (gi < np.asarray(dims)), axis=1)
    g, gi = g[ing], gi[ing]
    gu = U[gi[:, 0], gi[:, 1], gi[:, 2]]
    gtree = cKDTree(g)
    ws = z["worlds_q"]
    rows, surfs = [], []
    for k in range(len(ws)):
        q = ws[k]
        S = (np.abs(q.astype(np.int16)) <= Q_SURF) & D
        surfs.append(S & U)
        su = np.argwhere(S & U)
        r = {"k": k, "n_pred_unseen": int(len(su)), "n_gt_unseen": int(gu.sum())}
        if len(su):
            dpred, _ = gtree.query((su + 0.5) * MC.VS, k=1, workers=-1)
            stree = cKDTree((su + 0.5) * MC.VS)
            dgt, _ = stree.query(g[gu], k=1, workers=-1) if gu.any() else (np.zeros(0), None)
        else:
            dpred, dgt = np.zeros(0), np.full(int(gu.sum()), np.inf)
        for tau in (1, 2):
            P = float((dpred < tau * MC.VS).mean()) if len(dpred) else None
            R = float((dgt < tau * MC.VS).mean()) if len(dgt) else None
            r[f"P@{tau}"], r[f"R@{tau}"] = P, R
            r[f"F@{tau}"] = (2 * P * R / (P + R) if P is not None and R is not None and (P + R) > 0 else 0.0)
        w = q.astype(np.float64) / 127.0 * 3.0                  # voxels
        r["band_abs"] = float(np.abs(np.clip(w[B], -3, 3) - np.clip(p1[B], -3, 3)).mean()) if B.any() else None
        r["band_sign_agree"] = float((np.sign(w[B]) == np.sign(p1[B])).mean()) if B.any() else None
        r["empty_conflict"] = float((S & E).sum() / max(1, S.sum()))
        r["recall2_mask"] = dgt < 2 * MC.VS
        rows.append(r)
    union = np.zeros(int(gu.sum()), bool)
    for r in rows:
        union |= r.pop("recall2_mask")
    div = [float((surfs[i] ^ surfs[j]).sum() / max(1, (surfs[i] | surfs[j]).sum()))
           for i in range(len(surfs)) for j in range(i + 1, len(surfs))]
    keys = ("P@1", "R@1", "F@1", "P@2", "R@2", "F@2", "band_abs", "band_sign_agree", "empty_conflict")
    out.update({"per_world": rows, "K1": {k: rows[0][k] for k in keys},
                "K4": {k: float(np.mean([r[k] for r in rows if r[k] is not None])) for k in keys},
                "K4_union_R@2": float(union.mean()) if len(union) else None,
                "diversity": float(np.mean(div)) if div else None, "n_gt_unseen": int(gu.sum()),
                "unknown_frac_domain": float(U.sum() / max(1, D.sum()))})
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", choices=["frozen", "C"], required=True)
    ap.add_argument("--part", default="0/1")
    ap.add_argument("--states", default=os.path.join(F.W, "manifests", "indoor_states.json"))
    ap.add_argument("--root", default=os.path.join(F.W, "fixed_state"))
    ap.add_argument("--protocol", choices=["M", "R", "O"], default="M")
    a = ap.parse_args()
    if a.protocol == "R":
        TG.TRAJ_DIR = "traj_R"
    if a.protocol == "O":
        TG.TRAJ_DIR = "outdoor_traj"
    dev = torch.device("cuda")
    sts = json.load(open(a.states))["states"]
    pi, pn = (int(x) for x in a.part.split("/"))
    od = os.path.join(a.root, "eval3d", a.model)
    os.makedirs(od, exist_ok=True)
    for st in sts[pi::pn]:
        p = os.path.join(od, f"{st['id']}.json")
        if os.path.exists(p) or not os.path.exists(os.path.join(a.root, a.model, f"{st['id']}.json")):
            continue
        lk = p.replace(".json", ".lock")          # atomic claim (several workers on one list; reap_locks.py states)
        try:
            fd = os.open(lk, os.O_CREAT | os.O_EXCL | os.O_WRONLY)
        except FileExistsError:
            continue
        os.write(fd, f"{os.uname()[1]}:{os.getpid()}".encode())
        os.close(fd)
        if os.path.exists(p) or os.path.exists(p.replace(".json", ".failed.json")):
            continue
        r = run(a.model, st, dev, a.root)
        json.dump(r, open(p + ".tmp", "w"), indent=1, default=float)
        os.replace(p + ".tmp", p)
        print(st["id"], a.model, {k: round(v, 3) for k, v in r.get("K4", {}).items() if v is not None}, flush=True)


if __name__ == "__main__":
    main()
