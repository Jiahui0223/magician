"""EXP6 V5 -- surface-belief metrics on the evaluation multi-surface GT (seen2scene env; CPU or 1 GPU).

For every dumped ray (eval_g1_g2.py) with GT (eval_belief_gt.py) the V5 surface head b(t) = sigmoid(SurfaceMLP(z, t))
of the frozen checkpoint is queried at
  surface queries   every GT surface depth t_k inside [t_lo, t_hi] (k = 1 first surface, k >= 2 second / later)
  grid queries      64 log-uniform depths in [t_lo, t_hi] (fixed); label y = a GT surface within Delta / 2 (Delta = 8 VS)
Metrics (threshold 0.5): recall of the first surface, recall of the second+ surfaces (by depth bucket of the gap behind
the first surface and of the depth itself), grid precision / recall / Brier / ECE (10 bins), the false-surface rate
behind the first surface (grid negatives with t > t_1), all rays and the UNKNOWN-class rays. Rays without a valid
range are skipped. Output JSON. Usage: python eval_belief_metrics.py --ckpt V5_CKPT --dirs D1 [D2 ...] --out OUT.json"""
import argparse
import glob
import json
import os

import numpy as np
import torch

from seen2scene.futureview import model as M
from seen2scene.futureview import v5_model as V5

DELTA = 8 * M.VS
NG = 64
UNKNOWN = 0


def ece(p, y, bins=10):
    e, n = 0.0, len(p)
    for i in range(bins):
        m = (p >= i / bins) & (p < (i + 1) / bins if i < bins - 1 else p <= 1.0)
        if m.any():
            e += m.sum() / n * abs(p[m].mean() - y[m].mean())
    return float(e)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--ckpt", required=True)
    ap.add_argument("--dirs", nargs="+", required=True)
    ap.add_argument("--out", required=True)
    a = ap.parse_args()
    ck = torch.load(a.ckpt, map_location="cpu", weights_only=False)
    m = V5.V5Model(ck["model_cfg"]); m.load_state_dict(ck["ema"]); m.eval()
    dec = m.dec
    from seen2scene.futureview.geometry import CLS_UNKNOWN
    res = {}
    for d in a.dirs:
        acc = {k: [] for k in ("p_s", "first", "unk_s", "gap_s", "t_s", "p_g", "y_g", "unk_g", "behind_g", "n_surf", "unk_r")}
        for gp in sorted(glob.glob(os.path.join(d, "gt_*.npz"))):
            g = np.load(gp)
            b = np.load(gp.replace("gt_", "belief_"))
            z = torch.as_tensor(b["z"].astype(np.float32)).reshape(-1, b["z"].shape[-1])
            tcap = torch.as_tensor(b["tcap"].astype(np.float32)).reshape(-1)
            surf = g["surf"].reshape(-1, g["surf"].shape[-1]).astype(np.float64)
            lo, hi = g["t_lo"].reshape(-1).astype(np.float64), g["t_hi"].reshape(-1).astype(np.float64)
            unk = (b["cls"].reshape(-1) == CLS_UNKNOWN)
            ok = hi > lo
            z, tcap, surf, lo, hi, unk = z[ok], tcap[ok], surf[ok], lo[ok], hi[ok], unk[ok]
            n = len(z)
            if n == 0:
                continue
            ts = np.where(np.isfinite(surf), surf, 0.0)
            grid = np.exp(np.log(np.maximum(lo, 1e-4))[:, None] + (np.log(hi) - np.log(np.maximum(lo, 1e-4)))[:, None] * (np.arange(NG) + 0.5)[None] / NG)
            q = torch.as_tensor(np.concatenate([ts, grid], 1), dtype=torch.float32)
            with torch.no_grad():
                p = torch.sigmoid(torch.cat([dec.surface_logits(z[s:s + 4096], q[s:s + 4096], tcap[s:s + 4096]) for s in range(0, n, 4096)])).numpy()
            ks = surf.shape[1]
            fin = np.isfinite(surf)
            ps, pg = p[:, :ks], p[:, ks:]
            first = np.zeros_like(fin); first[:, 0] = True
            t1 = np.where(fin[:, 0], surf[:, 0], np.inf)
            gap = surf - t1[:, None]
            yg = (np.abs(grid[:, :, None] - np.where(fin, surf, np.inf)[:, None]) <= DELTA / 2).any(-1)
            acc["p_s"].append(ps[fin]); acc["first"].append(first[fin]); acc["unk_s"].append(np.broadcast_to(unk[:, None], fin.shape)[fin])
            acc["gap_s"].append(gap[fin] / M.VS); acc["t_s"].append(surf[fin] / M.VS)
            acc["p_g"].append(pg.ravel()); acc["y_g"].append(yg.ravel()); acc["unk_g"].append(np.repeat(unk, NG))
            acc["behind_g"].append((grid > t1[:, None]).ravel())
            acc["n_surf"].append(fin.sum(1)); acc["unk_r"].append(unk)
        if not acc["p_s"]:
            continue
        A = {k: np.concatenate(v) for k, v in acc.items()}
        out = {}
        for nm, rs, ss, gs in (("all", np.ones_like(A["unk_r"]), np.ones_like(A["unk_s"]), np.ones_like(A["unk_g"])),
                               ("unknown_rays", A["unk_r"], A["unk_s"], A["unk_g"])):
            ss, gs, rs = ss.astype(bool), gs.astype(bool), rs.astype(bool)
            hit = A["p_s"] > 0.5
            f, sp = ss & A["first"], ss & ~A["first"]
            pg, yg = A["p_g"][gs], A["y_g"][gs]
            pr = pg > 0.5
            beh = gs & A["behind_g"] & ~A["y_g"]
            buckets = {}
            for lab, lo_, hi_ in (("gap<8vox", 0, 8), ("gap8-32vox", 8, 32), ("gap32-128vox", 32, 128), ("gap>=128vox", 128, np.inf)):
                mm = sp & (A["gap_s"] >= lo_) & (A["gap_s"] < hi_)
                buckets[lab] = {"n": int(mm.sum()), "recall": float(hit[mm].mean()) if mm.any() else None}
            dbk = {}
            for lab, lo_, hi_ in (("t<64vox", 0, 64), ("t64-256vox", 64, 256), ("t>=256vox", 256, np.inf)):
                mm = sp & (A["t_s"] >= lo_) & (A["t_s"] < hi_)
                dbk[lab] = {"n": int(mm.sum()), "recall": float(hit[mm].mean()) if mm.any() else None}
            out[nm] = {"rays": int(rs.sum()), "rays_ge2_surfaces": float((A["n_surf"][rs] >= 2).mean()) if rs.any() else None,
                       "n_first": int(f.sum()), "recall_first": float(hit[f].mean()) if f.any() else None,
                       "n_second_plus": int(sp.sum()), "recall_second_plus": float(hit[sp].mean()) if sp.any() else None,
                       "recall_second_plus_by_gap": buckets, "recall_second_plus_by_depth": dbk,
                       "grid_n": int(gs.sum()), "grid_pos_rate": float(yg.mean()), "grid_precision": float((pr & yg).sum() / max(1, pr.sum())),
                       "grid_recall": float((pr & yg).sum() / max(1, yg.sum())), "grid_brier": float(((pg - yg) ** 2).mean()), "grid_ece": ece(pg, yg.astype(float)),
                       "false_surface_rate_behind_first": float((A["p_g"][beh] > 0.5).mean()) if beh.any() else None,
                       "first_hit_only_baseline_recall_second_plus": 0.0}
        res[os.path.relpath(d)] = out
        print(d, json.dumps(out["all"], default=float)[:400], flush=True)
    json.dump({"ckpt": a.ckpt, "delta_model_units": DELTA, "grid_queries": NG, "metrics": res}, open(a.out, "w"), indent=1, default=float)


if __name__ == "__main__":
    main()
