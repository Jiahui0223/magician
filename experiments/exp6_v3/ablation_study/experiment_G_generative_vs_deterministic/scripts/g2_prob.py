"""Experiment G2 items 1-2 on the eval_dev validation subsets (protocol G2-v1 sections 0-2; seen2scene env, 1 GPU).
--set in_val (VALSEL-in) | out_sc (S-c-out): the trainer's fixed subset (fixed_states("exp6-V3|eval|<set>", 2048, per_scene=True)), chunks of 32
views (eval_dev.chunks), the trainer's numeric settings (TF32 on, cudnn defaults). Per chunk:
  G  (V3-T0 final.pt): noise block 0 = eval_dev.eval_samples' own generator and call (128 rows) -> must reproduce T0 eval.jsonl step 40000
     samp_<set>; blocks 1..3 = generators keyed '<eval_dev key>|kb<j>' (one 128-row call each) -> 16 nested samples (G1 = k0, G4 = k0..3)
  D0 (G-v1 final.pt): det_sample on the same 4 x V repeated batch eval_samples used (-> must reproduce D0 eval.jsonl step 40000 samp_<set>)
Pass 0 (labels only, before any model forward): quartile edges of lu (labelled rays) and of l* (GT-hit rays) -> written first.
Outputs work_dirs/exp6_v3/experiment_G/g2/prob/<set>/: edges.json, rays.npz (labelled UNKNOWN rays: view, cluster, lu, l*, label, C-edge,
G ell[16] / hit[16], D0 ell / hit / b_hat), views.npz (per-view fair ES for G1 / G4 / G16 / D0), regression.json.
Usage: python g2_prob.py --set in_val|out_sc [--max-chunks N] [--out DIR]"""
import argparse
import json
import os
import sys
import time

import numpy as np
import torch

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
import d0_head as DH  # noqa: E402
from seen2scene.futureview import geometry as G  # noqa: E402
from seen2scene.futureview import model as M  # noqa: E402
from seen2scene.v3 import data as D  # noqa: E402
from seen2scene.v3 import eval_dev as E  # noqa: E402
from seen2scene.v3 import model as V3  # noqa: E402

R = "/bsuscratch/jiahuizhang/projects/occupancy/seen2scene"
T0 = (os.path.join(R, "work_dirs/exp6_v3/task0_vae/checkpoints/final.pt"), "7ed0ddd72bb91d1e8c6563f3192201c24a49fe862b49d2387ea8770915145ab6")
D0 = (os.path.join(R, "work_dirs/exp6_v3/experiment_G/d0/checkpoints/final.pt"), "aeefe098e86517a1ff3b7455f3a83338e3972fb0f4fcd95a22607dd77d83f205")
EVAL = {"T0": os.path.join(R, "work_dirs/exp6_v3/task0_vae/runs/main/eval.jsonl"), "D0": os.path.join(R, "work_dirs/exp6_v3/experiment_G/d0/runs/main/eval.jsonl")}
DOMAIN = {"in_val": ("indoor", "valsel"), "out_sc": ("outdoor", "sc")}
GH, GW = M.GH, M.GW


def edge_c(b):
    """b [V, R] int8 GT label (-1 / 0 / +1) -> [V, R] bool: labelled ray with a labelled 4-neighbour of the opposite label (64 x 114 grid)."""
    g = b.reshape(-1, GH, GW)
    out = torch.zeros_like(g, dtype=torch.bool)
    for dy, dx in ((1, 0), (-1, 0), (0, 1), (0, -1)):
        n = torch.roll(g, shifts=(dy, dx), dims=(1, 2))
        valid = torch.ones_like(out)
        if dy == 1:
            valid[:, 0, :] = False
        if dy == -1:
            valid[:, -1, :] = False
        if dx == 1:
            valid[:, :, 0] = False
        if dx == -1:
            valid[:, :, -1] = False
        out |= valid & (g != 0) & (n != 0) & (g != n)
    return out.reshape(b.shape)


def ell_of(b, y, lu):
    """eval_dev rule: hit = (b > 0) & (y < 1); ell = s * lu with s = clamp((y + 1) / 2) on hits, 1 otherwise."""
    hit = (b > 0) & (y < 1.0)
    s = ((y + 1) / 2).clamp(0, 1)
    return torch.where(hit, s, torch.ones_like(s)) * lu, hit


def es_view(ell, ell_t, m):
    """eval_dev fair ES of one view: ell [K, R], ell_t [R], m [R] -> e1 - pair / (2K(K-1)) (pair term 0 for K = 1)."""
    rms = lambda a: torch.sqrt((a[..., m] ** 2).mean(-1))  # noqa: E731
    K = ell.shape[0]
    e1 = rms(ell - ell_t[None]).mean()
    if K == 1:
        return float(e1)
    pair = sum(rms(ell[i] - ell[j]) for i in range(K) for j in range(K) if i != j)
    return float(e1 - pair / (2 * K * (K - 1)))


def samp_metrics_like_eval_dev(x, bt, V, K):
    """The eval_dev.eval_samples accumulators of one chunk (for the bitwise regression against eval.jsonl)."""
    x = M.unpad_rays(x)
    b, y = x[:, :, 0], x[:, :, 1]
    lm = M.unpad_rays(bt["lmask"])
    cat = bt["cat"].reshape(V, -1)
    lab = (cat == G.CAT["HIT"]) & lm
    hit = (b > 0) & (y < 1.0)
    p = hit.float().mean(0)
    s = ((y + 1) / 2).clamp(0, 1)
    lu = bt["lu_vox"].reshape(V, -1)
    ell = torch.where(hit, s, torch.ones_like(s)) * lu
    s_lab = ((M.unpad_rays(bt["x1"])[:, 1] + 1) / 2).clamp(0, 1)
    ell_t = torch.where(lab, s_lab, torch.ones_like(s_lab)) * lu
    nohit = (cat == G.CAT["NOHIT"]) & lm
    acc = {"brier": [0.0, 0.0], "fw": [0.0, 0.0], "ph": [0.0, 0.0], "lh": [0.0, 0.0], "es": [0.0, 0.0]}
    for v in range(V):
        m = lm[v]
        if not m.any():
            continue
        rms = lambda a: torch.sqrt((a[..., m] ** 2).mean(-1))  # noqa: E731
        e1 = rms(ell[:, v] - ell_t[v][None]).mean()
        pair = sum(rms(ell[i, v] - ell[j, v]) for i in range(K) for j in range(K) if i != j)
        acc["es"][0] += float(e1 - pair / (2 * K * (K - 1)))
        acc["es"][1] += 1
    acc["brier"][0] += float(((p - lab.float()) ** 2)[lm].sum()); acc["brier"][1] += float(lm.sum())
    acc["fw"][0] += float(p[nohit].sum()); acc["fw"][1] += float(nohit.sum())
    acc["ph"][0] += float(p[lm].sum()); acc["ph"][1] += float(lm.sum())
    acc["lh"][0] += float(lab[lm].float().sum()); acc["lh"][1] += float(lm.sum())
    return acc


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--set", required=True, choices=tuple(DOMAIN))
    ap.add_argument("--max-chunks", type=int, default=None, help="test only")
    ap.add_argument("--out", default=None)
    a = ap.parse_args()
    od = a.out or os.path.join(R, "work_dirs/exp6_v3/experiment_G/g2/prob", a.set)
    os.makedirs(od, exist_ok=True)
    torch.backends.cuda.matmul.allow_tf32 = True               # = trainer settings (cudnn deterministic / benchmark left at defaults)
    torch.backends.cudnn.allow_tf32 = True
    dev = torch.device("cuda")
    dom, purpose = DOMAIN[a.set]
    items = [x for x in D.items(dom, purpose) if os.path.exists(D.cache_path(dom, *x))]
    data = D.V3Views(dom, items, print)
    states = data.fixed_states(f"exp6-V3|eval|{a.set}", 2048, per_scene=True)
    chs = E.chunks(states, 32)
    if a.max_chunks:
        chs = chs[:a.max_chunks]
    # ---------------------------------------------------------------- pass 0: labels only -> quartile edges (no model involved)
    lus, lts = [], []
    for ch in chs:
        for ti, t, dlt, rows in ch:
            st = data.state_tensors(ti, t, dlt, rows, dev)
            lm = M.unpad_rays(st["lmask"])
            x1 = M.unpad_rays(st["x1"])
            lu = st["lu_vox"].reshape(lm.shape)
            hitl = lm & (x1[:, 0] > 0)
            lus.append(lu[lm].cpu().numpy())
            lts.append((((x1[:, 1] + 1) / 2).clamp(0, 1) * lu)[hitl].cpu().numpy())
    lus, lts = np.concatenate(lus), np.concatenate(lts)
    edges = {"set": a.set, "A_lu_quartiles": [float(np.quantile(lus, q)) for q in (0.25, 0.5, 0.75)],
             "B_lstar_quartiles": [float(np.quantile(lts, q)) for q in (0.25, 0.5, 0.75)], "n_labelled": int(len(lus)), "n_gt_hit": int(len(lts)),
             "time": time.ctime(), "note": "labels only, computed before any model forward (protocol G2-v1 section 2)"}
    json.dump(edges, open(os.path.join(od, "edges.json"), "w"), indent=1)
    print("edges", edges, flush=True)
    del lus, lts
    # ---------------------------------------------------------------- models
    mt, _, sha_t = DH.load(T0[0], dev, "flow", expect_sha=T0[1])
    md, _, sha_d = DH.load(D0[0], dev, "d0", expect_sha=D0[1])
    K4 = 4
    cols = {k: [] for k in ("view", "cluster", "lu", "lstar", "label", "edgeC", "g_ell", "g_hit", "d_ell", "d_hit", "d_b")}
    vrows = []
    reg = {"T0": {k: [0.0, 0.0] for k in ("brier", "fw", "ph", "lh", "es")}, "D0": {k: [0.0, 0.0] for k in ("brier", "fw", "ph", "lh", "es")}}
    nv = 0
    tb = time.time()
    with torch.no_grad():
        for ci, ch in enumerate(chs):
            key = f"exp6-V3|evalsample|{a.set}|{ch[0][0]}|{ch[0][1]}"
            # ---- G: block 0 exactly as eval_samples, blocks 1..3 keyed |kb<j>
            bt = D.build_cond(mt, data, ch, dev)
            V = bt["cond"].shape[0]
            cond, mu = bt["cond"].repeat(K4, 1, 1, 1), bt["m_unk"].repeat(K4, 1, 1)
            xs = []
            for j in range(4):
                g = E.gen(dev, key if j == 0 else f"{key}|kb{j}")
                eps = torch.randn(K4, V, 2, M.GH, M.GW_PAD, generator=g, device=dev).reshape(K4 * V, 2, M.GH, M.GW_PAD)
                xs.append(V3.sample(mt, cond, mu, eps, None, 10).reshape(K4, V, 2, M.GH, M.GW_PAD))
                if j == 0:
                    acc = samp_metrics_like_eval_dev(xs[0], bt, V, K4)
                    for k in acc:
                        reg["T0"][k][0] += acc[k][0]; reg["T0"][k][1] += acc[k][1]
            xg = M.unpad_rays(torch.cat(xs))                                      # [16, V, 2, R]
            # ---- D0: the 4 x V repeated batch of eval_samples (det_sample ignores eps)
            btd = D.build_cond(md, data, ch, dev)
            xd4 = DH.det_sample(md, btd["cond"].repeat(K4, 1, 1, 1), btd["m_unk"].repeat(K4, 1, 1), None).reshape(K4, V, 2, M.GH, M.GW_PAD)
            acc = samp_metrics_like_eval_dev(xd4, btd, V, K4)
            for k in acc:
                reg["D0"][k][0] += acc[k][0]; reg["D0"][k][1] += acc[k][1]
            xd = M.unpad_rays(xd4[:1])                                            # [1, V, 2, R]
            # ---- labels (model-independent; identical in bt and btd)
            lm = M.unpad_rays(bt["lmask"])
            x1 = M.unpad_rays(bt["x1"])
            lu = bt["lu_vox"].reshape(V, -1)
            lab = lm & (x1[:, 0] > 0)
            lstar = torch.where(lab, ((x1[:, 1] + 1) / 2).clamp(0, 1), torch.ones_like(lu)) * lu
            braw = torch.where(lm, x1[:, 0], torch.zeros_like(x1[:, 0])).to(torch.int8)
            ec = edge_c(braw)
            g_ell, g_hit = ell_of(xg[:, :, 0], xg[:, :, 1], lu[None])
            d_ell, d_hit = ell_of(xd[:, :, 0], xd[:, :, 1], lu[None])
            cl = torch.as_tensor([s[0] for s in ch for _ in range(len(s[3]))], device=dev)
            for v in range(V):
                m = lm[v]
                if not m.any():
                    continue
                vrows.append([nv + v, int(cl[v]), int(m.sum()), es_view(g_ell[:1, v], lstar[v], m), es_view(g_ell[:4, v], lstar[v], m),
                              es_view(g_ell[:, v], lstar[v], m), es_view(d_ell[:, v], lstar[v], m)])
            vi = torch.arange(V, device=dev)[:, None].expand_as(lm)
            cols["view"].append((vi[lm] + nv).cpu().numpy().astype(np.int32))
            cols["cluster"].append(cl[:, None].expand_as(lm)[lm].cpu().numpy().astype(np.int32))
            cols["lu"].append(lu[lm].cpu().numpy().astype(np.float32))
            cols["lstar"].append(lstar[lm].cpu().numpy().astype(np.float32))
            cols["label"].append(lab[lm].cpu().numpy())
            cols["edgeC"].append(ec[lm].cpu().numpy())
            cols["g_ell"].append(g_ell[:, lm].T.cpu().numpy().astype(np.float32))
            cols["g_hit"].append(g_hit[:, lm].T.cpu().numpy())
            cols["d_ell"].append(d_ell[0][lm].cpu().numpy().astype(np.float32))
            cols["d_hit"].append(d_hit[0][lm].cpu().numpy())
            cols["d_b"].append(xd[0, :, 0][lm].cpu().numpy().astype(np.float32))
            nv += V
            if ci % 10 == 0:
                print(f"chunk {ci + 1}/{len(chs)} views {nv} rays {sum(len(x) for x in cols['lu'])} {time.time() - tb:.0f}s", flush=True)
    np.savez(os.path.join(od, "rays.npz"), **{k: np.concatenate(v) for k, v in cols.items()})
    np.savez(os.path.join(od, "views.npz"), rows=np.asarray(vrows, np.float64), columns=np.asarray(["view", "cluster", "n_lm", "es_G1", "es_G4", "es_G16", "es_D0"]))
    # ---------------------------------------------------------------- regression vs eval.jsonl step 40000
    out = {"set": a.set, "chunks": len(chs), "views": nv, "ckpt_sha": {"T0": sha_t, "D0": sha_d}, "seconds": time.time() - tb, "host": os.uname()[1],
           "code_sha256": DH.code_hashes()}
    for arm in ("T0", "D0"):
        ref = next(e for e in map(json.loads, open(EVAL[arm])) if e["step"] == 40000)[f"samp_{a.set}"]
        r = {k: (v[0] / v[1] if v[1] else None) for k, v in reg[arm].items()}
        mine = {"brier": r["brier"], "false_wall": r["fw"], "pred_hit_rate": r["ph"], "label_hit_rate": r["lh"], "fair_es_arclength_vox": r["es"]}
        out[f"regression_{arm}"] = {k: {"mine": mine[k], "eval_jsonl": ref[k], "equal": mine[k] == ref[k]} for k in mine}
        out[f"regression_{arm}_pass"] = all(x["equal"] for x in out[f"regression_{arm}"].values()) if not a.max_chunks else None
    json.dump(out, open(os.path.join(od, "regression.json"), "w"), indent=1, default=str)
    print(json.dumps({k: v for k, v in out.items() if k != "code_sha256"}, default=str))


if __name__ == "__main__":
    main()
