"""EXP6 V3 / ablation E -- offline replay of the planner's predictions along a finished episode and their comparison with the
true novel surface of every legal candidate (seen2scene env, 1 GPU; diagnostic only, nothing is fed back to any planner).
Replay = the closed-loop V3 planner state (as experiments/exp6_v3/H/scripts/eval_fixed.py, with the CLOSED-LOOP noise keys):
P1 fusion of mode b2 (p1_full.patch("b2")), service records 0..t integrated (h_policy.capture -> the observed S10 points),
R10 refresh (V3 encode of the D100 partial TSDF grid at t_r = 10 * (t // 10), noise sha_seed(v2_common.seed_key(scene,
start, rep, t_r, k)), K = 4), then ONE frozen score_state(keep_hits=True) over the legal candidates of step t and the
H1-H3 per-sample value h_policy.value_cov (eps_cov 1.0, cell 1.0, w(z) = min(1, (z / z0)^2)) of every candidate.
Check: the recomputed per-candidate mean value must equal the h_mean of the planner step log when the log has one (H family).
Per candidate and sample k, the predicted novel cells are split with the evaluation-side oracle of the SAME state
(e_oracle.py: the true view's novel cells; the GT evaluation points):
  hit       predicted novel cell that is also a novel cell of the true view (correctly predicted new surface)
  seen      cell that holds GT evaluation points but is not novel in the true view (surface already observed / not visible)
  empty     cell without any GT evaluation point (phantom surface, ground or other non-evaluated geometry)
and precision = |hit| / |pred|, recall = |hit| / |true novel cells|, both in cells; values are weighted as the planner does.
Strict = exact 1.0 cell identity; tolerant (primary, "_tol") = cell centres within TOL = 2.0 runtime units (2 cells: absorbs a
few % depth error at 30-60 units) of a true novel cell (hit) / of a GT evaluation point (seen vs empty). The same tolerant
test is applied to the true view itself (true_rel_share = share of the true novel value within TOL of a GT point), the
baseline for the predicted empty share. Predicted union cells (max weight over samples) of every candidate are saved to OUT.npz.
v_inbox = the part of the predicted value whose cells lie inside the scene box of the MAGICIAN settings.json (x_min..x_max
x scene_scale_factor 10; the same box that defines z0, i.e. information the planner already uses), reported per candidate
as the mean over samples; it is a DIAGNOSTIC re-scoring only (no planner uses it here).
Usage: python e_predict.py --episode EP.json --ckpt CKPT --oracle ORACLE.json [--planner-log steps.jsonl] --out OUT.jsonl"""
import argparse
import hashlib
import json
import os
import sys
import time

HERE = os.path.dirname(os.path.abspath(__file__))
X = os.path.dirname(os.path.dirname(os.path.dirname(os.path.dirname(HERE))))
os.environ.setdefault("CUBLAS_WORKSPACE_CONFIG", ":4096:8")
sys.path.insert(0, os.path.join(X, "exp5_future_view", "task1", "scripts"))
import v2_common as LC  # noqa: E402
import v2_frozen as VF  # noqa: E402,F401  (path setup of the frozen planner)
sys.path.insert(0, os.path.join(X, "exp5_future_view", "task3_full_depth_p1", "scripts"))
import p1_full as PF  # noqa: E402
PF.patch("b2")
sys.path.insert(0, os.path.join(X, "exp6_v3", "shared"))
import planner_v3 as PV3  # noqa: E402
PC = PV3.install()
sys.path.insert(0, os.path.join(X, "exp6_v3", "H", "scripts"))
import h_policy as HP  # noqa: E402
import m_p1  # noqa: E402
import m_common as MC  # noqa: E402
import t5_diagnostic as TD  # noqa: E402
import numpy as np  # noqa: E402
import torch  # noqa: E402
import yaml  # noqa: E402
from seen2scene.futureview import predictor as PR  # noqa: E402
from seen2scene.futureview import rays as RY  # noqa: E402


TOL = 2.0


def centres(keys):
    keys = np.asarray(keys, np.int64)
    m = (1 << 20) - 1
    return np.stack([(keys >> 40) & m, (keys >> 20) & m, keys & m], 1).astype(np.float64) - (1 << 19) + 0.5


def recall_tol(pred_centres, true_ids):
    """share of the true novel cells that have a predicted novel cell within TOL (nan if the true view has none)."""
    from scipy.spatial import cKDTree
    if len(true_ids) == 0:
        return np.nan
    if len(pred_centres) == 0:
        return 0.0
    return float((cKDTree(pred_centres).query(centres(true_ids), k=1, distance_upper_bound=TOL)[0] < np.inf).mean())


def cell_keys(p):
    c = np.floor(np.asarray(p, np.float64) / 1.0).astype(np.int64) + (1 << 19)
    return (c[:, 0] << 40) | (c[:, 1] << 20) | c[:, 2]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--episode", required=True)
    ap.add_argument("--ckpt", required=True)
    ap.add_argument("--oracle", required=True)
    ap.add_argument("--planner-log", default=None)
    ap.add_argument("--out", required=True)
    a = ap.parse_args()
    t0 = time.time()
    torch.use_deterministic_algorithms(True, warn_only=True)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    dev = torch.device("cuda")
    ep = json.load(open(a.episode))
    s, k, rep = ep["scene"], int(ep["start"]), int(ep["rep"])
    svc = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(a.episode))), "svc", ep["tag"])
    cfg = yaml.safe_load(open(os.path.join(X, "exp6_v3", "H", "FROZEN_CONFIG.yaml")))["policies"]["h1"]
    sha = hashlib.sha256(open(a.ckpt, "rb").read()).hexdigest()
    core = PV3.V3Core(a.ckpt, dev, expect_sha=sha)
    fr, dims, dom, _ = TD.scene_geometry(s)
    fu = m_p1.P1Fusion(fr, dims, dom, device=dev)
    o = json.load(open(a.oracle))
    sb_ = json.load(open(os.path.join("/bsuscratch/jiahuizhang/projects/occupancy/MAGICIAN/data/Macarons++", s, "settings.json")))["scene"]
    z0 = sum((b - a_) * 10.0 for a_, b in zip(sb_["x_min"], sb_["x_max"])) / 3.0 / 2.0       # = abl_common.z0_of, as the closed loop sets LC.Z0
    if s in LC.Z0:
        assert abs(z0 - LC.Z0[s]) < 1e-3, (z0, LC.Z0[s])
    oz = np.load(a.oracle.replace(".json", ".npz"))
    assert abs(o["z0"] - z0) < 1e-3, (o["z0"], z0)
    from scipy.spatial import cKDTree
    sb = json.load(open(os.path.join("/bsuscratch/jiahuizhang/projects/occupancy/MAGICIAN/data/Macarons++", s, "settings.json")))["scene"]
    box_lo, box_hi = np.asarray(sb["x_min"], float) * 10.0, np.asarray(sb["x_max"], float) * 10.0
    gt_cells = set(cell_keys(oz["gt_points"]).tolist())
    gtree = cKDTree(oz["gt_points"].astype(np.float64))
    U_ids, U_w, U_off = [], [], []
    cid, cw = oz["cell_ids"], oz["cell_w"]
    orow = {r["t"]: r for r in o["steps"]}
    logs = {}
    if a.planner_log:
        for line in open(a.planner_log):
            r = json.loads(line)
            logs[r["t"]] = r
    HP.STATE.update({"obs": [], "tree": None, "n_tree": -1, "rec": None, "visited": set()})
    K = 4
    h = eps = None
    fout = open(a.out + ".tmp", "w")
    T = len(ep["steps"]) - 1
    maxrel = 0.0
    for t in range(T):
        rec = MC.load_service_record(os.path.join(svc, f"svc_{s}_s{k}_t{t}.npz"))
        assert int(rec["t"]) == t
        HP.capture(rec)
        fu.integrate_record(rec)
        if t % 10 == 0:                                   # R10 refresh times (v2_common.refresh_time of the V2M arm)
            geo = PC.Geometry.from_fusion(fr, dims, dom, fu, dev)
            h = core.pred.encode(geo.coarse())
            eps = PR.noise(dev, K, lambda kk, tr=t: RY.sha_seed(LC.seed_key(s, k, rep, tr, kk)))
        if t not in orow or len(rec["legal_R"]) == 0:
            continue
        geo = PC.Geometry.from_fusion(fr, dims, dom, fu, dev)
        out, tm, views = PC.score_state(core, geo, h, rec["legal_R"], rec["legal_T"], eps, z0, keep_hits=True)
        sets = [[HP.value_cov(out["hits"][m]["t"][kk], v, geo, z0, cfg) for kk in range(K)] for m, v in enumerate(views)]
        mu = [float(np.mean([float(x[1].sum()) for x in row])) for row in sets]
        lg = (logs.get(t) or {}).get("all_scores") or {}
        rel = None
        if lg.get("h_mean"):
            rel = max(abs(x - y) / max(abs(y), 1e-9) for x, y in zip(mu, lg["h_mean"]))
            maxrel = max(maxrel, rel)
        cands = []
        for j, c in enumerate(orow[t]["cands"]):
            g_ids = cid[c["cell_off"]:c["cell_off"] + c["gt_cells"]]
            g_w = cw[c["cell_off"]:c["cell_off"] + c["gt_cells"]]
            gset = set(g_ids.tolist())
            gtr = cKDTree(centres(g_ids)) if len(g_ids) else None
            true_rel = (gtree.query(centres(g_ids), k=1, distance_upper_bound=TOL)[0] < np.inf) if len(g_ids) else np.zeros(0, bool)
            per = []
            for kk in range(K):
                ids, w = sets[j][kk]
                hit = np.isin(ids, g_ids)
                surf = np.fromiter((x in gt_cells for x in ids.tolist()), bool, len(ids))
                cc = centres(ids)
                hit_t = (gtr.query(cc, k=1, distance_upper_bound=TOL)[0] < np.inf) if (gtr is not None and len(ids)) else np.zeros(len(ids), bool)
                surf_t = (gtree.query(cc, k=1, distance_upper_bound=TOL)[0] < np.inf) if len(ids) else np.zeros(0, bool)
                inb = np.all((cc >= box_lo) & (cc <= box_hi), 1) if len(ids) else np.zeros(0, bool)
                per.append({"v_inbox": float(w[inb].sum()), "v_hit_tol": float(w[hit_t].sum()), "v_seen_tol": float(w[~hit_t & surf_t].sum()), "v_empty_tol": float(w[~surf_t].sum()),
                            "prec_tol": float(hit_t.mean()) if len(ids) else np.nan,
                            "recall_tol": recall_tol(cc, g_ids),
                            "v": float(w.sum()), "v_hit": float(w[hit].sum()), "v_seen": float(w[~hit & surf].sum()), "v_empty": float(w[~surf].sum()),
                            "n": int(len(ids)), "n_hit": int(hit.sum()), "prec": float(hit.mean()) if len(ids) else np.nan,
                            "recall": float(hit.sum() / len(gset)) if gset else np.nan,
                            "ray_hit_frac": float(torch.isfinite(out["hits"][j]["t"][kk]).float().mean().item())})
            m_ = lambda q: float(np.nanmean([p[q] for p in per]))  # noqa: E731
            u = np.concatenate([x[0] for x in sets[j]]) if any(len(x[0]) for x in sets[j]) else np.zeros(0, np.int64)
            uw = np.concatenate([x[1] for x in sets[j]]) if len(u) else np.zeros(0)
            if len(u):
                u, inv = np.unique(u, return_inverse=True)
                w2 = np.zeros(len(u))
                np.maximum.at(w2, inv, uw)
                uw = w2
            U_off.append((t, j, sum(len(x) for x in U_ids), len(u)))
            U_ids.append(u)
            U_w.append(uw)
            cands.append({"pred_value": mu[j], "v_inbox": m_("v_inbox"), "v_inbox_std": float(np.std([p["v_inbox"] for p in per])), "v_hit_tol": m_("v_hit_tol"), "v_seen_tol": m_("v_seen_tol"), "v_empty_tol": m_("v_empty_tol"),
                          "prec_tol": m_("prec_tol"), "recall_tol": m_("recall_tol"),
                          "true_rel_share": float(g_w[true_rel].sum() / g_w.sum()) if len(g_w) and g_w.sum() > 0 else np.nan, "v_hit": m_("v_hit"), "v_seen": m_("v_seen"), "v_empty": m_("v_empty"), "prec": m_("prec"), "recall": m_("recall"),
                          "n_pred_cells": m_("n"), "n_true_cells": int(len(gset)), "ray_hit_frac": m_("ray_hit_frac"),
                          "sample_value_std": float(np.std([p["v"] for p in per])), "true_value": c["gt_value"], "true_dcov_pp": c["dcov_pp"]})
        fout.write(json.dumps({"t": t, "executed": orow[t]["executed"], "h_mean_rel_err_vs_log": rel, "cands": cands}, default=float) + "\n")
        del geo, out, views
    fout.close()
    np.savez_compressed(a.out.replace(".jsonl", "_cells.npz"), ids=np.concatenate(U_ids) if U_ids else np.zeros(0, np.int64),
                        w=np.concatenate(U_w) if U_w else np.zeros(0), index=np.asarray(U_off, np.int64).reshape(-1, 4))
    os.replace(a.out + ".tmp", a.out)
    print(f"predict done {a.out} max rel err vs log {maxrel:.2e} {time.time() - t0:.0f}s", flush=True)


if __name__ == "__main__":
    main()
