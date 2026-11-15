"""EXP9-SCONE Phase C -- smoke checks of one short SCONE episode (magician env, 1 GPU; EVALUATION SIDE ONLY, run after the
closed loop has finished; nothing here is visible to the planner).

out  (Macarons++): 1. official-coverage replay = task2/scripts/t25_out_cov.py run unchanged (the stored curve must be reproduced
                      bitwise; monotone 10 % coverage alongside), replay frames / metrics under --work;
                   2. legality: a fresh frozen EpisodeSim replays the executed keys; at every step the simulator legal list must
                      equal the one logged by the runner and the executed key must be in it;
                   3. oracle check of the candidate scoring at --eval-steps: the true one-step coverage gain of every legal
                      candidate (exp3b m_gain_eval.evaluate_actions: snapshot -> execute -> shared seeds of t+1 -> restore)
                      vs SCONE's logged predicted gain (Spearman, Top-1 hit, gain of the pick / best / mean);
                   4. occupancy sanity on the saved occupancy snapshots: proxy points with a SCONE prediction, split by their
                      distance to the GT surface sample (near < --near, far > --far runtime units): mean occupancy, AUROC.
in   (MP3D):       1. C_obs replay from the stored 10 % samples (collect_mp3d150.validate logic, any number of moves);
                   2. legality (executed[t+1] = legal[pick] of svc_t), planner guard log empty;
                   3. start pose = MAG-B / OURS start of the same level / start;
                   4. oracle candidate check (frozen mp3d_sim sensor renders each legal candidate of svc_t; sample seed of t+1;
                      C_obs gain), Spearman vs SCONE scores;
                   5. occupancy sanity as above, GT = the level's frozen evaluator points.
Usage: python scone_smoke_check.py out --scene S --start K --root ROOT --work W --eval-steps 1 5 10 15
       python scone_smoke_check.py in --scan S --level L --start K --ep-dir D --eval-steps 1 5 10 15 --out F
"""
import argparse
import glob
import json
import os
import sys

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
REPO = os.path.dirname(os.path.dirname(os.path.dirname(os.path.dirname(HERE))))
EXP = os.path.join(REPO, "experiments")


def auroc(score, label):
    from scipy.stats import rankdata
    score, label = np.asarray(score, np.float64), np.asarray(label, bool)
    n1, n0 = int(label.sum()), int((~label).sum())
    if n1 == 0 or n0 == 0:
        return None
    r = rankdata(score)
    return float((r[label].sum() - n1 * (n1 + 1) / 2) / (n1 * n0))


def occ_sanity(npz_files, G, near, far):
    from scipy.spatial import cKDTree
    tree, out = cKDTree(G), []
    for f in sorted(npz_files, key=lambda x: int(x.rsplit("_t", 1)[1].split(".")[0])):
        z = np.load(f)
        n = int(z["n_pred"])
        X, o = z["X"][:n].astype(np.float64), z["occ"][:n].astype(np.float64)
        if n == 0:
            out.append({"file": os.path.basename(f), "n_pred": 0})
            continue
        d, _ = tree.query(X, k=1, workers=16)
        nr, fr = d < near, d > far
        out.append({"file": os.path.basename(f), "n_pred": n, "n_out_of_field_default": int(len(z["occ"]) - n),
                    "occ_default_value": float(np.unique(z["occ"][n:])[0]) if len(z["occ"]) > n and len(np.unique(z["occ"][n:])) == 1 else None,
                    "occ_mean": float(o.mean()), "frac_occ_gt_0.5": float((o > 0.5).mean()),
                    "n_near": int(nr.sum()), "n_far": int(fr.sum()),
                    "occ_mean_near": float(o[nr].mean()) if nr.any() else None,
                    "occ_mean_far": float(o[fr].mean()) if fr.any() else None,
                    "auroc_near_vs_far": auroc(o[nr | fr], nr[nr | fr]),
                    "dist_median": float(np.median(d))})
    return out


def rank_check(pred, true):
    from scipy.stats import spearmanr
    p = np.asarray([np.nan if x is None else x for x in pred], np.float64)
    t = np.asarray(true, np.float64)
    ok = np.isfinite(p)
    res = {"n": int(ok.sum()), "pred": p.tolist(), "true": t.tolist()}
    if ok.sum() >= 3 and np.ptp(p[ok]) > 0 and np.ptp(t[ok]) > 0:
        res["spearman"] = float(spearmanr(p[ok], t[ok]).correlation)
    pick = int(np.nanargmax(np.where(ok, p, -np.inf))) if ok.any() else None
    if pick is not None:
        res.update({"pick": pick, "true_pick": float(t[pick]), "true_best": float(t[ok].max()), "true_mean": float(t[ok].mean()),
                    "top1_hit": bool(t[pick] >= t[ok].max() - 1e-12)})
    return res


def check_out(a):
    sys.path.insert(0, os.path.join(EXP, "exp5_future_view", "task2", "scripts"))
    import t25_out_cov as T25
    SIM, GE = T25.T6S, None
    sys.path.insert(0, os.path.join(EXP, "exp3b_macaronspp_transfer"))
    import m_gain_eval as GE  # noqa: F811
    import m_t6_common as T6
    tag = f"scone_{a.scene}_s{a.start}_r0"
    ep = json.load(open(os.path.join(a.root, "episodes", f"ep_{tag}.json")))
    res = {"tag": tag, "moves": ep["moves"], "coverage": ep["coverage"]}
    # 1. official-coverage replay (frozen T25, own work dirs)
    T25.C.W2 = a.work
    argv = sys.argv
    sys.argv = [argv[0], "--group", "scone", "--tag", tag, "--root", a.root, "--out", os.path.join(a.work, "metrics")]
    T25.main()
    sys.argv = argv
    mt = json.load(open(os.path.join(a.work, "metrics", "scone", f"{tag}.json")))
    res["replay"] = {k: mt[k] for k in ("official_equal_stored", "auc_official", "auc_mono", "cov_official_final", "cov_mono_final",
                                        "official_max_drop", "eps", "vis", "n_gt")}
    res["cov_mono"] = mt["cov_mono"]
    # 2. + 3. legality and oracle (fresh EpisodeSim)
    torch = SIM.torch
    sim = SIM.EpisodeSim(a.scene, a.start, os.path.join(a.work, "oracle_frames", tag))
    keys, steps = ep["executed_keys"], ep["steps"]
    legal_ok, cov_ok, oracle = True, True, {}
    for t in range(len(keys)):
        assert sim.key() == keys[t], (t, sim.key(), keys[t])
        ob = sim.observe(t)
        cov_ok &= ob["cov"] == ep["coverage"][t]
        if t + 1 < len(keys):
            legal, _ = sim.legal()
            lk = [T6.key_of(x[0]) for x in legal]
            legal_ok &= lk == steps[t]["legal_keys"] and keys[t + 1] in lk and steps[t]["action_key"] == keys[t + 1]
            if t in a.eval_steps and lk:
                g, sim.covered = GE.evaluate_actions(sim.camera, sim.mesh, sim.params, sim.dev, a.scene, a.start, t, sim.covered,
                                                     sim.gt_scene, sim.eps, sim.vis, ob["cov"], lk, sim.camera.get_idx_from_key)
                oracle[t] = rank_check(steps[t]["pred_gains"], [g[k] for k in lk])
                oracle[t]["true_unit"] = "percentage points of official coverage"
            sim.move(keys[t + 1])
    res["legality_replay_ok"], res["coverage_rereplay_equal"], res["oracle"] = bool(legal_ok), bool(cov_ok), oracle
    import csv
    c, others = np.asarray(ep["coverage"], np.float64), {}
    for r in csv.DictReader(open(os.path.join(EXP, "cvpr_results", "manifest", "episodes.csv"))):
        if r["scene_id"] == a.scene and int(r["start_id"]) == a.start and r["run_status"] == "COMPLETE" and \
                r["method_id"] in ("RANDOM", "MACARONS", "MAG-B", "V3-T0-D1"):
            c2 = np.asarray([float(r[f"c{t}"]) for t in range(101)])
            others[r["method_id"]] = {"same_c0": bool(c2[0] == c[0]),
                                      "auc": {f"AUC@{T}": float(np.trapz(c2[:T + 1]) / T) for T in (5, 10, 20, 50, 100) if T <= ep["moves"]},
                                      "cov_at": {f"Cov@{T}": float(c2[T]) for T in (5, 10, 20, 50, 100) if T <= ep["moves"]}}
    res["self"] = {"auc": {f"AUC@{T}": float(np.trapz(c[:T + 1]) / T) for T in (5, 10, 20, 50, 100) if T <= ep["moves"]},
                   "cov_at": {f"Cov@{T}": float(c[T]) for T in (5, 10, 20, 50, 100) if T <= ep["moves"]},
                   "decision_s_mean": float(np.mean([x["decision_s"] for x in steps[:-1]])), "peak_mem_gb": ep["peak_mem_gb"],
                   "n_hover": ep["n_hover"], "n_default_used": ep["n_default_used"], "info_flow": ep["info_flow"],
                   "occ_per_step": [x.get("occ") for x in steps[:-1]], "model": ep["model"]}
    res["other_methods"] = others
    # 4. occupancy sanity
    cells = [c for c in sim.gt_scene.cells if len(sim.gt_scene.cells[c].cell_pts) > 0]
    G = torch.cat([sim.gt_scene.cells[c].cell_pts for c in cells]).double().cpu().numpy()
    res["occupancy"] = occ_sanity(glob.glob(os.path.join(a.diag_dir or os.path.join(a.root, "diag"), f"{tag}_occ_t*.npz")), G,
                                  a.near, a.far)
    import shutil
    shutil.rmtree(os.path.join(a.work, "oracle_frames"), ignore_errors=True)
    out = a.out or os.path.join(a.work, f"smoke_check_{tag}.json")
    json.dump(res, open(out, "w"), indent=1)
    print(json.dumps({k: v for k, v in res.items() if k not in ("coverage", "cov_mono")}, indent=1)[:6000], flush=True)


def check_in(a):
    sys.path.insert(0, os.path.join(EXP, "exp5_future_view", "task0", "scripts"))
    import mp3d_sim as S
    import torch
    from scipy.spatial import cKDTree
    L = S.Level(a.scan, a.level)
    tag = f"{L.key}_S{a.start}"
    ep = json.load(open(os.path.join(a.ep_dir, "episode.json")))
    ans = [json.loads(x) for x in open(os.path.join(a.ep_dir, "planner.jsonl"))]
    info = json.load(open(os.path.join(a.ep_dir, "planner_info.json")))
    moves, c = ep["moves"], np.asarray(ep["coverage"], np.float64)
    ex = [tuple(int(x) for x in p) for p in ep["executed"]]
    z = np.load(os.path.join(S.W, "preprocessing", "coverage", f"{L.key}.npz"))
    pts, vis, eps = z["pts"], z["visible"], float(z["eps_rt"])
    tree = cKDTree(pts)
    cov = np.zeros(len(pts), bool)
    rep, legal_ok, svc = [], True, []
    for t in range(moves + 1):
        s = np.load(os.path.join(a.ep_dir, "svc", f"svc_t{t}.npz"))
        svc.append(s)
        sp = np.asarray(s["sample_points"], np.float64)
        if len(sp):
            dd, _ = cKDTree(sp).query(pts, k=1, distance_upper_bound=eps, workers=8)
            cov |= np.isfinite(dd)
        rep.append(float((cov & vis).sum() / max(1, vis.sum())))
        if tuple(int(x) for x in s["pose_idx"]) != ex[t]:
            legal_ok = False
        if t < moves:
            legal = [tuple(int(x) for x in r) for r in s["legal_idx"]]
            pk = ans[t]["pick"]
            legal_ok &= pk is not None and legal[pk] == ex[t + 1]
    res = {"tag": tag, "moves": moves, "coverage": c.tolist(), "replay_max_abs_diff": float(np.max(np.abs(np.asarray(rep) - c))),
           "monotone": bool(np.all(np.diff(c) >= -1e-15)), "legality_ok": bool(legal_ok),
           "guard_denied": sorted({p for x in ans for p in x.get("guard_denied", [])}),
           "fallbacks": int(sum(1 for x in ans if x.get("fallback"))), "model": info.get("model"),
           "auc": {f"AUC@{T}": float(np.trapz(c[:T + 1]) / T) for T in (5, 10, 20, 50, 100, 150) if T <= moves},
           "cov_at": {f"Cov@{T}": float(c[T]) for T in (5, 10, 20, 50, 100, 150) if T <= moves},
           "decision_s_mean": float(np.mean([x["timing"]["decision_s"] for x in ans])),
           "peak_mem_gb_max": float(max(x.get("peak_mem_gb", 0) for x in ans))}
    # 3. same start as the other methods of EXP7
    import csv
    others = {}
    for r in csv.DictReader(open(os.path.join(EXP, "cvpr_results", "exp7", "manifests", "mp3d150_manifest.csv"))):
        p = os.path.join(REPO, r["output_path"])
        if r["scene_id"] == L.key and int(r["start_id"]) == int(a.start) and os.path.exists(p):
            e2 = json.load(open(p))
            c2 = np.asarray(e2["coverage"], np.float64)
            others[r["method"]] = {"same_start": [int(x) for x in e2["executed"][0]] == list(ex[0]),
                                   "same_first_record_cov": bool(c2[0] == c[0]),
                                   "auc": {f"AUC@{T}": float(np.trapz(c2[:T + 1]) / T) for T in (5, 10, 20) if T <= moves},
                                   "cov_at": {f"Cov@{T}": float(c2[T]) for T in (5, 10, 20) if T <= moves}}
    res["other_methods"] = others
    # 4. oracle candidate check
    v, f = S.mesh_zup(a.scan)
    sen = S.Sensor(S.zup_m_to_rt(v), f, torch.device("cuda"))
    oracle = {}
    for t in a.eval_steps:
        if t >= moves:
            continue
        ct = np.zeros(len(pts), bool)
        for u in range(t + 1):
            sp = np.asarray(svc[u]["sample_points"], np.float64)
            if len(sp):
                dd, _ = cKDTree(sp).query(pts, k=1, distance_upper_bound=eps, workers=8)
                ct |= np.isfinite(dd)
        base = float((ct & vis).sum() / vis.sum())
        true = []
        for i in range(len(svc[t]["legal_idx"])):
            R = torch.as_tensor(svc[t]["legal_R"][i], dtype=torch.float32, device=sen.dev).view(1, 3, 3)
            T = torch.as_tensor(svc[t]["legal_T"][i], dtype=torch.float32, device=sen.dev).view(1, 3)
            fr = sen.render(R, T)
            p_, *_ = sen.sample(fr, S.derived_seed(L.key, a.start, t + 1, "sample"))
            p_ = p_.cpu().numpy().astype(np.float64)
            new = ct.copy()
            if len(p_):
                dd, _ = cKDTree(p_).query(pts, k=1, distance_upper_bound=eps, workers=8)
                new |= np.isfinite(dd)
            true.append(100.0 * (float((new & vis).sum() / vis.sum()) - base))
        oracle[t] = rank_check(ans[t]["scores"], true)
        oracle[t]["true_unit"] = "percentage points of C_obs"
    res["oracle"] = oracle
    # 5. occupancy sanity
    res["occupancy"] = occ_sanity(glob.glob(os.path.join(a.ep_dir, "diag", f"{tag}_occ_t*.npz")), pts, a.near, a.far)
    json.dump(res, open(a.out, "w"), indent=1)
    print(json.dumps({k: v for k, v in res.items() if k not in ("coverage",)}, indent=1)[:6000], flush=True)


def main():
    ap = argparse.ArgumentParser()
    sp = ap.add_subparsers(dest="mode", required=True)
    o = sp.add_parser("out")
    o.add_argument("--scene", required=True)
    o.add_argument("--start", type=int, required=True)
    o.add_argument("--root", required=True)
    o.add_argument("--work", required=True)
    o.add_argument("--diag-dir", default=None)
    o.add_argument("--eval-steps", type=int, nargs="*", default=[1, 5, 10, 15])
    o.add_argument("--near", type=float, default=2.0)
    o.add_argument("--far", type=float, default=10.0)
    o.add_argument("--out", default=None)
    i = sp.add_parser("in")
    i.add_argument("--scan", required=True)
    i.add_argument("--level", type=int, required=True)
    i.add_argument("--start", type=int, required=True)
    i.add_argument("--ep-dir", required=True)
    i.add_argument("--eval-steps", type=int, nargs="*", default=[1, 5, 10, 15])
    i.add_argument("--near", type=float, default=1.0)
    i.add_argument("--far", type=float, default=3.5)
    i.add_argument("--out", required=True)
    a = ap.parse_args()
    check_out(a) if a.mode == "out" else check_in(a)


if __name__ == "__main__":
    main()
