"""Replay a stored V4 closed-loop episode step by step with the FROZEN planner code and record, per step, what the
planner imagined (seen2scene env, 1 GPU). Nothing is re-planned: the executed records are read from the episode's
service records, the planner's own scores and choice are recomputed and asserted equal to the stored planner log.

  outdoor  task1/scripts/v2_planner.py loop (P1 = P1FusionB2 under p1_full.patch("b2"), R10 encoder refresh, noise
           LC.seed_key(scene, start, 0, t_r, k), DistP1 K = 4, m_score.choose)
  indoor   task0/scripts/t06_planner.Planner("V2K4") under the same patch + prefer-unvisited choice (v2nv nv_choice)
v2_plan_core.score_state is wrapped to keep the decoded hits (scores unchanged). Per step saved: all legal candidates'
runtime centres / forward directions and DistP1 scores (mean and per sample), the choice, and for the top-5 candidates
by score (the choice always included) the K = 4 predicted depth maps (runtime depth on the 64 x 114 scoring grid) and the
ray classes of the current map; for the chosen candidate also its predicted surface points (runtime frame).
Usage: python anim_replay.py --domain outdoor --scene S --start K --out FILE.npz
       python anim_replay.py --domain indoor --key SCAN_Lx --start K --out FILE.npz"""
import argparse
import json
import os
import sys

R = "/bsuscratch/jiahuizhang/projects/occupancy/seen2scene"
EXP5 = os.path.join(R, "experiments", "exp5_future_view")
CK = os.path.join(R, "work_dirs", "exp6_v4", "checkpoints", "final.pt")
W4 = os.path.join(R, "work_dirs", "exp6_v4", "eval", "closed_loop")
TOPN = 5


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--domain", choices=["outdoor", "indoor"], required=True)
    ap.add_argument("--scene", default=None)
    ap.add_argument("--key", default=None)
    ap.add_argument("--start", type=int, required=True)
    ap.add_argument("--out", required=True)
    a = ap.parse_args()
    sys.path.insert(0, os.path.join(EXP5, "task2", "scripts"))
    sys.path.insert(0, os.path.join(EXP5, "task1", "scripts"))
    if a.domain == "outdoor":
        import t24_v2_out_planner as P24          # registers the V2M arm (R10, K 4, distp1)
        LC, VF = P24.LC, P24.VF
    else:
        import t23_v2_in_planner as P23           # t06 planner paths, nv_choice
        TP, NV, VF = P23.TP, P23.NV, P23.VF
    sys.path.insert(0, os.path.join(EXP5, "task3_full_depth_p1", "scripts"))
    import p1_full as PF
    PF.patch("b2")
    import numpy as np
    import torch
    import m_common as MC
    import v2_plan_core as PC
    from seen2scene.futureview import predictor as PR
    from seen2scene.futureview import rays as RY
    torch.use_deterministic_algorithms(True, warn_only=True)
    dev = torch.device("cuda")
    sha = VF.sha256_file(CK)
    CAP = {}
    orig = PC.score_state

    def wrapped(*args, **kw):
        kw["keep_hits"] = True
        out, tm, views = orig(*args, **kw)
        CAP["hits"], CAP["views"], CAP["k_M"] = out.pop("hits"), views, float(args[1].fr.k_M)
        return out, tm, views
    PC.score_state = wrapped
    uu, vv = MC.scoring_pixel_grid()

    def fwd(Rm, Tm):
        d_y, _, _ = MC.pixel_rays(np.asarray(Rm, np.float64), np.asarray(Tm, np.float64), uu, vv)
        f = d_y.mean(0)
        return f / np.linalg.norm(f)

    GH, GW = 64, 114
    rng = np.random.default_rng(0)
    rows = []

    def pack(t, rec, scores, pick):
        """Turn one decision into compact arrays right away (GPU tensors freed)."""
        views, hits, k_M = CAP["views"], CAP["hits"], CAP["k_M"]
        n = len(scores)
        order = list(np.argsort(-np.asarray(scores), kind="stable")[:TOPN])
        if pick not in order:
            order[-1] = pick
        depth = np.full((TOPN, 4, GH, GW), np.nan, np.float16)
        cls = np.full((TOPN, GH, GW), 255, np.uint8)
        for q, j in enumerate(order):
            tt = hits[j]["t"].detach().cpu().numpy().astype(np.float64)                      # [K, R] model ray length
            dz = views[j]["dz"].detach().cpu().numpy()
            depth[q] = np.where(np.isfinite(tt), tt * k_M * dz[None], np.nan).reshape(4, GH, GW).astype(np.float16)
            cls[q] = views[j]["pr"]["cls"].detach().cpu().numpy().reshape(GH, GW).astype(np.uint8)
        tt = hits[pick]["t"].detach().cpu().numpy().astype(np.float64)
        unk = (views[pick]["pr"]["cls"] == 0).detach().cpu().numpy()
        d_y, _, c = MC.pixel_rays(np.asarray(rec["legal_R"][pick], np.float64), np.asarray(rec["legal_T"][pick], np.float64), uu, vv)
        P = [c[None] + d_y[unk & np.isfinite(tt[kk])] * (tt[kk, unk & np.isfinite(tt[kk])] * k_M)[:, None] for kk in range(tt.shape[0])]
        P = np.concatenate(P) if P else np.zeros((0, 3))
        if len(P) > 3000:
            P = P[rng.choice(len(P), 3000, replace=False)]
        pts = np.full((3000, 3), np.nan, np.float32)
        pts[:len(P)] = P
        rows.append({"t": t, "n": n, "scores": np.asarray(scores, np.float64), "pick": int(pick), "order": order, "depth": depth, "cls": cls, "pts": pts,
                     "cam_center": np.asarray(rec["cam_center"], np.float32), "cam_fwd": fwd(rec["R"], rec["T"]).astype(np.float32),
                     "cand_center": np.asarray(rec["legal_center"], np.float32), "cand_fwd": np.stack([fwd(rec["legal_R"][j], rec["legal_T"][j]) for j in range(n)]).astype(np.float32)})
        CAP.clear()

    if a.domain == "outdoor":
        import m_score as MS
        import t5_diagnostic as TD
        import abl_common as AC
        import m_p1
        s, k = a.scene, a.start
        tag = f"V2M_{s}_s{k}_r0"
        root = os.path.join(W4, "outdoor", "V4", "V2M")
        svc_dir = os.path.join(root, "svc", tag)
        log = json.load(open(os.path.join(root, "planner_logs", f"pl_{tag}.json")))["steps"]
        ep = json.load(open(os.path.join(root, "episodes", f"ep_{tag}.json")))
        core = PC.Core(CK, dev, expect_sha=sha)
        fr, dims, dom, win = TD.scene_geometry(s)
        fu = m_p1.P1Fusion(fr, dims, dom, device=dev)
        z0 = AC.z0_of(s)
        h = eps = None
        for t in range(len(log)):
            rec = MC.load_service_record(os.path.join(svc_dir, f"svc_{s}_s{k}_t{t}.npz"))
            fu.integrate_record(rec)
            geo = PC.Geometry.from_fusion(fr, dims, dom, fu, dev)
            if t % 10 == 0:
                h = core.pred.encode(geo.coarse())
                eps = PR.noise(dev, 4, lambda kk, tt=t: RY.sha_seed(LC.seed_key(s, k, 0, tt, kk)))
            out, tm, views = PC.score_state(core, geo, h, rec["legal_R"], rec["legal_T"], eps, z0)
            scores = out["distp1"]
            pick = MS.choose(scores, s, k, t, True)["pick"]
            assert scores == log[t]["scores"] and pick == log[t]["choice"]["pick"], ("replay differs from the planner log", t)
            pack(t, rec, scores, pick)
            del geo, out, views
        meta = {"domain": "outdoor", "scene": s, "start": k, "tag": tag, "coverage_official": ep["coverage"], "moves": ep["moves"], "executed_keys": ep["executed_keys"]}
    else:
        key, k = a.key, a.start
        tag = f"{key}_S{k}"
        root = os.path.join(W4, "indoor", "V4", "V2M", tag)
        log = [json.loads(x) for x in open(os.path.join(root, "planner.jsonl"))]
        ep = json.load(open(os.path.join(root, "episode.json")))
        fv0 = VF.frozen_values()
        VF.frozen_values = lambda: dict(fv0, checkpoint={"path": os.path.relpath(CK, VF.REPO), "sha256": sha})
        pl = TP.Planner("V2K4", key, tag, dev)
        for t in range(len(log)):
            rec = TP.MC.load_service_record(os.path.join(root, "svc", f"svc_t{t}.npz"))
            ans = pl.step(t, rec)
            pick = NV.nv_choice(ans["scores"], rec, f"T2|V2M|{tag}", t)[0]
            assert ans["scores"] == log[t]["scores"] and pick == log[t]["pick"], ("replay differs from the planner log", t)
            pack(t, rec, ans["scores"], pick)
        meta = {"domain": "indoor", "scene": key, "start": k, "tag": tag, "coverage_official": ep["coverage"], "moves": ep["moves"], "executed": ep["executed"]}
    Tn, n_max = len(rows), max(r["n"] for r in rows)
    out = {"meta": json.dumps(meta), "ckpt_sha": sha, "t": np.array([r["t"] for r in rows], np.int16), "n_legal": np.array([r["n"] for r in rows], np.int16),
           "pick": np.array([r["pick"] for r in rows], np.int16), "top": np.array([r["order"] + [-1] * (TOPN - len(r["order"])) for r in rows], np.int16),
           "top_depth": np.stack([r["depth"] for r in rows]), "top_cls": np.stack([r["cls"] for r in rows]), "pick_pts": np.stack([r["pts"] for r in rows]),
           "cam_center": np.stack([r["cam_center"] for r in rows]), "cam_fwd": np.stack([r["cam_fwd"] for r in rows])}
    for nm, shp in (("scores", ()), ("cand_center", (3,)), ("cand_fwd", (3,))):
        arr = np.full((Tn, n_max) + shp, np.nan, np.float32)
        for i, r in enumerate(rows):
            arr[i, :r["n"]] = r[nm if nm != "scores" else "scores"]
        out[nm] = arr
    os.makedirs(os.path.dirname(a.out), exist_ok=True)
    np.savez_compressed(a.out, **out)
    print("saved", a.out, "steps", Tn, "all scores and picks equal to the planner log")


if __name__ == "__main__":
    main()
