"""Extract everything needed to visualise one planning step of V4 (seen2scene env, 1 GPU).

For one fixed state (MP3D VALSEL or Macarons++ Dev-3; same states / candidates / K = 4 noise as the V4 G1/G2 rows),
replay the executed records with the V4 planner rules (D100 P1 input, encoder refreshed at t - t % 10, sparse DistP1
band_dil) and save per legal candidate:
  pose (origin o, ray directions d), ray classes of the current map (UNKNOWN / KNOWN / EMPTY) and the known depth,
  the K = 4 predicted depth maps (decoded exactly as the planner does), the per-sample "new surface" mask under the
  sparse band_dil, the predicted DistP1 score per sample,
  GT (evaluation only): mesh depth t* on the same rays, GT-new pixels (GT point farther than the coverage eps from every
  past 10 % sample point), true one-step coverage gain;
plus the current observation (depth image of the executed frame at t), a top-down slice of the current P1 and the
current camera. The scores are checked against the stored V4 rows (bitwise). Nothing here feeds back into planning.
Usage: python viz_extract.py --set valsel|dev3 --stratum B0|B2 --sid SID --out FILE.npz"""
import argparse
import glob
import json
import os
import sys

from seen2scene import PROJECT_ROOT

REPO = str(PROJECT_ROOT)
sys.path.insert(0, os.path.join(REPO, "experiments", "exp6_v3", "shared"))
import eval_fixed as EF  # noqa: E402

X, E2F = EF.X, EF.E2F
CK = os.path.join(REPO, "work_dirs", "exp6_v4", "checkpoints", "final.pt")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--set", choices=["valsel", "dev3"], required=True)
    ap.add_argument("--stratum", choices=["B0", "B2"], default="B0")
    ap.add_argument("--sid", required=True)
    ap.add_argument("--out", required=True)
    a = ap.parse_args()
    X.setup(a.stratum)
    sys.path.insert(0, os.path.join(REPO, "experiments", "exp5_future_view", "task2", "scripts"))
    np, torch, MC, PC, PR, RY, SCO, G = X.np, X.torch, X.MC, X.PC, X.PR, X.RY, X.SCO, X.G
    from scipy.spatial import cKDTree
    torch.use_deterministic_algorithms(True, warn_only=True)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    dev = torch.device("cuda")
    core = EF.V2Core(X.PC.Core(CK, dev))
    sts = None
    for tag, ss in E2F.states(a.set, a.stratum):
        for s in ss:
            if s["sid"] == a.sid:
                sts, st = ss, s
    assert sts is not None, f"sid {a.sid} not in {a.set}/{a.stratum}"
    rp = E2F.Rep(a.set, sts[0], dev)
    rp.sset = a.set
    t = st["t"]
    recs = [MC.load_service_record(rp.svc(q)) for q in range(t + 1)]
    r = t - t % 10
    region = rp.region_of(recs[r]) if a.set == "valsel" else rp.frame

    def build(x, upto):
        fus = X.D.fusion_class(x)(*region, device=dev)
        for q in range(upto + 1):
            fus.integrate_record(recs[q])
        return fus

    # D100 pass: encoder input at the refresh step r, geometry / candidates at t (the planner rule)
    fus = build(100, r)
    geo_r = PC.Geometry.from_fusion(*region, fus, dev)
    h = core.encode_for(fus, geo_r)
    eps = PR.noise(dev, X.A.K_MAIN, lambda kk, tr=r: RY.sha_seed(rp.key(tr, kk)))
    del geo_r
    for q in range(r + 1, t + 1):
        fus.integrate_record(recs[q])
    assert fus.n_frames == t + 1
    geo = PC.Geometry.from_fusion(*region, fus, dev)
    rec = recs[t]
    out, tm, views = PC.score_state(core, geo, h, rec["legal_R"], rec["legal_T"], eps, rp.z0, keep_hits=True)
    k_M = float(geo.fr.k_M)
    coarse = geo.coarse().cpu().numpy()                                    # [3, X', Y', Z'] uint8: EMPTY, BAND, in-domain
    # sparse pass: DistP1 band_dil of the planner (B2 mode)
    fus_s = build(10, t)
    geo_s = PC.Geometry.from_fusion(*region, fus_s, dev)
    hits = [hh["t"].detach() for hh in out["hits"]]
    scores = X.distp1_k(views, [x.cpu() for x in hits], geo_s, k_M, rp.z0)
    ref = None
    for f in glob.glob(os.path.join(REPO, "work_dirs", "exp6_v4", "eval_fixed", f"{a.set}_s_final_v4", a.stratum, "*.jsonl")):
        for line in open(f):
            row = json.loads(line)
            if row["sid"] == a.sid:
                ref = row["scores"]["V4"]
    assert ref is not None and scores == ref, "scores must reproduce the stored V4 G1/G2 row bitwise"
    newpred = []
    for v, hh in zip(views, hits):
        newpred.append(torch.stack([SCO.new_mask_from_depth(hh[k], v["o"], v["d"], geo_s.dims, geo_s.domain, geo_s.caster.band_dil)[0] for k in range(hh.shape[0])]).cpu().numpy())
    # evaluation side (GT): oracle of the state
    dname = "indoor" if a.set == "valsel" else "outdoor"
    root = os.path.join(E2F.EP.W, "valsel") if a.set == "valsel" else (X.ORIG_W if a.stratum == "B0" else X.P.ROOT_B2)
    orc = np.load(os.path.join(root, "states", dname, a.sid, "oracle.npz"))
    pts = np.concatenate([np.asarray(q["sample_points"], np.float64) for q in recs])
    tree = cKDTree(pts)
    ceps = 0.08 * float(orc["k_M"]) if dname == "indoor" else float(orc["eps"])
    uu, vv = MC.scoring_pixel_grid()
    gt_new, gt_ok, pred_new_eps = [], [], []
    for j in range(len(views)):
        ts = orc["tstar"][j].astype(np.float64)
        d_y, dz, c = MC.pixel_rays(np.asarray(rec["legal_R"][j], np.float64), np.asarray(rec["legal_T"][j], np.float64), uu, vv)
        ok = np.isfinite(ts) & (ts * k_M * dz < MC.SENSOR_RANGE)
        p = c[None] + d_y * np.where(ok, ts * k_M, 0.0)[:, None]
        dd, _ = tree.query(p[ok], k=1, distance_upper_bound=ceps)
        new = np.zeros(len(ts), bool)
        new[np.flatnonzero(ok)] = ~np.isfinite(dd)
        gt_new.append(new); gt_ok.append(ok)
        # the SAME novelty rule for the predictions (evaluation only): a predicted hit is "new" if it is farther than the
        # coverage eps from every past 10 % sample point (apples-to-apples with gt_new; DistP1 itself uses band_dil)
        tp = hits[j].cpu().numpy().astype(np.float64)                       # [K, R] model ray length
        pn = np.zeros(tp.shape, bool)
        for k in range(tp.shape[0]):
            okp = np.isfinite(tp[k]) & (tp[k] * k_M * dz < MC.SENSOR_RANGE)
            pp = c[None] + d_y * np.where(okp, tp[k] * k_M, 0.0)[:, None]
            ddp, _ = tree.query(pp[okp], k=1, distance_upper_bound=ceps)
            pn[k, np.flatnonzero(okp)] = ~np.isfinite(ddp)
        pred_new_eps.append(pn)
    cur_c = geo.caster.f.runtime_to_model(np.asarray(rec["cam_center"], np.float64).reshape(1, 3))[0]
    # current camera forward direction (z-up model frame) from the executed pose's central ray
    from seen2scene.futureview import rays as RYS
    cu, cv = RYS.grid_uv((0.0, 0.0))
    dcy, _, _ = MC.pixel_rays(np.asarray(rec["R"], np.float64), np.asarray(rec["T"], np.float64), cu, cv)
    cur_dir = MC.to_z_up(dcy).mean(0)
    save = {
        "set": a.set, "stratum": a.stratum, "sid": a.sid, "scene": st["scene"], "start": st["start"], "t": t, "k_M": k_M,
        "cand": np.asarray(st["cand"], int), "legal_idx": np.asarray(rec["legal_idx"]),
        "o": np.stack([v["o"].cpu().numpy() for v in views]), "d": np.stack([v["d"].cpu().numpy() for v in views]),
        "dz": np.stack([v["dz"].cpu().numpy() for v in views]), "tcap": np.stack([v["tcap"].cpu().numpy() for v in views]),
        "cls": np.stack([v["pr"]["cls"].cpu().numpy() for v in views]), "t_known": np.stack([v["pr"]["t_known"].cpu().numpy() for v in views]),
        "t_pred": np.stack([x.cpu().numpy() for x in hits]), "new_pred": np.stack(newpred), "scores": np.asarray(scores, float),
        "tstar": orc["tstar"].astype(np.float32), "gain": np.asarray(orc["gain"], float), "gt_new": np.stack(gt_new), "gt_ok": np.stack(gt_ok), "pred_new_eps": np.stack(pred_new_eps),
        "coarse": coarse, "block_m": 8 * 0.011, "cur_center": cur_c, "cur_dir": cur_dir, "cur_zbuf": np.asarray(rec["zbuf"], np.float32), "cur_mask": np.asarray(rec["mask"], bool),
        "cls_codes": np.asarray([G.CLS_UNKNOWN, G.CLS_KNOWN, G.CLS_EMPTY]), "z0": float(rp.z0), "eps_cov": ceps,
        "ckpt_sha": core.ckpt_sha, "svc_src": (st["sources"][0] if a.set == "valsel" else next(e for e in st["sources"] if e["source"] == "V2M"))["svc"]}
    os.makedirs(os.path.dirname(a.out), exist_ok=True)
    np.savez_compressed(a.out, **{k: (np.asarray(v) if not isinstance(v, str) else np.asarray(v)) for k, v in save.items()})
    print("saved", a.out, "candidates", len(views), "scores reproduce stored V4 row: True")


if __name__ == "__main__":
    main()
