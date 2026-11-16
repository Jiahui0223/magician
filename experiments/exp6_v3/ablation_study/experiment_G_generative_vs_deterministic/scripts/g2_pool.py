"""Experiment G2 items 1, 2, 4 on the 755-state Experiment F pool (protocol G2-v1; seen2scene env, 1 GPU per process; offline, GT only to
score). Replay = the UNCHANGED CT / U / G chain (ct_fixed.run_outdoor -> f_run -> mv_core; same replayed P1 state, R10 features, simulator
legal list); only ct_fixed.outdoor_state is replaced (monkeypatch, as g_fixed.py). No planner score is computed: per state every first-layer
legal candidate is projected (v2_plan_core.Geometry.view) and predicted
  G   V3-T0 final.pt, V3Predictor: noise maps k = 0..3 = the replay chain's eps4 (checked bitwise; one call, = the planner's G4 batch) and
      k = 4..15 = seed_key(scene, start, 0, t_r, k) (one more call) -> 16 nested samples (G1 = k0, G4 = k0..3)
  D0  D0 final.pt, DetPredictor (one forward)
Labels (offline): t* = FX.gt_first_hit of the view's runtime rays (as g_fixed / CT), geometry.encode_labels on the state's own map; labelled
UNKNOWN rays lmask = (n_u > 0) & (b != 0); lu = 0.5 n_u voxels; l* / ell by the eval_dev rule. Strata flags: C-edge (opposite labelled 4-neighbour),
D-edge (GT-hit 4-neighbour with relative z* jump > 0.10). Figure arrays (item 4): per state the first-layer view with the largest number of
labelled rays in C-edge u D-edge (ties: first in the legal list), with GT z*, labels, D0 / G k0..3 decoded runtime z-depth and back-projected
runtime points (float16).
Output work_dirs/exp6_v3/experiment_G/g2/pool/<tag>.npz (rays), <tag>_views.json, <tag>_fig.npz; claims; resumable.
Usage: python g2_pool.py --set confirm|explore [--only TAG] [--tmax T] [--out DIR]"""
import argparse
import json
import os
import sys
import time

R = "/bsuscratch/jiahuizhang/projects/occupancy/seen2scene"
HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.join(R, "experiments", "exp6_v3", "ablation_study", "experiment_CT_continued_training", "scripts"))
import ct_fixed as CT  # noqa: E402

sys.path.insert(0, HERE)
import d0_head as DH  # noqa: E402
import g2_prob as GP  # noqa: E402  (edge_c, ell_of, es_view)

C, FR, torch, np, G = CT.C, CT.FR, CT.torch, CT.np, CT.G
FX, MC = C.FX, C.MC
T0 = GP.T0
D0 = GP.D0
ACC = {}


def edge_d(zs, lab):
    """zs [R] runtime GT z-depth (inf = none), lab [R] GT hit (labelled) -> D-edge [R] bool."""
    z = torch.where(lab, zs, torch.full_like(zs, float("nan"))).reshape(GP.GH, GP.GW)
    out = torch.zeros_like(z, dtype=torch.bool)
    for dy, dx in ((1, 0), (-1, 0), (0, 1), (0, -1)):
        n = torch.roll(z, shifts=(dy, dx), dims=(0, 1))
        valid = torch.ones_like(out)
        if dy == 1:
            valid[0, :] = False
        if dy == -1:
            valid[-1, :] = False
        if dx == 1:
            valid[:, 0] = False
        if dx == -1:
            valid[:, -1] = False
        rel = (z - n).abs() / torch.minimum(z, n)
        out |= valid & torch.isfinite(rel) & (rel > 0.10)
    return out.reshape(-1) & lab


def pool_state(st_meta, P, geo, rec, h, eps4, cores, Frow, tcache, isect, tree, fr, kM):
    dev = C.INST["dev"]
    t = st_meta["t"]
    tr = 10 * (t // 10)
    seeds = [int(P["seed"](tr, kk)) for kk in range(16)]
    eps16 = C.PR.noise(dev, 16, lambda kk: seeds[kk])
    eps_ok = bool(torch.equal(eps16[:4], eps4))
    assert eps_ok, "noise maps 0..3 differ from the replay chain"
    keys1 = [tuple(int(x) for x in r) for r in np.asarray(rec["legal_idx"]).reshape(-1, 5)]
    views = [geo.view(R_, T_, (0.0, 0.0)) for R_, T_ in zip(rec["legal_R"], rec["legal_T"])]
    if not views:
        return [{"ckpt": "G2", "score": [0.0], "regression": {}, "peak_gb": 0.0}]
    vv = [(v["o"], v["d"], v["tcap"], v["pr"]) for v in views]
    pg, mu = cores["G"].pred.cond(h["G"], vv)
    b4, y4 = cores["G"].pred.sample(pg, mu, eps4)
    b12, y12 = cores["G"].pred.sample(pg, mu, eps16[4:])
    bg, yg = torch.cat([b4, b12], 1), torch.cat([y4, y12], 1)                 # [V, 16, R]
    pd, mud = cores["D0"].pred.cond(h["D0"], vv)
    bd, yd = cores["D0"].pred.sample(pd, mud, eps4[:1])                       # [V, 1, R]
    A = ACC[st_meta["tag"]]
    best = (-1, None)
    for j, (key, v) in enumerate(zip(keys1, views)):
        Pt = C.HP.STATE["pose"]
        c_rt = torch.as_tensor(np.asarray(fr.model_to_runtime(v["o"].double().cpu().numpy()[None]))[0], device=dev)
        d_rt = torch.as_tensor(MC.to_y_up(v["d"].double().cpu().numpy()), device=dev)
        d_rt = d_rt / d_rt.norm(dim=1, keepdim=True)
        tcap_rt = v["tcap"].double() * kM
        if key not in tcache:
            tt = torch.as_tensor(FX.gt_first_hit(isect, c_rt.cpu().numpy(), d_rt.cpu().numpy(), float(tcap_rt.max())), device=dev)
            tcache[key] = (torch.where(tt <= tcap_rt, tt, torch.full_like(tt, float("inf"))) / kM).to(v["d"].dtype)
        ts = tcache[key]
        pr = v["pr"]
        dz = v["dz"].to(ts.dtype)
        lab_ = G.encode_labels(v["o"], v["d"], pr, ts, dz, kM, geo.struct, geo.domain)
        b, y = lab_["b"], lab_["y"]
        lm = (pr["n_u"] > 0) & (b != 0)
        if not bool(lm.any()):
            continue
        lu = pr["n_u"].float() * 0.5
        lab = lm & (b == 1)
        lstar = torch.where(lab, ((y.float() + 1) / 2).clamp(0, 1), torch.ones_like(lu)) * lu
        zs = (ts.double() * kM * v["dz"].double()).float()
        ec = GP.edge_c(torch.where(lm, b, torch.zeros_like(b)).to(torch.int8)[None])[0]
        ed = edge_d(zs, lab)
        g_ell, g_hit = GP.ell_of(bg[j].float(), yg[j].float(), lu[None])
        d_ell, d_hit = GP.ell_of(bd[j].float(), yd[j].float(), lu[None])
        vid = A["nv"]
        A["views"].append({"t": t, "key": list(key), "view": vid, "n_lm": int(lm.sum()), "n_edgeC": int((ec & lm).sum()), "n_edgeD": int(ed.sum()),
                           "es_G1": GP.es_view(g_ell[:1], lstar, lm), "es_G4": GP.es_view(g_ell[:4], lstar, lm), "es_G16": GP.es_view(g_ell, lstar, lm),
                           "es_D0": GP.es_view(d_ell, lstar, lm)})
        n = int(lm.sum())
        for k_, val in (("view", np.full(n, vid, np.int32)), ("lu", lu[lm]), ("lstar", lstar[lm]), ("label", lab[lm]), ("edgeC", ec[lm]), ("edgeD", ed[lm]),
                        ("zstar", zs[lm]), ("g_ell", g_ell[:, lm].T), ("g_hit", g_hit[:, lm].T), ("d_ell", d_ell[0][lm]), ("d_hit", d_hit[0][lm]),
                        ("d_b", bd[j, 0][lm].float())):
            A["rays"].setdefault(k_, []).append(val if isinstance(val, np.ndarray) else val.cpu().numpy())
        A["nv"] += 1
        score = int((lm & (ec | ed)).sum())
        if score > best[0]:
            best = (score, j, key, v, ts, zs, lm, b, c_rt, d_rt)
    if best[1] is not None:                                                       # figure arrays of the state's most ambiguous view (GT-only score)
        score, j, key, v, ts, zs, lm, b, c_rt, d_rt = best
        dz = v["dz"].double()
        tg, hg = G.decode(bg[j, :4], yg[j, :4], v["o"], v["d"], v["pr"], geo.struct, geo.domain)
        td, hd = G.decode(bd[j], yd[j], v["o"], v["d"], v["pr"], geo.struct, geo.domain)

        def pts(tm):
            p = v["o"][None].double() + v["d"].double() * torch.where(torch.isfinite(tm), tm, torch.zeros_like(tm)).double()[:, None]
            q = np.asarray(fr.model_to_runtime(p.cpu().numpy()), np.float32)
            q[~torch.isfinite(tm).cpu().numpy()] = np.nan
            return q.astype(np.float16)
        A["fig"][f"t{t}"] = {"score": score, "key": np.asarray(key), "zstar": zs.cpu().numpy(), "b_gt": b.cpu().numpy(), "lm": lm.cpu().numpy(),
                             "n_u": v["pr"]["n_u"].cpu().numpy(), "z_g4": (tg.double() * kM * dz[None]).float().cpu().numpy(), "hitu_g4": hg.cpu().numpy(),
                             "z_d0": (td.double() * kM * dz[None]).float().cpu().numpy()[0], "hitu_d0": hd.cpu().numpy()[0], "b_d0": bd[j, 0].float().cpu().numpy(),
                             "p_gt": pts(ts), "p_d0": pts(torch.where(hd[0], td[0], torch.full_like(td[0], float("inf")))),
                             "p_g4": np.stack([pts(torch.where(hg[k], tg[k], torch.full_like(tg[k], float("inf")))) for k in range(4)]),
                             "c_rt": c_rt.cpu().numpy(), "tknown_rt": (v["pr"]["t_known"].double() * kM * dz).float().cpu().numpy()}
    return [{"ckpt": "G2", "score": [0.0], "regression": {"eps4_equal": eps_ok}, "peak_gb": torch.cuda.max_memory_allocated() / 2 ** 30}]


CT.outdoor_state = pool_state


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--set", required=True, choices=("explore", "confirm"))
    ap.add_argument("--only", default=None)
    ap.add_argument("--tmax", type=int, default=None)
    ap.add_argument("--out", default=None)
    a = ap.parse_args()
    od = a.out or os.path.join(R, "work_dirs/exp6_v3/experiment_G/g2/pool")
    claims = os.path.join(od, "_claims")
    os.makedirs(claims, exist_ok=True)
    torch.use_deterministic_algorithms(True, warn_only=True)               # = g_fixed / CT / U settings
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    dev = torch.device("cuda")
    J = FR.jobs(argparse.Namespace(set=a.set))
    if a.only:
        J = [j for j in J if j[0] == a.only]
    if not J:
        return
    kind0, spec0 = J[0][1]
    C.install(spec0[0] if kind0 in ("e3", "conf") else spec0[2][0]["scene"], dev)
    C.DV2.STATE["mode"] = "off"
    cores = {"G": FX.PV3.V3Core(T0[0], dev, expect_sha=T0[1])}

    class GCore:
        def __init__(self):
            self.model, self.pred, self.ckpt_sha = DH.load(D0[0], dev, "d0", 10, D0[1])
    cores["D0"] = GCore()
    for tag, (kind, spec) in J:
        po = os.path.join(od, f"{tag}.npz")
        if os.path.exists(po) and not a.only:
            continue
        if not a.only:
            try:
                fd = os.open(os.path.join(claims, f"{tag}.claim"), os.O_CREAT | os.O_EXCL | os.O_WRONLY)
                os.write(fd, f"{os.uname()[1]} {os.environ.get('CUDA_VISIBLE_DEVICES')} {time.strftime('%FT%T')}\n".encode())
                os.close(fd)
            except FileExistsError:
                continue
        tb = time.time()
        ACC[tag] = {"nv": 0, "views": [], "rays": {}, "fig": {}}
        C.DV2.STATE["mode"] = "off"
        CT.run_outdoor(a, tag, kind, spec, cores, dev)
        A = ACC.pop(tag)
        np.savez_compressed(os.path.join(od, f"{tag}_fig.npz"), **{f"{t}__{k}": v for t, d in A["fig"].items() for k, v in d.items()})
        json.dump({"tag": tag, "set": a.set, "views": A["views"], "ckpt_sha": {"G": cores["G"].ckpt_sha, "D0": cores["D0"].ckpt_sha}, "host": os.uname()[1],
                   "time": time.ctime(), "seconds": time.time() - tb}, open(os.path.join(od, f"{tag}_views.json"), "w"))
        np.savez(po + ".tmp.npz", **{k: np.concatenate(v) for k, v in A["rays"].items()})
        os.replace(po + ".tmp.npz", po)
        print(f"DONE {tag} views {A['nv']} {time.time() - tb:.0f}s", flush=True)


if __name__ == "__main__":
    main()
