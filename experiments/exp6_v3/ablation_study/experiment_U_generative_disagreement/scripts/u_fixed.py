"""Experiment U Phase D -- fixed-state diagnostic of the disagreement metrics on V3-T0 (seen2scene env, 1 GPU per process; offline,
GT only through the EXISTING oracle labels). Protocol U-v1 (configs/u_config.yaml).

State pool = the Experiment F outdoor pool (755 states: explore 395 = E3 D1 trajectories colosseum s0-s4 + pisa s1 and G2 Dev-3
B0 / B2 trajectories; confirm 360 = D-v2 Table 1 trajectories of 6 scenes, t = 0, 5, ..., 95; Dev-5 + Train scenes, NO Test-3,
no dunnottar). Replay / planner = the UNCHANGED Experiment CT evaluator chain (experiment_CT_continued_training/scripts/ct_fixed.py
-> f_run -> mv_core), exactly as EXP9 Table 4 V2 used it for V3-T0: same replayed P1 state, R10 features, K = 4 noise keys,
simulator legal list; D-v2 mode 'off' -> the planner is D1 (frozen d_policy d1, frozen d_config.yaml), recorded views include all
D1 second-layer children. Only ct_fixed.outdoor_state is replaced (monkeypatch in this file) by u_state below; nothing frozen is
edited. Per state: D1 two-step score, C1 one-step score (= mean_k value_cov, recorded per sample), U1 / U2 / U3 (+ offline-only
diagnostics) of every first-layer candidate and every recorded child, the cross-view noise-index agreement of every D1 path
(U4 rule), the official one-step coverage gain of every candidate (Experiment F oracle rows, dcov_pp) and the geometric one-step
truth (F geo1). Regression: the D1 scores must equal the EXP9 Table 4 V2 V3-T0 rows bitwise.
Output work_dirs/exp6_v3/experiment_U/fixed/states/<tag>.jsonl (+ arrays/<tag>_t<t>.npz with the first-layer predictions for the
figures). Usage: python u_fixed.py --set confirm|explore [--only TAG] [--tmax T] [--out DIR]"""
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
import u_metrics as UM  # noqa: E402

C, FR, torch, np, G = CT.C, CT.FR, CT.torch, CT.np, CT.G
FX, HP, MC = C.FX, C.HP, C.MC
W = os.path.join(R, "work_dirs", "exp6_v3", "experiment_U", "fixed")
T0 = ("work_dirs/exp6_v3/task0_vae/checkpoints/final.pt", "7ed0ddd72bb91d1e8c6563f3192201c24a49fe862b49d2387ea8770915145ab6")
REF_T4 = os.path.join(R, "work_dirs", "cvpr_results", "exp9_ablation_qualitative", "table4", "fixed", "states", "main")
_REF = {}


def ref_scores(tag):
    if tag not in _REF:
        p = os.path.join(REF_T4, f"{tag}.jsonl")
        _REF[tag] = {r["t"]: r["score"] for r in map(json.loads, open(p)) if r["ckpt"] == "T0"} if os.path.exists(p) else {}
    return _REF[tag]


def compact(m):
    return {k: m[k] for k in ("u1", "u2", "u3", "n_rc", "n_r2", "n_cells_union", "u3_sum", "u1_sum", "u2_rel", "jaccard_pair_mean") if k in m}


@torch.no_grad()
def gt_attr(t, hit_u, ts, v, geo, fr, kM, sets_k):
    """OFFLINE ONLY. Where does the disagreement of one candidate come from? ts [R] = GT first hit (model units, inf = none).
    GT category per ray = geometry.encode_labels (training rule) on the state's own map: HIT = true surface on an UNKNOWN sample;
    NOHIT / KNOWN = no true surface inside the UNKNOWN part; other = BLOCKED / NEAR / INF_* / OUT. Returns per-category U1 mass,
    U2 mass (R2 rays), hit accuracy on GT-HIT rays (|t_k - t*| <= 1.0 runtime), and the share of U3 disagreement mass on cells
    that contain a true UNKNOWN-surface point (GT-HIT rays back-projected into the same 1.0 grid)."""
    pr = v["pr"]
    lab = G.encode_labels(v["o"], v["d"], pr, ts, v["dz"].to(ts.dtype), kM, geo.struct, geo.domain)
    cat = lab["cat"]
    rc = UM.ray_set(v)
    h = hit_u & rc[None]
    K = h.shape[0]
    p = h.double().mean(0)
    bern = 4.0 * p * (1.0 - p)
    groups = {"HIT": cat == G.CAT["HIT"], "NOHIT": cat == G.CAT["NOHIT"], "KNOWN": cat == G.CAT["KNOWN"]}
    groups["OTHER"] = ~(groups["HIT"] | groups["NOHIT"] | groups["KNOWN"])
    o = {"n_rc": int(rc.sum())}
    for nm, m in groups.items():
        mm = m & rc
        o[f"n_{nm}"] = int(mm.sum())
        o[f"u1mass_{nm}"] = float(bern[mm].sum())
    o["u1mass_singleton"] = float(bern[(p > 0) & (p <= 1.0 / K + 1e-9) & rc].sum())     # exactly one of K samples hits
    # hit accuracy on GT-HIT rays
    gh = groups["HIT"] & rc
    tr = ts.double() * kM
    tk = t.double() * kM
    ok = h & gh[None] & ((tk - tr[None]).abs() <= 1.0)
    o["n_hits_on_gthit"] = int((h & gh[None]).sum())
    o["n_hits_correct"] = int(ok.sum())
    o["n_hits_on_gtnohit"] = int((h & (groups["NOHIT"] | groups["KNOWN"])[None]).sum())
    # U2 mass by category + depth error vs spread on GT-HIT R2 rays
    r2 = h.sum(0) >= 2
    if bool(r2.any()):
        hh = h[:, r2].double()
        n = hh.sum(0)
        tt = torch.where(h, tk, torch.zeros_like(tk))[:, r2]
        mu = tt.sum(0) / n
        var = (hh * (tt - mu[None]) ** 2).sum(0) / (n - 1.0)
        for nm, m in groups.items():
            o[f"u2mass_{nm}"] = float(var[m[r2]].sum())
        g2 = gh[r2]
        if int(g2.sum()) >= 10:
            from scipy.stats import spearmanr
            err = (mu[g2] - tr[r2][g2]).abs().cpu().numpy()
            rr = spearmanr(var[g2].sqrt().cpu().numpy(), err).correlation
            o["u2_rho_spread_err_gthit"] = float(rr) if np.isfinite(rr) else None
    # U3: disagreement mass on cells holding a true UNKNOWN-surface point vs phantom cells
    sel = torch.nonzero(gh & torch.isfinite(ts))[:, 0]
    if sel.numel():
        pg = v["o"][None].double() + v["d"][sel].double() * ts[sel].double()[:, None]
        tcell = np.unique(UM.cell_keys(np.asarray(fr.model_to_runtime(pg.cpu().numpy()), np.float64)))
    else:
        tcell = np.zeros(0, np.int64)
    allk = np.concatenate(sets_k) if any(len(s) for s in sets_k) else np.zeros(0, np.int64)
    o["n_true_cells"] = int(len(tcell))
    if len(allk):
        u, cnt = np.unique(allk, return_counts=True)
        pv = cnt / float(K)
        b = 4.0 * pv * (1.0 - pv)
        tru = np.isin(u, tcell)
        o["u3mass_true"], o["u3mass_phantom"] = float(b[tru].sum()), float(b[~tru].sum())
        o["n_cells_true"], o["n_cells_phantom"] = int(tru.sum()), int((~tru).sum())
        o["true_cells_recalled"] = int(np.isin(tcell, u).sum())
    return o


def u_state(st_meta, P, geo, rec, h, eps4, cores, Frow, tcache, isect, tree, fr, kM):
    z0 = P["z0"]
    core = cores["T0"]
    t0 = C.sync()
    run = C.run_planner(core, geo, h["T0"], rec, eps4, z0)
    t_plan = C.sync() - t0
    keys1 = run["keys1"]
    ch = {c1: C.children((c1,), geo) for c1 in keys1}
    paths = [(a, b) for a in keys1 for b in ch[a]]
    S = run["SETS"]
    V1k = [[float(S[c1][k][1].sum()) for k in range(4)] for c1 in keys1]
    d1 = [float(x) for x in run["out"]["distp1"]]
    tb = time.time()
    met, sets, pts1 = {}, {}, {}
    for key in run["V"]:
        v, ht = run["V"][key], run["HT"][key]
        m, s_, p_ = UM.view_metrics(ht["t"], ht["hit_u"], v, fr, y=ht.get("y"), diag=True)
        met[key], sets[key] = m, s_
        if key in keys1:
            pts1[key] = p_
    agree = []
    for a, b in paths:
        r = UM.overlap_agreement(sets[a], sets[b])
        r["c1"], r["c2"] = list(a), list(b)
        agree.append(r)
    t_u = time.time() - tb
    # OFFLINE GT attribution of the disagreement (first-layer candidates only; GT never reaches any planner score)
    gtat, tst = [], {}
    for key in keys1:
        v, ht = run["V"][key], run["HT"][key]
        try:
            Vc = C.ViewC(key, v, ht, fr, kM, C.INST["dev"])
            tt = torch.as_tensor(FX.gt_first_hit(isect, Vc.c.cpu().numpy(), Vc.d.cpu().numpy(), float(Vc.tcap.max())), device=Vc.tcap.device)
            ts = (torch.where(tt <= Vc.tcap, tt, torch.full_like(tt, float("inf"))) / kM).to(v["d"].dtype)
            tst[key] = ts
            gtat.append(gt_attr(ht["t"], ht["hit_u"], ts, v, geo, fr, kM, sets[key]))
        except Exception as e:
            sys.stderr.write(f"gt_attr failed {st_meta['tag']} t={st_meta['t']} {key}: {e!r}\n")
            gtat.append(None)
    # first-layer arrays for the figures (runtime frame; predictions only, no GT)
    try:
        ad = os.path.join(W, "arrays")
        os.makedirs(ad, exist_ok=True)
        dep, hu, rcm, tkn, orr, drr = [], [], [], [], [], []
        for key in keys1:
            v, ht = run["V"][key], run["HT"][key]
            dep.append((ht["t"].double() * kM).cpu().numpy().astype(np.float32))
            hu.append(ht["hit_u"].cpu().numpy())
            rcm.append(UM.ray_set(v).cpu().numpy())
            tkn.append((v["pr"]["t_known"].double() * kM).cpu().numpy().astype(np.float32))
            orr.append(np.asarray(fr.model_to_runtime(v["o"].double().cpu().numpy()[None]))[0])
            drr.append(MC.to_y_up(v["d"].double().cpu().numpy()).astype(np.float32))
        obs = np.concatenate(HP.STATE["obs"]) if HP.STATE["obs"] else np.zeros((0, 3))
        obs = obs[::max(1, int(np.ceil(len(obs) / 40000)))].astype(np.float32)
        np.savez_compressed(os.path.join(ad, f"{st_meta['tag']}_t{st_meta['t']}.npz"), keys1=np.asarray(keys1), depth_rt=np.stack(dep), hit_u=np.stack(hu),
                            rc=np.stack(rcm), tknown_rt=np.stack(tkn), o_rt=np.stack(orr), d_rt=np.stack(drr), obs_rt=obs,
                            oracle_pp=np.asarray(Frow["oracle"], np.float64),
                            gt_rt_offline=np.stack([(tst[k].double() * kM).cpu().numpy().astype(np.float32) if k in tst else np.full(len(dep[0][0]), np.nan, np.float32) for k in keys1]))
    except Exception as e:                                                    # figures are optional; never fail the state
        sys.stderr.write(f"arrays failed {st_meta['tag']} t={st_meta['t']}: {e!r}\n")
    ref = ref_scores(st_meta["tag"]).get(st_meta["t"])
    row = {**st_meta, "ckpt": "T0", "ckpt_sha": core.ckpt_sha, "n_legal": len(keys1), "keys1": [list(k) for k in keys1],
           "children": {json.dumps(list(a)): [list(b) for b in ch[a]] for a in keys1},
           "oracle_pp": [float(x) for x in Frow["oracle"]], "geo1": Frow.get("geo1"),
           "score": d1, "d1": d1, "c1": [float(np.mean(v)) for v in V1k], "V1k": V1k,
           "u": [met[k] for k in keys1], "u_children": {json.dumps(list(k)): compact(met[k]) for k in met if k not in keys1},
           "agree": agree, "gt_attr": gtat, "seconds_plan": t_plan, "seconds_u": t_u, "predicted_views": run["out"]["d"]["predicted_views"],
           "peak_gb": torch.cuda.max_memory_allocated() / 2 ** 30,
           "regression": {"keys_eq_F": [list(k) for k in keys1] == Frow["keys1"], "paths_eq_F": [[list(a), list(b)] for a, b in paths] == Frow["paths"],
                          "d1_bitwise_T4V2_T0": (d1 == [float(x) for x in ref]) if ref is not None else None}}
    return [row]


CT.outdoor_state = u_state


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--set", required=True, choices=("explore", "confirm"))
    ap.add_argument("--only", default=None)
    ap.add_argument("--tmax", type=int, default=None)
    ap.add_argument("--out", default=None)
    a = ap.parse_args()
    od = a.out or os.path.join(W, "states")
    claims = os.path.join(W, "claims")
    os.makedirs(od, exist_ok=True)
    os.makedirs(claims, exist_ok=True)
    torch.use_deterministic_algorithms(True, warn_only=True)
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
    C.DV2.STATE["mode"] = "off"                     # D1 (frozen d_policy d1, value_cov = the frozen function)
    cores = {"T0": FX.PV3.V3Core(os.path.join(R, T0[0]), dev, expect_sha=T0[1])}
    for tag, (kind, spec) in J:
        po = os.path.join(od, f"{tag}.jsonl")
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
        C.DV2.STATE["mode"] = "off"
        rows = CT.run_outdoor(a, tag, kind, spec, cores, dev)
        with open(po + ".tmp", "w") as f:
            for r in rows:
                f.write(json.dumps(r, default=float) + "\n")
        os.replace(po + ".tmp", po)
        print(f"DONE {tag} rows {len(rows)} {time.time() - tb:.0f}s", flush=True)


if __name__ == "__main__":
    main()
