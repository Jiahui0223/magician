"""Experiment G3 -- fixed-state evaluation of G4 / G1 / D0 / M-raw / M-depth on the 755-state Experiment F pool (protocol G3-v1 section 5;
seen2scene env, 1 GPU per process; offline, GT only to score). Replay = the UNCHANGED CT / U / G chain (ct_fixed.run_outdoor -> f_run ->
mv_core, D-v2 off -> D1); only ct_fixed.outdoor_state is replaced (monkeypatch, as g_fixed.py). The m_combine hooks are installed BEFORE
mv_core.install (so the D1 policy captures them); for G4 / G1 / D0 they are identities (mode off).
Per state and arm: D1 scores, pick (m_score.choose), official one-step gain, Top-1, Spearman (mv_core.rank_metrics vs the F oracle); per
first-layer candidate, on labelled UNKNOWN rays: l(t) = 0.5 x #UNKNOWN samples with sample distance <= t (no hit -> lu), fair CRPS, fair ES
(eval_dev formula), hit counts, Brier, TP depth error |t - t*| k_M; on all predicted UNKNOWN hits: floating-surface counts S (> 1.0 runtime
from every G4 sample hit on the ray, or none hits), T (> 1.0 from the GT first hit, or none), S & T. Checks: G4 D1 scores == EXP9 Table 4 T0
rows (STOP otherwise); the 4 samples used inside M-raw / M-depth == G4's samples bitwise (first layer, every state).
Figure arrays: per state the view with the largest M-depth S & T count (GT z*, the 4 samples, M-raw, M-depth, masks, runtime points).
Output work_dirs/exp6_v3/experiment_G/g3/fixed/<tag>.jsonl (+ <tag>_fig.npz); claims; resumable.
Usage: python g3_fixed.py --set confirm|explore [--only TAG] [--tmax T] [--out DIR]"""
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
import m_combine as MC  # noqa: E402

C, FR, torch, np, G = CT.C, CT.FR, CT.torch, CT.np, CT.G
FX, MCm = C.FX, C.MC
from seen2scene.futureview import scoring as SCO  # noqa: E402

T0 = (os.path.join(R, "work_dirs/exp6_v3/task0_vae/checkpoints/final.pt"), "7ed0ddd72bb91d1e8c6563f3192201c24a49fe862b49d2387ea8770915145ab6")
D0 = (os.path.join(R, "work_dirs/exp6_v3/experiment_G/d0/checkpoints/final.pt"), "aeefe098e86517a1ff3b7455f3a83338e3972fb0f4fcd95a22607dd77d83f205")
REF_T4 = os.path.join(R, "work_dirs", "cvpr_results", "exp9_ablation_qualitative", "table4", "fixed", "states", "main")
ARMS = [("G4", "off", 4), ("G1", "off", 1), ("D0", "off", 1), ("MRAW", "raw", 1), ("MDEPTH", "depth", 1)]
TAU = 1.0
_REF, FIG = {}, {}


def ref_scores(tag):
    if tag not in _REF:
        p = os.path.join(REF_T4, f"{tag}.jsonl")
        _REF[tag] = {r["t"]: r["score"] for r in map(json.loads, open(p)) if r["ckpt"] == "T0"} if os.path.exists(p) else {}
    return _REF[tag]


class GCore:
    def __init__(self, path, dev, mode, sha):
        self.model, self.pred, self.ckpt_sha = DH.load(path, dev, mode, 10, sha)


def ell_of_t(v, tlist, geo, chunk=2048):
    """tlist [A, R] ray distances (inf = none) -> l [A, R] = 0.5 x #UNKNOWN samples with distance <= t (inf -> all = lu)."""
    pr = v["pr"]
    A, Rr = tlist.shape
    out = torch.zeros(A, Rr, dtype=torch.float64, device=tlist.device)
    for s in range(0, Rr, chunk):
        sl = slice(s, min(Rr, s + chunk))
        unk, ts = G.unknown_samples(v["o"], v["d"][sl], pr["tmin"][sl], pr["tmax"][sl], pr["valid"][sl], pr["M"], pr["t_known"][sl], geo.struct, geo.domain)
        tt = tlist[:, sl].to(ts.dtype)
        out[:, sl] = 0.5 * (unk[None] & (ts[None] <= tt[..., None])).sum(-1).double()
    return out


def crps(ell, lt):
    K = ell.shape[0]
    t1 = (ell - lt[None]).abs().mean(0)
    if K == 1:
        return t1
    s = torch.sort(ell, 0).values
    w = (2 * torch.arange(1, K + 1, device=ell.device, dtype=ell.dtype) - K - 1)[:, None]
    return t1 - (s * w).sum(0) / (K * (K - 1))


def es_view(ell, lt, m):
    rms = lambda a: torch.sqrt((a[..., m] ** 2).mean(-1))  # noqa: E731
    K = ell.shape[0]
    e1 = rms(ell - lt[None]).mean()
    if K == 1:
        return float(e1)
    return float(e1 - sum(rms(ell[i] - ell[j]) for i in range(K) for j in range(K) if i != j) / (2 * K * (K - 1)))


def g3_state(st_meta, P, geo, rec, h, eps4, cores, Frow, tcache, isect, tree, fr, kM):
    dev = C.INST["dev"]
    z0 = P["z0"]
    runs, rows = {}, []
    for name, mode, K in ARMS:
        MC.STATE["mode"] = mode
        MC.STATE["eps4"] = eps4 if mode != "off" else None
        MC.STATE["calls"] = [] if mode != "off" else None
        eps = eps4 if K == 4 else eps4[:1]
        t0 = C.sync()
        run = C.run_planner(cores[name], geo, h[name], rec, eps, z0)
        run["t_plan"] = C.sync() - t0
        run["calls"] = MC.STATE["calls"]
        runs[name] = run
    MC.STATE.update({"mode": "off", "eps4": None, "calls": None})
    keys1 = runs["G4"]["keys1"]
    same4 = True
    for name in ("MRAW", "MDEPTH"):
        first = runs[name]["calls"][0]
        for j, key in enumerate(keys1):
            g4 = runs["G4"]["HT"][key]
            same4 &= bool(torch.equal(first["samples"][j]["t4"], g4["t"]) and torch.equal(first["samples"][j]["hit_u4"], g4["hit_u"]))
    assert same4, "M arms did not see the G4 samples"
    acc = {n: {k: 0.0 for k in ("n_lm", "n_lab", "crps", "es", "n_views", "p", "brier", "tp", "hit", "tp_err", "tp_n", "tp_le_tau",
                                "pred", "S", "T", "ST")} for n, _, _ in ARMS}
    best = (-1, None)
    for j, key in enumerate(keys1):
        v = runs["G4"]["V"][key]
        Vc = C.ViewC(key, v, runs["G4"]["HT"][key], fr, kM, dev)
        if key not in tcache:
            tt = torch.as_tensor(FX.gt_first_hit(isect, Vc.c.cpu().numpy(), Vc.d.cpu().numpy(), float(Vc.tcap.max())), device=dev)
            tcache[key] = (torch.where(tt <= Vc.tcap, tt, torch.full_like(tt, float("inf"))) / kM).to(v["d"].dtype)
        ts = tcache[key]
        pr = v["pr"]
        lab_ = G.encode_labels(v["o"], v["d"], pr, ts, v["dz"].to(ts.dtype), kM, geo.struct, geo.domain)
        lm = (pr["n_u"] > 0) & (lab_["b"] != 0)
        lab = lm & (lab_["b"] == 1)
        lu = pr["n_u"].double() * 0.5
        preds = {n: (runs[n]["HT"][key]["t"].double(), runs[n]["HT"][key]["hit_u"]) for n, _, _ in ARMS}
        allt = torch.cat([torch.where(preds[n][1], preds[n][0], torch.full_like(preds[n][0], float("inf"))) for n, _, _ in ARMS]
                         + [torch.where(lab, ts.double(), torch.full_like(ts.double(), float("inf")))[None]])
        ell_all = ell_of_t(v, allt, geo)
        lstar = torch.where(lab, ell_all[-1], lu)
        g4t, g4h = preds["G4"]
        tstar = ts.double()
        i0 = 0
        for n, mode, K in ARMS:
            t_, h_ = preds[n]
            e = torch.where(h_, ell_all[i0:i0 + K], lu[None].expand(K, -1))
            i0 += K
            a = acc[n]
            if bool(lm.any()):
                a["n_lm"] += float(lm.sum()); a["n_lab"] += float(lab.sum())
                a["crps"] += float(crps(e[:, lm], lstar[lm]).sum())
                a["es"] += es_view(e, lstar, lm); a["n_views"] += 1
                p = h_.double().mean(0)
                a["p"] += float(p[lm].sum()); a["brier"] += float(((p - lab.double()) ** 2)[lm].sum())
                a["tp"] += float((h_ & lab[None]).sum()) / K; a["hit"] += float((h_ & lm[None]).sum()) / K
            tpm = h_ & lab[None]
            err = (t_ - tstar[None]).abs() * kM
            a["tp_err"] += float(err[tpm].sum()); a["tp_n"] += float(tpm.sum()); a["tp_le_tau"] += float((tpm & (err <= TAU)).sum())
            # floating surfaces over all predicted UNKNOWN hits
            Tm = h_ & ~(torch.isfinite(tstar)[None] & (err <= TAU))
            a["pred"] += float(h_.sum()); a["T"] += float(Tm.sum())
            if n in ("D0", "MRAW", "MDEPTH"):
                near = (g4h & ((g4t - t_).abs() * kM <= TAU)).any(0)[None]
                Sm = h_ & ~near
                a["S"] += float(Sm.sum()); a["ST"] += float((Sm & Tm).sum())
                if n == "MDEPTH":
                    fl_d = (Sm & Tm)[0]
                if n == "MRAW":
                    fl_r = (Sm & Tm)[0]
        sc = int(fl_d.sum())
        if sc > best[0]:
            best = (sc, j, key, v, ts, preds, fl_d, fl_r, Vc)
    # figure arrays of the view with the most M-depth floating rays
    if best[1] is not None:
        sc, j, key, v, ts, preds, fl_d, fl_r, Vc = best
        dz = v["dz"].double()

        def pts(tm, hm):
            tm = torch.where(hm, tm, torch.full_like(tm, float("inf")))
            p = v["o"][None].double() + v["d"].double() * torch.where(torch.isfinite(tm), tm, torch.zeros_like(tm)).double()[:, None]
            q = np.asarray(fr.model_to_runtime(p.cpu().numpy()), np.float32)
            q[~torch.isfinite(tm).cpu().numpy()] = np.nan
            return q.astype(np.float16)
        g4t, g4h = preds["G4"]
        FIG[f"t{st_meta['t']}"] = {"score": sc, "key": np.asarray(key), "zstar": (ts.double() * kM * dz).float().cpu().numpy(),
                                   "n_u": v["pr"]["n_u"].cpu().numpy(), "z_g4": (g4t * kM * dz[None]).float().cpu().numpy(), "hitu_g4": g4h.cpu().numpy(),
                                   "z_mraw": (preds["MRAW"][0] * kM * dz[None]).float().cpu().numpy()[0], "hitu_mraw": preds["MRAW"][1].cpu().numpy()[0],
                                   "z_mdepth": (preds["MDEPTH"][0] * kM * dz[None]).float().cpu().numpy()[0], "hitu_mdepth": preds["MDEPTH"][1].cpu().numpy()[0],
                                   "float_mdepth": fl_d.cpu().numpy(), "float_mraw": fl_r.cpu().numpy(), "p_gt": pts(ts.double(), torch.isfinite(ts)),
                                   "p_g4": np.stack([pts(g4t[k], g4h[k]) for k in range(4)]), "p_mdepth": pts(preds["MDEPTH"][0][0], preds["MDEPTH"][1][0]),
                                   "p_mraw": pts(preds["MRAW"][0][0], preds["MRAW"][1][0]), "c_rt": Vc.c.cpu().numpy()}
    for name, mode, K in ARMS:
        run = runs[name]
        d1 = [float(x) for x in run["out"]["distp1"]]
        V1k = [[float(run["SETS"][c1][k][1].sum()) for k in range(K)] for c1 in keys1]
        reg = {"keys_eq_F": [list(k) for k in run["keys1"]] == Frow["keys1"]}
        if name == "G4":
            ref = ref_scores(st_meta["tag"]).get(st_meta["t"])
            reg["d1_bitwise_T4V2_T0"] = (d1 == [float(x) for x in ref]) if ref is not None else None
            assert reg["d1_bitwise_T4V2_T0"] is not False, "STOP: G4 D1 scores differ from the Table 4 T0 rows"
        if name in ("MRAW", "MDEPTH"):
            reg["samples_equal_G4"] = same4
        rows.append({**st_meta, "ckpt": name, "arm": name, "K": K, "mode": mode, "ckpt_sha": run_sha(name), "n_legal": len(keys1),
                     "keys1": [list(k) for k in keys1], "oracle_pp": [float(x) for x in Frow["oracle"]], "score": d1, "V1k": V1k,
                     "rank_d1": C.rank_metrics(d1, keys1, Frow["oracle"], st_meta["scene"], st_meta["start"], st_meta["t"]),
                     "rank_one_step": C.rank_metrics([float(np.mean(x)) for x in V1k], keys1, Frow["oracle"], st_meta["scene"], st_meta["start"], st_meta["t"]),
                     "acc": acc[name], "seconds_plan": run["t_plan"], "regression": reg, "peak_gb": torch.cuda.max_memory_allocated() / 2 ** 30})
    return rows


SHA = {}


def run_sha(name):
    return SHA.get(name)


CT.outdoor_state = g3_state


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--set", required=True, choices=("explore", "confirm"))
    ap.add_argument("--only", default=None)
    ap.add_argument("--tmax", type=int, default=None)
    ap.add_argument("--out", default=None)
    a = ap.parse_args()
    od = a.out or os.path.join(R, "work_dirs/exp6_v3/experiment_G/g3/fixed")
    claims = os.path.join(od, "_claims")
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
    MC.install(C.PC, C.PR, G, SCO)                     # BEFORE mv_core.install -> d_policy captures the hook
    kind0, spec0 = J[0][1]
    C.install(spec0[0] if kind0 in ("e3", "conf") else spec0[2][0]["scene"], dev)
    C.DV2.STATE["mode"] = "off"
    t0core = FX.PV3.V3Core(T0[0], dev, expect_sha=T0[1])
    d0core = GCore(D0[0], dev, "d0", D0[1])
    cores = {"G4": t0core, "G1": t0core, "D0": d0core, "MRAW": t0core, "MDEPTH": t0core}
    SHA.update({n: c.ckpt_sha for n, c in cores.items()})
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
        FIG.clear()
        C.DV2.STATE["mode"] = "off"
        rows = CT.run_outdoor(a, tag, kind, spec, cores, dev)
        np.savez_compressed(os.path.join(od, f"{tag}_fig.npz"), **{f"{t}__{k}": v for t, d in FIG.items() for k, v in d.items()})
        with open(po + ".tmp", "w") as f:
            for r in rows:
                f.write(json.dumps(r, default=float) + "\n")
        os.replace(po + ".tmp", po)
        print(f"DONE {tag} rows {len(rows)} {time.time() - tb:.0f}s", flush=True)


if __name__ == "__main__":
    main()
