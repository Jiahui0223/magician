"""Experiment G -- fixed-state evaluation of D0 / G0 / G1 / G4 (protocol G-v1 section 7; seen2scene env, 1 GPU per process; offline,
GT only to score). State pool = the Experiment F outdoor pool (755 states: explore 395 + confirm 360; no Test-3, no dunnottar), replayed
by the UNCHANGED Experiment CT / U chain (ct_fixed.run_outdoor -> f_run -> mv_core): same replayed P1 state, R10 features, noise keys,
simulator legal list; D-v2 mode 'off' -> the planner is D1 (frozen d_policy d1, frozen d_config.yaml). Only ct_fixed.outdoor_state is
replaced (monkeypatch in this file, as u_fixed.py does). Per state and arm (all arms in one process, same state / candidates / noise keys):
  G4  V3-T0 final.pt, V3Predictor, noise maps 0..3 (= Table 1 V3-T0-D1)        regression: D1 scores == EXP9 Table 4 V2 T0 rows bitwise
  G1  V3-T0 final.pt, V3Predictor, noise map 0      (= Table 3 V3-T0-K1D2)
  G0  V3-T0 final.pt, d0_head.ZeroNoisePredictor (eps = 0, Euler 10), K = 1
  D0  D0 checkpoint, d0_head.DetPredictor (one forward), K = 1
Stored per arm: the D1 two-step score of every legal candidate, the one-step per-sample values, the pick (m_score.choose) with official
one-step gain / Top-1 / within-state Spearman (mv_core.rank_metrics vs the Experiment F oracle row) and per first-layer camera the CT
camera_metrics (labels = geometry.encode_labels of the GT first hit on the state's own map: TP / FP / FN / TN on labelled UNKNOWN rays,
Brier, false-wall counts, depth errors on true positives, ...).
Output work_dirs/exp6_v3/experiment_G/fixed/states/<tag>.jsonl (claims; resumable).
Usage: python g_fixed.py --set confirm|explore [--arms G4,G1,G0,D0] [--d0-ckpt PATH] [--only TAG] [--tmax T] [--out DIR]"""
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

C, FR, torch, np, G = CT.C, CT.FR, CT.torch, CT.np, CT.G
FX = C.FX
W = os.path.join(R, "work_dirs", "exp6_v3", "experiment_G", "fixed")
T0 = (os.path.join(R, "work_dirs/exp6_v3/task0_vae/checkpoints/final.pt"), "7ed0ddd72bb91d1e8c6563f3192201c24a49fe862b49d2387ea8770915145ab6")
D0_DEFAULT = os.path.join(R, "work_dirs/exp6_v3/experiment_G/d0/checkpoints/final.pt")
REF_T4 = os.path.join(R, "work_dirs", "cvpr_results", "exp9_ablation_qualitative", "table4", "fixed", "states", "main")
_REF = {}
ARM_K = {"G4": 4, "G1": 1, "G0": 1, "D0": 1}


def ref_scores(tag):
    if tag not in _REF:
        p = os.path.join(REF_T4, f"{tag}.jsonl")
        _REF[tag] = {r["t"]: r["score"] for r in map(json.loads, open(p)) if r["ckpt"] == "T0"} if os.path.exists(p) else {}
    return _REF[tag]


class GCore:
    def __init__(self, path, dev, mode, expect_sha=None):
        self.model, self.pred, self.ckpt_sha = DH.load(path, dev, mode, 10, expect_sha)
        self.mode, self.ckpt = mode, path


def g_state(st_meta, P, geo, rec, h, eps4, cores, Frow, tcache, isect, tree, fr, kM):
    rows = []
    z0 = P["z0"]
    for name, core in cores.items():
        K = ARM_K[name]
        eps = eps4 if K == 4 else eps4[:1]                     # K = 1: noise map 0 of the K = 4 set (G1); G0 / D0 predictors ignore it
        t0 = C.sync()
        run = C.run_planner(core, geo, h[name], rec, eps, z0)
        t_plan = C.sync() - t0
        keys1 = run["keys1"]
        ch = {c1: C.children((c1,), geo) for c1 in keys1}
        paths = [(a, b) for a in keys1 for b in ch[a]]
        S = run["SETS"]
        V1k = [[float(S[c1][k][1].sum()) for k in range(K)] for c1 in keys1]
        d1 = [float(x) for x in run["out"]["distp1"]]
        rank = C.rank_metrics(d1, keys1, Frow["oracle"], st_meta["scene"], st_meta["start"], st_meta["t"])
        rank1 = C.rank_metrics([float(np.mean(v)) for v in V1k], keys1, Frow["oracle"], st_meta["scene"], st_meta["start"], st_meta["t"])
        cams = []
        for key in keys1:
            v, ht = run["V"][key], run["HT"][key]
            Vc = C.ViewC(key, v, ht, fr, kM, C.INST["dev"])
            if key not in tcache:
                tt = torch.as_tensor(FX.gt_first_hit(isect, Vc.c.cpu().numpy(), Vc.d.cpu().numpy(), float(Vc.tcap.max())), device=Vc.tcap.device)
                tcache[key] = (torch.where(tt <= Vc.tcap, tt, torch.full_like(tt, float("inf"))) / kM).to(v["d"].dtype)
            ts = tcache[key]
            new_true = CT.outdoor_truth_new(ts, v, geo, tree, kM)
            new_pred = torch.stack([C.novel_rays(Vc, k, geo, z0, tree)[0] for k in range(K)])
            cams.append(CT.camera_metrics(ht["t"], ht["hit_u"], ts, v, geo, kM, new_pred, new_true, 1.0))
        row = {**st_meta, "ckpt": name, "arm": name, "K": K, "ckpt_sha": core.ckpt_sha, "mode": getattr(core, "mode", "flow"), "n_legal": len(keys1),
               "keys1": [list(k) for k in keys1], "oracle_pp": [float(x) for x in Frow["oracle"]], "score": d1, "V1k": V1k,
               "rank_d1": rank, "rank_one_step": rank1, "cams": cams, "seconds_plan": t_plan,
               "predicted_views": run["out"]["d"]["predicted_views"], "peak_gb": torch.cuda.max_memory_allocated() / 2 ** 30}
        reg = {"keys_eq_F": row["keys1"] == Frow["keys1"], "paths_eq_F": [[list(a), list(b)] for a, b in paths] == Frow["paths"]}
        if name == "G4":
            ref = ref_scores(st_meta["tag"]).get(st_meta["t"])
            reg["d1_bitwise_T4V2_T0"] = (d1 == [float(x) for x in ref]) if ref is not None else None
        row["regression"] = reg
        rows.append(row)
    return rows


CT.outdoor_state = g_state


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--set", required=True, choices=("explore", "confirm"))
    ap.add_argument("--arms", default="G4,G1,G0,D0")
    ap.add_argument("--d0-ckpt", default=D0_DEFAULT)
    ap.add_argument("--d0-sha", default=None)
    ap.add_argument("--only", default=None)
    ap.add_argument("--tmax", type=int, default=None)
    ap.add_argument("--out", default=None)
    a = ap.parse_args()
    od = a.out or os.path.join(W, "states")
    claims = os.path.join(od, "_claims")
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
    arms = a.arms.split(",")
    assert set(arms) <= set(ARM_K), arms
    t0core = FX.PV3.V3Core(T0[0], dev, expect_sha=T0[1]) if {"G4", "G1"} & set(arms) else None
    mk = {"G4": lambda: t0core, "G1": lambda: t0core, "G0": lambda: GCore(T0[0], dev, "g0", T0[1]), "D0": lambda: GCore(a.d0_ckpt, dev, "d0", a.d0_sha)}
    cores = {n: mk[n]() for n in arms}
    meta = {"arms": arms, "ckpts": {n: c.ckpt_sha for n, c in cores.items()}, "d0_ckpt": a.d0_ckpt if "D0" in arms else None,
            "code_sha256": DH.code_hashes(), "host": os.uname()[1], "time": time.ctime()}
    with open(os.path.join(od, f"_run_{a.set}_{os.uname()[1]}_{os.getpid()}.json"), "w") as f:
        json.dump(meta, f, indent=1)
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
