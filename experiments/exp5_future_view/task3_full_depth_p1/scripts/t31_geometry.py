"""EXP5 Task 3 / Experiment 1 -- Stage 1 geometry smoke + fixed-state diagnostics (seen2scene env, 1 GPU per part).

States: the A1/A2 fixed states reached by the Task 2 V2-Mix closed loop (MP3D t in {0, 5, 20, 25, 50, 55, 100, 105, 140,
145}, Macarons++ t in {0, 5, 10, 15, 20, 25, 50, 55, 80, 85}: refresh steps and 5 steps after); the V2-Mix service records are replayed and TWO P1 maps are maintained side
by side with the planner's own rules (indoor: n_win = 5 region rebuilt at t % 10 == 0 + increments; outdoor: one
scene grid): Sparse-P1 (frozen m_p1.P1Fusion, B0) and FullDepth-P1 (p1_full.P1FusionFull, B1).
Per state and map: voxel counts (BAND / EMPTY / UNKNOWN, BAND-over-carved conflicts), coarse-grid statistics, and for the
current camera and every candidate of the candidate set the planner casting: t_known valid ratio, known-surface depth
vs ground truth (current camera: its own stored depth; candidates: S1 t*), pseudo-surface rate (P1 claims a surface in
front of the true first surface), UNKNOWN segments / length, the 11-channel means; timings.
Fixed-state scores of the frozen V2-Mix with the planner noise: B0 (sparse input + sparse DistP1), B1 (full + full),
B2 (full input, DistP1 novelty from the sparse map), B3 (sparse input, novelty from the full map). B0 must reproduce
the stored ablation scores bitwise (else STOP). GT (t*, oracle gains) is read only after scoring (evaluation side).
Output W3/geometry/<dataset>/part<i>.jsonl
"""
import json
import os
import sys
import time

HERE = os.path.dirname(os.path.abspath(__file__))
EXP5 = os.path.dirname(os.path.dirname(HERE))
sys.path.insert(0, os.path.join(EXP5, "ablation_study", "common"))
import ab_common as A  # noqa: E402
import s2_infer as S2  # noqa: E402

sys.path.insert(0, HERE)
import p1_full as PF  # noqa: E402

np, torch = S2.np, S2.torch
MC, G, PC, PR, RY, SCO, m_p1 = S2.MC, S2.G, S2.PC, S2.PR, S2.RY, S2.SCO, S2.m_p1
W3 = os.path.join(A.REPO, "work_dirs", "exp5_future_view", "task3_full_depth_p1")
MODES = (("s", PF._P1), ("f", PF.P1FusionFull))


def sync():
    torch.cuda.synchronize()
    return time.time()


def states_of(dataset):
    out = {}
    for st in A.load_manifest(dataset)["states"]:
        if any(e["source"] == "V2M" for e in st["sources"]):          # brief times + the t+5 states between refreshes
            out.setdefault(st["tag"], []).append(st)
    return sorted(out.items())


def run_episode(dataset, tag, sts, core, dev):
    ep = S2.episode_of(dataset, sts[0])
    rp = S2.StateReplay(dataset, ep, "V2M", {}, dev)
    want = {s["t"]: s for s in sts}
    recs, fus, region, h, eps = [], {}, None, {}, None
    tm = {m: {"step": [], "rebuild": []} for m, _ in MODES}
    rows = []
    for t in range(max(want) + 1):
        rec = MC.load_service_record(A.svc_path(dataset, "V2M", ep, t))
        assert int(rec["t"]) == t
        recs.append(rec)
        if dataset == "indoor" and t % 10 == 0:
            fr, dims, dom, win, box = rp.F.region_geometry(rp.card, rec["cam_center"])
            region = (fr, dims, dom)
            for m, cls in MODES:
                fus.pop(m, None)
                torch.cuda.empty_cache()
                t0 = sync()
                fus[m] = cls(*region, device=dev)
                for r in recs:
                    fus[m].integrate_record(r)
                tm[m]["rebuild"].append(sync() - t0)
        else:
            if not fus:
                fr, dims, dom, _ = rp.frame
                region = (fr, dims, dom)
                fus = {m: cls(*region, device=dev) for m, cls in MODES}
            for m, _ in MODES:
                t0 = sync()
                fus[m].integrate_record(rec)
                tm[m]["step"].append(sync() - t0)
        if t % 10 != 0 and t not in want:
            continue
        geos, enc_s = {}, {}
        for m, _ in MODES:
            geos[m] = PC.Geometry.from_fusion(*region, fus[m], dev)
        if t % 10 == 0:
            for m, _ in MODES:
                t0 = sync()
                h[m] = core.pred.encode(geos[m].coarse())
                enc_s[m] = sync() - t0
            eps = PR.noise(dev, A.K_MAIN, lambda kk, tr=t: RY.sha_seed(rp.seed_key(tr, kk)))
        if t in want:
            rows.append(state_row(dataset, want[t], rec, geos, fus, h, eps, core, rp.z0, enc_s))
            f2 = lambda x: "n/a" if x is None else f"{x:.2f}"                          # noqa: E731
            print(dataset, tag, t, "t_known cur", f2(rows[-1]["s"]["cur"]["t_known_valid"]), "->", f2(rows[-1]["f"]["cur"]["t_known_valid"]),
                  "| cand", f2(rows[-1]["s"]["cand"]["t_known_valid"]), "->", f2(rows[-1]["f"]["cand"]["t_known_valid"]),
                  "| repro", rows[-1]["repro_b0"], flush=True)
        del geos
    for r in rows:
        r["timing_fusion"] = {m: {"step_s": float(np.mean(v["step"])) if v["step"] else None,
                                  "rebuild_s": float(np.mean(v["rebuild"])) if v["rebuild"] else None} for m, v in tm.items()}
    return rows


def ray_block(geo, R, T, ztrue, k_M, u, trunc_rt):
    t0 = sync()
    v = geo.view(R, T)
    cast_s = sync() - t0
    pr = v["pr"]
    dz = v["dz"].double().cpu().numpy()
    zk = pr["t_known"].double().cpu().numpy() * k_M * dz
    gt = np.isfinite(ztrue) & (ztrue < MC.SENSOR_RANGE)
    known = np.isfinite(zk)
    kg = known & gt
    tol = trunc_rt + 0.01 * np.where(gt, ztrue, 0.0)
    err = zk - ztrue
    st16 = G.stats_fp16(pr, v["tcap"]).float().cpu().numpy()
    cls = pr["cls"].cpu().numpy()
    return {"rays": int(len(zk)), "gt_rays": int(gt.sum()), "t_known_valid": float(known[gt].mean()) if gt.any() else None,
            "t_known_valid_all": float(known.mean()),
            "match": float(np.mean(np.abs(err[kg]) <= tol[kg])) if kg.any() else None,
            "pseudo_front": float(np.mean(err[kg] < -tol[kg])) if kg.any() else None,
            "behind_unobserved": float(np.mean(err[kg] > tol[kg])) if kg.any() else None,
            "med_abs_err": float(np.median(np.abs(err[kg])) / u) if kg.any() else None,
            "n_seg_mean": float(pr["n_seg"].float().mean()), "lu_mean": float((pr["n_u"].double().cpu().numpy() * G.H_STEP * k_M).mean() / u),
            "cls_frac": [float(np.mean(cls == c)) for c in (G.CLS_UNKNOWN, G.CLS_KNOWN, G.CLS_EMPTY)],
            "ch_mean": [float(x) for x in st16.mean(1)], "cast_s": cast_s}


def mean_blocks(bs):
    keys = [k for k in bs[0] if not isinstance(bs[0][k], list)]
    out = {}
    for k in keys:
        v = [b[k] for b in bs if b[k] is not None]
        out[k] = float(np.mean(v)) if v else None
    for k in ("cls_frac", "ch_mean"):
        out[k] = np.mean([b[k] for b in bs], 0).tolist()
    return out


def distp1_from_hits(views, hits, geo, k_M, z0):
    out = []
    for v, hh in zip(views, hits):
        ws = []
        for k in range(hh["t"].shape[0]):
            _, w = SCO.new_set_from_depth(hh["t"][k], v["o"], v["d"], v["dz"], k_M, geo.dims, geo.domain, geo.caster.band_dil, z0)
            ws.append(float(w.sum()))
        out.append(ws)
    return out


def state_row(dataset, st, rec, geos, fus, h, eps, core, z0, enc_s):
    u = 7.0 if dataset == "indoor" else 1.0
    k_M = float(geos["s"].fr.k_M)
    trunc_rt = MC.TRUNC * k_M
    row = {"sid": st["sid"], "scene": st["scene"], "start": st["start"], "t": st["t"], "tag": st["tag"], "encode_s": enc_s}
    # ---- planner side first: scores (no GT read yet)
    outs, views = {}, {}
    for m in ("s", "f"):
        t0 = sync()
        outs[m], tmm, views[m] = PC.score_state(core, geos[m], h[m], rec["legal_R"], rec["legal_T"], eps, z0, keep_hits=True)
        row.setdefault("score_timing", {})[m] = dict(tmm, total_s=sync() - t0)
    sc = {"B0": outs["s"]["distp1_k"], "B1": outs["f"]["distp1_k"],
          "B2": distp1_from_hits(views["f"], outs["f"]["hits"], geos["s"], k_M, z0),
          "B3": distp1_from_hits(views["s"], outs["s"]["hits"], geos["f"], k_M, z0)}
    row["scores_k"] = sc
    row["scores"] = {k: [float(np.mean(x)) for x in v] for k, v in sc.items()}
    stored = json.load(open(os.path.join(A.state_dir(dataset, st["sid"]), "v2.json")))["models"]["V2M"]["distp1_k"]
    row["repro_b0"] = bool(np.asarray(sc["B0"]).tolist() == stored)
    if not row["repro_b0"]:
        raise SystemExit(f"STOP: Sparse-P1 rebuild does not reproduce the stored V2-Mix scores ({dataset} {st['sid']})")
    # ---- map statistics
    for m in ("s", "f"):
        s = geos[m].struct
        dom = geos[m].domain
        fu = fus[m]
        coarse = geos[m].coarse().float() / 255.0
        cd = coarse[2] > 0
        row[m] = {"band": int((s == G.BAND).sum()), "empty": int((s == G.EMPTY).sum()), "unknown": int(((s == G.UNKNOWN) & dom).sum()),
                  "band_over_carved": int(((fu.cnt > 0) & fu.empty).sum()),
                  "coarse_empty_mean": float(coarse[0][cd].mean()), "coarse_band_mean": float(coarse[1][cd].mean()),
                  "coarse_band_cells": float((coarse[1][cd] > 0).float().mean())}
    # ---- evaluation side: GT depth of the rays
    orc = np.load(os.path.join(A.state_dir(dataset, st["sid"]), "oracle.npz"))
    uu, vv = MC.scoring_pixel_grid()
    zcur = np.asarray(rec["zbuf"], np.float64)[np.round(vv).astype(int), np.round(uu).astype(int)]
    zcur = np.where(np.asarray(rec["mask"], bool)[np.round(vv).astype(int), np.round(uu).astype(int)], zcur, np.inf)
    for m in ("s", "f"):
        row[m]["cur"] = ray_block(geos[m], rec["R"], rec["T"], zcur, k_M, u, trunc_rt)
        cb = []
        for j in st["cand"]:
            v = geos[m].view(rec["legal_R"][j], rec["legal_T"][j])
            zt = orc["tstar"][j].astype(np.float64) * k_M * v["dz"].double().cpu().numpy()
            cb.append(ray_block(geos[m], rec["legal_R"][j], rec["legal_T"][j], zt, k_M, u, trunc_rt))
        row[m]["cand"] = mean_blocks(cb)
    row["s"]["peak_gb"] = torch.cuda.max_memory_allocated() / 2 ** 30
    return row


def main():
    import argparse
    ap = argparse.ArgumentParser()
    ap.add_argument("--dataset", choices=["indoor", "outdoor"], required=True)
    ap.add_argument("--part", default="0/1")
    a = ap.parse_args()
    pi, pn = map(int, a.part.split("/"))
    torch.use_deterministic_algorithms(True, warn_only=True)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    dev = torch.device("cuda")
    core = S2.load_cores(dev)["V2M"]
    od = os.path.join(W3, "geometry", a.dataset)
    os.makedirs(od, exist_ok=True)
    p = os.path.join(od, f"part{pi}.jsonl")
    done = {json.loads(x)["sid"] for x in open(p)} if os.path.exists(p) else set()
    for tag, sts in states_of(a.dataset)[pi::pn]:
        sts = [s for s in sts if s["sid"] not in done]
        if not sts:
            continue
        rows = run_episode(a.dataset, tag, sts, core, dev)
        with open(p, "a") as f:
            for r in rows:
                f.write(json.dumps(r) + "\n")
        torch.cuda.empty_cache()


if __name__ == "__main__":
    main()
