"""EXP5 Task 1.3 fixed-state diagnostics (seen2scene env, 1 GPU per part; protocol §4.1, §4.6, §4.8). Descriptive only:
nothing here selects or tunes anything (delta / lambda come from S-c, t13_calibrate.py).

  tests : T-S1 -- on the 73 S-a states, every legal pose x the 4 archived T1-C worlds: new_set_from_depth(cast t) ==
          abl_score.new_set (ids and weights bitwise), and the per-state DistP1 one-step scores reproduce the archived
          abl_1 fixed-state values exactly
  sa    : V2 on the S-a states (K = 4 seeds exp5-V2|sa|..): DistP1 (K4 / K1), U_raw / DistU (all deltas), U_flow,
          U_feat, Frontier DistP1, T-S2 (decoded hits only in UNKNOWN voxels); evaluation side afterwards: true-depth
          DistP1 from dev_tstar
  sb    : the same on the B0 closed-loop states (Dev-3, 15 trajectories x 100 steps; --part over trajectories)
  g1es  : G1 future-depth ES. (1) reproduction: frame_scores_x fed with the casts of the archived frozen / T1-C worlds
          must reproduce g1_common.frame_scores and the archived per-state union ES (rel 1e-9); (2) only then V2 on the
          same 48 states and future frames (integer-pixel rays = scoring grid shifted by (0.5, 0.5))
Output W/fixed_state/<mode>/...
"""
import argparse
import json
import os
import sys
import time

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
import v2_frozen as VF  # noqa: E402
import v2_common as LC  # noqa: E402

import numpy as np  # noqa: E402
import torch  # noqa: E402

import m_common as MC  # noqa: E402
SCF = VF.load_cfg("scoring_v1.json")
DELTAS = tuple(SCF["U"]["delta_grid_vox"])
OUT = os.path.join(VF.WORK, "fixed_state")


def seed(key):
    from seen2scene.futureview import rays as RY
    return RY.sha_seed(key)


# ---------------------------------------------------------------------------------------------------- T-S1
def mode_tests(part):
    import m_score as MS
    import abl_score as SC
    import abl_common as AC
    import t3_common as C3
    from seen2scene.futureview import scoring as SCO
    dev = torch.device("cuda")
    pi, pn = part
    arch = {}
    for p in sorted(os.listdir(os.path.join(AC.ABL, "_fixed_state"))):
        if p.startswith("part_") and p.endswith(".json"):
            for r in json.load(open(os.path.join(AC.ABL, "_fixed_state", p))):
                arch[(r["scene"], r["start"], r["t"])] = r
    rows = []
    for s, k, t in C3.STATES[pi::pn]:
        tb = time.time()
        st = C3.planner_state(s, k, t, dev)
        worlds = [torch.as_tensor(w).to(dev) for w in C3.load_worlds("t1c", s, k, t)]
        caster = MS.SceneCaster(st["frame"], st["dims"], st["dom"], st["vol"], device=dev)
        z0 = AC.z0_of(s)
        n_eq, n_all, dist = 0, 0, []
        for R, T in zip(st["legal_R"], st["legal_T"]):
            rays = SC.pose_rays(caster, np.asarray(R, np.float64), np.asarray(T, np.float64))
            o, d, tcap, dz = rays
            ws = []
            for w in worlds:
                vol = caster.cast_volume(w)
                ids_r, w_r = SC.new_set(caster, vol, rays, z0)
                r = caster.cast(vol, o, d, tcap)
                ids, wv = SCO.new_set_from_depth(r["t"], o, d, dz, st["frame"].k_M, caster.dims, caster.domain, caster.band_dil, z0)
                n_all += 1
                n_eq += int(torch.equal(ids, ids_r) and torch.equal(wv, w_r))
                ws.append(float(wv.sum()))
                del vol
            dist.append(float(np.mean(ws)))
        a = arch[(s, k, t)]
        row = {"scene": s, "start": k, "t": t, "n_pose_world": n_all, "n_bitwise_equal": n_eq,
               "distp1_equal_archived": dist == a["distp1"], "seconds": time.time() - tb}
        rows.append(row)
        print(row, flush=True)
        del worlds, caster, st
        torch.cuda.empty_cache()
    od = os.path.join(OUT, "checks")
    os.makedirs(od, exist_ok=True)
    json.dump({"rows": rows, "pass": all(r["n_bitwise_equal"] == r["n_pose_world"] and r["distp1_equal_archived"] for r in rows)},
              open(os.path.join(od, f"TS1_part{pi}.json"), "w"), indent=1)


# ---------------------------------------------------------------------------------------------------- V2 on a state
def score_v2(core, geo, h, rec, eps, z0, want_feat, tstar=None):
    """Planner-side V2 scores (+ Frontier) of one state; afterwards (evaluation side) true-depth DistP1 from tstar."""
    import v2_plan_core as PC
    from seen2scene.futureview import geometry as G
    from seen2scene.futureview import scoring as SCO
    out, tm, views = PC.score_state(core, geo, h, rec["legal_R"], rec["legal_T"], eps, z0, deltas=DELTAS,
                                    want_flow=True, want_feat=want_feat, keep_hits=True)
    front, viol, nh = [], 0, 0
    for v, hh in zip(views, out["hits"]):
        pr = v["pr"]
        tf = torch.where(pr["n_u"] > 0, pr["d_lo"], pr["t_known"])
        _, wf = SCO.new_set_from_depth(tf, v["o"], v["d"], v["dz"], geo.fr.k_M, geo.dims, geo.domain, geo.caster.band_dil, z0)
        front.append(float(wf.sum()))
        hu = hh["hit_u"]
        p = v["o"][None, None] + v["d"][None] * torch.where(hu, hh["t"], torch.zeros_like(hh["t"]))[..., None]
        ijk = torch.floor(p / G.VS).long()[hu]
        nh += len(ijk)
        if len(ijk):
            stt = geo.struct[ijk[:, 0], ijk[:, 1], ijk[:, 2]]
            viol += int(((stt != G.UNKNOWN) | ~geo.domain[ijk[:, 0], ijk[:, 1], ijk[:, 2]]).sum())
    row = {k: v for k, v in out.items() if k != "hits"}
    row.update({"frontier_distp1": front, "TS2_violations": viol, "TS2_hits": nh, "timing": tm,
                "keys": [str([int(x) for x in idx]) for idx in rec["legal_idx"]]})
    if tstar is not None:
        tru = []
        for v, ts in zip(views, tstar):
            ts = torch.as_tensor(ts, device=v["d"].device, dtype=v["d"].dtype)
            tt = torch.where(ts <= v["pr"]["tmax"], ts, torch.full_like(ts, float("inf")))
            _, wt = SCO.new_set_from_depth(tt, v["o"], v["d"], v["dz"], geo.fr.k_M, geo.dims, geo.domain, geo.caster.band_dil, z0)
            tru.append(float(wt.sum()))
        row["truedepth_distp1"] = tru
    return row


def load_core(dev):
    import v2_plan_core as PC
    ck = VF.frozen_values()["checkpoint"]
    return PC.Core(os.path.join(VF.REPO, ck["path"]), dev, expect_sha=ck["sha256"])


def dev_tstar(setname, s, k):
    p = os.path.join(OUT, "dev_tstar", f"{setname}_{s}_s{k}.npz")
    return np.load(p) if os.path.exists(p) else None


def mode_sa(part):
    import abl_common as AC
    import t3_common as C3
    import t5_diagnostic as TD
    import v2_plan_core as PC
    from seen2scene.futureview import predictor as PR
    dev = torch.device("cuda")
    core = load_core(dev)
    pi, pn = part
    od = os.path.join(OUT, "sa")
    os.makedirs(od, exist_ok=True)
    f = open(os.path.join(od, f"part{pi}.jsonl"), "w")
    for s, k, t in C3.STATES[pi::pn]:
        tb = time.time()
        st = C3.planner_state(s, k, t, dev)
        geo = PC.Geometry(st["frame"], st["dims"], st["dom"], st["vol"], st["structure"], dev)
        h = core.pred.encode(geo.coarse())
        rec = MC.load_service_record(TD.svc_path(s, k, t))
        eps = PR.noise(dev, 4, lambda kk: seed(f"exp5-V2|sa|{s}|s{k}|t{t}|k{kk}"))
        dt = dev_tstar("sa", s, k)
        ts = dt["tstar"][dt["t"] == t] if dt is not None else None
        row = score_v2(core, geo, h, rec, eps, AC.z0_of(s), True, ts)
        row.update({"scene": s, "start": k, "t": t, "seconds": time.time() - tb})
        f.write(json.dumps(row, default=float) + "\n")
        f.flush()
        print(s, k, t, f"{row['seconds']:.1f}s TS2 viol {row['TS2_violations']}", flush=True)
        del geo, h, st
        torch.cuda.empty_cache()


def mode_sb(part):
    import m_p1
    import t5_diagnostic as TD
    import v2_plan_core as PC
    from seen2scene.futureview import predictor as PR
    dev = torch.device("cuda")
    core = load_core(dev)
    pi, pn = part
    od = os.path.join(OUT, "sb")
    os.makedirs(od, exist_ok=True)
    items = [(s, k) for s in LC.SCENES for k in LC.STARTS][pi::pn]
    for s, k in items:
        p = os.path.join(od, f"sb_{s}_s{k}.jsonl")
        if os.path.exists(p):
            continue
        fr, dims, dom, win = TD.scene_geometry(s)
        fu = m_p1.P1Fusion(fr, dims, dom, device=dev)
        svc = os.path.join(LC.B0_ROOT, "svc", f"A2_{s}_s{k}_r0")
        dt = dev_tstar("sb", s, k)
        rows = []
        for t in range(LC.MOVES):
            rec = MC.load_service_record(os.path.join(svc, f"svc_{s}_s{k}_t{t}.npz"))
            fu.integrate_record(rec)
            if len(rec["legal_R"]) == 0:
                rows.append({"scene": s, "start": k, "t": t, "n": 0})
                continue
            geo = PC.Geometry.from_fusion(fr, dims, dom, fu, dev)
            h = core.pred.encode(geo.coarse())
            eps = PR.noise(dev, 4, lambda kk: seed(f"exp5-V2|sb|{s}|s{k}|t{t}|k{kk}"))
            ts = dt["tstar"][dt["t"] == t] if dt is not None else None
            row = score_v2(core, geo, h, rec, eps, LC.Z0[s], False, ts)
            row.update({"scene": s, "start": k, "t": t})
            rows.append(row)
            del geo, h
        with open(p + ".tmp", "w") as f:
            for r in rows:
                f.write(json.dumps(r, default=float) + "\n")
        os.replace(p + ".tmp", p)
        print(f"sb {s}/s{k}: {len(rows)} states", flush=True)
        del fu
        torch.cuda.empty_cache()


# ---------------------------------------------------------------------------------------------------- G1 ES
def frame_scores_x(caster, samples, rays):
    """g1_common.frame_scores with the K casts replaced by given samples [(t [R], hit [R])]: identical ray set, valid /
    blocked / out-of-domain rules, observation y, no-hit -> segment end, voxel units, fair ES / CRPS / Brier / FC."""
    import g1_common as GC
    o_np, d_np, tcap_np, tobs_np = rays
    dev, dt = caster.dev, caster.dt
    o = torch.as_tensor(o_np, dtype=dt, device=dev)
    d = torch.as_tensor(d_np, dtype=dt, device=dev)
    tcap = torch.as_tensor(tcap_np, dtype=dt, device=dev)
    y = torch.as_tensor(tobs_np, dtype=dt, device=dev)
    tmin, tmax = caster._entry_exit(o[None].expand_as(d), d)
    tmax = torch.minimum(tmax, tcap)
    valid = tmax > tmin
    blocked = y < tmin - MC.VS
    y_in = torch.isfinite(y) & (y <= tmax)
    p = o[None] + d * torch.where(y_in, y, torch.zeros_like(y))[:, None]
    ijk = torch.floor(p / MC.VS).long()
    for a in range(3):
        ijk[:, a] = ijk[:, a].clamp(0, caster.dims[a] - 1)
    in_dom = caster.domain[ijk[:, 0], ijk[:, 1], ijk[:, 2]]
    keep = valid & ~blocked & ~(y_in & ~in_dom)
    n = int(keep.sum())
    if n < GC.MIN_RAYS:
        return None
    y_hit = y_in & keep
    yc = torch.where(y_in, y, tmax)
    novel = keep & (~y_hit | ~caster.band_dil[ijk[:, 0], ijk[:, 1], ijk[:, 2]])
    X = torch.stack([torch.where(hh, tt, tmax) for tt, hh in samples])[:, keep].double() / MC.VS
    Hh = torch.stack([hh for _, hh in samples])[:, keep].double()
    yv = yc[keep].double() / MC.VS
    Kk = X.shape[0]
    rms = lambda a: torch.sqrt((a ** 2).mean(-1))  # noqa: E731
    es = rms(X - yv[None]).mean()
    crps = (X - yv[None]).abs().mean(0)
    pair_es, pair_crps = 0.0, torch.zeros_like(yv)
    for a in range(Kk):
        for b in range(Kk):
            if a != b:
                pair_es = pair_es + rms(X[a] - X[b])
                pair_crps = pair_crps + (X[a] - X[b]).abs()
    es = es - pair_es / (2 * Kk * (Kk - 1))
    crps = (crps - pair_crps / (2 * Kk * (Kk - 1))).mean()
    ph = Hh.mean(0)
    yh = y_hit[keep].double()
    nov = novel[keep]
    brier = ((ph - yh) ** 2)
    fc = ((ph == 0) & (yh == 1)) | ((ph == 1) & (yh == 0))
    return {"n_rays": n, "union": {"es": float(es), "crps": float(crps), "brier": float(brier.mean()),
                                   "brier_novel": float(brier[nov].mean()) if nov.any() else None,
                                   "fc_novel": float(fc[nov].double().mean()) if nov.any() else None,
                                   "fc_all": float(fc.double().mean())}}


def mode_g1es(part):
    import g1_common as GC
    import g1_eval as GE1
    import m_score as MS
    import t3_common as C3
    import t5_diagnostic as TD
    import v2_plan_core as PC
    from seen2scene.futureview import geometry as G
    from seen2scene.futureview import predictor as PR
    dev = torch.device("cuda")
    pi, pn = part
    od = os.path.join(OUT, "g1es")
    os.makedirs(od, exist_ok=True)
    core = load_core(dev) if os.path.exists(os.path.join(od, "REPRODUCTION_PASS")) or "--v2" in sys.argv else None
    rows = []
    for s, k, t in GC.STATES_B_FULL[pi::pn]:
        tb = time.time()
        st = C3.planner_state(s, k, t, dev)
        fr = st["frame"]
        caster = MS.SceneCaster(fr, st["dims"], st["dom"], st["vol"], device=dev)
        fut = [MC.load_service_record(TD.svc_path(s, k, tt)) for tt in range(t + 1, min(t + GC.HORIZON, GC.T_LAST) + 1)]
        rays = [GC.future_rays(r, fr) for r in fut]
        ev = json.load(open(os.path.join(GC.WORK, "g1", "eval", f"eval_{GC.tag(s, k, t)}.json")))
        row = {"scene": s, "start": k, "t": t, "n_frames": len(fut)}
        for model in ("frozen", "C2000"):
            worlds = GE1.load_worlds(s, k, t, model, dev)
            vols = [caster.cast_volume(w) for w in worlds]
            es_x, es_g, same = [], [], True
            for ry in rays:
                ref = GC.frame_scores(caster, vols, ry)
                o = torch.as_tensor(ry[0], dtype=caster.dt, device=dev)
                d = torch.as_tensor(ry[1], dtype=caster.dt, device=dev)
                tc = torch.as_tensor(ry[2], dtype=caster.dt, device=dev)
                smp = []
                for vol in vols:
                    c = caster.cast(vol, o, d, tc)
                    smp.append((c["t"], c["hit"]))
                mine = frame_scores_x(caster, smp, ry)
                if (ref is None) != (mine is None):
                    same = False
                    continue
                if ref is None:
                    continue
                same &= ref["union"] == mine["union"] and ref["n_rays"] == mine["n_rays"]
                es_x.append(mine["union"]["es"])
            arch = ev[model]["union_es"]
            mx = float(np.mean(es_x)) if es_x else None
            row[f"repro_{model}"] = {"frames_equal_frame_scores": same, "es": mx, "es_archived": arch,
                                     "equal_archived": (mx is None and arch is None) or (mx is not None and arch is not None and abs(mx - arch) <= 1e-9 * max(1.0, abs(arch)))}
            del worlds, vols
        if core is not None:
            geo = PC.Geometry(fr, st["dims"], st["dom"], st["vol"], st["structure"], dev)
            h = core.pred.encode(geo.coarse())
            eps = PR.noise(dev, 4, lambda kk: seed(f"exp5-V2|g1|{s}|s{k}|t{t}|k{kk}"))
            es_v, fr_rows = [], []
            for j, (r_, ry) in enumerate(zip(fut, rays)):
                v = geo.view(r_["R"], r_["T"], (0.5, 0.5))
                same_rays = (torch.equal(v["o"], torch.as_tensor(ry[0], dtype=caster.dt, device=dev)) and
                             torch.equal(v["d"], torch.as_tensor(ry[1], dtype=caster.dt, device=dev)) and
                             torch.equal(v["tcap"], torch.as_tensor(ry[2], dtype=caster.dt, device=dev)))
                if not same_rays:
                    raise VF.InvalidRun("INVALID: V2 query rays differ from the G1 future rays")
                cond, m_unk = core.pred.cond(h, [(v["o"], v["d"], v["tcap"], v["pr"])])
                b, y = core.pred.sample(cond, m_unk, eps)
                t_dec, hit_u = G.decode(b[0], y[0], v["o"], v["d"], v["pr"], geo.struct, geo.domain)
                smp = [(t_dec[kk], torch.isfinite(t_dec[kk])) for kk in range(4)]
                sc = frame_scores_x(caster, smp, ry)
                if sc is not None:
                    es_v.append(sc["union"]["es"])
                    fr_rows.append({"t_frame": t + 1 + j, **sc["union"]})
            row["v2"] = {"es": float(np.mean(es_v)) if es_v else None, "frames": fr_rows}
            del geo, h
        row["seconds"] = time.time() - tb
        rows.append(row)
        print(s, k, t, {m: row[f"repro_{m}"]["equal_archived"] for m in ("frozen", "C2000")},
              (row.get("v2") or {}).get("es"), f"{row['seconds']:.0f}s", flush=True)
        del caster, st
        torch.cuda.empty_cache()
    json.dump(rows, open(os.path.join(od, f"part{pi}{'_v2' if core is not None else ''}.json"), "w"), indent=1, default=float)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("mode", choices=["tests", "sa", "sb", "g1es"])
    ap.add_argument("--part", default="0/1")
    ap.add_argument("--v2", action="store_true", help="g1es: also score V2 (after the reproduction passed)")
    a = ap.parse_args()
    VF.verify_frozen()
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    part = tuple(int(x) for x in a.part.split("/"))
    {"tests": mode_tests, "sa": mode_sa, "sb": mode_sb, "g1es": mode_g1es}[a.mode](part)


if __name__ == "__main__":
    main()
