"""EXP5 Ablation S2 -- V2 Original / V2-Mix on the fixed states (seen2scene env, 1 GPU).

Planner side (records only, no GT): per stored source episode the planner state of the Task 2 V2 planners is rebuilt
exactly (indoor: t06_planner V2K4 with the Task 2 n_win = 5 region, R10 refresh, P1 of records 0..t, encoder input of
the last refresh t_r, noise seed "exp5-V2|task0cl|<tag>|t<t_r>|k<k>"; outdoor: v2_planner, whole-scene frame, R10,
seed closed_loop_v1 seed_key(scene, start, rep 0, t_r, k)). Both models score the SAME geometry with the SAME noise
(v2_plan_core.score_state, K = 4, 10 Euler steps, DistP1 frozen). For V2 source episodes the scores of the source
model must reproduce the online planner log (reported as max |diff|).
Evaluation side (afterwards, offline diagnostics only): t* of S1 -> GT-depth DistP1 (T1.3 truedepth_distp1 rule),
labels (futureview.geometry.encode_labels) -> per candidate prediction metrics (t12_train.eval_samples formulas per view),
per sample hit / depth statistics and K-sample diversity.
mode main : W/states/<dataset>/<sid>/v2.npz (+ v2.json)
mode kabl : K in {1, 4, 8, 16} with 4 independent seed sets of 16 samples (seed "exp5-ABL|K|<sid>|r<r>|k<k>")
            -> W/states/<dataset>/<sid>/kabl.npz
Usage: python s2_infer.py --dataset indoor|outdoor --mode main|kabl --part i/n
"""
import argparse
import json
import os
import sys
import time

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
import ab_common as A  # noqa: E402

A.add_paths()
os.environ.setdefault("CUBLAS_WORKSPACE_CONFIG", ":4096:8")
import v2_frozen as VF  # noqa: E402
import v2_common as LC  # noqa: E402
import t2_common as C2  # noqa: E402

import numpy as np  # noqa: E402
import torch  # noqa: E402

import m_common as MC  # noqa: E402
import m_p1  # noqa: E402
import v2_plan_core as PC  # noqa: E402
from seen2scene.futureview import geometry as G  # noqa: E402
from seen2scene.futureview import predictor as PR  # noqa: E402
from seen2scene.futureview import rays as RY  # noqa: E402
from seen2scene.futureview import scoring as SCO  # noqa: E402


# ---------------------------------------------------------------------------------------------------- planner state
class StateReplay:
    """Yields (t, rec, geo, h{model}, eps) at the requested times of one stored episode, as the Task 2 planners."""

    def __init__(self, dataset, ep, src, cores, dev):
        self.dataset, self.ep, self.src, self.cores, self.dev = dataset, ep, src, cores, dev
        if dataset == "indoor":
            import mp3d_frame as F
            F.region_box_m = lambda card, center_rt: C2.region_box_m(card, center_rt, C2.N_WIN)
            self.F = F
            self.card = F.load_card(ep["scene"])
            self.z0 = F.z0_of_card(self.card)
        else:
            import abl_common as AC
            import t5_diagnostic as TD
            self.frame = TD.scene_geometry(ep["scene"])
            self.z0 = AC.z0_of(ep["scene"])

    def seed_key(self, t_r, k):
        if self.dataset == "indoor":
            return f"exp5-V2|task0cl|{self.ep['tag']}|t{t_r}|k{k}"
        return LC.seed_key(self.ep["scene"], self.ep["start"], 0, t_r, k)

    def run(self, times):
        recs, fu, region, h, eps = [], None, None, None, None
        for t in range(max(times) + 1):
            rec = MC.load_service_record(A.svc_path(self.dataset, self.src, self.ep, t))
            assert int(rec["t"]) == t
            recs.append(rec)
            if self.dataset == "indoor":
                if t % 10 == 0:
                    fr, dims, dom, win, box = self.F.region_geometry(self.card, rec["cam_center"])
                    region = (fr, dims, dom)
                    fu = m_p1.P1Fusion(fr, dims, dom, device=self.dev)
                    for r in recs:
                        fu.integrate_record(r)
                else:
                    fu.integrate_record(rec)
            else:
                if fu is None:
                    fr, dims, dom, _ = self.frame
                    region = (fr, dims, dom)
                    fu = m_p1.P1Fusion(fr, dims, dom, device=self.dev)
                fu.integrate_record(rec)
            if t % 10 == 0 or t in times:
                geo = PC.Geometry.from_fusion(*region, fu, self.dev)
                if t % 10 == 0:
                    coarse = geo.coarse()
                    h = {m: c.pred.encode(coarse) for m, c in self.cores.items()}
                    eps = PR.noise(self.dev, A.K_MAIN, lambda kk, tr=t: RY.sha_seed(self.seed_key(tr, kk)))
                if t in times:
                    yield t, rec, geo, h, eps
                del geo


# ---------------------------------------------------------------------------------------------------- metrics
def ray_points(v, t, mask, k_M):
    """3D points (runtime units, frame coordinates) of the rays in mask at distance t [R]."""
    return (v["o"][None] + v["d"][mask] * t[mask][:, None]) * float(k_M)


def chamfer(a, b):
    if len(a) == 0 or len(b) == 0:
        return None
    da = torch.cdist(a[None].float(), b[None].float())[0]
    return float(0.5 * (da.min(1).values.mean() + da.min(0).values.mean()))


@torch.no_grad()
def candidate_metrics(v, hh, ts, geo, z0, k_M):
    """Evaluation-side metrics of one candidate view: labels from t*, prediction metrics of the K samples (formulas of
    t12_train.eval_samples restricted to this view), per-sample statistics, diversity, GT-depth DistP1."""
    pr = v["pr"]
    K = hh["t"].shape[0]
    ts = torch.as_tensor(ts, device=v["d"].device, dtype=v["d"].dtype)
    tt = torch.where(ts <= pr["tmax"], ts, torch.full_like(ts, float("inf")))
    ids_g, w_g = SCO.new_set_from_depth(tt, v["o"], v["d"], v["dz"], k_M, geo.dims, geo.domain, geo.caster.band_dil, z0)
    new_g, _, hit_g = SCO.new_mask_from_depth(tt, v["o"], v["d"], geo.dims, geo.domain, geo.caster.band_dil)
    lab = G.encode_labels(v["o"], v["d"], pr, ts, v["dz"], k_M, geo.struct, geo.domain)
    m_unk = pr["cls"] == G.CLS_UNKNOWN
    lm = m_unk & (lab["b"] != 0)
    cat = lab["cat"]
    lab_hit = (cat == G.CAT["HIT"]) & lm
    nohit = (cat == G.CAT["NOHIT"]) & lm
    b, y = hh["b"].float(), hh["y"].float()
    hit = (b > 0) & (y < 1.0)                                              # [K, R] (eval_samples definition)
    p = hit.float().mean(0)
    s = ((y + 1) / 2).clamp(0, 1)
    lu = (pr["n_u"].to(v["tcap"].dtype) * G.H_STEP / v["tcap"]).clamp(0, 1) * v["tcap"] / G.VS
    ell = torch.where(hit, s, torch.ones_like(s)) * lu
    s_lab = ((lab["y"].float() + 1) / 2).clamp(0, 1)
    ell_t = torch.where(lab_hit, s_lab, torch.ones_like(s_lab)) * lu
    out = {"n_rays": int(len(m_unk)), "n_unknown_rays": int(m_unk.sum()), "n_label_rays": int(lm.sum()),
           "n_label_hit": int(lab_hit.sum()), "n_label_nohit": int(nohit.sum()),
           "gtd_distp1": float(w_g.sum()), "gtd_new_vox": int(len(ids_g)), "gtd_new_rays": int(new_g.sum()),
           "gtd_hit_rays": int(hit_g.sum())}
    if lm.any():
        rms = lambda a: torch.sqrt((a[..., lm] ** 2).mean(-1))          # noqa: E731
        e1 = rms(ell - ell_t[None]).mean()
        pair = sum(rms(ell[i] - ell[j]) for i in range(K) for j in range(K) if i != j)
        out["fair_es"] = float(e1 - pair / (2 * K * (K - 1)))
        out["brier"] = float(((p - lab_hit.float()) ** 2)[lm].mean())
        out["pred_hit_rate"] = float(p[lm].mean())
        out["label_hit_rate"] = float(lab_hit[lm].float().mean())
        pd = torch.stack([(ell[i] - ell[j]).abs() for i in range(K) for j in range(i + 1, K)]).mean(0)
        out["spread_vox"] = float(pd[lm].mean())
        dis = (hit[:, lm].any(0) & ~hit[:, lm].all(0)).float()
        out["hit_disagree_rays"] = float(dis.mean())
        out["hit_disagree_pair"] = float(np.mean([float((hit[i, lm] != hit[j, lm]).float().mean()) for i in range(K) for j in range(i + 1, K)]))
    out["false_surface"] = float(p[nohit].mean()) if nohit.any() else None
    out["missed_surface"] = float(1 - p[lab_hit].mean()) if lab_hit.any() else None
    # decoded samples (the scored depths): per-sample categories, errors, new sets
    t_dec, hit_u = hh["t"], hh["hit_u"]
    valid = pr["valid"]
    per = []
    news, pts = [], []
    for k in range(K):
        new_k, flat_k, _ = SCO.new_mask_from_depth(t_dec[k], v["o"], v["d"], geo.dims, geo.domain, geo.caster.band_dil)
        ids_k, w_k = SCO.new_set_from_depth(t_dec[k], v["o"], v["d"], v["dz"], k_M, geo.dims, geo.domain, geo.caster.band_dil, z0)
        fin = torch.isfinite(t_dec[k])
        both = hit_u[k] & lab_hit
        rel = ((t_dec[k][both] - ts[both]).abs() / ts[both]) if both.any() else None
        ab = ((t_dec[k][both] - ts[both]).abs() * float(k_M)) if both.any() else None
        hk = hit_u[k]
        per.append({"distp1": float(w_k.sum()), "new_vox": int(len(ids_k)), "new_rays": int(new_k.sum()),
                    "pred_unknown_hit": int(hk.sum()), "pred_unknown_hit_not_new": int((hk & ~new_k).sum()),
                    "known_hit": int((fin & ~hk).sum()), "no_hit": int((~fin & valid).sum()), "invalid": int((~valid).sum()),
                    "false_surface": float(hk[nohit].float().mean()) if nohit.any() else None,
                    "missed_surface": float((~hk[lab_hit]).float().mean()) if lab_hit.any() else None,
                    "precision_new": float(lab_hit[hk & lm].float().mean()) if (hk & lm).any() else None,
                    "depth_rel_err": float(rel.mean()) if rel is not None else None,
                    "depth_abs_err_rt": float(ab.mean()) if ab is not None else None,
                    "n_cohit": int(both.sum())})
        news.append(set(ids_k.tolist()))
        pts.append(ray_points(v, t_dec[k], new_k, k_M))
    jac, chf, dd = [], [], []
    for i in range(K):
        for j in range(i + 1, K):
            u = news[i] | news[j]
            jac.append(1.0 - len(news[i] & news[j]) / len(u) if u else None)
            chf.append(chamfer(pts[i], pts[j]))
            both = hit_u[i] & hit_u[j]
            if both.any():
                dd.append(float(((t_dec[i][both] - t_dec[j][both]).abs() * float(k_M)).mean()))
    clean = lambda xs: [x for x in xs if x is not None]                  # noqa: E731
    out["div_new_vox_jaccard"] = float(np.mean(clean(jac))) if clean(jac) else None
    out["div_new_surface_chamfer_rt"] = float(np.mean(clean(chf))) if clean(chf) else None
    out["div_cond_depth_rt"] = float(np.mean(dd)) if dd else None
    out["n_pairs_new_empty_both"] = int(sum(1 for x in jac if x is None))
    out["per_sample"] = per
    gpts = ray_points(v, tt, new_g, k_M)
    out["chamfer_to_gt_new_rt"] = [chamfer(pp, gpts) for pp in pts]
    arrays = {"lab_cat": cat.to(torch.uint8), "lab_b": lab["b"].to(torch.int8), "m_unk": m_unk, "lm": lm, "gt_new": new_g,
              "valid": valid}
    return out, arrays


# ---------------------------------------------------------------------------------------------------- modes
def load_cores(dev):
    cores = {}
    for m, ck in A.MODELS.items():
        cores[m] = PC.Core(os.path.join(A.REPO, ck["path"]), dev, expect_sha=ck["sha256"])
    return cores


def groups(man):
    g = {}
    for st in man["states"]:
        g.setdefault((st["sources"][0]["source"], st["tag"]), []).append(st)
    return sorted(g.items())


def episode_of(dataset, st):
    if dataset == "indoor":
        return {"scene": st["scene"], "tag": st["tag"], "start": st["start"]}
    return {"scene": st["scene"], "start": st["start"], "tag": st["tag"]}


def do_main(dataset, st, t, rec, geo, h, eps, cores, z0, k_M):
    od = A.state_dir(dataset, st["sid"])
    res = {"sid": st["sid"], "t": t, "models": {}, "repro": {}}
    arrs = {}
    views = None
    for m, core in cores.items():
        tb = time.time()
        out, tm, views = PC.score_state(core, geo, h[m], rec["legal_R"], rec["legal_T"], eps, z0, keep_hits=True)
        res["models"][m] = {"distp1": out["distp1"], "distp1_k": out["distp1_k"], "timing": tm, "seconds": time.time() - tb}
        for e in st["sources"]:
            if e["source"] == m and "online" in e:
                d = float(np.max(np.abs(np.asarray(out["distp1"]) - np.asarray(e["online"]["scores"]))))
                res["repro"][m] = max(res["repro"].get(m, 0.0), d)
        arrs[m] = out["hits"]
    # ---- evaluation side from here on (t* of S1)
    orc = np.load(os.path.join(od, "oracle.npz"))
    assert np.array_equal(orc["legal_idx"], rec["legal_idx"])
    save = {}
    for m in cores:
        cm = []
        for j, (v, hh) in enumerate(zip(views, arrs[m])):
            mt, ar = candidate_metrics(v, hh, orc["tstar"][j], geo, z0, k_M)
            cm.append(mt)
            if m == "V2O":
                for kk, a in ar.items():
                    save.setdefault(kk, []).append(a.cpu().numpy())
            save.setdefault(f"{m}_t", []).append(hh["t"].float().cpu().numpy())
            save.setdefault(f"{m}_hit_u", []).append(hh["hit_u"].cpu().numpy())
            save.setdefault(f"{m}_b", []).append(hh["b"].half().cpu().numpy())
            save.setdefault(f"{m}_y", []).append(hh["y"].half().cpu().numpy())
        res["models"][m]["cand"] = cm
    res["gtd_distp1"] = [c["gtd_distp1"] for c in res["models"]["V2O"]["cand"]]
    geo_np = {"o": np.stack([v["o"].cpu().numpy() for v in views]), "d": np.stack([v["d"].cpu().numpy() for v in views]),
              "t_known": np.stack([v["pr"]["t_known"].float().cpu().numpy() for v in views]),
              "cls": np.stack([v["pr"]["cls"].cpu().numpy().astype(np.uint8) for v in views])}
    np.savez_compressed(os.path.join(od, "v2.tmp.npz"), **{k: np.stack(v) for k, v in save.items()}, **geo_np,
                        k_M=np.asarray(k_M), origin=np.asarray(geo.fr.origin_z, np.float64), z0=np.asarray(z0))
    os.replace(os.path.join(od, "v2.tmp.npz"), os.path.join(od, "v2.npz"))
    json.dump(res, open(os.path.join(od, "v2.json.tmp"), "w"), default=float)
    os.replace(os.path.join(od, "v2.json.tmp"), os.path.join(od, "v2.json"))
    return res


def do_kabl(dataset, st, t, rec, geo, h, cores, z0, dev):
    od = A.state_dir(dataset, st["sid"])
    out = {}
    tms = {}
    for m, core in cores.items():
        for r in range(A.N_SEEDSETS):
            eps = PR.noise(dev, max(A.K_ABL), lambda kk: A.sha_seed(f"exp5-ABL|K|{st['sid']}|r{r}|k{kk}"))
            o, tm, _ = PC.score_state(core, geo, h[m], rec["legal_R"], rec["legal_T"], eps, z0)
            out[f"{m}_r{r}"] = np.asarray(o["distp1_k"], np.float64)            # [n_legal, 16]
            if r == 0:
                for K in A.K_ABL:
                    _, tmk, _ = PC.score_state(core, geo, h[m], rec["legal_R"], rec["legal_T"], eps[:K], z0)
                    tms[f"{m}_K{K}"] = tmk
    np.savez_compressed(os.path.join(od, "kabl.tmp.npz"), **out)
    os.replace(os.path.join(od, "kabl.tmp.npz"), os.path.join(od, "kabl.npz"))
    json.dump(tms, open(os.path.join(od, "kabl_timing.json"), "w"))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--dataset", choices=["indoor", "outdoor"], required=True)
    ap.add_argument("--mode", choices=["main", "kabl"], default="main")
    ap.add_argument("--part", default="0/1")
    ap.add_argument("--primary-only", action="store_true")
    a = ap.parse_args()
    pi, pn = map(int, a.part.split("/"))
    VF.verify_frozen()
    torch.use_deterministic_algorithms(True, warn_only=True)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    dev = torch.device("cuda")
    cores = load_cores(dev)
    man = A.load_manifest(a.dataset)
    fn = "v2.json" if a.mode == "main" else "kabl.npz"
    for (src, tag), sts in groups(man)[pi::pn]:
        sts = [s for s in sts if (s["primary_t"] or not a.primary_only)
               and not os.path.exists(os.path.join(A.state_dir(a.dataset, s["sid"]), fn))
               and os.path.exists(os.path.join(A.state_dir(a.dataset, s["sid"]), "oracle.npz"))]
        if not sts:
            continue
        ep = episode_of(a.dataset, sts[0])
        rp = StateReplay(a.dataset, ep, src, cores, dev)
        want = {s["t"]: s for s in sts}
        for t, rec, geo, h, eps in rp.run(sorted(want)):
            tb = time.time()
            if a.mode == "main":
                k_M = float(np.load(os.path.join(A.state_dir(a.dataset, want[t]["sid"]), "oracle.npz"))["k_M"])
                assert abs(k_M - float(geo.fr.k_M)) < 1e-9 * max(1.0, k_M), (k_M, geo.fr.k_M)
                res = do_main(a.dataset, want[t], t, rec, geo, h, eps, cores, rp.z0, float(geo.fr.k_M))
                print(a.dataset, src, tag, t, "repro", res["repro"], "%.1fs" % (time.time() - tb), flush=True)
            else:
                do_kabl(a.dataset, want[t], t, rec, geo, h, cores, rp.z0, dev)
                print("kabl", a.dataset, src, tag, t, "%.1fs" % (time.time() - tb), flush=True)
        torch.cuda.empty_cache()


if __name__ == "__main__":
    main()
