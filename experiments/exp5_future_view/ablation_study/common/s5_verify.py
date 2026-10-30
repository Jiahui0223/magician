"""EXP5 Ablation Stage 5 -- independent verification (seen2scene env; GPU for the re-score subset).

V1 oracle gain of the executed move == the stored Task 2 coverage increment (indoor: episode.json C_obs curve; outdoor:
   the monotone curve of task2 t25_out_cov metrics, an independent computation), and cov_t == the stored curve at t
V2 indoor oracle gains of ALL candidates recomputed with an independent numpy / cKDTree implementation from the cached
   GT points + visibility of the frozen Coverage, the stored samples of records 0..t and the saved candidate samples
V3 re-score subset (seeded 12 states / dataset): planner state rebuilt, both models re-scored, GT-depth DistP1 from t*
   -> must equal v2.json bitwise
V4 per-state picks / Top-1 / regret / Spearman / pairwise re-derived in plain numpy from a1_candidates_<d>.csv
V5 K = 4 mean == mean of the 4 per-sample scores; K-ablation regret / Top-1 re-derived from kabl.npz for all states
V6 figure annotations == the source tables / files
Output W/verify/verify.json (and STOP on any failure)
"""
import csv
import json
import os
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
ABL = os.path.dirname(HERE)
sys.path.insert(0, HERE)
import ab_common as A  # noqa: E402

import numpy as np  # noqa: E402

TAU = A.TIE_TOL * 100


def fails(name, bad, n):
    print(f"{name}: {n - len(bad)}/{n} ok" + (f"; FAIL e.g. {bad[:3]}" if bad else ""), flush=True)
    return {"checked": n, "failed": len(bad), "examples": bad[:10]}


def v1():
    out = {}
    for d in ("indoor", "outdoor"):
        bad, n = [], 0
        for st in A.load_manifest(d)["states"]:
            o = np.load(os.path.join(A.state_dir(d, st["sid"]), "oracle.npz"))
            for e in st["sources"]:
                pk, t = e["executed_pick"], st["t"]
                if pk is None:
                    continue
                if d == "indoor":
                    c = json.load(open(os.path.join(A.CL_IN, e["source"], st["tag"], "episode.json")))["coverage"]
                else:
                    c = json.load(open(os.path.join(A.T2W, "eval", "outdoor", "metrics", e["source"],
                                                    f"{e['source']}_{st['scene']}_s{st['start']}_r0.json")))["cov_mono"]
                n += 1
                if abs(float(o["gain"][pk]) - (c[t + 1] - c[t])) > 1e-12 or abs(float(o["cov_t"]) - c[t]) > 1e-12:
                    bad.append([st["sid"], e["source"], float(o["gain"][pk]), c[t + 1] - c[t]])
        out[d] = fails(f"V1 {d} executed gain == stored coverage increment", bad, n)
    return out


def v2():
    from scipy.spatial import cKDTree
    bad, n = [], 0
    man = A.load_manifest("indoor")
    by = {}
    for st in man["states"]:
        by.setdefault((st["sources"][0]["source"], st["tag"]), []).append(st)
    for (src, tag), sts in sorted(by.items()):
        cz = np.load(os.path.join(A.REPO, "work_dirs", "exp5_future_view", "task0", "preprocessing", "coverage", f"{sts[0]['scene']}.npz"))
        G, vis, eps = cz["pts"], cz["visible"].astype(bool), float(cz["eps_rt"])
        tree = cKDTree(G)
        cov = np.zeros(len(G), bool)
        want = {s["t"]: s for s in sts}

        def near(P):
            m = np.zeros(len(G), bool)
            if len(P):
                d, _ = cKDTree(np.asarray(P, np.float64)).query(G, k=1, distance_upper_bound=eps)
                m = np.isfinite(d)
            return m
        for t in range(max(want) + 1):
            cov |= near(np.load(os.path.join(A.CL_IN, src, tag, "svc", f"svc_t{t}.npz"))["sample_points"])
            if t in want:
                o = np.load(os.path.join(A.state_dir("indoor", want[t]["sid"]), "oracle.npz"))
                off = o["cand_off"]
                g = [float((near(o["cand_pts"][off[j]:off[j + 1]]) & vis & ~cov).sum() / vis.sum()) for j in range(len(off) - 1)]
                n += 1
                if not np.allclose(g, o["gain"], rtol=0, atol=1e-12):
                    bad.append([want[t]["sid"], float(np.max(np.abs(np.asarray(g) - o["gain"])))])
        del tree
    return fails("V2 indoor all-candidate gains (independent numpy)", bad, n)


def v3():
    import s2_infer as S2
    torch = S2.torch
    torch.use_deterministic_algorithms(True, warn_only=True)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    dev = torch.device("cuda")
    cores = S2.load_cores(dev)
    out = {}
    for d in ("indoor", "outdoor"):
        sts = A.load_manifest(d)["states"]
        rng = np.random.default_rng(12345)
        pick = [sts[i] for i in sorted(rng.choice(len(sts), 12, replace=False))]
        bad = []
        for st in pick:
            rp = S2.StateReplay(d, S2.episode_of(d, st), st["sources"][0]["source"], cores, dev)
            v2 = json.load(open(os.path.join(A.state_dir(d, st["sid"]), "v2.json")))
            orc = np.load(os.path.join(A.state_dir(d, st["sid"]), "oracle.npz"))
            for t, rec, geo, h, eps in rp.run([st["t"]]):
                for m, core in cores.items():
                    o, _, views = S2.PC.score_state(core, geo, h[m], rec["legal_R"], rec["legal_T"], eps, rp.z0)
                    if np.asarray(o["distp1_k"]).tolist() != v2["models"][m]["distp1_k"]:
                        bad.append([st["sid"], m])
                gtd = []
                for j, v in enumerate(views):
                    ts = torch.as_tensor(orc["tstar"][j], device=v["d"].device, dtype=v["d"].dtype)
                    tt = torch.where(ts <= v["pr"]["tmax"], ts, torch.full_like(ts, float("inf")))
                    _, w = S2.SCO.new_set_from_depth(tt, v["o"], v["d"], v["dz"], float(geo.fr.k_M), geo.dims, geo.domain,
                                                     geo.caster.band_dil, rp.z0)
                    gtd.append(float(w.sum()))
                if gtd != v2["gtd_distp1"]:
                    bad.append([st["sid"], "GTD"])
        out[d] = fails(f"V3 {d} re-score (12 seeded states, both models + GTD) bitwise", bad, len(pick) * 3)
    return out


def spearman_np(a, b):
    from scipy.stats import rankdata
    if len(a) < 2 or np.std(a) == 0 or np.std(b) == 0:
        return None
    return float(np.corrcoef(rankdata(a), rankdata(b))[0, 1])


def v4():
    out = {}
    for d in ("indoor", "outdoor"):
        C = list(csv.DictReader(open(os.path.join(ABL, "A1_prediction_scoring", "results", f"a1_candidates_{d}.csv"))))
        S = list(csv.DictReader(open(os.path.join(ABL, "A1_prediction_scoring", "results", f"a1_states_{d}.csv"))))
        by = {}
        for r in C:
            if r["in_candidate_set"] == "True":
                by.setdefault(r["sid"], []).append(r)
        bad, n = [], 0
        col = {"V2O": "V2O_pred", "V2M": "V2M_pred", "GTD": "GTD"}
        for r in S:
            if r["scorer"] not in col:
                continue
            c = by[r["sid"]]
            g = np.asarray([float(x["oracle_gain_pp"]) for x in c])
            s = np.asarray([float(x[col[r["scorer"]]]) for x in c])
            sel = [i for i, x in enumerate(c) if x[f"selected_{r['scorer']}"] == "True"]
            n += 1
            if len(sel) != 1:
                bad.append([r["sid"], r["scorer"], "selected count", len(sel)])
                continue
            gp = g[sel[0]]
            reg, top = g.max() - gp, float(gp >= g.max() - TAU)
            rho = spearman_np(s, g)
            num = den = 0.0
            for i in range(len(g)):
                for j in range(i + 1, len(g)):
                    if abs(g[i] - g[j]) > TAU:
                        den += 1
                        num += 0.5 if s[i] == s[j] else float((s[i] - s[j]) * (g[i] - g[j]) > 0)
            pw = num / den if den else None
            ok = abs(reg - float(r["regret"])) < 1e-9 and top == float(r["top1"])
            ok &= (rho is None) == (r["spearman"] in ("", "None")) and (rho is None or abs(rho - float(r["spearman"])) < 1e-9)
            ok &= (pw is None) == (r["pairwise"] in ("", "None")) and (pw is None or abs(pw - float(r["pairwise"])) < 1e-9)
            # the pick must be an argmax of the score over the candidate set unless the all-zero / <=0 fallback was used
            if r["fallback"] != "True":
                ok &= s[sel[0]] == s.max()
            if not ok:
                bad.append([r["sid"], r["scorer"]])
        out[d] = fails(f"V4 {d} picks / Top-1 / regret / Spearman / pairwise (numpy re-derivation)", bad, n)
    return out


def v5():
    sys.path.insert(0, HERE)
    import ab_load as AL
    out = {}
    for d in ("indoor", "outdoor"):
        bad, n = [], 0
        kr = {}
        p = os.path.join(ABL, "A2_stochastic_future_views", "results", f"a2_kabl_{d}.csv")
        for r in (csv.DictReader(open(p)) if os.path.exists(p) else []):
            kr[(r["sid"], r["model"], int(r["K"]))] = r
        for r in AL.load_states(d):
            for m in ("V2O", "V2M"):
                n += 1
                if not np.array_equal(np.asarray([np.mean(x) for x in r["distp1_k"][m]]), r["scores"][m]):
                    bad.append([r["sid"], m, "K4 mean"])
            kp = os.path.join(A.state_dir(d, r["sid"]), "kabl.npz")
            if not r["primary_t"] or not os.path.exists(kp):
                continue
            z = np.load(kp)
            Cs = r["C"]
            for m in ("V2O", "V2M"):
                for K in A.K_ABL:
                    regs, tops = [], []
                    for rr in range(A.N_SEEDSETS):
                        s = z[f"{m}_r{rr}"][:, :K].mean(1)
                        pk = AL.pick_of(d, r, s, r["rec"], m)[0]
                        g = r["gain"][Cs]
                        regs.append(g.max() - r["gain"][pk])
                        tops.append(float(r["gain"][pk] >= g.max() - TAU))
                    x = kr.get((r["sid"], m, K))
                    n += 1
                    if x is None or abs(np.mean(regs) - float(x["regret"])) > 1e-9 or abs(np.mean(tops) - float(x["top1"])) > 1e-9:
                        bad.append([r["sid"], m, K])
        out[d] = fails(f"V5 {d} K-means / K-ablation re-derivation", bad, n)
    return out


def v6():
    bad, n = [], 0
    S1 = json.load(open(os.path.join(ABL, "A1_prediction_scoring", "results", "a1_summary.json")))
    ann = json.load(open(os.path.join(ABL, "A1_prediction_scoring", "figures", "annotations.json")))
    for k, v in ann.items():
        p = k.split("|")
        if p[0] == "fig1":
            n += 1
            if abs(S1[p[1]]["scorers"][p[2]][p[3]]["mean"] - v) > 1e-12:
                bad.append(k)
        elif p[0] == "fig4":
            n += 1
            if abs(S1[p[1]]["per_scene"][p[3]][p[4]][p[2]] - v) > 1e-12:
                bad.append(k)
    a2 = json.load(open(os.path.join(ABL, "A2_stochastic_future_views", "figures", "annotations.json")))
    S2s = json.load(open(os.path.join(ABL, "A2_stochastic_future_views", "results", "a2_summary.json")))
    for k, v in a2.items():
        p = k.split("|")
        if p[0] == "depth":
            d, sid_case, m, c = p[1], p[2], p[3], int(p[4][1:])
            sid = json.load(open(os.path.join(ABL, "A2_stochastic_future_views", "results", "cases.json")))[d][sid_case]["sid"]
            v2 = json.load(open(os.path.join(A.state_dir(d, sid), "v2.json")))
            o = np.load(os.path.join(A.state_dir(d, sid), "oracle.npz"))
            ref = {"gtd": v2["gtd_distp1"][c], "gain": float(o["gain"][c] * 100)}.get(p[5])
            if ref is None:
                ref = v2["models"][m]["distp1_k"][c][int(p[5][1:])]
            n += 1
            if abs(ref - v) > 1e-9:
                bad.append(k)
        elif p[0] == "summary" and p[3] == "kabl":
            continue
        elif p[0] == "summary" and p[3] != "kabl":
            n += 1
            if abs(S2s[p[1]]["models"][p[2]][p[3]]["mean"] - v) > 1e-12:
                bad.append(k)
    pv = os.path.join(ABL, "A2_stochastic_future_views", "figures", "v2", "annotations.json")
    cases = json.load(open(os.path.join(ABL, "A2_stochastic_future_views", "results", "cases.json")))
    for k, v in (json.load(open(pv)).items() if os.path.exists(pv) else []):
        _, d, case, m, jj, what = k.split("|")
        sid, j = cases[d][case]["sid"], int(jj[1:])
        v2 = json.load(open(os.path.join(A.state_dir(d, sid), "v2.json")))
        if what == "gain":
            ref = float(np.load(os.path.join(A.state_dir(d, sid), "oracle.npz"))["gain"][j] * 100)
        elif what == "gtd":
            ref = v2["gtd_distp1"][j]
        else:
            ref = v2["models"][m]["distp1_k"][j][int(what[1:])]
        n += 1
        if abs(ref - v) > 1e-9:
            bad.append(k)
    return fails("V6 figure annotations == source values", bad, n)


def main():
    res = {"V1": v1(), "V2": v2(), "V4": v4(), "V5": v5(), "V6": v6()}
    if "--no-gpu" not in sys.argv:
        res["V3"] = v3()
    os.makedirs(os.path.join(A.W, "verify"), exist_ok=True)
    json.dump(res, open(os.path.join(A.W, "verify", "verify.json"), "w"), indent=1)
    tot = sum(x["failed"] for x in [res["V2"], res["V6"]] + [y for k in ("V1", "V4", "V5", "V3") if k in res for y in res[k].values()])
    print("TOTAL FAILED", tot)


if __name__ == "__main__":
    main()
