"""EXP Temporal Refresh -- analysis (seen2scene env, CPU; REFRESH_PROTOCOL §3-§4).
fixed : per state and (checkpoint, R): G2 under the historical choice rule (ab_load.pick_of), G1 on the D100-UNKNOWN rays,
        selection change vs R10, staleness s = t % 10, timings; hierarchical means, scene-level bootstrap (+ house-level
        indoor / episode-level outdoor) of R - R10 per checkpoint; early-phase subset.
g3    : historical R10 (B2, E2-D100 Task 3 roots) + new R5 / R2 / R1 (work_dirs/exp_temporal_refresh/g3), readers and
        paired scene bootstrap of exp_density_ab/scripts/da_g3_analyze.py; only complete configs are summarised.
Outputs results/fixed_summary.json, results/fixed_states.csv, results/g3_summary.json, results/g3_episodes.csv, figures/*.png
Usage: python tr_analyze.py fixed|g3
"""
import csv
import glob
import json
import os
import sys

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
EXP = os.path.dirname(HERE)
REPO = os.path.dirname(os.path.dirname(EXP))
sys.path.insert(0, os.path.join(REPO, "experiments", "exp_density_ab", "scripts"))
CK, RS = ("B2", "E2D"), (10, 5, 2, 1)


def fixed():
    import da_analyze_fixed as DF
    EA, AL, EP, P = DF.EA, DF.AL, DF.EP, DF.P
    FIX = os.path.join(REPO, "work_dirs", "exp_temporal_refresh", "fixed")
    recs = []
    for sset, dname in (("valsel", "indoor"), ("dev3", "outdoor")):
        rows = []
        for f in sorted(glob.glob(os.path.join(FIX, sset, "*", "*.jsonl"))):
            rows += [json.loads(x) for x in open(f)]
        if sset == "valsel":
            man = {s["sid"]: s for s in json.load(open(os.path.join(EP.W, "valsel", "manifests", "states_valsel.json")))["states"]}
        else:
            m0 = {s["sid"]: s for s in json.load(open(os.path.join(DF.ORIG_W, "manifests", "states_outdoor.json")))["states"]}
            m2 = {s["sid"]: s for s in json.load(open(os.path.join(P.ROOT_B2, "manifests", "states_outdoor.json")))["states"]}
            man = {r["sid"]: (m0 if r["stratum"] == "B0" else m2)[r["sid"]] for r in rows}
        for r in rows:
            st = man[r["sid"]]
            C = np.asarray(st["cand"], int)
            g = np.asarray(r["oracle_gain_pp"], float)[C]
            src = st["sources"][0] if sset == "valsel" else next(e for e in st["sources"] if e["source"] == "V2M")
            rec = {k: np.asarray(x) for k, x in np.load(os.path.join(EP.REPO, src["svc"])).items() if k in ("legal_idx", "executed_pose_idx_history")}
            loc = {int(c): i for i, c in enumerate(C)}
            x = {"set": sset, "dataset": dname, "sid": r["sid"], "scene": r["scene"], "start": r["start"], "t": r["t"], "phase": r["phase"],
                 "src": r["stratum"], "tag": r["tag"], "stale10": r["stale10"], "repro_ok": bool(all(r["repro"].values())), "n_repro": len(r["repro"]),
                 "oracle_max": float(g.max()), "encode_s": r.get("encode_s_mean"), "peak_gb": r.get("episode_peak_gb")}
            for m in CK:
                pk10 = None
                for R in RS:
                    c = f"{m}|{R}"
                    s = np.asarray([float(np.mean(k)) for k in r["scores"][c]])
                    pk = AL.pick_of(dname, st, s, rec, "V2M")[0]
                    gp = g[loc[pk]]
                    pk10 = pk if R == 10 else pk10
                    x.update({f"{c}#top1": float(gp >= g.max() - DF.TAU), f"{c}#sel_gain": float(gp), f"{c}#regret": float(g.max() - gp),
                              f"{c}#rho": EA.corr(EA.spearmanr, s[C], g), f"{c}#pick": int(pk), f"{c}#chg_vs_R10": float(pk != pk10),
                              f"{c}#score_s": r["timing"].get(c, {}).get("total_s")})
                    for sname in ("unk", "new"):
                        q = r["g1"][c][sname]
                        x[f"{c}#{sname}#mae"] = DF.ratio(q["sae"], q["tp"])
                        x[f"{c}#{sname}#hitacc"] = DF.ratio(q["tp"] + q["tn"], q["n"])
                        x[f"{c}#{sname}#fn_rate"] = DF.ratio(q["fn"], q["gt"])
                        x[f"{c}#{sname}#acc_first"] = DF.ratio(q["acc"], q["gt"])
            recs.append(x)
    out = {}
    for d in ("indoor", "outdoor"):
        rr = [r for r in recs if r["dataset"] == d]
        res = {"n_states": len(rr), "repro_ok": int(sum(r["repro_ok"] for r in rr)), "n_repro": int(sum(r["n_repro"] for r in rr))}
        for sub, cond in (("all", lambda r: True), ("early", lambda r: r["phase"] == "early"), ("stale_ge5", lambda r: r["stale10"] >= 5),
                          ("stale_0", lambda r: r["stale10"] == 0)):
            S = [r for r in rr if cond(r)]
            keys = sorted({k for r in S for k in r if "#" in k and not k.endswith("#pick")})
            M = {k: AL.boot(AL.hier_mean(S, k)) for k in keys}
            Pd = {}
            for m in CK:
                for R in (5, 2, 1):
                    for met in ("top1", "sel_gain", "rho", "unk#mae", "unk#hitacc", "unk#acc_first", "unk#fn_rate", "new#mae", "new#acc_first"):
                        ha, hb = AL.hier_mean(S, f"{m}|{R}#{met}"), AL.hier_mean(S, f"{m}|10#{met}")
                        dd = {s: ha[s] - hb[s] for s in ha if s in hb}
                        ea, eb = DF.ep_mean(S, f"{m}|{R}#{met}"), DF.ep_mean(S, f"{m}|10#{met}")
                        de = {s: ea[s] - eb[s] for s in ea if s in eb}
                        bo = AL.boot(dd)
                        hv = {}
                        for k, v in dd.items():
                            hv.setdefault(k.split("_")[0], []).append(v)
                        hb_ = AL.boot({k: float(np.mean(v)) for k, v in hv.items()}) if d == "indoor" else DF.boot_ep(de)
                        bo.update({"wins": int(sum(v > 1e-12 for v in dd.values())), "losses": int(sum(v < -1e-12 for v in dd.values())),
                                   "aux": hb_, "aux_kind": "house" if d == "indoor" else "episode"})
                        Pd[f"{m}|R{R}-R10|{met}"] = bo
            res[sub] = {"n_states": len(S), "means": M, "paired": Pd}
        stale = {}
        for s in range(10):
            S = [r for r in rr if r["stale10"] == s]
            if S:
                stale[str(s)] = {"n": len(S), **{f"{m}|{R}#{met}": float(np.nanmean([r[f'{m}|{R}#{met}'] for r in S if r[f'{m}|{R}#{met}'] is not None]))
                                                   for m in CK for R in (10, 1) for met in ("unk#mae", "sel_gain", "top1")}}
        res["by_staleness"] = stale
        out[d] = res
    os.makedirs(os.path.join(EXP, "results"), exist_ok=True)
    json.dump(out, open(os.path.join(EXP, "results", "fixed_summary.json"), "w"), indent=1, default=float)
    cols = ["dataset", "sid", "scene", "start", "t", "phase", "src", "stale10", "oracle_max"] + \
           [f"{m}|{R}#{k}" for m in CK for R in RS for k in ("pick", "top1", "sel_gain", "rho", "chg_vs_R10", "unk#mae", "unk#acc_first", "score_s")]
    with open(os.path.join(EXP, "results", "fixed_states.csv"), "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(cols)
        for r in recs:
            w.writerow([r.get(c) for c in cols])
    for d in ("indoor", "outdoor"):
        A = out[d]["all"]
        print(d, "states", out[d]["n_states"], "repro ok", out[d]["repro_ok"], "checks", out[d]["n_repro"])
        for m in CK:
            for R in RS:
                c = f"{m}|{R}"
                M = A["means"]
                print(f"  {c:6s} top1 {M[c + '#top1']['mean']:.3f} sel {M[c + '#sel_gain']['mean']:.3f} rho {M[c + '#rho']['mean']:.3f} "
                      f"mae {M[c + '#unk#mae']['mean']:.4f} accf {M[c + '#unk#acc_first']['mean']:.3f} chg {M[c + '#chg_vs_R10']['mean']:.3f}")


def g3():
    import da_g3_analyze as DG
    W3, NEW = DG.W3, os.path.join(REPO, "work_dirs", "exp_temporal_refresh", "g3")
    HIST = {"B2_R10": "B2", "E2D_R10": "E2D"}
    CFGS = [f"{m}_R{R}" for m in CK for R in RS]

    def roots(cfg):
        if cfg in HIST:
            g = HIST[cfg]
            return (os.path.join(W3, "closed_loop", "indoor", g, "V2M"), os.path.join(W3, "closed_loop", "outdoor", g, "V2M"),
                    os.path.join(W3, "metrics", "outdoor", g, "V2M"))
        return (os.path.join(NEW, "closed_loop", "indoor", cfg, "V2M"), os.path.join(NEW, "closed_loop", "outdoor", cfg, "V2M"),
                os.path.join(NEW, "metrics", "outdoor", cfg, "V2M"))

    def pairs():
        return [(f"{m}_R{R}", f"{m}_R10", f"R{R} - R10 ({m})") for m in CK for R in (5, 2, 1)] + [("E2D_R1", "B2_R1", "E2D - B2 at R1"), ("E2D_R2", "B2_R2", "E2D - B2 at R2")]
    DG.roots, DG.pairs = roots, pairs
    out, csv_rows = {}, []
    for dataset in ("indoor", "outdoor"):
        rows = []
        for ep in DG.A.episodes(dataset):
            r = {"dataset": dataset, "scene": ep["scene"], "start": ep["start"]}
            for c in CFGS:
                x = DG.ep_in(c, ep) if dataset == "indoor" else DG.ep_out(c, ep)
                if x is not None:
                    r[c] = x
            rows.append(r)
        avail = [c for c in CFGS if all(c in r for r in rows)]
        S = DG.summarize(rows, dataset, avail)
        S["partial_configs"] = {c: sum(c in r for r in rows) for c in CFGS if c not in avail}
        S["n_refresh_encodes_per_episode"] = {c: (150 if dataset == "indoor" else 100) // int(c.split("_R")[1]) for c in avail}
        out[dataset] = S
        for r in rows:
            for c in avail:
                csv_rows.append([dataset, r["scene"], r["start"], c] + [r[c].get(k) for k in ("auc", "final", "cov20", "cov50", "time_per_move", "peak_gb")])
        plt = DG.T.plt
        DG.T.ST.setup()
        fig, ax = plt.subplots(1, 1, figsize=(9, 5))
        col = {"B2": {10: "#bcd4f2", 5: "#86b6ef", 2: "#3f86d9", 1: "#1c5cab"}, "E2D": {10: "#f9d3b8", 5: "#f6ad7f", 2: "#eb6834", 1: "#b8471a"}}
        for c in avail:
            m, R = c.split("_R")
            ax.plot(S["mean_curve"][c], color=col[m][int(R)], lw=1.8, ls="-" if m == "B2" else "--", label=f"{m} R{R}  AUC {S['means']['auc'][c]['mean']:.3f}")
        ax.set_xlabel("move")
        ax.set_ylabel("coverage")
        ax.set_title(f"Temporal refresh G3 -- {'MP3D (C_obs)' if dataset == 'indoor' else 'Macarons++ (monotone)'}", loc="left", fontsize=10)
        ax.legend(fontsize=8, loc="lower right")
        DG.T.ST.clean(ax, "both")
        fig.tight_layout()
        os.makedirs(os.path.join(EXP, "figures"), exist_ok=True)
        fig.savefig(os.path.join(EXP, "figures", f"g3_curves_{dataset}.png"), dpi=130)
        plt.close(fig)
    json.dump(out, open(os.path.join(EXP, "results", "g3_summary.json"), "w"), indent=1, default=float)
    with open(os.path.join(EXP, "results", "g3_episodes.csv"), "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["dataset", "scene", "start", "config", "auc", "final", "cov20", "cov50", "time_per_move_s", "peak_gb"])
        w.writerows(csv_rows)
    for d in ("indoor", "outdoor"):
        S = out[d]
        print(d, "available", S["configs"], "partial", S["partial_configs"])
        for c in S["configs"]:
            M = S["means"]
            print(f"  {c:8s} AUC {M['auc'][c]['mean']:.4f} Final {M['final'][c]['mean']:.4f} Cov@20 {M['cov20'][c]['mean']:.4f} t/move {M['time_per_move'][c]['mean']:.2f}s")


if __name__ == "__main__":
    {"fixed": fixed, "g3": g3}[sys.argv[1]]()
