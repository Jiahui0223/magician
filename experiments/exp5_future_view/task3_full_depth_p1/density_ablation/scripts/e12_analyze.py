"""EXP5 Task 3 / E1.2 -- analysis of the fixed-state density ablation (seen2scene env, CPU; protocol.md v1.0).

Ranking (Task 2 choice rule, ab_load.pick_of) of every scorer variant: I10..I100 (input density, sparse DistP1),
S10..S100 (100 % input, scoring density), O50 / O75 (own density for both); geometry, prediction, novelty-mask and
phantom statistics per density; aggregation state -> episode -> scene, scene-level bootstrap; strata S_B0, S_B2 and
pooled (shared states counted once, in S_B0); phases early / middle / late (+ Cov@20 neighbourhood); the frozen
closed-loop selection rule (protocol §4). Checks: I10 == own D10, S10 == I100, S100 == own D100; the I10 pick equals the
executed B0 move on S_B0 states and the I100 pick the executed B2 move on S_B2 states.
Outputs analysis/e12_summary.json, analysis/tables.md, figures/fig1_geometry.png, fig1b_prediction.png,
fig2_ranking_input.png, fig2b_ranking_scoring.png, fig6_summary.png
"""
import glob
import json
import os
import sys

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
import e12_paths as P  # noqa: E402

A = P.A
ORIG_W = A.W
import ab_load as AL  # noqa: E402
import ab_style as ST  # noqa: E402
from scipy.stats import kendalltau, spearmanr  # noqa: E402

DA = os.path.dirname(HERE)
OUT, FIG = os.path.join(DA, "analysis"), os.path.join(DA, "figures")
DENS = (10, 50, 75, 100)
VARS = [f"I{x}" for x in DENS] + [f"S{x}" for x in DENS] + ["O50", "O75"]
PHASES = ("early", "middle", "late")
COV20 = {"indoor": (20, 25), "outdoor": (15, 20, 25)}
TAU = 1e-4 * 100
NAME = {"indoor": "MP3D", "outdoor": "Macarons++"}
plt = ST.plt
DCOL = {10: ST.BLUE[3], 50: ST.BLUE[6], 75: ST.BLUE[9], 100: ST.BLUE[12]}
PCOL = {"early": "#d03b3b", "middle": "#eb6834", "late": "#1baf7a", "all": ST.INK}


def man(d, root):
    return {s["sid"]: s for s in json.load(open(os.path.join(root, "manifests", f"states_{d}.json")))["states"]}


def corr(f, a, b):
    if len(a) < 2 or np.ptp(a) == 0 or np.ptp(b) == 0:
        return None
    r = f(a, b)
    r = r.correlation if hasattr(r, "correlation") else r[0]
    return float(r) if np.isfinite(r) else None


def ratio(a, b):
    return float(a) / float(b) if b else None


def derive(dr):
    """Per-state scalars of one density block."""
    o = {}
    m = dr["map"]
    o.update({"band_ratio": ratio(m["band"], m["domain"]), "empty_ratio": ratio(m["empty"], m["domain"]),
              "unknown_ratio": ratio(m["unknown"], m["domain"]), "coarse_band": m["coarse_band_mean"], "band_vox": m["band"]})
    for cam in ("cur", "cand"):
        c = dr[cam]
        o.update({f"{cam}_tk": ratio(c["known_gt"], c["gt"]), f"{cam}_match": ratio(c["match"], c["known_gt"]),
                  f"{cam}_err": c["abs_err_med_u"], f"{cam}_entry": ratio(c["phantom_entry"], c["phantom@1"]),
                  f"{cam}_edge_share": ratio(c["phantom_edge"], c["phantom@1"]), f"{cam}_edge_base": ratio(c["edge_gt"], c["gt"]),
                  f"{cam}_ph_edge": ratio(c["phantom_edge"], c["known_gt_edge"]),
                  f"{cam}_ph_int": ratio(c["phantom@1"] - c["phantom_edge"], c["known_gt"] - c["known_gt_edge"]),
                  f"{cam}_off_bf": c["off_band_first_med"], f"{cam}_off_zero": c["off_zero_med"], f"{cam}_off_ee": c["off_empty_end_med"],
                  f"{cam}_bf_front": ratio(c["band_first_front"], c["band_first_gt"]), f"{cam}_ph_mag": c["phantom_mag_med_trunc"],
                  f"{cam}_nseg": ratio(c["n_seg_sum"], c["rays"]), f"{cam}_lu": ratio(c["lu_sum_u"], c["rays"])})
        for mm in (1, 2, 3):
            o[f"{cam}_ph{mm}_known"] = ratio(c[f"phantom@{mm}"], c["known_gt"])
            o[f"{cam}_ph{mm}_all"] = ratio(c[f"phantom@{mm}"], c["gt"])
    p = dr["pred"]
    o.update({"mae": ratio(p["sae"], p["n_both"]), "rmse": float(np.sqrt(p["sse"] / p["n_both"])) if p["n_both"] else None,
              "absrel": ratio(p["srel"], p["n_both"]), "hit_agree": ratio(p["hit_agree"], p["n_rays"]),
              "mae_new": ratio(p["sae_new"], p["n_new"]), "mae_old": ratio(p["sae_old"], p["n_old"]),
              "rel_std": ratio(p.get("rel_std_sum", 0.0), p["n_allhit"]), "hit_dis": ratio(p["hit_disagree"], p["n_rays"] / 4),
              "spread_err_rho": p.get("spread_err_rho_mean")})
    for b in ("b0", "b1", "b2", "b3"):
        o[f"mae_{b}"] = ratio(p[f"sae_{b}"], p[f"n_{b}"])
    for src, mk in dr["masks"].items():
        o.update({f"{src}_excl_dil": ratio(mk["excl_dil"], mk["hit"]), f"{src}_excl_band": ratio(mk["excl_band"], mk["hit"]),
                  f"{src}_fexcl_dil": ratio(mk["false_excl_dil"], mk["hit_truth_new"]), f"{src}_fexcl_band": ratio(mk["false_excl_band"], mk["hit_truth_new"]),
                  f"{src}_frew_dil": ratio(mk["false_reward_dil"], mk["hit_truth_old"]), f"{src}_frew_band": ratio(mk["false_reward_band"], mk["hit_truth_old"]),
                  f"{src}_truth_new": ratio(mk["hit_truth_new"], mk["hit_truth_new"] + mk["hit_truth_old"])})
        if "false_excl_dil_acc" in mk:
            o[f"{src}_fexcl_acc_share"] = ratio(mk["false_excl_dil_acc"], mk["false_excl_dil"])
    for k in ("projection_s", "cond_s", "sampling_s", "decode_score_s", "total_s"):
        o[f"time_{k}"] = dr["score_timing"][k]
    return o


def load(d):
    m0, m2 = man(d, ORIG_W), man(d, P.ROOT_B2)
    raw = {}
    for stratum in ("B0", "B2"):
        for f in sorted(glob.glob(os.path.join(P.E12, "fixed", stratum, d, "*.jsonl"))):
            for line in open(f):
                r = json.loads(line)
                raw[r["sid"]] = r
    eps_info = {}
    for stratum in ("B0", "B2"):
        for f in glob.glob(os.path.join(P.E12, "fixed", stratum, d, "*.episode.json")):
            eps_info[(stratum, os.path.basename(f)[:-len(".episode.json")])] = json.load(open(f))
    recs, checks = [], {"I10_eq_own10": True, "S10_eq_I100": True, "S100_eq_own100": True, "pick_B0": [0, 0], "pick_B2": [0, 0]}
    for sid, r in raw.items():
        st = (m0 if r["stratum"] == "B0" else m2)[sid]
        root = ORIG_W if r["stratum"] == "B0" else P.ROOT_B2
        o = np.load(os.path.join(root, "states", d, sid, "oracle.npz"))
        C = np.asarray(st["cand"], int)
        g = np.asarray(o["gain"], float)[C] * 100.0
        src = next(e for e in st["sources"] if e["source"] == "V2M")
        rec = {k: np.asarray(x) for k, x in np.load(os.path.join(A.REPO, src["svc"])).items() if k in ("legal_idx", "executed_pose_idx_history")}
        sk = {f"I{x}": r["I_k"][str(x)] for x in DENS}
        sk.update({f"S{x}": r["d"][str(x)]["S_k"] for x in DENS})
        sk.update({f"O{x}": r["d"][str(x)]["own_k"] for x in (50, 75)})
        checks["I10_eq_own10"] &= sk["I10"] == r["d"]["10"]["own_k"]
        checks["S10_eq_I100"] &= sk["S10"] == sk["I100"]
        checks["S100_eq_own100"] &= sk["S100"] == r["d"]["100"]["own_k"]
        base = {"sid": sid, "scene": st["scene"], "start": st["start"], "t": st["t"], "phase": r["phase"], "tag": st["tag"],
                "split": st.get("split"), "cov20": st["t"] in COV20[d]}
        loc = {int(c): i for i, c in enumerate(C)}
        met = {}
        picks = {}
        for v, s in sk.items():
            s = np.asarray([float(np.mean(x)) for x in s])
            pk = AL.pick_of(d, st, s, rec, "V2M")[0]
            picks[v] = pk
            gp = g[loc[pk]]
            met[f"{v}_pick"] = int(pk)
            met.update({f"{v}_top1": float(gp >= g.max() - TAU), f"{v}_regret": float(g.max() - gp), f"{v}_gain": float(gp),
                        f"{v}_rho": corr(spearmanr, s[C], g), f"{v}_tau": corr(kendalltau, s[C], g)})
        for v in sk:
            met[f"{v}_chg10"] = float(picks[v] != picks["I10"])
            met[f"{v}_chg100"] = float(picks[v] != picks["I100"])
        met["oracle_max"] = float(g.max())
        geo = {}
        for x in DENS:
            for k, val in derive(r["d"][str(x)]).items():
                geo[f"D{x}_{k}"] = val
        for x in (50, 75, 100):
            for s2 in ("gt", "pred100"):
                e = r["dense_vs_sparse_excl"].get(f"{s2}_{x}")
                if e:
                    geo[f"D{x}_{s2}_onlydense_newshare"] = ratio(e["only_dense_truth_new"], e["only_dense"])
                    geo[f"D{x}_{s2}_onlydense_edgeshare"] = ratio(e["only_dense_edge"], e["only_dense"])
                    geo[f"D{x}_{s2}_onlydense_rate"] = ratio(e["only_dense"], r["d"]["10"]["masks"][s2]["hit"])
        row = dict(base, **met, **geo)
        # memberships: S_B0 states (incl. shared) and S_B2 states (B2-only rows + shared S_B0 rows)
        if r["stratum"] == "B0":
            ex = src["executed_pick"]
            checks["pick_B0"][0] += 1
            checks["pick_B0"][1] += int(picks["I10"] == ex)
            recs.append(dict(row, src="B0", pooled=True))
            if sid in m2:
                ex2 = m2[sid]["sources"][0]["executed_pick"]
                checks["pick_B2"][0] += 1
                checks["pick_B2"][1] += int(picks["I100"] == ex2)
                recs.append(dict(row, src="B2", pooled=False))
        else:
            ex2 = st["sources"][0]["executed_pick"]
            checks["pick_B2"][0] += 1
            checks["pick_B2"][1] += int(picks["I100"] == ex2)
            recs.append(dict(row, src="B2", pooled=True))
    return recs, eps_info, checks


def sel(recs, stratum, sub):
    rr = [r for r in recs if (r["pooled"] if stratum == "pooled" else r["src"] == stratum)]
    if sub == "all":
        return rr
    if sub == "cov20":
        return [r for r in rr if r["cov20"]]
    return [r for r in rr if r["phase"] == sub]


def hm(rr, key):
    return AL.hier_mean(rr, key)


def bootm(rr, key):
    return AL.boot(hm(rr, key))


def paired(rr, ka, kb, better="lower"):
    a, b = hm(rr, ka), hm(rr, kb)
    dd = {s: a[s] - b[s] for s in a if s in b}
    out = AL.boot(dd)
    sgn = -1 if better == "lower" else 1
    out.update({"wins": int(sum(sgn * v > 1e-12 for v in dd.values())), "losses": int(sum(sgn * v < -1e-12 for v in dd.values())),
                "per_scene": dd})
    return out


def selection(S):
    """Protocol §4 (frozen)."""
    res = {}
    for x in (50, 75):
        per = {}
        for d in ("indoor", "outdoor"):
            R = S[d]["rank"]["pooled"]
            e = R["early"]["paired_vs_I100"][f"I{x}_regret"]
            base = R["early"]["means"]["I100_regret"]["mean"]
            c1 = bool(e["mean"] is not None and base is not None and e["mean"] <= -0.10 * base and e["wins"] >= 4)
            reg_x, reg_b = R["all"]["means"][f"I{x}_regret"]["mean"], R["all"]["means"]["I100_regret"]["mean"]
            t_x, t_b = R["all"]["means"][f"I{x}_top1"]["mean"], R["all"]["means"]["I100_top1"]["mean"]
            c2 = bool(reg_x <= 1.05 * reg_b and t_x >= t_b - 0.03)
            per[d] = {"C1": c1, "C2": c2, "early_d_regret": e["mean"], "early_rel": e["mean"] / base if base and e["mean"] is not None else None,
                      "early_wins": e["wins"], "all_regret": [reg_x, reg_b], "all_top1": [t_x, t_b]}
        q = (per["indoor"]["C1"] and per["indoor"]["C2"] and per["outdoor"]["C2"]) or \
            (per["outdoor"]["C1"] and per["outdoor"]["C2"] and per["indoor"]["C2"])
        res[f"D{x}"] = {"per": per, "qualifies": bool(q),
                        "mean_rel_early": float(np.mean([per[d]["early_rel"] for d in per if per[d]["early_rel"] is not None]))}
    ql = [k for k in res if res[k]["qualifies"]]
    choice = min(ql, key=lambda k: res[k]["mean_rel_early"]) if ql else None
    return {"candidates": res, "closed_loop": choice}


def summarize(d):
    recs, eps_info, checks = load(d)
    S = {"checks": checks, "n": {s: len(sel(recs, s, "all")) for s in ("B0", "B2", "pooled")}, "rank": {}, "geo": {}, "timing": {}}
    for stratum in ("pooled", "B0", "B2"):
        S["rank"][stratum] = {}
        for sub in ("all",) + PHASES + ("cov20",):
            rr = sel(recs, stratum, sub)
            if not rr:
                continue
            means = {f"{v}_{m}": bootm(rr, f"{v}_{m}") for v in VARS for m in ("top1", "regret", "rho", "tau", "gain", "chg10", "chg100")}
            means["oracle_max"] = bootm(rr, "oracle_max")
            pv = {}
            for v in VARS:
                for m, b in (("regret", "lower"), ("top1", "higher"), ("rho", "higher"), ("tau", "higher"), ("gain", "higher")):
                    pv[f"{v}_{m}"] = paired(rr, f"{v}_{m}", f"I100_{m}" if v.startswith("I") or v.startswith("O") else f"S10_{m}", b)
            p10 = {f"{v}_{m}": paired(rr, f"{v}_{m}", f"I10_{m}", b) for v in VARS for m, b in (("regret", "lower"), ("top1", "higher"))}
            S["rank"][stratum][sub] = {"n_states": len(rr), "means": means, "paired_vs_I100": pv, "paired_vs_I10": p10}
    geokeys = sorted({k for r in recs for k in r if k.startswith("D")})
    for stratum in ("pooled",):
        for sub in ("all",) + PHASES:
            rr = sel(recs, stratum, sub)
            S["geo"][sub] = {k: bootm(rr, k) for k in geokeys}
            S["geo"][sub]["per_scene"] = {k: hm(rr, k) for k in geokeys if any(k.endswith(s) for s in (
                "_cand_tk", "_cand_ph1_known", "_mae", "_cand_err", "_band_ratio"))}
    for x in DENS:
        tm = {}
        for k in ("rebuild_s", "step_s", "encode_s", "geometry_s", "peak_gb"):
            v = [e["passes"][str(x)][k] for e in eps_info.values() if e["passes"][str(x)].get(k) is not None]
            tm[k] = float(np.mean(v)) if v else None
        S["timing"][str(x)] = tm
    fr = np.asarray([f for e in eps_info.values() for f in e["frames"]], float)
    fv = fr[fr[:, 0] > 0]
    S["frames"] = {"n_frames": int(len(fr)), "n_frames_no_valid_pixel": int((fr[:, 0] == 0).sum()), "mean_valid": float(fr[:, 0].mean()),
                   "fraction": {str(x): float((fv[:, i + 1] / fv[:, 0]).mean()) for i, x in enumerate(DENS)}}
    S["episode_seconds"] = float(np.sum([e.get("seconds", 0) for e in eps_info.values()]))
    return S, recs


# ------------------------------------------------------------------------------------------------------------ figures
def line(ax, xs, ys, lo=None, hi=None, **kw):
    ax.plot(xs, ys, marker="o", ms=4, lw=1.8, **kw)
    if lo is not None:
        ax.fill_between(xs, lo, hi, color=kw.get("color"), alpha=0.12, lw=0)


def ser(G, key_fmt, sub="all", scale=1.0):
    m, lo, hi = [], [], []
    for x in DENS:
        b = G[sub].get(key_fmt.format(x=x)) or {}
        mv = b.get("mean")
        m.append(np.nan if mv is None else scale * mv)
        ci = b.get("ci95") or [None, None]
        lo.append(np.nan if ci[0] is None else scale * ci[0])
        hi.append(np.nan if ci[1] is None else scale * ci[1])
    return np.asarray(m), np.asarray(lo), np.asarray(hi)


def fig_geometry(S):
    fig, axs = plt.subplots(2, 4, figsize=(17, 7.6))
    for i, d in enumerate(("indoor", "outdoor")):
        G = S[d]["geo"]
        un = "m" if d == "indoor" else "scene units"
        panels = [("known-surface hit ratio (t_known valid)\namong rays with a true surface", [("D{x}_cand_tk", "candidate cameras", "-"), ("D{x}_cur_tk", "current camera", "--")], 100, "%"),
                  ("median |known depth - true depth|", [("D{x}_cand_err", "candidate cameras", "-"), ("D{x}_cur_err", "current camera", "--")], 1, un),
                  ("phantom rate (known surface > tol in FRONT of GT)\ntol = TRUNC + 1% z", [("D{x}_cand_ph1_known", "candidates / known-GT rays", "-"),
                                                                                      ("D{x}_cand_ph1_all", "candidates / all GT rays", "--"),
                                                                                      ("D{x}_cur_ph1_known", "current / known-GT rays", ":")], 100, "%"),
                  ("3D-encoder input: mean coarse BAND fraction", [("D{x}_coarse_band", "8^3 blocks with a domain", "-")], 100, "%")]
        for j, (tt, ks, sc, unit) in enumerate(panels):
            ax = axs[i, j]
            for key, lab, ls in ks:
                m, lo, hi = ser(G, key, scale=sc)
                line(ax, DENS, m, lo, hi, color=ST.BLUE[11] if ls == "-" else ST.INK2, ls=ls, label=lab)
            ax.set_xticks(DENS)
            ax.set_xticklabels([f"{x}%" for x in DENS])
            ax.set_title(f"{NAME[d]}: {tt}", loc="left")
            ax.set_ylabel(unit)
            ST.clean(ax)
            ax.legend(fontsize=7.5, loc="best")
    fig.supxlabel("BAND density (share of the valid depth pixels of every frame written into the known-surface band)")
    fig.suptitle("Task 3 / E1.2 Figure 1 — density vs geometry quality (measured; fixed states of the B0 and B2 trajectories, scene-level mean and 95% CI)",
                 x=0.01, ha="left", fontsize=11)
    fig.tight_layout()
    fig.savefig(os.path.join(FIG, "fig1_geometry.png"))
    plt.close(fig)


def fig_prediction(S):
    fig, axs = plt.subplots(2, 4, figsize=(17, 7.6))
    for i, d in enumerate(("indoor", "outdoor")):
        G = S[d]["geo"]
        un = "m" if d == "indoor" else "scene units"
        ax = axs[i, 0]
        for key, lab, c in (("D{x}_mae", "all rays with both hits", ST.INK), ("D{x}_mae_old", "true surface already observed", ST.BLUE[8]),
                            ("D{x}_mae_new", "true surface NEW", "#d03b3b")):
            m, lo, hi = ser(G, key)
            line(ax, DENS, m, lo, hi, color=c, label=lab)
        ax.set_title(f"{NAME[d]}: future-depth MAE (K = 4, candidate views)", loc="left")
        ax.set_ylabel(un)
        ax = axs[i, 1]
        for b, lab, c in (("b0", "z < 0.5 z0", ST.BLUE[4]), ("b1", "0.5-1 z0", ST.BLUE[7]), ("b2", "1-2 z0", ST.BLUE[10]), ("b3", "> 2 z0", ST.BLUE[12])):
            m, lo, hi = ser(G, "D{x}_mae_" + b)
            line(ax, DENS, m, None, None, color=c, label=lab)
        ax.set_title(f"{NAME[d]}: MAE by true distance", loc="left")
        ax.set_ylabel(un)
        ax = axs[i, 2]
        for key, lab, c, sc in (("D{x}_hit_agree", "hit / no-hit agreement with GT", ST.INK, 100), ("D{x}_hit_dis", "K samples disagree on hit", "#eb6834", 100)):
            m, lo, hi = ser(G, key, scale=sc)
            line(ax, DENS, m, lo, hi, color=c, label=lab)
        ax.set_title(f"{NAME[d]}: hit validity", loc="left")
        ax.set_ylabel("%")
        ax = axs[i, 3]
        for key, lab, c in (("D{x}_rel_std", "relative spread of K depths", ST.BLUE[9]), ("D{x}_spread_err_rho", "Spearman(spread, |error|)", "#1baf7a")):
            m, lo, hi = ser(G, key)
            line(ax, DENS, m, lo, hi, color=c, label=lab)
        ax.set_title(f"{NAME[d]}: diversity and its calibration", loc="left")
        for ax in axs[i]:
            ax.set_xticks(DENS)
            ax.set_xticklabels([f"{x}%" for x in DENS])
            ST.clean(ax)
            ax.legend(fontsize=7.5)
    fig.supxlabel("input BAND density (frozen V2-Mix, same noise)")
    fig.suptitle("Task 3 / E1.2 Figure 1b — density vs future-depth prediction (measured vs t*, evaluation only)", x=0.01, ha="left", fontsize=11)
    fig.tight_layout()
    fig.savefig(os.path.join(FIG, "fig1b_prediction.png"))
    plt.close(fig)


def fig_ranking(S, fam, fn, title):
    fig, axs = plt.subplots(2, 3, figsize=(15, 7.6))
    for i, d in enumerate(("indoor", "outdoor")):
        R = S[d]["rank"]["pooled"]
        for j, (m, lab, sc) in enumerate((("regret", "oracle regret (pp of coverage, lower = better)", 1), ("top1", "Top-1 accuracy", 100),
                                          ("rho", "Spearman(score, oracle gain)", 1))):
            ax = axs[i, j]
            for sub in ("early", "middle", "late", "all"):
                if sub not in R:
                    continue
                M = R[sub]["means"]
                nz = lambda v: np.nan if v is None else sc * v  # noqa: E731
                ys = [nz(M[f"{fam}{x}_{m}"]["mean"]) for x in DENS]
                lo = [nz(M[f"{fam}{x}_{m}"]["ci95"][0]) for x in DENS] if sub in ("early", "all") else None
                hi = [nz(M[f"{fam}{x}_{m}"]["ci95"][1]) for x in DENS] if sub in ("early", "all") else None
                line(ax, DENS, ys, lo, hi, color=PCOL[sub], label=f"{sub} ({R[sub]['n_states']} states)", ls="-" if sub != "all" else "--")
            ax.set_xticks(DENS)
            ax.set_xticklabels([f"{x}%" for x in DENS])
            ax.set_title(f"{NAME[d]}: {lab}", loc="left")
            ST.clean(ax)
            ax.legend(fontsize=7.5)
    fig.supxlabel("input BAND density, DistP1 scored with the sparse 10% map" if fam == "I" else "DistP1 scoring BAND density, model input = 100% map")
    fig.suptitle(title, x=0.01, ha="left", fontsize=11)
    fig.tight_layout()
    fig.savefig(os.path.join(FIG, fn))
    plt.close(fig)


def fig_summary(S):
    rows = [("Early regret (pp) [I]", lambda d, x: S[d]["rank"]["pooled"]["early"]["means"][f"I{x}_regret"]["mean"], "lower"),
            ("All regret (pp) [I]", lambda d, x: S[d]["rank"]["pooled"]["all"]["means"][f"I{x}_regret"]["mean"], "lower"),
            ("Top-1 [I]", lambda d, x: S[d]["rank"]["pooled"]["all"]["means"][f"I{x}_top1"]["mean"], "higher"),
            ("Spearman [I]", lambda d, x: S[d]["rank"]["pooled"]["all"]["means"][f"I{x}_rho"]["mean"], "higher"),
            ("All regret (pp) [S]", lambda d, x: S[d]["rank"]["pooled"]["all"]["means"][f"S{x}_regret"]["mean"], "lower"),
            ("Pred. MAE", lambda d, x: S[d]["geo"]["all"][f"D{x}_mae"]["mean"], "lower"),
            ("t_known valid (cand.)", lambda d, x: S[d]["geo"]["all"][f"D{x}_cand_tk"]["mean"], "higher"),
            ("phantom / known (cand.)", lambda d, x: S[d]["geo"]["all"][f"D{x}_cand_ph1_known"]["mean"], "lower"),
            ("GT new surface wrongly excluded", lambda d, x: S[d]["geo"]["all"][f"D{x}_gt_fexcl_dil"]["mean"], "lower"),
            ("fusion time / step (s)", lambda d, x: S[d]["timing"][str(x)]["step_s"], "lower"),
            ("peak GPU memory (GB)", lambda d, x: S[d]["timing"][str(x)]["peak_gb"], "lower")]
    fig, axs = plt.subplots(1, 2, figsize=(15, 6.2))
    for ax, d in zip(axs, ("indoor", "outdoor")):
        V = np.asarray([[f(d, x) if f(d, x) is not None else np.nan for x in DENS] for _, f, _ in rows], float)
        rel = np.zeros_like(V)
        for r_, (_, _, b) in enumerate(rows):
            ref = V[r_, 0]
            s = 1 if b == "higher" else -1
            rel[r_] = np.clip(s * (V[r_] - ref) / (abs(ref) + 1e-12), -1, 1) if np.isfinite(ref) else 0
        ax.imshow(rel, cmap=ST.DIV.reversed(), vmin=-0.6, vmax=0.6, aspect="auto")
        for r_ in range(V.shape[0]):
            for c in range(V.shape[1]):
                ax.text(c, r_, f"{V[r_, c]:.3g}", ha="center", va="center", fontsize=8.5, color=ST.INK)
        ax.set_xticks(range(len(DENS)))
        ax.set_xticklabels([f"D{x}" for x in DENS])
        ax.set_yticks(range(len(rows)))
        ax.set_yticklabels([r[0] for r in rows])
        ax.set_title(f"{NAME[d]} (all numbers measured on fixed states; colour = better (blue) / worse (red) than D10)", loc="left", fontsize=9)
    fig.suptitle("Task 3 / E1.2 Figure 6 — summary: [I] input-density ablation (sparse scoring), [S] scoring-density ablation (100% input); no closed-loop numbers",
                 x=0.01, ha="left", fontsize=11)
    fig.tight_layout()
    fig.savefig(os.path.join(FIG, "fig6_summary.png"))
    plt.close(fig)


# ------------------------------------------------------------------------------------------------------------- tables
def f3(b, nd=3, sc=1.0):
    if b is None or b.get("mean") is None:
        return "—"
    if b.get("ci95", [None])[0] is None or b.get("scenes", 0) < 4:
        return f"{sc * b['mean']:.{nd}f}"
    return f"{sc * b['mean']:.{nd}f} [{sc * b['ci95'][0]:.{nd}f}, {sc * b['ci95'][1]:.{nd}f}]"


def tables(S):
    L = ["# Task 3 / E1.2 — 固定状态结果表（自动生成，`e12_analyze.py`）", ""]
    for d in ("indoor", "outdoor"):
        X = S[d]
        L += [f"## {NAME[d]}", "", f"- 状态数：S_B0 {X['n']['B0']}，S_B2 {X['n']['B2']}，合并去重 {X['n']['pooled']}",
              f"- 检查：{json.dumps(X['checks'])}",
              f"- 每帧实际比例：{json.dumps({k: round(v, 4) for k, v in X['frames']['fraction'].items()})}（{X['frames']['n_frames']} 帧）", ""]
        for stratum in ("pooled", "B0", "B2"):
            R = X["rank"][stratum]
            L += [f"### 排序（{stratum}）— 输入消融 I（稀疏评分）与评分消融 S（100% 输入）", "",
                  "| 阶段 | 状态数 | 指标 | " + " | ".join(f"I{x}" for x in DENS) + " | " + " | ".join(f"S{x}" for x in DENS) + " |",
                  "|---|---|---|" + "---|" * 8]
            for sub in ("early", "middle", "late", "cov20", "all"):
                if sub not in R:
                    continue
                for m, sc, nd in (("regret", 1, 3), ("top1", 1, 3), ("rho", 1, 3)):
                    L.append(f"| {sub} | {R[sub]['n_states']} | {m} | " + " | ".join(f3(R[sub]['means'][f'I{x}_{m}'], nd) for x in DENS) + " | "
                             + " | ".join(f3(R[sub]['means'][f'S{x}_{m}'], nd) for x in DENS) + " |")
            L += ["", f"**配对差（{stratum}）**：输入消融相对 I100，评分消融相对 S10（场景 bootstrap；胜/负 = 更好/更差的场景数）", "",
                  "| 阶段 | 比较 | Δregret | 胜/负 | ΔTop-1 | 胜/负 | ΔSpearman |", "|---|---|---|---|---|---|---|"]
            for sub in ("early", "middle", "late", "cov20", "all"):
                if sub not in R:
                    continue
                for v in ("I10", "I50", "I75", "S50", "S75", "S100"):
                    pr, pt, ps = (R[sub]["paired_vs_I100"][f"{v}_{m}"] for m in ("regret", "top1", "rho"))
                    ref = "I100" if v.startswith("I") else "S10"
                    L.append(f"| {sub} | {v} − {ref} | {f3(pr)} | {pr['wins']}/{pr['losses']} | {f3(pt)} | {pt['wins']}/{pt['losses']} | {f3(ps)} |")
            L.append("")
        G = X["geo"]["all"]
        L += ["### 几何（合并，全部状态；候选相机 = cand，当前相机 = cur）", "", "| 指标 | " + " | ".join(f"D{x}" for x in DENS) + " |", "|---|" + "---|" * 4]
        for k, lab, sc in (("band_ratio", "BAND / domain 体素", 100), ("empty_ratio", "EMPTY / domain", 100), ("unknown_ratio", "UNKNOWN / domain", 100),
                           ("coarse_band", "粗网格 BAND 占比", 100), ("cur_tk", "t_known 有效率（当前）", 100), ("cand_tk", "t_known 有效率（候选）", 100),
                           ("cand_match", "深度一致率（候选）", 100), ("cand_err", "已知深度中位误差（候选）", 1),
                           ("cand_ph1_known", "伪表面 / 已知∩GT（候选，tol×1）", 100), ("cand_ph1_all", "伪表面 / 全部 GT（候选，tol×1）", 100),
                           ("cand_ph2_known", "伪表面 / 已知∩GT（tol×2）", 100), ("cand_ph3_known", "伪表面 / 已知∩GT（tol×3）", 100),
                           ("cur_ph1_known", "伪表面 / 已知∩GT（当前）", 100), ("cand_entry", "伪表面中入口零交叉比例", 100),
                           ("cand_edge_share", "伪表面中深度边缘射线比例", 100), ("cand_edge_base", "深度边缘射线基础比例", 100),
                           ("cand_ph_edge", "边缘射线的伪表面率", 100), ("cand_ph_int", "非边缘射线的伪表面率", 100),
                           ("cand_ph_mag", "伪表面偏前中位数（TRUNC）", 1), ("cand_off_bf", "首个 BAND − GT 中位（TRUNC）", 1),
                           ("cand_off_zero", "零交叉 − GT 中位（TRUNC）", 1), ("cand_off_ee", "EMPTY 结束 − GT 中位（TRUNC）", 1),
                           ("cur_off_bf", "首个 BAND − GT（当前，TRUNC）", 1), ("cur_off_zero", "零交叉 − GT（当前，TRUNC）", 1),
                           ("cur_off_ee", "EMPTY 结束 − GT（当前，TRUNC）", 1), ("cand_bf_front", "首个 BAND 在 GT 前 > tol 的比例", 100),
                           ("cand_nseg", "UNKNOWN 区段数（候选）", 1), ("cand_lu", "UNKNOWN 长度（候选）", 1)):
            L.append(f"| {lab} | " + " | ".join(f3(G.get(f'D{x}_{k}'), 3, sc) for x in DENS) + " |")
        L += ["", "### 预测（本密度输入，K = 4，对比 t*）", "", "| 指标 | " + " | ".join(f"D{x}" for x in DENS) + " |", "|---|" + "---|" * 4]
        for k, lab, sc in (("mae", "MAE", 1), ("rmse", "RMSE", 1), ("absrel", "AbsRel", 1), ("hit_agree", "命中一致率 %", 100),
                           ("mae_old", "MAE（GT 已观测表面）", 1), ("mae_new", "MAE（GT 新表面）", 1), ("mae_b0", "MAE z<0.5z0", 1),
                           ("mae_b1", "MAE 0.5–1 z0", 1), ("mae_b2", "MAE 1–2 z0", 1), ("mae_b3", "MAE ≥2 z0", 1),
                           ("rel_std", "K 间相对离散度", 1), ("hit_dis", "K 间命中不一致 %", 100), ("spread_err_rho", "离散度–误差 Spearman", 1)):
            L.append(f"| {lab} | " + " | ".join(f3(G.get(f'D{x}_{k}'), 3, sc) for x in DENS) + " |")
        for sub in PHASES:
            Gp = X["geo"][sub]
            L.append(f"| MAE（{sub}） | " + " | ".join(f3(Gp.get(f'D{x}_mae')) for x in DENS) + " |")
        L += ["", "### 新颖性掩码（评分端；真值 = GT 点到历史 10% 采样点的距离 > 覆盖 eps）", "",
              "| 来源 | 指标 | " + " | ".join(f"D{x}" for x in DENS) + " |", "|---|---|" + "---|" * 4]
        for src, nm in (("gt", "GT 深度"), ("pred100", "D100 预测")):
            for k, lab in (("excl_band", "被 BAND 排除 %"), ("excl_dil", "被 band_dil 排除 %"), ("fexcl_band", "真值新但被 BAND 排除 %"),
                           ("fexcl_dil", "真值新但被 band_dil 排除 %"), ("frew_band", "真值旧但未被 BAND 排除 %"),
                           ("frew_dil", "真值旧但未被 band_dil 排除 %"), ("fexcl_acc_share", "错误排除中预测准确的比例 %")):
                if any(G.get(f"D{x}_{src}_{k}") for x in DENS):
                    L.append(f"| {nm} | {lab} | " + " | ".join(f3(G.get(f'D{x}_{src}_{k}'), 2, 100) for x in DENS) + " |")
            for k, lab in (("onlydense_rate", "仅稠密排除（相对 D10 命中）%"), ("onlydense_newshare", "其中真值为新 %"), ("onlydense_edgeshare", "其中在深度边缘 %")):
                L.append(f"| {nm} | {lab} | — | " + " | ".join(f3(G.get(f'D{x}_{src}_{k}'), 2, 100) for x in (50, 75, 100)) + " |")
        L += ["", "### 时间与显存（每条 episode 的均值，非独占计时）", "", "| 项目 | " + " | ".join(f"D{x}" for x in DENS) + " |", "|---|" + "---|" * 4]
        for k, lab in (("rebuild_s", "区域重建 s"), ("step_s", "单步融合 s"), ("geometry_s", "几何构建 s"), ("encode_s", "3D 编码 s"), ("peak_gb", "峰值显存 GB")):
            L.append(f"| {lab} | " + " | ".join("—" if X['timing'][str(x)][k] is None else f"{X['timing'][str(x)][k]:.3f}" for x in DENS) + " |")
        for k, lab in (("time_projection_s", "射线特征（投影）s"), ("time_cond_s", "条件编码 s"), ("time_sampling_s", "Flow 采样 s"), ("time_total_s", "评分总计 s")):
            L.append(f"| {lab} | " + " | ".join(f3(G.get(f'D{x}_{k}')) for x in DENS) + " |")
        L.append("")
    return "\n".join(L)


def main():
    os.makedirs(OUT, exist_ok=True)
    os.makedirs(FIG, exist_ok=True)
    ST.setup()
    S, R = {}, {}
    for d in ("indoor", "outdoor"):
        S[d], R[d] = summarize(d)
        print(d, S[d]["n"], S[d]["checks"], flush=True)
    S["selection"] = selection(S)
    json.dump(S, open(os.path.join(OUT, "e12_summary.json"), "w"), indent=1, default=float)
    json.dump({d: R[d] for d in R}, open(os.path.join(OUT, "e12_states.json"), "w"), default=float)
    open(os.path.join(OUT, "tables.md"), "w").write(tables(S))
    fig_geometry(S)
    fig_prediction(S)
    fig_ranking(S, "I", "fig2_ranking_input.png", "Task 3 / E1.2 Figure 2 — input-density ablation: candidate ranking by phase (measured; pooled states, scene-level mean, 95% CI shaded for early / all)")
    fig_ranking(S, "S", "fig2b_ranking_scoring.png", "Task 3 / E1.2 Figure 2b — scoring-density ablation (100% input): candidate ranking by phase (measured)")
    fig_summary(S)
    print(json.dumps(S["selection"], indent=1, default=float))


if __name__ == "__main__":
    main()
