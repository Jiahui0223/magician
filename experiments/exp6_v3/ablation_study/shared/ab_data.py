"""EXP6 V3 Ablation -- unified fixed-state records of both pools for the analysis scripts (CPU; numpy / scipy).

A record = one state: oracle gains (frozen evaluators), GT-depth DistP1 (GTD), the K = 4 mean DistP1 of every scorer
(historical V2O / V2M from EXP5 v2.json; B2 / E2D / T0 / T1 from the new cache), the online pick of every scorer and
the ranking metrics of protocol §2 (tie tolerance TAU on the coverage-fraction gains; regret in percentage points).
"""
import glob
import hashlib
import json
import os
import sys

import numpy as np
from scipy.stats import spearmanr

HERE = os.path.dirname(os.path.abspath(__file__))
REPO = os.path.dirname(os.path.dirname(os.path.dirname(os.path.dirname(HERE))))
W5 = os.path.join(REPO, "work_dirs", "exp5_future_view", "ablation_study")
WA = os.path.join(REPO, "work_dirs", "exp6_v3", "ablation_study")
E2W = os.path.join(REPO, "work_dirs", "exp5_future_view", "task3_full_depth_p1", "e2")
B2W = os.path.join(REPO, "work_dirs", "exp5_future_view", "task3_full_depth_p1", "e12", "abroot_B2")
T2CL = os.path.join(REPO, "work_dirs", "exp5_future_view", "task2", "eval")
A1R = os.path.join(REPO, "experiments", "exp5_future_view", "ablation_study", "A1_prediction_scoring", "results")
sys.path.insert(0, os.path.join(REPO, "experiments", "exp3b_macaronspp_transfer"))

TAU = 1e-4
NEW = ("B2", "E2D", "T0", "T1")
HIST = ("V2O", "V2M")
SCENES_IN = ("2azQ1b91cZZ_L0", "QUCTc6BB5sX_L1", "TbHJrupSAjP_L1", "Z6MFQCViBuw_L0", "oLBMNvg9in8_L1")
TEST3, DEV3 = ("alhambra", "bridge", "neuschwanstein"), ("barts", "pantheon", "colosseum")


def phase_cl(dataset, t):
    lo, hi = {"indoor": (25, 55), "outdoor": (15, 55)}[dataset]
    return "early" if t <= lo else ("mid" if t <= hi else "late")


def phase_fx(dataset, t):
    T = 150 if dataset == "indoor" else 100
    return "early" if t < 0.2 * T else ("late" if t >= 0.8 * T else "mid")


# ------------------------------------------------------------------------------------------------- online choice rules
def pick_indoor(scores, cand, name, tag, t):
    """v2nv_planner.nv_choice on the candidate set C (ab_load.pick_of key 'T2|<name>|<tag>'); returns (legal index, fallback)."""
    C = list(cand)
    s = np.asarray(scores, float)[C]
    if s.max() <= 0:
        rng = np.random.default_rng(int(hashlib.sha256(f"exp5-T0.10|NV|T2|{name}|{tag}|{t}".encode()).hexdigest()[:15], 16))
        return C[int(rng.integers(len(C)))], True
    return C[int(np.flatnonzero(s == s.max())[0])], False


def pick_outdoor(scores, scene, start, t):
    """m_score.choose: argmax (lowest index); all-zero -> m_common.fallback_choice."""
    s = np.asarray(scores, float)
    if s.max() == 0.0:
        import m_common as MC
        return MC.fallback_choice(scene, int(start), int(t), len(s)), True
    return int(np.flatnonzero(s == s.max())[0]), False


def pick(dataset, scores, st, name):
    if dataset == "indoor":
        return pick_indoor(scores, st["cand"], name, st["tag"], st["t"])
    return pick_outdoor(scores, st["scene"], st["start"], st["t"])


# ------------------------------------------------------------------------------------------------- metrics
def rank_metrics(s, g, pk_pos, tau=TAU):
    """s, g over the candidate set; pk_pos = position of the pick inside the candidate set."""
    s, g = np.asarray(s, float), np.asarray(g, float)
    gmax = g.max()
    out = {"top1": bool(g[pk_pos] >= gmax - tau), "top1_strict": bool(g[pk_pos] >= gmax), "regret": float((gmax - g[pk_pos]) * 100),
           "gmax": float(gmax * 100), "n_cand": int(len(g)), "n_best": int((g >= gmax - tau).sum())}
    out["nregret"] = float((gmax - g[pk_pos]) / gmax) if gmax > tau else None
    out["rho"] = None
    if len(g) >= 2 and s.std() > 0 and g.std() > 0:
        r = spearmanr(s, g).correlation
        out["rho"] = float(r) if np.isfinite(r) else None
    agree, n = 0.0, 0
    for i in range(len(g)):
        for j in range(i + 1, len(g)):
            if abs(g[i] - g[j]) > tau:
                n += 1
                agree += 0.5 if s[i] == s[j] else float(np.sign(s[i] - s[j]) == np.sign(g[i] - g[j]))
    out["pairwise"] = agree / n if n else None
    return out


def random_reference(g, tau=TAU):
    g = np.asarray(g, float); gmax = g.max()
    return {"top1": float((g >= gmax - tau).mean()), "regret": float((gmax - g.mean()) * 100)}


def spearman(a, b):
    a, b = np.asarray(a, float), np.asarray(b, float)
    if len(a) < 2 or a.std() == 0 or b.std() == 0:
        return None
    r = spearmanr(a, b).correlation
    return float(r) if np.isfinite(r) else None


# ------------------------------------------------------------------------------------------------- loading
def _cache(pool, dataset):
    out = {}
    for f in glob.glob(os.path.join(WA, "cache", pool, dataset, "*.json")):
        j = json.load(open(f))
        out[j["sid"]] = j
    return out


def episode_curve_indoor(source, tag):
    p = os.path.join(T2CL, "indoor", "closed_loop", source, tag, "episode.json")
    return np.asarray(json.load(open(p))["coverage"], float) if os.path.exists(p) else None


def load_pool(pool, dataset, scorers=None):
    """List of state records. pool 'cl' adds the historical V2O / V2M scores and the EXP5 GTD; 'fx' = valsel (indoor) /
    dev3 stratum B2 (outdoor) from the cache alone (GTD computed by ab_score)."""
    cache = _cache(pool, dataset)
    recs, missing = [], 0
    if pool == "cl":
        man = json.load(open(os.path.join(W5, "manifests", f"states_{dataset}.json")))["states"]
        region = json.load(open(os.path.join(A1R, "a1_region_diag.json")))["states"] if dataset == "indoor" else {}
        rel = {r["sid"]: r for r in json.load(open(os.path.join(A1R, "a1_relevance_diag.json")))[dataset]["rows"]}
        curves = {}
        for st in man:
            sid = st["sid"]
            if sid not in cache:
                missing += 1
                continue
            c = cache[sid]
            v2 = json.load(open(os.path.join(W5, "states", dataset, sid, "v2.json")))
            orc = np.load(os.path.join(W5, "states", dataset, sid, "oracle.npz"))
            src = st["sources"][0]["source"]
            r = {"pool": pool, "dataset": dataset, "sid": sid, "scene": st["scene"], "start": st["start"], "tag": st["tag"], "t": int(st["t"]),
                 "source": src, "episode": f"{st['tag']}|{src}", "phase": phase_cl(dataset, int(st["t"])), "primary_t": bool(st["primary_t"]),
                 "cand": [int(x) for x in st["cand"]], "n_legal": int(st["n_legal"]), "gain": np.asarray(orc["gain"], float),
                 "gtd": np.asarray(v2["gtd_distp1"], float), "gtd_new": np.asarray(c["gtd_distp1"], float),
                 "scores": {m: np.asarray(v2["models"][m]["distp1"], float) for m in HIST}, "distp1_k": {},
                 "regression": c.get("regression", {}), "split": st.get("split", "")}
            if dataset == "outdoor":
                r["gain_official"] = np.asarray(orc["gain_official"], float)
                r["split"] = "test" if st["scene"] in TEST3 else "dev"
            for m in NEW:
                r["scores"][m] = np.asarray(c["models"][m]["distp1"], float)
                r["distp1_k"][m] = np.asarray(c["models"][m]["distp1_k"], float)
            r["gtd_eq_exp5"] = bool(np.array_equal(r["gtd"], r["gtd_new"]))
            if dataset == "indoor" and sid in region:
                r["gain_in"] = np.asarray(region[sid]["gain_in"], float)
                r["remaining_in_region_share"] = float(region[sid]["remaining_in_region_share"])
                key = (src, st["tag"])
                if key not in curves:
                    curves[key] = episode_curve_indoor(src, st["tag"])
                cv = curves[key]
                t = int(st["t"])
                r["stall"] = bool(cv is not None and t >= 30 and (cv[t] - cv[t - 30]) < 0.01)
                r["cov_gain_30"] = float(cv[t] - cv[t - 30]) if (cv is not None and t >= 30) else None
            if sid in rel:
                r["relevance"] = rel[sid]
            recs.append(r)
    else:
        for sid, c in cache.items():
            orc_dir = os.path.join(E2W, "valsel", "states", "indoor", sid) if dataset == "indoor" else os.path.join(B2W, "states", "outdoor", sid)
            orc = np.load(os.path.join(orc_dir, "oracle.npz"))
            r = {"pool": pool, "dataset": dataset, "sid": sid, "scene": c["scene"], "start": c["start"], "tag": c["tag"].split("|", 1)[1], "t": int(c["t"]),
                 "source": c["source"], "episode": c["tag"], "phase": phase_fx(dataset, int(c["t"])), "primary_t": True,
                 "cand": [int(x) for x in c["cand"]], "n_legal": int(c["n_legal"]), "gain": np.asarray(orc["gain"], float),
                 "gtd": np.asarray(c["gtd_distp1"], float), "gtd_new": np.asarray(c["gtd_distp1"], float), "gtd_eq_exp5": None,
                 "scores": {m: np.asarray(c["models"][m]["distp1"], float) for m in NEW},
                 "distp1_k": {m: np.asarray(c["models"][m]["distp1_k"], float) for m in NEW}, "regression": c.get("regression", {}),
                 "split": "mp3d_valsel" if dataset == "indoor" else "dev"}
            if dataset == "outdoor" and "gain_official" in orc.files:
                r["gain_official"] = np.asarray(orc["gain_official"], float)
            recs.append(r)
    for r in recs:
        r["scores"]["GTD"] = r["gtd"]
        r["picks"], r["fallback"], r["metrics"] = {}, {}, {}
        cand = r["cand"]
        g = r["gain"][cand]
        for name, s in r["scores"].items():
            pk, fb = pick(dataset, s, r, name)
            r["picks"][name], r["fallback"][name] = int(pk), bool(fb)
            r["metrics"][name] = rank_metrics(s[cand], g, cand.index(pk))
        r["random"] = random_reference(g)
        r["gstar"] = float(g.max() * 100)
        r["fidelity"] = {m: spearman(r["scores"][m][cand], r["gtd"][cand]) for m in r["scores"] if m != "GTD"}
    recs.sort(key=lambda r: (r["scene"], r["start"], r["episode"], r["t"]))
    return recs, missing


# ------------------------------------------------------------------------------------------------- aggregation
def hier_mean(recs, value):
    """state -> episode -> scene means; value(r) -> float or None (skipped). Returns {scene: mean}."""
    ep = {}
    for r in recs:
        v = value(r)
        if v is None:
            continue
        ep.setdefault((r["scene"], r["episode"]), []).append(float(v))
    sc = {}
    for (scene, _), vs in ep.items():
        sc.setdefault(scene, []).append(float(np.mean(vs)))
    return {s: float(np.mean(v)) for s, v in sc.items()}


def boot(scene_vals, n=10000, seed=0):
    v = np.asarray(list(scene_vals), float)
    if len(v) == 0:
        return None
    rng = np.random.default_rng(seed)
    idx = rng.integers(0, len(v), size=(n, len(v)))
    m = v[idx].mean(1)
    return {"mean": float(v.mean()), "lo": float(np.percentile(m, 2.5)), "hi": float(np.percentile(m, 97.5)), "n_scenes": int(len(v))}


def paired(sa, sb, n=10000, seed=0):
    """Scene-paired difference a - b over the scenes present in both."""
    sc = sorted(set(sa) & set(sb))
    d = np.asarray([sa[s] - sb[s] for s in sc], float)
    if len(d) == 0:
        return None
    b = boot(d, n, seed)
    b.update({"wins": int((d < 0).sum()), "losses": int((d > 0).sum()), "per_scene": {s: float(x) for s, x in zip(sc, d)}})
    return b


def fmt(b, nd=3, pct=False):
    if b is None:
        return "—"
    k = 100 if pct else 1
    return f"{b['mean'] * k:.{nd}f} [{b['lo'] * k:.{nd}f}, {b['hi'] * k:.{nd}f}]"
