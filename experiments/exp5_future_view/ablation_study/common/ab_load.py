"""EXP5 Ablation -- per-state records for analysis / figures / verification (seen2scene env; CPU).

load_states(dataset) -> list of dicts with the manifest entry, oracle gains, the three scores, per-candidate metrics and
the picks of every scorer under the online choice rule (protocol §3.2). For V2 source states the pick of the source
model must equal the executed move of the stored closed loop (asserted).
"""
import json
import os
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
import ab_common as A  # noqa: E402

A.add_paths()
import numpy as np  # noqa: E402

SCORERS = ("V2O", "V2M", "GTD", "GTDn")      # GTDn: unweighted GT new-voxel count (diagnostic only)


def _rules():
    import v2nv_planner as NV
    import m_score as MS
    return NV.nv_choice, MS.choose


def pick_of(dataset, st, scores, rec, name):
    nv_choice, choose = _rules()
    if dataset == "indoor":
        pk, fb, _ = nv_choice(scores, rec, f"T2|{name}|{st['tag']}", st["t"])
        return pk, bool(fb)
    ch = choose(list(scores), st["scene"], st["start"], st["t"], True)
    return ch["pick"], ch.get("fallback") is not None


def load_states(dataset, need_v2=True):
    man = A.load_manifest(dataset)
    out = []
    for st in man["states"]:
        d = A.state_dir(dataset, st["sid"])
        po, pv = os.path.join(d, "oracle.npz"), os.path.join(d, "v2.json")
        if not os.path.exists(po) or (need_v2 and not os.path.exists(pv)):
            continue
        o = np.load(po)
        r = dict(st)
        r["src"] = st["sources"][0]["source"]
        r["gain"] = np.asarray(o["gain"], float) * 100.0                    # percentage points
        r["gain_official"] = np.asarray(o["gain_official"], float) * 100.0 if "gain_official" in o else None
        r["cov_t"] = float(o["cov_t"])
        r["C"] = np.asarray(st["cand"], int)
        if need_v2:
            v = json.load(open(pv))
            r["scores"] = {m: np.asarray(v["models"][m]["distp1"], float) for m in ("V2O", "V2M")}
            r["scores"]["GTD"] = np.asarray(v["gtd_distp1"], float)
            r["scores"]["GTDn"] = np.asarray([c["gtd_new_vox"] for c in v["models"]["V2O"]["cand"]], float)
            r["distp1_k"] = {m: np.asarray(v["models"][m]["distp1_k"], float) for m in ("V2O", "V2M")}
            r["cand"] = {m: v["models"][m]["cand"] for m in ("V2O", "V2M")}
            r["repro"] = v["repro"]
            r["timing"] = {m: v["models"][m]["timing"] for m in ("V2O", "V2M")}
            rec = {k: np.asarray(x) for k, x in np.load(os.path.join(A.REPO, st["sources"][0]["svc"])).items()
                   if k in ("legal_idx", "executed_pose_idx_history")}
            r["rec"] = rec
            r["picks"], r["fallback"] = {}, {}
            for m in SCORERS:
                r["picks"][m], r["fallback"][m] = pick_of(dataset, st, r["scores"][m], rec, m)
                assert r["picks"][m] is None or r["picks"][m] in set(r["C"].tolist()), (st["sid"], m)
            for e in st["sources"]:
                if e["source"] in ("V2O", "V2M"):
                    assert r["picks"][e["source"]] == e["executed_pick"], ("online pick not reproduced", st["sid"], e["source"])
        out.append(r)
    return out


def t_bin(dataset, t):
    lo, hi = {"indoor": (25, 55), "outdoor": (15, 55)}[dataset]
    return "early" if t <= lo else ("mid" if t <= hi else "late")


def scene_of(r):
    return r["scene"]


def episode_of(r):
    return f"{r['scene']}|s{r['start']}|{r['src']}"


def hier_mean(rows, key, group=scene_of):
    """rows: state dicts with value r[key] (None skipped) -> {scene: mean over episodes of the mean over states}."""
    ep = {}
    for r in rows:
        x = r.get(key) if isinstance(r, dict) else None
        if x is None or (isinstance(x, float) and not np.isfinite(x)):
            continue
        ep.setdefault((group(r), episode_of(r)), []).append(float(x))
    sc = {}
    for (g, _), v in ep.items():
        sc.setdefault(g, []).append(float(np.mean(v)))
    return {g: float(np.mean(v)) for g, v in sc.items()}


def boot(scene_means, n=10000, seed=0):
    g = np.asarray(list(scene_means.values()), float)
    if len(g) == 0:
        return {"mean": None, "ci95": [None, None], "scenes": 0}
    rng = np.random.default_rng(seed)
    bs = g[rng.integers(0, len(g), (n, len(g)))].mean(1)
    return {"mean": float(g.mean()), "ci95": [float(np.quantile(bs, 0.025)), float(np.quantile(bs, 0.975))], "scenes": int(len(g))}
