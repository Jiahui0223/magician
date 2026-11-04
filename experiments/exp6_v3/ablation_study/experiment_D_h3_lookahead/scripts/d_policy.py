"""EXP6 V3 / ablation D -- H3 lookahead extensions on the frozen V3-T1 (seen2scene env, planner process only).
Same mechanism as experiments/exp6_v3/H/scripts/h_policy.py (imported read-only for value_cov, union, the settings-grid
neighbour rule, the known-surface segment test and the observation capture): the frozen score_state runs unchanged
(keep_hits=True), the policy score replaces out["distp1"], m_score.choose is unchanged. Policy (env D_POLICY, config env
D_CONFIG = configs/d_config.yaml): depth / beam / lambda_risk of the per-root beam search described in the config.
d0ref (depth 2, lambda 0.25, exhaustive) must reproduce H3 bitwise."""
import json
import os
import sys
import time

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
X = os.path.dirname(os.path.dirname(os.path.dirname(os.path.dirname(HERE))))
sys.path.insert(0, os.path.join(X, "exp6_v3", "H", "scripts"))
import h_policy as HP  # noqa: E402


def load_cfg():
    import yaml
    p = os.environ["D_CONFIG"]
    return yaml.safe_load(open(p)), HP.sha256_file(p)


def q_of(sets_k, lam):
    vals = [float(s[1].sum()) for s in sets_k]
    return float(np.mean(vals)) - lam * float(np.std(vals))


def union_sets(a, b):
    """Per-sample union (ids, weights) of two cell sets (max weight per cell)."""
    if len(b[0]) == 0:
        return a
    if len(a[0]) == 0:
        return b
    ids = np.concatenate([a[0], b[0]])
    w = np.concatenate([a[1], b[1]])
    u, inv = np.unique(ids, return_inverse=True)
    wv = np.zeros(len(u))
    np.maximum.at(wv, inv, w)
    return u, wv


def lookahead(core, geo, h, eps, z0, sets, s_one, pc, orig_score, stats):
    """Per-root beam lookahead. sets: per legal view, per sample (ids, weights) of the frozen score_state predictions."""
    rec, P = HP.STATE["rec"], HP.STATE["pose"]
    legal = [tuple(int(x) for x in r) for r in np.asarray(rec["legal_idx"]).reshape(-1, 5)]
    depth, beam, lam = int(pc["depth"]), pc.get("beam"), float(pc["lambda_risk"])
    K = eps.shape[0]
    pred = {c: sets[i] for i, c in enumerate(legal)}
    vcfg = {"eps_cov": pc["eps_cov"], "cell_size": pc["cell_size"]}

    def ensure(poses):
        need = [c for c in dict.fromkeys(poses) if c not in pred]
        chunk = int(pc.get("c2_chunk", 16))
        for s0 in range(0, len(need), chunk):
            part = need[s0:s0 + chunk]
            o2, _, v2 = orig_score(core, geo, h, [P["R"][P["lut"][c]] for c in part], [P["T"][P["lut"][c]] for c in part], eps, z0, keep_hits=True)
            for q, c in enumerate(part):
                tk = o2["hits"][q]["t"]
                pred[c] = [HP.value_cov(tk[k], v2[q], geo, z0, vcfg) for k in range(K)]
            del o2, v2
        stats["predicted_views"] += len(need)

    def children(path):
        vis = HP.STATE["visited"] | set(path)
        nb = [tuple(int(x) for x in r) for r in HP.neighbours(path[-1], P["grid"])]
        nb = [n for n in nb if n not in path]
        new = [n for n in nb if n not in vis]
        cand = new if new else nb
        out = []
        for c in cand:
            if HP.segment_blocked(geo, P["C"][P["lut"][path[-1]]], P["C"][P["lut"][c]]):
                stats["blocked"] += 1
                continue
            out.append(c)
        return out

    score = np.asarray(s_one, float).copy()
    per_root = []
    # layer 2 for all roots first, predictions requested in the same order / chunks as H3 (bitwise reproduction of H3)
    ch2 = [children((c1,)) for c1 in legal]
    ensure([c for lst in ch2 for c in lst])
    for i, c1 in enumerate(legal):
        frontier = [((c1,), pred[c1])]                # (path, per-sample union sets)
        best, best_path, nodes = None, None, 0
        for L in range(2, depth + 1):
            exp = []
            for path, U in frontier:
                ch = ch2[i] if L == 2 else children(path)
                if L > 2:
                    ensure(ch)
                for c in ch:
                    U2 = [union_sets(U[k], pred[c][k]) for k in range(K)]
                    exp.append((path + (c,), U2, q_of(U2, lam)))
            nodes += len(exp)
            if not exp:
                break
            layer_best = max(exp, key=lambda e: e[2])  # first maximal element (generation order) on ties
            best, best_path = layer_best[2], layer_best[0]
            if L < depth:
                order = sorted(range(len(exp)), key=lambda j: -exp[j][2])   # stable: ties keep generation order
                keep = order if beam is None else order[:int(beam)]
                frontier = [(exp[j][0], exp[j][1]) for j in keep]
        if best is not None:
            score[i] = best
        per_root.append({"c1": list(c1), "nodes": nodes, "best_path": [list(p) for p in best_path] if best_path else None, "q": best, "s_one": float(s_one[i])})
    return score, per_root


def install(policy):
    import v2_plan_core as PC
    import m_p1
    import v2_common as LC
    cfg, cfg_sha = load_cfg()
    pc = dict(cfg["policies"][policy], **cfg["value"])
    pc["c2_chunk"] = cfg["search"].get("c2_chunk", 16)
    scene = sys.argv[sys.argv.index("--scene") + 1]
    z = np.load(os.path.join(cfg["pose_tables"], f"{scene}.npz"))
    HP.STATE["pose"] = {"lut": {tuple(int(x) for x in r): j for j, r in enumerate(z["idx"])}, "R": z["R"], "T": z["T"], "C": z["C"], "grid": json.loads(str(z["grid"]))}
    info = {"d_policy": policy, "d_policy_file_sha256": HP.sha256_file(os.path.abspath(__file__)), "h_policy_file_sha256": HP.sha256_file(HP.__file__),
            "config_sha256": cfg_sha, "config_version": cfg["version"], "policy_cfg": pc}
    sys.stderr.write("D_POLICY " + json.dumps(info) + "\n")
    for g in ("V2M", "V2O"):
        if g in LC.ARMS:
            LC.ARMS[g]["d_policy"] = info
    cls = m_p1.P1Fusion
    orig_int = cls.integrate_record

    def integrate_record(self, rec, *a, **k):
        HP.capture(rec)
        return orig_int(self, rec, *a, **k)
    cls.integrate_record = integrate_record
    orig_score = PC.score_state
    lam = float(pc["lambda_risk"])

    def score_state(core, geo, h, legal_R, legal_T, eps, z0, *a, **kw):
        want_hits = kw.pop("keep_hits", False)
        tt = time.time()
        out, tm, views = orig_score(core, geo, h, legal_R, legal_T, eps, z0, *a, keep_hits=True, **kw)
        t_frozen = time.time() - tt
        if not views:
            if not want_hits:
                out.pop("hits", None)
            return out, tm, views
        K = eps.shape[0]
        sets = [[HP.value_cov(out["hits"][m]["t"][k], v, geo, z0, pc) for k in range(K)] for m, v in enumerate(views)]
        g = np.asarray([[float(s_[1].sum()) for s_ in row] for row in sets])
        mu = np.asarray([float(np.mean(list(r))) for r in g])
        sd = g.std(1)
        s_one = mu - lam * sd
        stats = {"predicted_views": 0, "blocked": 0}
        score, per_root = lookahead(core, geo, h, eps, z0, sets, s_one, pc, orig_score, stats)
        out["h0_distp1"] = list(out["distp1"])
        out["h_value_k"] = g.tolist()
        out["h_mean"], out["h_std"], out["d_one_step"] = mu.tolist(), sd.tolist(), s_one.tolist()
        out["d"] = {"per_root": per_root, **stats}
        out["distp1"] = [float(x) for x in score]
        tm = dict(tm, frozen_score_state_s=t_frozen, policy_s=time.time() - tt - t_frozen)
        if not want_hits:
            out.pop("hits", None)
        return out, tm, views
    PC.score_state = score_state
    return info
