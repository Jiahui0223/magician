"""EXP5 Ablation A1 diagnostic (DEVIATIONS #8; evaluation side, CPU): which part of the GT-depth DistP1 "new surface"
does the coverage evaluator actually count?

Per state and candidate, the GT new-surface rays of S2 (gt_new: first GT hit in the planning domain, outside the
dilated BAND -- exactly the rays DistP1 counts with true depth) are back-projected (o + d t*, world coordinates) and
classified against the coverage evaluator's GT point set G (MP3D: frozen Coverage points that are visible from some
pose; Macarons++: the EpisodeSim GT cell points, a1_gtpoints_out.py), with the evaluator's eps:
  relevant  a GT point of G lies within eps of the hit
  useful    the nearest GT point within eps is not yet covered at t (coverage state replayed from the stored samples
            of records 0..t with the evaluator's rule)
Scorers compared (ray counts, no distance weight): GTDr = all GT-new rays, GTDrel = relevant ones, GTDuse = useful ones;
plus GTD (frozen DistP1) for reference. Output results/a1_relevance_diag.json
"""
import json
import os
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.join(os.path.dirname(HERE), "common"))
import ab_common as A  # noqa: E402

A.add_paths()
import t13_analyze as T13  # noqa: E402

import numpy as np  # noqa: E402
from scipy.spatial import cKDTree  # noqa: E402

TAU = A.TIE_TOL * 100


def eval_points(dataset, st):
    if dataset == "indoor":
        cz = np.load(os.path.join(A.REPO, "work_dirs", "exp5_future_view", "task0", "preprocessing", "coverage", f"{st['scene']}.npz"))
        return cz["pts"].astype(np.float64), cz["visible"].astype(bool), float(cz["eps_rt"])
    z = np.load(os.path.join(A.W, "gt_points", f"{st['scene']}_s{st['start']}.npz"))
    G = z["G"].astype(np.float64)
    return G, np.ones(len(G), bool), float(z["eps"])


def main():
    out = {}
    for dataset in ("indoor", "outdoor"):
        man = A.load_manifest(dataset)
        by = {}
        for st in man["states"]:
            by.setdefault((st["sources"][0]["source"], st["tag"]), []).append(st)
        rows = []
        for (src, tag), sts in sorted(by.items()):
            G, vis, eps = eval_points(dataset, sts[0])
            tree = cKDTree(G)
            cov = np.zeros(len(G), bool)
            want = {s["t"]: s for s in sts}
            ep = {"scene": sts[0]["scene"], "start": sts[0]["start"], "tag": tag}
            for t in range(max(want) + 1):
                P = np.load(A.svc_path(dataset, src, ep, t))["sample_points"].astype(np.float64)
                if len(P):
                    d, _ = cKDTree(P).query(G, k=1, distance_upper_bound=eps)
                    cov |= np.isfinite(d)
                if t not in want:
                    continue
                st = want[t]
                z = np.load(os.path.join(A.state_dir(dataset, st["sid"]), "v2.npz"))
                o = np.load(os.path.join(A.state_dir(dataset, st["sid"]), "oracle.npz"))
                v = json.load(open(os.path.join(A.state_dir(dataset, st["sid"]), "v2.json")))
                k_M, origin = float(z["k_M"]), z["origin"]
                C = np.asarray(st["cand"])
                sc = {"GTDr": [], "GTDrel": [], "GTDuse": []}
                rel_frac = []
                for j in range(len(o["gain"])):
                    m = z["gt_new"][j].astype(bool)
                    f = (z["o"][j][None] + z["d"][j][m] * o["tstar"][j][m][:, None].astype(np.float64)) * k_M + origin
                    W = np.stack([f[:, 0], f[:, 2], -f[:, 1]], 1)               # frame z-up -> world y-up (runtime units)
                    if len(W):
                        dd, ii = tree.query(W, k=1, distance_upper_bound=eps)
                        ok = np.isfinite(dd)
                        idx = ii[ok]
                        rel = ok.copy()
                        rel[ok] = vis[idx]
                        use = rel.copy()
                        use[ok] = vis[idx] & ~cov[idx]
                    else:
                        rel = use = np.zeros(0, bool)
                    sc["GTDr"].append(float(m.sum()))
                    sc["GTDrel"].append(float(rel.sum()))
                    sc["GTDuse"].append(float(use.sum()))
                    rel_frac.append(float(rel.mean()) if len(rel) else None)
                g = o["gain"][C] * 100
                row = {"sid": st["sid"], "scene": st["scene"], "t": t, "rel_frac": rel_frac}
                ob = set(np.flatnonzero(g >= g.max() - TAU).tolist())
                for nm, s in list(sc.items()) + [("GTD", v["gtd_distp1"])]:
                    s = np.asarray(s, float)[C]
                    pk = int(np.argmax(s))
                    row[nm] = {"top1": float(pk in ob), "regret": float(g.max() - g[pk]), "spearman": T13.spearman(s, g), "pick": int(C[pk])}
                gb = row["GTD"]["pick"]
                best = int(C[int(np.argmax(g))])
                row["rel_frac_gtd_pick"] = rel_frac[gb]
                row["rel_frac_oracle_best"] = rel_frac[best]
                rows.append(row)
            print(dataset, src, tag, "done", flush=True)
        S = {}
        for nm in ("GTD", "GTDr", "GTDrel", "GTDuse"):
            r = [x[nm] for x in rows]
            S[nm] = {k: float(np.nanmean([x[k] for x in r if x[k] is not None])) for k in ("top1", "regret", "spearman")}
        wrong = [x for x in rows if x["GTD"]["top1"] == 0]
        right = [x for x in rows if x["GTD"]["top1"] == 1]
        mean = lambda xs: float(np.nanmean([x for x in xs if x is not None])) if any(x is not None for x in xs) else None  # noqa: E731
        S["rel_frac_all_candidates"] = mean([f for x in rows for f in x["rel_frac"]])
        S["GTD_wrong"] = {"states": len(wrong), "rel_frac_gtd_pick": mean([x["rel_frac_gtd_pick"] for x in wrong]),
                          "rel_frac_oracle_best": mean([x["rel_frac_oracle_best"] for x in wrong])}
        S["GTD_right"] = {"states": len(right), "rel_frac_gtd_pick": mean([x["rel_frac_gtd_pick"] for x in right])}
        S["n_states"] = len(rows)
        out[dataset] = {"summary": S, "rows": rows}
        print(dataset, json.dumps(S, indent=1))
    json.dump(out, open(os.path.join(HERE, "results", "a1_relevance_diag.json"), "w"))


if __name__ == "__main__":
    main()
