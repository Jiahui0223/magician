"""EXP Density A/B -- Task 0: S10 vs D100 training shards, ray by ray (seen2scene env, CPU, read only).

Pairs every training shard of the E2-S10 arm (Task 2 MP3D shards task2/mp3d/views, T1.1 Macarons++ shards task1/data/views)
with its D100 rebuild (e2/w2_d100/mp3d/views, e2/out_views_d100). Per pair:
  structure   arrays that drive the sampler and the rays (t, a, R, T, jitter, block, heldout, k_M, [tstar]) bitwise equal
  supervision lmask = (cls == UNKNOWN) & (b != 0) -- the exact training mask (task1 v2_data.state_tensors)
              counts: sup in both / only S10 / only D100; HIT (b = +1) per arm; label transitions among rays supervised in
              both (HIT->HIT, HIT->NOHIT, NOHIT->HIT, NOHIT->NOHIT); |dy| on HIT->HIT; ray-class transition matrix
  input       coarse-grid BAND / EMPTY fraction over in-domain cells (encoder input statistics)
Output: results/t0_labels.json (aggregates per domain and split) + work_dirs/exp_density_ab/t0/labels_pairs.jsonl
Usage: python da_t0_labels.py [--workers 48]
"""
import argparse
import glob
import json
import os
import sys
from multiprocessing import Pool

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
EXP = os.path.dirname(HERE)
REPO = os.path.dirname(os.path.dirname(EXP))
W5 = os.path.join(REPO, "work_dirs", "exp5_future_view")
PAIRS_SRC = {"indoor": (os.path.join(W5, "task2", "mp3d", "views"), os.path.join(W5, "task3_full_depth_p1", "e2", "w2_d100", "mp3d", "views")),
             "outdoor": (os.path.join(W5, "task1", "data", "views"), os.path.join(W5, "task3_full_depth_p1", "e2", "out_views_d100"))}
CLS_UNKNOWN = 0  # overwritten from seen2scene.futureview.geometry in main()
STRUCT = ("t", "a", "R", "T", "jitter", "block", "heldout", "k_M", "tstar", "n_blocks")


def one(args):
    dom, p10, p100, cls_unk, purpose = args
    a, b = np.load(p10), np.load(p100)
    st = {k: bool(np.array_equal(a[k], b[k])) for k in STRUCT if k in a.files}
    out = {"domain": dom, "file": os.path.relpath(p10, W5), "purpose": purpose, "struct_equal": st, "n_views": int(a["t"].shape[0]),
           "heldout": bool(a["heldout"])}
    c10, c100 = a["cls"], b["cls"]
    s10 = (c10 == cls_unk) & (a["b"] != 0)
    s100 = (c100 == cls_unk) & (b["b"] != 0)
    both = s10 & s100
    h10, h100 = a["b"] > 0, b["b"] > 0
    out["n_rays"] = int(c10.size)
    out["sup10"], out["sup100"] = int(s10.sum()), int(s100.sum())
    out["sup_both"], out["sup_only10"], out["sup_only100"] = int(both.sum()), int((s10 & ~s100).sum()), int((s100 & ~s10).sum())
    out["hit10"], out["hit100"] = int((s10 & h10).sum()), int((s100 & h100).sum())
    out["both_hh"], out["both_hn"] = int((both & h10 & h100).sum()), int((both & h10 & ~h100).sum())
    out["both_nh"], out["both_nn"] = int((both & ~h10 & h100).sum()), int((both & ~h10 & ~h100).sum())
    hh = both & h10 & h100
    dy = np.abs(a["y"][hh].astype(np.float64) - b["y"][hh].astype(np.float64))
    out["dy_sum_hh"] = float(dy.sum())
    out["only10_hit"] = int((s10 & ~s100 & h10).sum())
    out["only10_to_cls100"] = [int((s10 & ~s100 & (c100 == k)).sum()) for k in range(3)]
    out["cls_trans"] = [[int(((c10 == i) & (c100 == j)).sum()) for j in range(3)] for i in range(3)]
    ck = "coarse" if "coarse" in a.files else None
    keys = [ck] if ck else sorted(k for k in a.files if k.startswith("coarse_b"))
    bf = {}
    for x, z in (("10", a), ("100", b)):
        num_b = num_e = n = 0.0
        for k in keys:
            cg = z[k].astype(np.float64) / 255.0
            m = cg[:, 2] > 0
            num_b += float(cg[:, 1][m].sum())
            num_e += float(cg[:, 0][m].sum())
            n += float(m.sum())
        bf[x] = {"band_sum": num_b, "empty_sum": num_e, "cells": n}
    out["coarse"] = bf
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--workers", type=int, default=48)
    a = ap.parse_args()
    sys.path.insert(0, REPO)
    from seen2scene.futureview import geometry as G
    cls_unk = int(G.CLS_UNKNOWN)
    jobs = json.load(open(os.path.join(W5, "task2", "mp3d", "manifests", "traj_jobs.json")))["jobs"]
    purpose = {f"{j['key']}/{j['traj']}.npz": j["purpose"] for j in jobs}
    tasks, missing = [], []
    for dom, (r10, r100) in PAIRS_SRC.items():
        for p in sorted(glob.glob(os.path.join(r10, "*", "*.npz"))):
            rel = os.path.relpath(p, r10)
            q = os.path.join(r100, rel)
            if not os.path.exists(q):
                missing.append(f"{dom}:{rel}")
                continue
            tasks.append((dom, p, q, cls_unk, purpose.get(rel, "train6" if dom == "outdoor" else "unknown")))
    print(len(tasks), "pairs;", len(missing), "S10 shards without D100 twin", flush=True)
    od = os.path.join(REPO, "work_dirs", "exp_density_ab", "t0")
    os.makedirs(od, exist_ok=True)
    with Pool(a.workers) as pool:
        rows = pool.map(one, tasks, chunksize=1)
    with open(os.path.join(od, "labels_pairs.jsonl"), "w") as f:
        for r in rows:
            f.write(json.dumps(r) + "\n")
    agg = {}
    for r in rows:
        grp = f"{r['domain']}|{'heldout' if r['heldout'] else r['purpose']}"
        g = agg.setdefault(grp, {"pairs": 0, "struct_all_equal": 0})
        g["pairs"] += 1
        g["struct_all_equal"] += int(all(r["struct_equal"].values()))
        for k, v in r.items():
            if isinstance(v, (int, float)) and not isinstance(v, bool) and k not in ("pairs",):
                g[k] = g.get(k, 0) + v
            elif k in ("only10_to_cls100",):
                g[k] = (np.asarray(g.get(k, [0] * 3)) + np.asarray(v)).tolist()
            elif k == "cls_trans":
                g[k] = (np.asarray(g.get(k, np.zeros((3, 3), int))) + np.asarray(v)).tolist()
            elif k == "coarse":
                for x in ("10", "100"):
                    for kk in ("band_sum", "empty_sum", "cells"):
                        g[f"coarse{x}_{kk}"] = g.get(f"coarse{x}_{kk}", 0.0) + v[x][kk]
    for g in agg.values():
        g["sup_rate10"], g["sup_rate100"] = g["sup10"] / g["n_rays"], g["sup100"] / g["n_rays"]
        g["sup_ratio_100_over_10"] = g["sup100"] / g["sup10"]
        g["hit_ratio_100_over_10"] = g["hit100"] / max(g["hit10"], 1)
        g["hit_frac10"], g["hit_frac100"] = g["hit10"] / g["sup10"], g["hit100"] / max(g["sup100"], 1)
        g["frac_only10_of_sup10"] = g["sup_only10"] / g["sup10"]
        g["frac_only100_of_sup100"] = g["sup_only100"] / max(g["sup100"], 1)
        g["both_label_agree"] = (g["both_hh"] + g["both_nn"]) / max(g["sup_both"], 1)
        g["dy_mean_hh"] = g["dy_sum_hh"] / max(g["both_hh"], 1)
        for x in ("10", "100"):
            g[f"coarse{x}_band_frac"] = g[f"coarse{x}_band_sum"] / g[f"coarse{x}_cells"]
            g[f"coarse{x}_empty_frac"] = g[f"coarse{x}_empty_sum"] / g[f"coarse{x}_cells"]
    res = {"cls_codes": {"UNKNOWN": cls_unk, "KNOWN": int(G.CLS_KNOWN), "EMPTY": int(G.CLS_EMPTY)}, "n_pairs": len(rows), "missing_d100": missing,
           "struct_mismatch_files": [r["file"] for r in rows if not all(r["struct_equal"].values())], "groups": agg}
    json.dump(res, open(os.path.join(EXP, "results", "t0_labels.json"), "w"), indent=1)
    for k, g in sorted(agg.items()):
        print(k, {x: (round(g[x], 4) if isinstance(g[x], float) else g[x]) for x in ("pairs", "struct_all_equal", "n_views", "sup_rate10", "sup_rate100",
                                                                                   "frac_only10_of_sup10", "frac_only100_of_sup100", "hit_ratio_100_over_10",
                                                                                   "both_label_agree", "dy_mean_hh", "coarse10_band_frac", "coarse100_band_frac")})


if __name__ == "__main__":
    main()
