"""EXP9 Fig. 4 v6 -- replacement candidates for row 1 (Bannerman s3 t0) whose v5 zoom window holds NO real surface (GT = sky) while 3 of
the 4 generations predict a wall there (user 2026-11-16: "the candidate view is empty but the predictions show surfaces -> change it").

Window rule v6 = the v5 rule (fig4_v5.py: 21 x 38 scoring-cell window, no GT hit on an unknown ray outside the scene ROI box, score =
#unknown rays whose 4 generations disagree on hit / no-hit, ties smaller row then column) + ONE added validity condition, the fix already
proposed in the v5 README before this request: at least GT_MIN = 30 % of the window's rays have a real surface (GT hit within range).
Rows 2-3 of v5 (Colosseum, MP3D Scene3) satisfy it with their v5 windows (asserted: their v6 window == v5 window).
Candidate pool = the 69 eligible outdoor states of manifests/fig4_selection_outdoor_maxp90_inbox.csv (unchanged eligibility of v4/v5).
Ranking = v6 best-window score descending (ties: flip share on unknown rays descending, pool order); excluded: Colosseum s3 t0 (row 2);
at most one state per (scene, start); take the top TOP, plus the best state of each Unseen-6 scene in the pool (alhambra, neuschwanstein)
if not already in.  Bannerman s3 t0 itself is kept as an option with its v6 window.
Phases:  --scan (CPU, seen2scene env)  ->  --render (GPU, magician env; fig4_v4_render.run_case unchanged, output redirected to
work_dirs/.../fig4_v6/render/)  ->  --sheet (CPU)  candidate sheets drawn with the v5 composite (same colours, zoom x3).
Output: experiments/cvpr_results/fig/fig4/fig4_v6/{ranking_v6.csv, candidates.json, candidates_sheet_<k>.{pdf,png}, README.md}"""
import argparse
import csv
import json
import os
import sys

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
import fig4_select as FS  # noqa: E402
import fig4_panels as FP  # noqa: E402

OUT = FS.REPO + "experiments/cvpr_results/fig/fig4/fig4_v6/"
W6 = FS.REPO + "work_dirs/cvpr_results/exp9_ablation_qualitative/fig4_v6/render/"
SEL = FS.MAN + "fig4_selection_outdoor_maxp90_inbox.csv"
WIN_H, WIN_W = 21, 38
GT_MIN = 0.30
TOP = 6
ROW1, ROW2 = "outdoor_bannerman_s3_t0", "outdoor_colosseum_s3_t0"
UNSEEN = ("alhambra", "neuschwanstein")


def disagreement(hv, unk):
    """v4 / v5 code convention (fig4_v4_final3.disagreement_codes): -2 known, -1 unknown and agreeing, 1..3 = hits on disagreeing rays."""
    import fig4_v4_final3 as F3
    return F3.disagreement_codes(hv, unk)[0]


def state_arrays(d):
    S = FS.load_state(d)
    A = FP.depth_arrays(d, S)
    lo, hi = FS.roi_box(d)
    P = FS.gt_hit_points(S)
    inb = ((P >= lo) & (P <= hi)).all(1).reshape(FS.GH, FS.GW)
    code = disagreement(A["hyp_valid"], A["unknown"])
    return S, A, inb, code


def best_window(unk, gtv, inb, code, gt_min):
    dis, bad = code > 0, gtv & unk & ~inb
    best, n_valid = None, 0
    for y in range(FS.GH - WIN_H + 1):
        for x in range(FS.GW - WIN_W + 1):
            sl = (slice(y, y + WIN_H), slice(x, x + WIN_W))
            if bad[sl].any() or gtv[sl].mean() < gt_min:
                continue
            n_valid += 1
            c = (int(dis[sl].sum()), y, x)
            if best is None or (-c[0], c[1], c[2]) < (-best[0], best[1], best[2]):
                best = c
    return best, n_valid


def win_stats(A, code, y, x):
    sl = (slice(y, y + WIN_H), slice(x, x + WIN_W))
    n = WIN_H * WIN_W
    return {"win_row0": y, "win_col0": x, "win_disagree_rays": int((code[sl] > 0).sum()), "win_unknown_share": float(A["unknown"][sl].mean()),
            "win_gt_hit_share": float(A["gt_valid"][sl].mean()),
            **{f"win_gen{k + 1}_hit_share": float(A["hyp_valid"][k][sl].sum() / n) for k in range(FS.K)}}


def scan():
    os.makedirs(OUT, exist_ok=True)
    rows = [r for r in csv.DictReader(open(SEL)) if r["row_type"] == "state" and r["eligible"] == "True"]
    assert len(rows) == 69, len(rows)
    out = []
    for r in rows:
        d = {"domain": "outdoor", "group": r["scene"], "start": int(r["start"]), "t": int(r["t"]), "case": r["case"], "pick": int(r["pick"])}
        S, A, inb, code = state_arrays(d)
        m = FS.metrics(S)
        assert abs(m["flip_share_on_unknown"] - float(r["flip_share_on_unknown"])) < 1e-6, r["case"]
        b6, nv6 = best_window(A["unknown"], A["gt_valid"], inb, code, GT_MIN)
        b5, _ = best_window(A["unknown"], A["gt_valid"], inb, code, 0.0)
        e = dict(d, pool_order=int(r["pool_order"]), split=r["split"], flip_unk=m["flip_share_on_unknown"], unknown_share=m["unknown_share"],
                 v5_window=[b5[1], b5[2]] if b5 else None, v5_score=b5[0] if b5 else -1, v6_valid_windows=nv6,
                 v6_score=b6[0] if b6 else -1, **(win_stats(A, code, b6[1], b6[2]) if b6 else {}))
        out.append(e)
    # v5 rows 1-2 (outdoor) reproduce the v5 windows of zoom_windows.csv
    zw = {x["case"]: x for x in csv.DictReader(open(FS.REPO + "experiments/cvpr_results/fig/fig4/fig4_v5/zoom_windows.csv"))}
    for e in out:
        if e["case"] in zw:
            assert e["v5_window"] == [int(zw[e["case"]]["win_row0"]), int(zw[e["case"]]["win_col0"])], (e["case"], e["v5_window"])
            print(e["case"], "v5 window reproduced", e["v5_window"], "| v6 window", [e.get("win_row0"), e.get("win_col0")], "score", e["v6_score"])
    col = next(e for e in out if e["case"] == ROW2)
    assert [col["win_row0"], col["win_col0"]] == col["v5_window"], "row 2 window changes under v6"
    out.sort(key=lambda e: (-e["v6_score"], -e["flip_unk"], e["pool_order"]))
    seen, k = set(), 0
    for i, e in enumerate(out):
        e["rank_v6"] = i + 1
        g = (e["group"], e["start"])
        if e["case"] == ROW2:
            e["status"] = "excluded: row 2"
        elif e["case"] == ROW1:
            e["status"] = "current row 1 (kept as an option with its v6 window)"
        elif g in seen:
            e["status"] = "excluded: (scene, start) already represented"
        elif k < TOP and e["v6_score"] > 0:
            k += 1
            seen.add(g)
            e["status"] = f"candidate {k}"
        else:
            e["status"] = ""
    for sc in UNSEEN:
        if not any(e["status"].startswith("candidate") and e["group"] == sc for e in out):
            e = next((e for e in out if e["group"] == sc and e["v6_score"] > 0 and not e["status"]), None)
            if e:
                k += 1
                e["status"] = f"candidate {k} (best {sc}, unseen scene)"
    keys = ["rank_v6", "case", "group", "split", "start", "t", "status", "v6_score", "v6_valid_windows", "win_row0", "win_col0", "win_unknown_share",
            "win_gt_hit_share"] + [f"win_gen{k + 1}_hit_share" for k in range(FS.K)] + ["v5_window", "v5_score", "flip_unk", "unknown_share", "pick"]
    with open(OUT + "ranking_v6.csv", "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=keys, extrasaction="ignore")
        w.writeheader()
        for e in out:
            w.writerow({kk: (f"{v:.4f}" if isinstance(v, float) else v) for kk, v in e.items()})
    cands = [e for e in out if e["status"].startswith("candidate")] + [e for e in out if e["case"] == ROW1]
    json.dump(cands, open(OUT + "candidates.json", "w"), indent=1, default=float)
    for e in cands:
        print(f"{e['status']:40s} {e['case']:34s} v6 score {e['v6_score']:4d} (v5 {e['v5_score']:4d}) win {e.get('win_row0')},{e.get('win_col0')} "
              f"GT {e.get('win_gt_hit_share', float('nan')):.2f} unk {e['unknown_share']:.2f} flip {e['flip_unk']:.3f}")


def render():
    import fig4_v4_render as R4
    R4.W4 = W6
    dev = R4.torch.device("cuda", 0)
    scenes = {}
    for e in sorted(json.load(open(OUT + "candidates.json")), key=lambda e: (e["group"], e["case"])):
        d = {k: e[k] for k in ("domain", "group", "start", "t", "case", "pick")}
        d["sets"] = ["v6"]
        if os.path.exists(W6 + f"{d['case']}.npz"):
            continue
        R4.run_case(d, dev, scenes)


def sheet():
    import fig4_v5 as V5
    plt = V5.plt
    cands = json.load(open(OUT + "candidates.json"))
    rows = []
    for e in cands:
        d = {k: e[k] for k in ("domain", "group", "start", "t", "case", "pick")}
        S, A, inb, code = state_arrays(d)
        z = np.load(W6 + f"{e['case']}.npz")
        lab = f"{FS.OUT_NAME[e['group']]} s{e['start']} t{e['t']}"
        rows.append({"meta": {"domain": "outdoor", "depth_unit": A["unit"], "case": e["case"]}, "cur_rgb": z["cur_rgb"], "cand_rgb": z["cand_rgb"],
                     "unk": A["unknown"], "gt": A["gt_depth"], "gtv": A["gt_valid"], "hyp": A["hyp_depth"], "hypv": A["hyp_valid"],
                     "vmin": A["vmin"], "vmax": A["vmax"], "code": code, "win": (e["win_row0"], e["win_col0"]),
                     "label": ("Option: " if e["case"] == ROW1 else f"#{e['status'].split()[1]} ") + lab})
    for i in range(0, len(rows), 3):
        V5.composite(rows[i:i + 3], OUT + f"candidates_sheet_{i // 3 + 1}", True)
        print("sheet", i // 3 + 1, [r["label"] for r in rows[i:i + 3]])
    plt.close("all")


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("phase", choices=["scan", "render", "sheet"])
    a = ap.parse_args()
    {"scan": scan, "render": render, "sheet": sheet}[a.phase]()
