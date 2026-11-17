"""EXP9 Fig. 4 v7 -- more row-1 candidates (user 2026-11-16: "all t0? none I really want; generate more").
Pool = the v4-v6 outdoor pool (bannerman, redeemer, colosseum, pantheon, alhambra, neuschwanstein; 180 states) + the 4 Unseen-6 scenes it
lacked (eiffel, bridge, barts, sestino_museum; 120 states from fig4_v7_scan.py = fig4_scan.py unchanged, output redirected).
Eligibility = exactly max_p90_inbox (fig4_select.py): exactness guards true, no inside-mesh exclusion (sources 1-3; back faces from
fig4_v7_backface.py for the new scenes), UNKNOWN share >= 0.20, GT hits on UNKNOWN rays >= 50 % inside the ROI box.
Window = the v6 rule (fig4_v6.best_window: v5 rule + >= 30 % of the window's rays hit a real surface).
Candidate sets (declared before ranking; already shown states excluded = v5 rows 1-2 and the 7 v6 states):
  U  for each Unseen-6 scene the eligible state with the highest v6 window score (any t)
  L  later steps: eligible states with t >= 10 not in U, v6 score descending, <= 1 per (scene, start), top 6
Phases: --rank (CPU) -> --render (GPU, fig4_v4_render.run_case unchanged, output work_dirs/.../fig4_v7/render/) -> --sheet (CPU, v5 style).
Output: experiments/cvpr_results/fig/fig4/fig4_v7/{ranking_v7.csv, candidates.json, sheet_U*.{pdf,png}, sheet_L*.{pdf,png}, README.md}"""
import argparse
import csv
import json
import os
import sys

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
import fig4_select as FS  # noqa: E402
import fig4_v6 as V6  # noqa: E402

NEWD = FS.REPO + "work_dirs/cvpr_results/exp9_ablation_qualitative/fig4_v7/states/"
W7 = FS.REPO + "work_dirs/cvpr_results/exp9_ablation_qualitative/fig4_v7/render/"
OUT = FS.REPO + "experiments/cvpr_results/fig/fig4/fig4_v7/"
NEW = ["eiffel", "bridge", "barts", "sestino_museum"]
UNSEEN = ["alhambra", "neuschwanstein", "eiffel", "bridge", "barts", "sestino_museum"]
NAME = {"bannerman": "Bannerman", "redeemer": "Redeemer", "colosseum": "Colosseum", "pantheon": "Pantheon", "alhambra": "Alhambra",
        "neuschwanstein": "Neuschwanstein", "eiffel": "Eiffel Tower", "bridge": "Manhattan Bridge", "barts": "Barts", "sestino_museum": "Sestino Museum"}
SPLIT_NEW = {"eiffel": "dev", "barts": "dev", "sestino_museum": "dev", "bridge": "test"}
_state_npz = FS.state_npz
FS.state_npz = lambda d: (NEWD + f"{d['group']}_s{d['start']}.npz") if d["group"] in NEW else _state_npz(d)
TOP_L = 6


def context():
    js, bf = {}, {}
    for s in sum(FS.POOL_OUT.values(), []) + NEW:
        root = NEWD if s in NEW else FS.OUT_IN
        for r in map(json.loads, open(root + f"{s}.jsonl")):
            js[(s, r["start"], r["t"])] = r
        for r in map(json.loads, open(root + f"backface_{s}.jsonl")):
            bf[(s, r["start"], r["t"])] = r
    man = {}
    for r in csv.DictReader(open(FS.BF_CSV)):
        man.setdefault((r["scene_id"], int(r["start_id"])), []).append((int(r["step_of_V3-T0-D1_episode"]), float(r["back_face_share"])))
    inside = set(json.load(open(FS.BF_SUM))["start_pose_inside_mesh"])
    return js, bf, man, inside


def pool():
    out = FS.pool("outdoor")
    n = len(out)
    for s in NEW:
        for k in range(5):
            for t in FS.STEPS["outdoor"]:
                out.append({"domain": "outdoor", "group": s, "split": SPLIT_NEW[s], "start": k, "t": t, "case": f"outdoor_{s}_s{k}_t{t}", "pool_order": n})
                n += 1
    return out


def rank():
    os.makedirs(OUT, exist_ok=True)
    ctx = context()
    shown = {"outdoor_bannerman_s3_t0", "outdoor_colosseum_s3_t0"} | {e["case"] for e in json.load(open(V6.OUT + "candidates.json"))}
    failed = json.load(open(OUT + "render_failed.json")) if os.path.exists(OUT + "render_failed.json") else {}
    shown |= set(failed)                                         # a state whose RGB render fails a v4 guard is dropped; the next one moves up
    old = {r["case"]: r for r in csv.DictReader(open(FS.MAN + "fig4_selection_outdoor_maxp90_inbox.csv")) if r["row_type"] == "state"}
    states = []
    for d in pool():
        FS.annotate_outdoor(d, ctx)
        S = FS.load_state(d)
        d.update(FS.metrics(S))
        d["pick"] = S["pick"]
        assert d["flip_share_on_unknown"] == d["flip_share_on_unknown_jsonl"], d["case"]
        d.update(FS.inbox_metrics(d, S))
        why = []
        if d["exclusion"]:
            why.append("inside mesh")
        if not d["guards_ok"]:
            why.append("exactness guard failed")
        if d["unknown_share"] < FS.UNK_MIN:
            why.append("UNKNOWN share < 0.20")
        if not FS.inbox_ok(d):
            why.append("GT hits on UNKNOWN rays < 50 % inside the ROI box")
        d["eligible"], d["why"] = not why, "; ".join(why)
        if d["case"] in old:                                             # the old pool must reproduce the v4 eligibility exactly
            assert d["eligible"] == (old[d["case"]]["eligible"] == "True"), (d["case"], d["why"], old[d["case"]]["ineligible_reason"])
        if d["eligible"]:
            _, A, inb, code = V6.state_arrays(d)
            b6, nv = V6.best_window(A["unknown"], A["gt_valid"], inb, code, V6.GT_MIN)
            d.update(v6_score=b6[0] if b6 else -1, v6_valid_windows=nv, **(V6.win_stats(A, code, b6[1], b6[2]) if b6 else {}))
        states.append(d)
    el = [d for d in states if d["eligible"]]
    print("pool", len(states), "eligible", len(el), "| old pool eligible", sum(1 for d in el if d["group"] not in NEW), "| new", sum(1 for d in el if d["group"] in NEW))
    key = lambda d: (-d["v6_score"], -d["flip_share_on_unknown"], d["pool_order"])  # noqa: E731
    el.sort(key=key)
    free = [d for d in el if d["case"] not in shown and d["v6_score"] > 0]
    cands = []
    for s in UNSEEN:
        d = next((d for d in free if d["group"] == s), None)
        if d:
            d["set"] = f"U: best {NAME[s]}"
            cands.append(d)
    seen = {(d["group"], d["start"]) for d in cands}
    k = 0
    for d in free:
        if k >= TOP_L:
            break
        if d["t"] >= 10 and "set" not in d and (d["group"], d["start"]) not in seen:
            k += 1
            seen.add((d["group"], d["start"]))
            d["set"] = f"L{k}: later step"
            cands.append(d)
    for i, d in enumerate(el):
        d["rank_v7"] = i + 1
    keys = ["rank_v7", "case", "group", "split", "start", "t", "set", "v6_score", "v6_valid_windows", "win_row0", "win_col0", "win_unknown_share",
            "win_gt_hit_share", "flip_share_on_unknown", "unknown_share", "gt_hits_unknown_in_roi_share", "pick"]
    with open(OUT + "ranking_v7.csv", "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=keys, extrasaction="ignore")
        w.writeheader()
        for d in el:
            w.writerow({kk: (f"{v:.4f}" if isinstance(v, float) else v) for kk, v in d.items()})
    by_t = {t: [d["v6_score"] for d in el if d["t"] == t] for t in FS.STEPS["outdoor"]}
    print("eligible per t:", {t: len(v) for t, v in by_t.items()})
    json.dump([{k: d[k] for k in ("case", "group", "start", "t", "pick", "domain", "set", "v6_score", "win_row0", "win_col0", "win_gt_hit_share",
                                  "unknown_share", "flip_share_on_unknown")} for d in cands], open(OUT + "candidates.json", "w"), indent=1, default=float)
    for d in cands:
        print(f"{d['set']:28s} {d['case']:34s} score {d['v6_score']:4d} GT {d['win_gt_hit_share']:.2f} unknown {d['unknown_share']:.2f} "
              f"flip {d['flip_share_on_unknown']:.3f} rank {d['rank_v7']}/{len(el)}")


def render():
    import fig4_v4_render as R4
    R4.W4 = W7
    dev = R4.torch.device("cuda", 0)
    scenes = {}
    for e in sorted(json.load(open(OUT + "candidates.json")), key=lambda e: (e["group"], e["case"])):
        d = {k: e[k] for k in ("domain", "group", "start", "t", "case", "pick")}
        d["sets"] = ["v7"]
        if not os.path.exists(W7 + f"{d['case']}.npz"):
            try:
                R4.run_case(d, dev, scenes)
            except SystemExit as ex:                                 # v4 guard failed (C1 / A1-A4): record, drop, re-rank
                p = OUT + "render_failed.json"
                f = json.load(open(p)) if os.path.exists(p) else {}
                f[d["case"]] = str(ex)
                json.dump(f, open(p, "w"), indent=1)
                print("FAILED guard, dropped:", d["case"], str(ex)[:160], flush=True)


def sheet():
    import fig4_v5 as V5
    cands = json.load(open(OUT + "candidates.json"))
    rows = {"U": [], "L": []}
    for e in cands:
        d = {k: e[k] for k in ("domain", "group", "start", "t", "case", "pick")}
        _, A, inb, code = V6.state_arrays(d)
        z = np.load(W7 + f"{e['case']}.npz")
        tag = e["set"].split(":")[0]
        rows[tag[0]].append({"meta": {"domain": "outdoor", "depth_unit": A["unit"], "case": e["case"]}, "cur_rgb": z["cur_rgb"], "cand_rgb": z["cand_rgb"],
                             "unk": A["unknown"], "gt": A["gt_depth"], "gtv": A["gt_valid"], "hyp": A["hyp_depth"], "hypv": A["hyp_valid"],
                             "vmin": A["vmin"], "vmax": A["vmax"], "code": code, "win": (e["win_row0"], e["win_col0"]),
                             "label": f"{tag} {NAME[e['group']]} s{e['start']} t{e['t']}"})
    for s, rr in rows.items():
        for i in range(0, len(rr), 3):
            V5.composite(rr[i:i + 3], OUT + f"sheet_{s}{i // 3 + 1}", True)
            print("sheet", f"{s}{i // 3 + 1}", [r["label"] for r in rr[i:i + 3]])
    V5.plt.close("all")


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("phase", choices=["rank", "render", "sheet"])
    {"rank": rank, "render": render, "sheet": sheet}[ap.parse_args().phase]()
