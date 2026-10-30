"""EXP5 Task 2 / T2.1b -- V2-Mix checkpoint selection by the rule frozen before training (protocol §3; seen2scene env, CPU).

score(step) = ES_in_val(step) / ES_in_val(0) + ES_out_sc(step) / ES_out_sc(0)   (fair ES, unknown-arclength voxels, K = 4)
V2-Mix = the evaluated step > 0 with the minimum score (ties -> later step); step 0 (= V2 Original) is reported as the
reference. The chosen EMA is hard-linked into work_dirs/weights/W21_exp5_v2mix/W21-1_V2Mix_best/ (+ the last EMA as
W21-2_V2Mix_last/, all evaluated EMA as W21-3_V2Mix_intermediate_ema/), each with MANIFEST.json, and its path / sha256
is written into configs/eval_indoor_v1.json and configs/eval_outdoor_v1.json (group V2M).
Usage: python t22_select.py --run W2/training/mix_v1
"""
import argparse
import glob
import hashlib
import json
import os
import sys
import time

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
import t2_common as C  # noqa: E402


def sha256(p):
    h = hashlib.sha256()
    with open(p, "rb") as f:
        for b in iter(lambda: f.read(1 << 23), b""):
            h.update(b)
    return h.hexdigest()


def link(src, folder, title, extra=None):
    d = os.path.join(C.REPO, "work_dirs", "weights", "W21_exp5_v2mix", folder)
    os.makedirs(d, exist_ok=True)
    dst = os.path.join(d, os.path.basename(src))
    if not os.path.exists(dst):
        os.link(src, dst)
    man_p = os.path.join(d, "MANIFEST.json")
    man = json.load(open(man_p)) if os.path.exists(man_p) else {"folder": f"W21_exp5_v2mix/{folder}", "title": title, "files": []}
    if not any(f["name"] == os.path.basename(src) for f in man["files"]):
        man["files"].append({"name": os.path.basename(src), "sha256": sha256(dst), "size": os.path.getsize(dst),
                             "source_path": os.path.relpath(src, C.REPO), **(extra or {})})
    json.dump(man, open(man_p, "w"), indent=1)
    return dst


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--run", required=True)
    a = ap.parse_args()
    ev = [json.loads(x) for x in open(os.path.join(a.run, "eval.jsonl"))]
    ev = {e["step"]: e for e in ev}
    e0 = ev[0]
    es = lambda e, k: e[f"samp_{k}"]["fair_es_arclength_vox"]  # noqa: E731
    rows = []
    for st in sorted(ev):
        e = ev[st]
        sc = es(e, "in_val") / es(e0, "in_val") + es(e, "out_sc") / es(e0, "out_sc")
        rows.append({"step": st, "score": sc, "es_in_val": es(e, "in_val"), "es_out_sc": es(e, "out_sc"),
                     "brier_in_val": e["samp_in_val"]["brier"], "brier_out_sc": e["samp_out_sc"]["brier"],
                     "false_wall_in_val": e["samp_in_val"]["false_wall"], "false_wall_out_sc": e["samp_out_sc"]["false_wall"],
                     "fm_in_val": e["fm_in_val"], "fm_out_sc": e["fm_out_sc"]})
    cand = [r for r in rows if r["step"] > 0]
    best = min(cand, key=lambda r: (r["score"], -r["step"]))
    last = max(cand, key=lambda r: r["step"])
    title = "EXP5 Task 2 V2-Mix (indoor + outdoor mixed fine-tuning of V2 Original), selected by the frozen rule"
    pb = link(os.path.join(a.run, f"ema_{best['step']:06d}.pt"), "W21-1_V2Mix_best", title, {"step": best["step"], "score": best["score"]})
    pl = link(os.path.join(a.run, f"ema_{last['step']:06d}.pt"), "W21-2_V2Mix_last", "EXP5 Task 2 V2-Mix, last EMA", {"step": last["step"]})
    for p in sorted(glob.glob(os.path.join(a.run, "ema_*.pt"))):
        link(p, "W21-3_V2Mix_intermediate_ema", "EXP5 Task 2 V2-Mix, every evaluated EMA (step 0 = V2 Original)")
    sb = sha256(pb)
    for cfgn in ("eval_indoor_v1.json", "eval_outdoor_v1.json"):
        p = os.path.join(C.T2, "configs", cfgn)
        cfg = json.load(open(p))
        cfg["groups"]["V2M"].update({"checkpoint": os.path.relpath(pb, C.REPO), "sha256": sb, "step": best["step"]})
        json.dump(cfg, open(p, "w"), indent=1)
    out = {"rule": __doc__.split("\n")[2].strip(), "rows": rows, "step0_reference": rows[0], "selected": best,
           "selected_path": os.path.relpath(pb, C.REPO), "selected_sha256": sb, "last": last, "last_path": os.path.relpath(pl, C.REPO),
           "step0_better_than_all": all(r["score"] >= rows[0]["score"] for r in cand), "time": time.strftime("%Y-%m-%d %H:%M:%S")}
    json.dump(out, open(os.path.join(a.run, "selection.json"), "w"), indent=1)
    for r in rows:
        print(r["step"], "score %.4f  ES in %.3f out %.3f  Brier in %.4f out %.4f" % (r["score"], r["es_in_val"], r["es_out_sc"], r["brier_in_val"], r["brier_out_sc"]))
    print("selected", best["step"], sb[:16])


if __name__ == "__main__":
    main()
