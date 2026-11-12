"""FisherRF reproduction -- validation + aggregation of F2 (MP3D-150) and F3 (Macarons++ All-14) (CPU, seen2scene env).

Outdoor (F3): per episode the official coverage curve of the closed loop (episodes/ep_FisherRF_<scene>_s<k>_r0.json) must equal the
official replay bitwise (metrics/FisherRF/FisherRF_<scene>_s<k>_r0.json, official_equal_stored = True; fisherrf_out_cov.py =
table1 t1_out_cov.py procedure). Metrics on the official coverage C[0..100]: AUC@T = trapz(C[0..T]) / T (20 / 50 / 100), Cov@20 /
Cov@50, Final = C[100]; scene-equal means over All-14; paired scene bootstrap (10,000, seed 0) against the CVPR Table 1 canonical
records of V3-T0-D1 (Ours), MAG-B, MACARONS, RANDOM.
Indoor (F2): the same five checks as EXP7 collect_mp3d150.validate (151 values; C_obs replay from the stored 10 % samples bitwise;
legality against each record's legal list; provenance n/a) + same start pose as the EXP7 OURS record of the level / start.
Metrics C_obs: AUC@20 / 50 / 100 / 150, Cov@20 / 50 / 100, Final = C[150]; level-equal means over the 22 levels; paired level
bootstrap against EXP7 OURS / MAG-B / RANDOM. Behaviour: distinct positions visited, fallbacks / ties / skipped frames (planner logs).
Outputs: experiments/cvpr_results/fisherrf_reproduction/results/*.csv|json."""
import csv
import glob
import json
import os
import sys

import numpy as np

R = "/bsuscratch/jiahuizhang/projects/occupancy/seen2scene"
F = os.path.join(R, "work_dirs", "cvpr_results", "fisherrf_reproduction")
OUT = os.path.join(R, "experiments", "cvpr_results", "fisherrf_reproduction", "results")
sys.path.insert(0, os.path.join(R, "experiments", "cvpr_results", "scripts"))
sys.path.insert(0, os.path.join(R, "experiments", "cvpr_results", "exp7", "scripts"))
import cvpr_lib as L  # noqa: E402


def boot(d, n=10000):
    d = np.asarray(d, float)
    b = d[np.random.default_rng(0).integers(0, len(d), (n, len(d)))].mean(1)
    return float(d.mean()), float(np.percentile(b, 2.5)), float(np.percentile(b, 97.5)), int((d > 0).sum())


def behaviour(log_glob):
    fb, ties, skipped, n = {}, 0, 0, 0
    for p in glob.glob(log_glob):
        for line in open(p):
            r = json.loads(line)
            n += 1
            fb[r.get("fallback")] = fb.get(r.get("fallback"), 0) + 1
            ties += int(r.get("ties") or 0) > 0
            skipped += bool(r.get("skipped_no_valid_depth"))
    return {"steps": n, "fallbacks": {str(k): v for k, v in fb.items()}, "steps_with_ties": ties, "skipped_no_valid_depth": skipped}


def outdoor():
    K = ["AUC@20", "AUC@50", "AUC@100", "Cov@20", "Cov@50", "Final"]
    met = lambda c: {"AUC@20": np.trapz(c[:21]) / 20, "AUC@50": np.trapz(c[:51]) / 50, "AUC@100": np.trapz(c[:101]) / 100, "Cov@20": c[20], "Cov@50": c[50], "Final": c[100]}
    eps, bad = {}, []
    for s in L.SCENE_ORDER:
        for k in range(5):
            u = f"{s}_s{k}"
            e = os.path.join(F, "outdoor", "closed_loop", u, "episodes", f"ep_FisherRF_{u}_r0.json")
            mtr = os.path.join(F, "outdoor", "closed_loop", u, "metrics", "FisherRF", f"FisherRF_{u}_r0.json")
            if not os.path.exists(e):
                continue
            ep = json.load(open(e))
            c = np.asarray(ep["coverage"], float)
            ok = len(c) == 101 and os.path.exists(mtr) and json.load(open(mtr)).get("official_equal_stored") is True
            if not ok:
                bad.append(u)
                continue
            keys = ep.get("executed_keys") or [x["pose_key"] for x in ep["steps"]]
            eps[(s, k)] = dict(met(c), positions=len({tuple(json.loads(x)[:3]) for x in keys}), curve=c.tolist())
    sc = {s: {kk: float(np.mean([eps[(s, k)][kk] for k in range(5)])) for kk in K} for s in L.SCENE_ORDER if all((s, k) in eps for k in range(5))}
    res = {"episodes_valid": len(eps), "episodes_invalid_or_unreplayed": bad, "scenes_complete": len(sc),
           "FisherRF": {kk: float(np.mean([v[kk] for v in sc.values()])) for kk in K} if sc else None,
           "positions_mean": float(np.mean([v["positions"] for v in eps.values()])) if eps else None}
    t1 = {r["method_id"]: r for r in csv.DictReader(open(os.path.join(R, "experiments", "cvpr_results", "fig", "table1", "table1_all14_full.csv")))}
    res["table1_rows"] = {m: {kk: float(t1[m][kk]) for kk in K} for m in ("RANDOM", "MACARONS", "MAG-B", "V3-T0-D1") if t1[m]["status"] == "COMPLETE"}
    if len(sc) == 14:
        res["paired"] = {}
        for m in ("V3-T0-D1", "MAG-B", "MACARONS", "RANDOM"):
            E = L.load(m)
            other = {s: {kk: float(np.mean([L.metrics(E[(s, k)]["coverage_official"])[kk] if hasattr(L, "metrics") else met(np.asarray(E[(s, k)]["coverage_official"]))[kk] for k in range(5)])) for kk in K} for s in L.SCENE_ORDER}
            res["paired"][f"FisherRF - {m}"] = {kk: boot([sc[s][kk] - other[s][kk] for s in L.SCENE_ORDER]) for kk in K}
    with open(os.path.join(OUT, "outdoor_episodes.csv"), "w", newline="") as f:
        w = csv.writer(f); w.writerow(["scene", "start"] + K + ["distinct_positions"])
        for (s, k), v in sorted(eps.items()):
            w.writerow([s, k] + [f"{v[kk]:.5f}" for kk in K] + [v["positions"]])
    res["behaviour"] = behaviour(os.path.join(F, "outdoor", "closed_loop", "*", "planner_logs", "*.steps.jsonl"))
    return res


def indoor():
    import collect_mp3d150 as CM
    K = ["AUC@20", "AUC@50", "AUC@100", "AUC@150", "Cov@20", "Cov@50", "Cov@100", "Final"]
    man = [r for r in csv.DictReader(open(os.path.join(R, "experiments", "cvpr_results", "exp7", "manifests", "mp3d150_manifest.csv"))) if r["method"] == "OURS"]
    eps, bad = {}, []
    for r in man:
        tag = f"{r['scene_id']}_S{r['start_id']}"
        e = os.path.join(F, "indoor", "closed_loop", "FisherRF", tag, "episode.json")
        if not os.path.exists(e):
            continue
        row = dict(r, method="FISHERRF", output_path=os.path.relpath(e, R))
        v = CM.validate(row)
        ours = json.load(open(os.path.join(R, "work_dirs", "cvpr_results", "exp7", "episodes", "mp3d150", "OURS", f"{tag}.json")))
        v["checks"]["V4_same_start_as_OURS"] = list(v["start"]) == list(ours["start_pose"])
        if not all(v["checks"].values()):
            bad.append((tag, v["checks"]))
            continue
        eps[tag] = dict(v["metrics"], scene=r["scene_id"], split=r["split"], positions=len({tuple(x[:3]) for x in v["executed"]}))
    lv = {}
    for t, v in eps.items():
        lv.setdefault(v["scene"], []).append(v)
    lvm = {s: {kk: float(np.mean([x[kk] for x in xs])) for kk in K} for s, xs in lv.items() if len(xs) == 2}
    split = {r["scene_id"]: r["split"] for r in man}
    res = {"episodes_valid": len(eps), "episodes_invalid": bad, "levels_complete": len(lvm)}
    for name, levels in (("ALL-22", sorted(split)), ("HO-12", sorted(s for s in split if split[s] == "HO")), ("VALSEL-10", sorted(s for s in split if split[s] == "VALSEL"))):
        done = [s for s in levels if s in lvm]
        res[name] = {"levels": f"{len(done)}/{len(levels)}", "FisherRF": {kk: float(np.mean([lvm[s][kk] for s in done])) for kk in K} if done else None}
    other = {}
    for r in csv.DictReader(open(os.path.join(R, "experiments", "cvpr_results", "exp7", "results", "mp3d150_levels.csv"))):
        other.setdefault(r["method"], {})[r["scene_id"]] = {kk: float(r[kk]) for kk in K}
    if len(lvm) == 22:
        res["paired_ALL-22"] = {f"FisherRF - {m}": {kk: boot([lvm[s][kk] - other[m][s][kk] for s in sorted(lvm)]) for kk in K} for m in ("OURS", "MAG-B", "RANDOM")}
    res["exp7_rows_ALL-22"] = {m: {kk: float(np.mean([other[m][s][kk] for s in sorted(split)])) for kk in K} for m in ("OURS", "MAG-B", "RANDOM")}
    res["positions_mean"] = float(np.mean([v["positions"] for v in eps.values()])) if eps else None
    with open(os.path.join(OUT, "indoor_episodes.csv"), "w", newline="") as f:
        w = csv.writer(f); w.writerow(["tag", "split"] + K + ["distinct_positions"])
        for t, v in sorted(eps.items()):
            w.writerow([t, v["split"]] + [f"{v[kk]:.5f}" for kk in K] + [v["positions"]])
    res["behaviour"] = behaviour(os.path.join(F, "indoor", "closed_loop", "FisherRF", "*", "planner_fisherrf.steps.jsonl"))
    return res


if __name__ == "__main__":
    os.makedirs(OUT, exist_ok=True)
    o, i = outdoor(), indoor()
    json.dump({"outdoor": o, "indoor": i}, open(os.path.join(OUT, "summary.json"), "w"), indent=1, default=float)
    print(json.dumps({"outdoor": {k: o[k] for k in o if k not in ("paired",)}, "indoor": {k: i[k] for k in i if not k.startswith("paired")}}, indent=1, default=float)[:4000])
