"""CVPR Table 1 -- read-only audit of the historical OUTDOOR (Macarons++) closed-loop episodes of every registered model.

For each model (MODEL_REGISTRY.yaml ids) the primary historical G3 root is scanned; every (scene, start) episode file
episodes/ep_<tag>.json and its coverage file metrics/.../<tag>.json are checked:
  episode : exists, len(executed_keys) == 101, len(coverage) == 101, moves == 100, len(steps) == 101, planner_bye present,
            resume field, n_hover, planner checkpoint sha256 (planner_hello) vs registry, P1 mode (planner stderr), timings
  metrics : exists, cov_official / cov_mono (101 values), auc_*/cov_*_final, official_equal_stored, cov_official == coverage
AUC variants are recomputed on the stored curves (no simulator): trapz(c[0..100], dx=1)/100 (historical AUC@100),
mean of the 101 values, and MAGICIAN evaluation_lmdb.py's unnormalised trapz(c[:100]) (x = 0..99).
Secondary (duplicate / replay) sources are listed separately and never counted as extra episodes.
CPU only, nothing is written outside experiments/cvpr_results/table1/audit/.
Usage: /bsuscratch/jiahuizhang/envs/seen2scene/bin/python audit_model_episodes.py
"""
import csv
import glob
import json
import os
import re

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
T1 = os.path.dirname(HERE)
REPO = os.path.dirname(os.path.dirname(os.path.dirname(T1)))
W = os.path.join(REPO, "work_dirs")
OUT = os.path.join(T1, "audit")
SCENES = ["alhambra", "bridge", "neuschwanstein", "barts", "pantheon", "colosseum"]   # Test-3 + Dev-3 (task2 eval_outdoor_v1)
SPLIT = {"alhambra": "Test-3", "bridge": "Test-3", "neuschwanstein": "Test-3", "barts": "Dev-3", "pantheon": "Dev-3", "colosseum": "Dev-3"}
T2 = os.path.join(W, "exp5_future_view", "task2", "eval", "outdoor")
T3 = os.path.join(W, "exp5_future_view", "task3_full_depth_p1")

SHA = {"MAG-B": None, "MAG-G": None,
       "V2O": "a379c0fb9d7d80266a430ab6f48f39315fad54fb650c6c3740a999a02db079fc",
       "B0": "343f41ab8d566c52007c881b5f32375c3253dd8f1c7a0d8a906d412a165e4516",
       "B2": "343f41ab8d566c52007c881b5f32375c3253dd8f1c7a0d8a906d412a165e4516",
       "B1": "343f41ab8d566c52007c881b5f32375c3253dd8f1c7a0d8a906d412a165e4516",
       "E2D": "61439612cb6132afbe74976b82a531c8f827802b06b4f87ad6de1a197d0a2fa5",
       "E2S": "578855bb149956262f3eba976db837f09e61172542a4412a0404efcb69f1521b",
       "V3": "639c9d3ae3f3c069dc0399dc4cdd44102be2bf20e70d666a6a529c6affc62bdb",
       "V3.1": "13df0a45df56d125d474a27f0998b9be596be33d0fad1defbb9e82c863263265",
       "V4": "1b31afffbe4da2d94344e0a509b711b30561a34bc4aa521773f1595aafac00d4",
       "V4C": "c4190f1d05dd29d999f23657ac9be381910ad06e3e66d8b6ffbd77b8235cdbe4",
       "V5": "5ef30ac4fd0b4a175d1ddd72d78c504c82b31b40b56da68b1d9454533918d543"}
P1_EXPECTED = {"V2O": "sparse", "B0": "sparse", "B2": "b2", "B1": "full", "E2D": "b2", "E2S": "sparse", "V3": "b2",
               "V3.1": "b2", "V4": "b2", "V4C": "b2", "V5": "b2"}


def v2m(root):          # V2-family layout: <root>/episodes/ep_V2M_<s>_s<k>_r0.json
    return lambda s, k: (os.path.join(root, "episodes", f"ep_V2M_{s}_s{k}_r0.json"), os.path.join(root, "planner_logs", f"pl_V2M_{s}_s{k}_r0"))


PRIMARY = {   # model -> (description, episode locator, metrics locator)
    "MAG-B": ("task2 group MAG-B", lambda s, k: (os.path.join(T2, "closed_loop", "MAG-B", "episodes", f"ep_MAG-B_{s}_s{k}_r0.json"), None),
              lambda s, k: os.path.join(T2, "metrics", "MAG-B", f"MAG-B_{s}_s{k}_r0.json")),
    "MAG-G": ("task2 group MAG-G", lambda s, k: (os.path.join(T2, "closed_loop", "MAG-G", "episodes", f"ep_MAG-G_{s}_s{k}_r0.json"), None),
              lambda s, k: os.path.join(T2, "metrics", "MAG-G", f"MAG-G_{s}_s{k}_r0.json")),
    "V2O": ("task2 group V2O", lambda s, k: (os.path.join(T2, "closed_loop", "V2O", "episodes", f"ep_V2O_{s}_s{k}_r0.json"),
                                             os.path.join(T2, "closed_loop", "V2O", "planner_logs", f"pl_V2O_{s}_s{k}_r0")),
            lambda s, k: os.path.join(T2, "metrics", "V2O", f"V2O_{s}_s{k}_r0.json")),
    "B0": ("task2 group V2M (= Task 3 B0; task3 closed_loop/outdoor/B0/V2M is a symlink to it)",
           lambda s, k: (os.path.join(T2, "closed_loop", "V2M", "episodes", f"ep_V2M_{s}_s{k}_r0.json"),
                         os.path.join(T2, "closed_loop", "V2M", "planner_logs", f"pl_V2M_{s}_s{k}_r0")),
           lambda s, k: os.path.join(T2, "metrics", "V2M", f"V2M_{s}_s{k}_r0.json")),
    "B2": ("task3 B2", v2m(os.path.join(T3, "closed_loop", "outdoor", "B2", "V2M")),
           lambda s, k: os.path.join(T3, "metrics", "outdoor", "B2", "V2M", f"V2M_{s}_s{k}_r0.json")),
    "B1": ("task3 B1", v2m(os.path.join(T3, "closed_loop", "outdoor", "B1", "V2M")),
           lambda s, k: os.path.join(T3, "metrics", "outdoor", "B1", "V2M", f"V2M_{s}_s{k}_r0.json")),
    "E2D": ("task3 / E2 pilot E2D", v2m(os.path.join(T3, "closed_loop", "outdoor", "E2D", "V2M")),
            lambda s, k: os.path.join(T3, "metrics", "outdoor", "E2D", "V2M", f"V2M_{s}_s{k}_r0.json")),
    "E2S": ("task3 / E2 pilot E2S", v2m(os.path.join(T3, "closed_loop", "outdoor", "E2S", "V2M")),
            lambda s, k: os.path.join(T3, "metrics", "outdoor", "E2S", "V2M", f"V2M_{s}_s{k}_r0.json")),
    "V3": ("exp6_v3 G3 group T1", v2m(os.path.join(W, "exp6_v3", "eval", "closed_loop", "outdoor", "T1", "V2M")),
           lambda s, k: os.path.join(W, "exp6_v3", "eval", "metrics", "outdoor", "T1", "V2M", f"V2M_{s}_s{k}_r0.json")),
    "V3.1": ("exp6_v4 G3 group V31", v2m(os.path.join(W, "exp6_v4", "eval", "closed_loop", "outdoor", "V31", "V2M")),
             lambda s, k: os.path.join(W, "exp6_v4", "eval", "metrics", "outdoor", "V31", "V2M", f"V2M_{s}_s{k}_r0.json")),
    "V4": ("exp6_v4 G3 group V4", v2m(os.path.join(W, "exp6_v4", "eval", "closed_loop", "outdoor", "V4", "V2M")),
           lambda s, k: os.path.join(W, "exp6_v4", "eval", "metrics", "outdoor", "V4", "V2M", f"V2M_{s}_s{k}_r0.json")),
    "V4C": ("exp6_v5 G3 group V4C", v2m(os.path.join(W, "exp6_v5", "eval", "closed_loop", "outdoor", "V4C", "V2M")),
            lambda s, k: os.path.join(W, "exp6_v5", "eval", "metrics", "outdoor", "V4C", "V2M", f"V2M_{s}_s{k}_r0.json")),
    "V5": ("exp6_v5 G3 group V5", v2m(os.path.join(W, "exp6_v5", "eval", "closed_loop", "outdoor", "V5", "V2M")),
           lambda s, k: os.path.join(W, "exp6_v5", "eval", "metrics", "outdoor", "V5", "V2M", f"V2M_{s}_s{k}_r0.json")),
}
EXTRA_METRICS = {"B0": lambda s, k: os.path.join(T3, "metrics", "outdoor", "B0", "V2M", f"V2M_{s}_s{k}_r0.json")}
SECONDARY = {   # duplicate / replay outdoor episodes (documented as bitwise equal to the primary); never counted
    "B0": ["exp_density_ab/g0_replay/closed_loop/outdoor/B2_S10_S10/V2M", "exp5_future_view/task3_full_depth_p1/timing/outdoor_B0/V2M"],
    "B2": ["exp_density_ab/g0_replay/closed_loop/outdoor/B2_D100_S10/V2M", "exp5_future_view/task3_full_depth_p1/timing/outdoor_B2/V2M",
           "exp6_v2/smoke_g3/closed_loop/outdoor/B2/V2M", "exp_camera_pose/closed_loop/B2_C0/closed_loop/outdoor/B2_C0/V2M"],
    "B1": ["exp_density_ab/g0_replay/closed_loop/outdoor/B2_D100_D100/V2M"],
    "E2S": ["exp_density_ab/g0_replay/closed_loop/outdoor/E2S_S10_S10/V2M"],
    "E2D": ["exp_density_ab/g0_replay/closed_loop/outdoor/E2D_D100_S10/V2M", "exp_temporal_refresh/g3/closed_loop/outdoor/E2D_R10/V2M",
            "exp6_v2/smoke_g3/closed_loop/outdoor/E2D/V2M"],
    "V3": ["exp6_v3/eval/smoke/outdoor/T1/V2M"],
    "V3.1": ["exp6_v4/eval/smoke/outdoor/V31/V2M"],
    "V4": ["exp6_v4/eval/smoke/outdoor/V4/V2M", "exp_camera_pose/closed_loop/V4_C0/closed_loop/outdoor/V4_C0/V2M"],
    "V5": ["exp6_v5/eval/smoke/outdoor/V2M", "exp_camera_pose/closed_loop/V5_C0/closed_loop/outdoor/V5_C0/V2M"],
}


def aucs(c):
    c = np.asarray(c, np.float64)
    return {"trapz100": float(np.trapz(c, dx=1.0) / 100.0), "mean101": float(c.mean()),
            "evlmdb_unnorm": float(np.trapz(c[:100], dx=1.0)), "evlmdb_div99": float(np.trapz(c[:100], dx=1.0) / 99.0)}


def p1_mode(pl):
    if pl is None:
        return None
    f = pl + ".stderr"
    if not os.path.exists(f):
        return "no-stderr"
    head = open(f, errors="replace").read(4000)
    m = re.search(r"(?:T3_P1_MODE|DA_P1_MODE)=(\w+)", head)
    return m.group(1) if m else "sparse(no patch line)"


def one(model, s, k):
    ep_path, pl = PRIMARY[model][1](s, k)
    mt_path = PRIMARY[model][2](s, k)
    r = {"model": model, "scene": s, "split": SPLIT[s], "start": k, "episode": ep_path.replace(REPO + "/", ""),
         "metrics": mt_path.replace(REPO + "/", ""), "ep_exists": os.path.exists(ep_path), "mt_exists": os.path.exists(mt_path)}
    if r["ep_exists"]:
        e = json.load(open(ep_path))
        hello = (e.get("planner_hello") or [{}])[0]
        bye = e.get("planner_bye") or {}
        steps = e.get("steps") or []
        dec = [st.get("planner", {}).get("decision_s") for st in steps if isinstance(st.get("planner"), dict)]
        dec = [d for d in dec if d is not None]
        r.update({"n_keys": len(e.get("executed_keys", [])), "n_cov": len(e.get("coverage", [])), "moves": e.get("moves"),
                  "n_steps": len(steps), "bye": bool(bye.get("bye")), "resume": e.get("resume"), "n_hover": e.get("n_hover"),
                  "host": e.get("host"), "ckpt_sha256": hello.get("ckpt_sha256"),
                  "planner_info": hello.get("info") or hello.get("planner"),
                  "p1_mode": p1_mode(pl), "decision_s_total": bye.get("decision_s_total"),
                  "decision_s_mean": float(np.mean(dec)) if dec else None, "n_decisions": len(dec),
                  "decide_wall_sum": float(sum(st.get("decide_wall_s") or 0 for st in steps)),
                  "observe_sum": float(sum(st.get("observe_s") or 0 for st in steps)),
                  "seconds_this_process": e.get("seconds_this_process"), "peak_mem_gb": bye.get("peak_mem_gb"),
                  "auc_episode": e.get("auc"), "cov_final_episode": e.get("cov_final")})
        r["sha_ok"] = (SHA[model] is None) or (r["ckpt_sha256"] == SHA[model])
        r["p1_ok"] = (model not in P1_EXPECTED) or (r["p1_mode"] == P1_EXPECTED[model]) or \
                     (P1_EXPECTED[model] == "sparse" and str(r["p1_mode"]).startswith("sparse"))
        r["ep_complete"] = r["n_keys"] == 101 and r["n_cov"] == 101 and r["moves"] == 100 and r["n_steps"] == 101 and r["bye"]
        r.update({f"ep_auc_{a}": v for a, v in aucs(e["coverage"]).items()} if r["n_cov"] == 101 else {})
        cov_ep = e.get("coverage")
    if r["mt_exists"]:
        m = json.load(open(mt_path))
        need = ["cov_official", "cov_mono", "auc_mono", "auc_official", "cov_mono_final", "cov_official_final"]
        r["mt_fields_ok"] = all(x in m for x in need) and len(m.get("cov_official", [])) == 101 and len(m.get("cov_mono", [])) == 101
        r["official_equal_stored"] = m.get("official_equal_stored")
        r["mt_official_eq_episode"] = bool(r["ep_exists"] and m.get("cov_official") == cov_ep)
        for x in ("auc_mono", "auc_official", "cov_mono_final", "cov_official_final", "eps", "vis", "n_gt", "seconds"):
            r[x] = m.get(x)
        if r["mt_fields_ok"]:
            r.update({f"mono_auc_{a}": v for a, v in aucs(m["cov_mono"]).items()})
            r["mono_monotone"] = bool(np.all(np.diff(m["cov_mono"]) >= -1e-12))
    if model in EXTRA_METRICS:
        f2 = EXTRA_METRICS[model](s, k)
        r["metrics_alt"] = f2.replace(REPO + "/", "")
        if os.path.exists(f2) and r["mt_exists"]:
            m2 = json.load(open(f2))
            r["metrics_alt_equal"] = (m2.get("cov_mono") == m.get("cov_mono") and m2.get("cov_official") == m.get("cov_official"))
    r["reusable"] = bool(r.get("ep_complete") and r.get("mt_fields_ok") and r.get("sha_ok", True) and r.get("p1_ok", True)
                         and r.get("official_equal_stored") and r.get("mt_official_eq_episode"))
    return r


def scene_mean(rows, key):
    v = [np.mean([r[key] for r in rows if r["scene"] == s and r.get(key) is not None]) for s in SCENES
         if any(r["scene"] == s and r.get(key) is not None for r in rows)]
    return float(np.mean(v)) if v else None


def main():
    os.makedirs(OUT, exist_ok=True)
    rows, summ, sec = [], {}, {}
    for model in PRIMARY:
        rs = [one(model, s, k) for s in SCENES for k in range(5)]
        rows += rs
        tot = [r.get("seconds_this_process") for r in rs if r.get("seconds_this_process") and r.get("resume") is None]
        dts = [r.get("decision_s_total") for r in rs if r.get("decision_s_total")]
        summ[model] = {"source": PRIMARY[model][0], "n_expected": len(rs), "n_episode_files": sum(r["ep_exists"] for r in rs),
                       "n_complete": sum(bool(r.get("ep_complete")) for r in rs), "n_metrics": sum(r["mt_exists"] for r in rs),
                       "n_reusable": sum(r["reusable"] for r in rs), "n_resumed": sum(r.get("resume") is not None for r in rs if r["ep_exists"]),
                       "sha_ok_all": all(r.get("sha_ok", True) for r in rs), "p1_modes": sorted({str(r.get("p1_mode")) for r in rs}),
                       "hosts": sorted({str(r.get("host")) for r in rs}),
                       "official_equal_stored_all": all(bool(r.get("official_equal_stored")) for r in rs if r["mt_exists"]),
                       "scene_mean": {x: scene_mean(rs, x) for x in ("auc_mono", "cov_mono_final", "auc_official", "cov_official_final",
                                                                      "ep_auc_mean101", "ep_auc_evlmdb_div99", "mono_auc_mean101")},
                       "split_mean": {sp: {x: float(np.mean([np.mean([r[x] for r in rs if r["scene"] == s]) for s in SCENES if SPLIT[s] == sp]))
                                           for x in ("auc_official", "cov_official_final", "auc_mono", "cov_mono_final")}
                                      for sp in ("Dev-3", "Test-3") if all(r.get("auc_official") is not None for r in rs)},
                       "runtime_s": {"episode_wall_median": float(np.median(tot)) if tot else None,
                                     "episode_wall_min": float(np.min(tot)) if tot else None,
                                     "episode_wall_max": float(np.max(tot)) if tot else None, "n_wall": len(tot),
                                     "planner_decision_total_median": float(np.median(dts)) if dts else None,
                                     "planner_decision_per_step_mean": float(np.mean([r["decision_s_mean"] for r in rs if r.get("decision_s_mean")])) if dts else None}}
        if model in EXTRA_METRICS:
            summ[model]["metrics_alt_all_equal"] = all(r.get("metrics_alt_equal") for r in rs)
    for model, dirs in SECONDARY.items():
        sec[model] = []
        for d in dirs:
            p = os.path.join(W, d, "episodes")
            for f in sorted(glob.glob(os.path.join(p, "ep_*.json"))):
                e = json.load(open(f))
                tag = os.path.basename(f)[3:-5]
                s, k = e.get("scene"), e.get("start")
                prim = json.load(open(PRIMARY[model][1](s, k)[0])) if os.path.exists(PRIMARY[model][1](s, k)[0]) else None
                sec[model].append({"dir": d, "tag": tag, "moves": e.get("moves"), "n_keys": len(e.get("executed_keys", [])),
                                   "same_keys_as_primary": bool(prim and prim["executed_keys"] == e.get("executed_keys")),
                                   "same_curve_as_primary": bool(prim and prim["coverage"] == e.get("coverage")),
                                   "same_prefix_as_primary": bool(prim and prim["executed_keys"][:len(e.get("executed_keys", []))] == e.get("executed_keys"))})
    json.dump({"scenes": SCENES, "split": SPLIT, "summary": summ, "secondary": sec, "episodes": rows},
              open(os.path.join(OUT, "historical_outdoor_episodes.json"), "w"), indent=1, default=str)
    cols = ["model", "scene", "split", "start", "ep_exists", "ep_complete", "n_keys", "n_cov", "moves", "n_steps", "bye", "resume",
            "n_hover", "ckpt_sha256", "sha_ok", "p1_mode", "p1_ok", "mt_exists", "mt_fields_ok", "official_equal_stored",
            "mt_official_eq_episode", "auc_official", "cov_official_final", "auc_mono", "cov_mono_final", "ep_auc_mean101",
            "ep_auc_evlmdb_unnorm", "decision_s_total", "decision_s_mean", "seconds_this_process", "host", "reusable", "episode", "metrics"]
    with open(os.path.join(OUT, "historical_outdoor_episodes.csv"), "w", newline="") as fh:
        w = csv.DictWriter(fh, fieldnames=cols, extrasaction="ignore")
        w.writeheader()
        for r in rows:
            w.writerow(r)
    for m, v in summ.items():
        print(m, v["n_episode_files"], v["n_complete"], v["n_metrics"], v["n_reusable"], "resumed", v["n_resumed"], v["p1_modes"],
              "sha", v["sha_ok_all"], {k: (round(x, 4) if x is not None else None) for k, x in v["scene_mean"].items()},
              {k: (round(x, 1) if x else None) for k, x in v["runtime_s"].items()})
    for m, v in sec.items():
        for x in v:
            print("  sec", m, x["dir"], x["tag"], x["moves"], x["n_keys"], x["same_keys_as_primary"], x["same_curve_as_primary"], x["same_prefix_as_primary"])


if __name__ == "__main__":
    main()
