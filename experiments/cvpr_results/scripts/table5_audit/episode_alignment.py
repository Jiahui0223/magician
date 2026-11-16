"""Table 5 alignment audit (2026-11-16): per Unseen-6 episode, compare weights, planner config, start pose, cov0, evaluator settings, restarts, hovers and fallbacks of every row against Ours."""
import os, sys, json, collections, hashlib
R = "/bsuscratch/jiahuizhang/projects/occupancy/seen2scene/"
sys.path.insert(0, R + "experiments/cvpr_results/scripts")
sys.path.insert(0, R + "experiments/exp6_v3/ablation_study/experiment_G_generative_vs_deterministic/scripts")
import cvpr_lib as L, g3_collect as G3
U6 = ["alhambra", "neuschwanstein", "eiffel", "bridge", "barts", "sestino_museum"]
E = {"D0": G3.load_dir(G3.WG, "D0")[0], "MRAW": G3.load_dir(os.path.join(G3.WG, "g3"), "MRAW")[0],
     "MDEPTH": G3.load_dir(os.path.join(G3.WG, "g3"), "MDEPTH")[0], "G4": L.load_manifest("V3-T0-D1")}
for j, m in enumerate(["V3-T0-K1D2", "V3-T0-K1D2-N1", "V3-T0-K1D2-N2", "V3-T0-K1D2-N3"]):
    E[f"G1n{j}"] = L.load_manifest(m)
def J(p): return json.load(open(p if os.path.isabs(p) else R + p))
def strip(pl):
    pl = json.loads(json.dumps(pl))
    for k in ("K",): pl.pop(k, None)
    return pl
info = collections.defaultdict(lambda: collections.Counter())
ref = {}
diffs = collections.defaultdict(list)
for a in E:
    for s in U6:
        for k in range(5):
            e = E[a][(s, k)]
            d, m = J(e["episode_file"]), J(e["metrics_file"])
            hs = d["planner_hello"]
            h = hs[0]
            st = d["steps"]
            key = (s, k)
            rec = dict(start=st[0]["pose_key"], cov0=round(st[0]["cov"], 12), n_gt=m["n_gt"], eps=m["eps"], vis=json.dumps(m["vis"])[:200],
                       planner=json.dumps({kk: vv for kk, vv in strip(h["planner"]).items()}, sort_keys=True), seed=d["seed"], exp=d["exp"])
            info[a]["ckpt " + h["ckpt_sha256"][:8]] += 1
            info[a][f"K={h['planner'].get('K')}"] += 1
            info[a][f"n_hello={len(hs)}"] += 1
            info[a][f"moves={d['moves']}"] += 1
            info[a][f"resume={d['resume'] is not None}"] += 1
            info[a][f"hover={d['n_hover']}"] += 1
            info[a][f"official_equal_stored={m['official_equal_stored']}"] += 1
            nfb = sum(1 for x in st if isinstance(x.get("planner"), dict) and x["planner"].get("fallback"))
            info[a][f"fallback_steps>0={nfb > 0}"] += 1
            if a == "G4":
                ref[key] = rec
    for s in U6:
        for k in range(5):
            pass
for a in E:
    if a == "G4": continue
    for s in U6:
        for k in range(5):
            e = E[a][(s, k)]; d, m = J(e["episode_file"]), J(e["metrics_file"]); h = d["planner_hello"][0]; st = d["steps"]
            rec = dict(start=st[0]["pose_key"], cov0=round(st[0]["cov"], 12), n_gt=m["n_gt"], eps=m["eps"], vis=json.dumps(m["vis"])[:200],
                       planner=json.dumps(strip(h["planner"]), sort_keys=True), seed=d["seed"], exp=d["exp"])
            for f in rec:
                if rec[f] != ref[(s, k)][f]:
                    diffs[(a, f)].append((s, k))
print("== per-arm counters (30 episodes each)")
for a in E: print(a, dict(info[a]))
print("== fields that differ from Ours (same scene/start)")
for (a, f), v in sorted(diffs.items()): print(a, f, len(v), "episodes")
print("planner field example diff:")
for a in ("D0", "MRAW", "G1n0"):
    e = E[a][("alhambra", 0)]; h = J(e["episode_file"])["planner_hello"][0]["planner"]
    g = J(E["G4"][("alhambra", 0)]["episode_file"])["planner_hello"][0]["planner"]
    print(" ", a, {k: (h.get(k), g.get(k)) for k in set(h) | set(g) if json.dumps(h.get(k), sort_keys=True) != json.dumps(g.get(k), sort_keys=True)})
