"""Table 5 audit (2026-11-16): planner fallback steps per row on Unseen-6 (fallback all_zero = every candidate has zero predicted gain -> seeded random legal move, same rule for every row)."""
import os, sys, json, collections
R = "/bsuscratch/jiahuizhang/projects/occupancy/seen2scene/"
sys.path.insert(0, R + "experiments/cvpr_results/scripts")
sys.path.insert(0, R + "experiments/exp6_v3/ablation_study/experiment_G_generative_vs_deterministic/scripts")
import cvpr_lib as L, g3_collect as G3
U6 = ["alhambra", "neuschwanstein", "eiffel", "bridge", "barts", "sestino_museum"]
E = {"D0": G3.load_dir(G3.WG, "D0")[0], "MRAW": G3.load_dir(os.path.join(G3.WG, "g3"), "MRAW")[0],
     "MDEPTH": G3.load_dir(os.path.join(G3.WG, "g3"), "MDEPTH")[0], "G4": L.load_manifest("V3-T0-D1"), "G1n0": L.load_manifest("V3-T0-K1D2"), "G1n1": L.load_manifest("V3-T0-K1D2-N1"), "G1n2": L.load_manifest("V3-T0-K1D2-N2"), "G1n3": L.load_manifest("V3-T0-K1D2-N3")}
for a in E:
    c, steps_by_t = collections.Counter(), collections.Counter()
    tot = 0
    for s in U6:
        for k in range(5):
            st = json.load(open(R + E[a][(s, k)]["episode_file"]))["steps"]
            for x in st:
                p = x.get("planner")
                if isinstance(p, dict) and p.get("fallback"):
                    c[str(p["fallback"])[:60]] += 1; tot += 1; steps_by_t["<50" if x["t"] < 50 else ">=50"] += 1
    print(a, "fallback steps", tot, "of 3000;", dict(steps_by_t), dict(c.most_common(4)))
