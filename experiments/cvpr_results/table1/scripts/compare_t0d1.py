"""Paired comparison of V3-T0-D1 against V3-D1-H3-NORISK (V3-T1 + D1) and MAG-B on the canonical Table 1 records.
Scene-level pairing (mean of the 5 shared starts), scene bootstrap 95% CI (10,000 draws, seed 0). Prints a Markdown table."""
import glob, json, os, sys
import numpy as np, yaml
R = "/bsuscratch/jiahuizhang/projects/occupancy/seen2scene"; TB = os.path.join(R, "work_dirs", "cvpr_results", "table1", "episodes")
reg = yaml.safe_load(open(os.path.join(R, "experiments", "cvpr_results", "table1", "SCENE_REGISTRY.yaml")))["scenes"]
SPLIT = {s["local_id"]: s["split"] for s in reg}; NAME = {s["local_id"]: s["official_name"] for s in reg}
NEW = sys.argv[1] if len(sys.argv) > 1 else "V3-T0-D1"; REFS = ("V3-D1-H3-NORISK", "MAG-B", "V3")
rng = np.random.default_rng(0)
def load(m):
    out = {}
    for f in glob.glob(os.path.join(TB, m, "*.json")):
        r = json.load(open(f)); out[(r["scene"], int(r["start"]))] = np.asarray(r["coverage_official"], float)
    return out
E = {m: load(m) for m in (NEW,) + REFS}
scenes = sorted({k[0] for k in E[NEW]}); full = [s for s in scenes if sum(1 for k in E[NEW] if k[0] == s) == 5]
print(f"{NEW}: {len(E[NEW])} episodes, {len(full)} complete scenes of {len(scenes)}")
MET = {"AUC@20": lambda c: np.trapz(c[:21]) / 20, "Cov@20": lambda c: c[20], "AUC@50": lambda c: np.trapz(c[:51]) / 50, "AUC@100": lambda c: np.trapz(c[:101]) / 100, "Final": lambda c: c[100]}
GROUPS = {"Train-6": [s for s in full if SPLIT[s].startswith("train")], "Dev-5": [s for s in full if SPLIT[s].startswith("dev")], "Test-3": [s for s in full if SPLIT[s].startswith("test")], "Unseen-8": [s for s in full if not SPLIT[s].startswith("train")], "All-14": full}
sc = {m: {s: np.mean([E[m][(s, k)] for k in range(5)], 0) for s in full} for m in E}
def boot(v):
    v = np.asarray(v); b = v[rng.integers(0, len(v), (10000, len(v)))].mean(1); return v.mean(), np.percentile(b, 2.5), np.percentile(b, 97.5)
print("\n| 组 | 指标 | " + NEW + " | V3-T1 + D1 | Δ(T0 − T1) [95% CI] | MAGICIAN | Δ(T0 − MAG) [95% CI] |\n|---|---|---|---|---|---|---|")
for g, ss in GROUPS.items():
    if not ss: continue
    for met, fn in MET.items():
        a = [fn(sc[NEW][s]) for s in ss]; b = [fn(sc["V3-D1-H3-NORISK"][s]) for s in ss]; c = [fn(sc["MAG-B"][s]) for s in ss]
        d1 = boot(np.subtract(a, b)); d2 = boot(np.subtract(a, c))
        print(f"| {g} ({len(ss)}) | {met} | {np.mean(a):.3f} | {np.mean(b):.3f} | {d1[0]:+.3f} [{d1[1]:+.3f}, {d1[2]:+.3f}] | {np.mean(c):.3f} | {d2[0]:+.3f} [{d2[1]:+.3f}, {d2[2]:+.3f}] |")
print("\n| 场景 | 划分 | AUC@20 T0-D1 / T1-D1 / MAG | AUC@100 T0-D1 / T1-D1 / MAG | Final T0-D1 / T1-D1 / MAG |\n|---|---|---|---|---|")
for s in full:
    row = [f"{MET[m](sc[x][s]):.3f}" for m in ("AUC@20", "AUC@100", "Final") for x in (NEW, "V3-D1-H3-NORISK", "MAG-B")]
    print(f"| {NAME[s]} | {SPLIT[s]} | {' / '.join(row[0:3])} | {' / '.join(row[3:6])} | {' / '.join(row[6:9])} |")
# per-start paired on the T1 row: how many of the 70 episodes does T0 beat T1 on AUC@100
w = sum(1 for s in full for k in range(5) if MET["AUC@100"](E[NEW][(s, k)]) > MET["AUC@100"](E["V3-D1-H3-NORISK"][(s, k)]))
print(f"\nT0-D1 > T1-D1 on AUC@100 in {w} / {5 * len(full)} paired episodes")
