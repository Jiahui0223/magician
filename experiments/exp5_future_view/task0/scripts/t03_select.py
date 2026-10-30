"""EXP5 Task 0.3 -- indoor scene / start selection (magician env; rule fixed before any model output):
for every official-val house and every level with panoramas: panorama graph; eligible starts = panoramas of the
largest connected component with degree >= 1; levels whose largest component has < MIN_COMPONENT panoramas are
excluded (reported); 2 starts per level drawn without replacement with numpy default_rng(sha256 seed
"exp5-T0|starts|<level key>"). Writes W/manifests/indoor_scenes.json and W/manifests/traj_jobs.txt.
"""
import hashlib
import json
import os
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
import mp3d_sim as S  # noqa: E402

import numpy as np  # noqa: E402

MIN_COMPONENT = 4
N_STARTS = 2


def components(g):
    seen, comps = set(), []
    for s in sorted(g):
        if s in seen:
            continue
        c, st = [], [s]
        while st:
            u = st.pop()
            if u in seen:
                continue
            seen.add(u)
            c.append(u)
            st += g[u]
        comps.append(sorted(c))
    return sorted(comps, key=lambda c: (-len(c), c[0]))


def main():
    val = [x.strip() for x in open(os.path.join(S.W, "download_logs", "official_raw", "scenes_val.txt")) if x.strip()]
    out, jobs = {"rule": __doc__, "houses": {}}, []
    for scan in sorted(val):
        h = S.parse_house(scan)
        levels = sorted(set(p["level"] for p in h["panos"] if p["level"] >= 0))
        out["houses"][scan] = {}
        for lv in levels:
            L = S.Level(scan, lv)
            g = L.build_graph()
            S.write_scene_card(L)
            comp = components(g)[0]
            elig = [i for i in comp if len(g[i]) >= 1]
            rec = {"n_panos": len(L.panos), "largest_component": len(comp), "eligible": len(elig),
                   "height_m": L.height_m, "k_rt": L.k, "extent_model": L.frame.extent_model.tolist()}
            if len(comp) < MIN_COMPONENT:
                rec["excluded"] = f"largest component {len(comp)} < {MIN_COMPONENT}"
            else:
                seed = int(hashlib.sha256(f"exp5-T0|starts|{L.key}".encode()).hexdigest()[:15], 16)
                pick = np.random.default_rng(seed).choice(len(elig), min(N_STARTS, len(elig)), replace=False)
                starts = sorted(L.panos[elig[i]]["idx"] for i in pick)
                rec["starts_pano_idx"] = starts
                jobs += [f"{scan} {lv} {s}" for s in starts]
            out["houses"][scan][str(lv)] = rec
            print(scan, lv, rec, flush=True)
    md = os.path.join(S.W, "manifests")
    json.dump(out, open(os.path.join(md, "indoor_scenes.json"), "w"), indent=1, default=float)
    open(os.path.join(md, "traj_jobs.txt"), "w").write("\n".join(jobs) + "\n")
    print(len(jobs), "trajectories")


if __name__ == "__main__":
    main()
