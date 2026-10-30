"""EXP5 Task 2 / T2.1a step 1 -- MP3D training / selection levels and trajectory jobs (magician env, CPU).

For every level of the TRAIN houses and of the VALSEL houses with >= MIN_PANOS panoramas: the Task 0 Level (floor /
ceiling from vertical rays, level box, scale k) -> a scene card in W2/mp3d/scenes/<key>.json (same fields as the Task 0
cards + grid / region parameters). The mesh cache of these houses is written to W2/mp3d/mesh_cache (not into Task 0).
Jobs (W2/mp3d/manifests/traj_jobs.json), all with 150 moves on the T0.12 grid:
  train levels  : 2 trajectories, policy 'rnv' (seeded uniform among unvisited legal poses) and 'ogr' (GT one-step greedy
                  on newly covered GT points, GT used only to generate training states), each from its own seeded start
                  panorama and seeded start azimuth
  valsel levels : 1 trajectory, policy seeded from {'rnv', 'ogr'}
"""
import json
import os
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
import t2_common as C  # noqa: E402

sys.path.insert(0, os.path.join(C.EXP5, "task0", "scripts"))
import mp3d_sim as S  # noqa: E402

import numpy as np  # noqa: E402

_mesh0 = S.mesh_zup


def mesh_zup(scan):
    """S.mesh_zup with the cache in W2 for houses Task 0 never cached (Task 0 files are reused read-only)."""
    if os.path.exists(os.path.join(S.W, "preprocessing", "mesh_cache", f"{scan}.npz")):
        return _mesh0(scan)
    cp = os.path.join(C.W2, "mp3d", "mesh_cache", f"{scan}.npz")
    if os.path.exists(cp):
        z = np.load(cp)
        return z["v"], z["f"]
    import glob
    import hashlib
    import trimesh
    obj = glob.glob(os.path.join(C.MP, scan, "matterport_mesh", "*", "*.obj"))
    assert len(obj) == 1, obj
    m = trimesh.load(obj[0], process=False, force="mesh", skip_materials=True)
    v, f = np.asarray(m.vertices, np.float64), np.asarray(m.faces, np.int64)
    os.makedirs(os.path.dirname(cp), exist_ok=True)
    np.savez(cp + ".tmp.npz", v=v, f=f, obj_sha256=hashlib.sha256(open(obj[0], "rb").read()).hexdigest())
    os.replace(cp + ".tmp.npz", cp)
    return v, f


S.mesh_zup = mesh_zup


def main():
    part = sys.argv[1] if len(sys.argv) > 1 else "0/1"           # "i/n": cards of every n-th house only (no job file)
    pi, pn = (int(x) for x in part.split("/"))
    out_dir = os.path.join(C.W2, "mp3d", "scenes")
    os.makedirs(out_dir, exist_ok=True)
    jobs, summary = [], {}
    for purpose, houses in (("train", C.TRAIN_HOUSES), ("valsel", C.VALSEL_HOUSES)):
        for hi, scan in enumerate(houses):
            if pn > 1 and hi % pn != pi:
                continue
            C.assert_mesh_allowed(scan, purpose)
            h = S.parse_house(scan)
            for lv in sorted({p["level"] for p in h["panos"] if p["level"] >= 0}):
                n = sum(p["level"] == lv for p in h["panos"])
                key = f"{scan}_L{lv}"
                if n < C.MIN_PANOS:
                    summary[key] = {"purpose": purpose, "excluded": f"{n} panoramas < {C.MIN_PANOS}"}
                    continue
                cp = os.path.join(out_dir, f"{key}.json")
                if os.path.exists(cp):
                    card = json.load(open(cp))
                else:
                    L = S.Level(scan, lv)
                    card = {**L.describe(), "scan": scan, "level": lv, "purpose": purpose, "pano_idx": [p["idx"] for p in L.panos],
                            "pano_rt": L.pano_rt.tolist(), "elevs": list(S.ELEVS), "n_azim": S.N_AZIM, "s_rt": S.S_RT, "n_win": C.N_WIN}
                    json.dump(card, open(cp + ".tmp", "w"), indent=1)
                    os.replace(cp + ".tmp", cp)
                rng = np.random.default_rng(C.seed_of(f"exp5-T2|traj|{key}"))
                pols = ["rnv", "ogr"] if purpose == "train" else [["rnv", "ogr"][int(rng.integers(2))]]
                starts = rng.choice(len(card["pano_idx"]), len(pols), replace=False)
                for pol, si in zip(pols, starts):
                    jobs.append({"key": key, "scan": scan, "level": lv, "purpose": purpose, "policy": pol,
                                 "start_pano": int(card["pano_idx"][si]), "start_azim": int(rng.integers(8)),
                                 "traj": f"{pol}_p{int(card['pano_idx'][si])}"})
                summary[key] = {"purpose": purpose, "n_panos": n, "height_m": card["height_m"], "box_m": card["box_m"]}
                print(purpose, key, n, "panos", f"{card['height_m']:.2f} m", flush=True)
    if pn > 1:
        return
    md = os.path.join(C.W2, "mp3d", "manifests")
    os.makedirs(md, exist_ok=True)
    json.dump({"doc": __doc__, "levels": summary, "jobs": jobs}, open(os.path.join(md, "traj_jobs.json"), "w"), indent=1)
    print(len(jobs), "trajectory jobs;", sum(j["purpose"] == "train" for j in jobs), "train,", sum(j["purpose"] == "valsel" for j in jobs), "valsel")


if __name__ == "__main__":
    main()
