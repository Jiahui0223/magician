"""EXP5 Task 0.5 -- outdoor matched set (magician env; simulator / evaluation side; protocol_v1.2_outdoor.md).

prep  : Dev-5 scene cards (settings scene box, z-up OBJ units, S_RT = 10; k asserted = frozen E-M2 k_M, z0 asserted =
        abl_common.z0_of), read-only symlinks W/preprocessing/outdoor_traj/OUT_<scene>_S<k>/svc_t<t>.npz -> the frozen
        EXP3b T5 MAGICIAN replay records (t = 0..50), C_obs per step (same GT sampling / eps as indoor; normalised by
        the scene's MAGICIAN visibility_ratio, capped at 1), states = first entry per coverage stratum.
label : per outdoor state and legal candidate: GT t* on the scoring rays (T1.1 label renderer, Macarons++ mesh x 10)
        and the true gain (frozen sensor on MAGICIAN's own mesh, seed EXP3b derived_seed(scene, start, t + 1, sample)).
Outputs W/preprocessing/scenes/OUT_<scene>.json, W/preprocessing/outdoor_traj/*, W/manifests/outdoor_states.json,
W/fixed_state_out/labels/<state>.npz|json.
"""
import argparse
import json
import os
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
import mp3d_sim as S  # noqa: E402
import t04_states as TS  # noqa: E402

import numpy as np  # noqa: E402

sys.path.insert(0, os.path.join(S.REPO, "experiments", "exp3b_macaronspp_transfer"))
sys.path.insert(0, os.path.join(S.REPO, "experiments", "exp4_multiworld_belief", "ablation_study", "abl_1", "common"))
sys.path.insert(0, os.path.join(S.REPO, "experiments", "exp5_future_view", "task1", "scripts"))
DEV5 = ["barts", "colosseum", "eiffel", "pantheon", "sestino_museum"]
S_OUT = 10.0
OUTD = os.path.join(S.W, "fixed_state_out")


def card(scene):
    import abl_common as AC
    import t5_diagnostic as TD
    st = json.load(open(os.path.join(S.MACARONS, scene, "settings.json")))["scene"]
    lo, hi = np.asarray(st["x_min"], np.float64), np.asarray(st["x_max"], np.float64)          # y-up OBJ units
    z = lambda p: np.array([p[0], -p[2], p[1]])                                                 # noqa: E731
    a, b = z(lo), z(hi)
    box = np.stack([np.minimum(a, b), np.maximum(a, b)])
    fr = TD.frame_of(scene)
    k = S_OUT * (box[1, 2] - box[0, 2]) / (S.MC.PATCH * S.MC.VS)
    assert abs(k - fr.k_M) < 1e-6, (scene, k, fr.k_M)
    rt = S_OUT * (hi - lo)
    z0 = float(np.sum(np.abs(rt)) / 3.0 / 2.0)
    assert abs(z0 - AC.z0_of(scene)) < 1e-6, (scene, z0, AC.z0_of(scene))
    c = {"key": f"OUT_{scene}", "scan": f"OUT_{scene}", "box_m": box.tolist(), "height_m": float(box[1, 2] - box[0, 2]),
         "k_I": k, "z0_rt": z0, "s_rt": S_OUT, "n_win": 3, "unit": "OBJ (Macarons++), runtime = OBJ x 10",
         "visibility_ratio": float(json.load(open(os.path.join(S.MACARONS, scene, "settings.json")))["scene"]["visibility_ratio"])}
    p = os.path.join(S.W, "preprocessing", "scenes", f"OUT_{scene}.json")
    json.dump(c, open(p, "w"), indent=1)
    return c


class OutCoverage:
    """GT samples in the scene box (as indoor), covered within eps, normalised by MAGICIAN's visibility_ratio."""

    def __init__(self, c):
        from scipy.spatial import cKDTree
        v, f = S.mesh_zup(c["key"])
        box = np.asarray(c["box_m"])
        cen = v[f].mean(1)
        inb = np.all((cen >= box[0]) & (cen <= box[1]), axis=1)
        tri = v[f[inb]]
        area = 0.5 * np.linalg.norm(np.cross(tri[:, 1] - tri[:, 0], tri[:, 2] - tri[:, 0]), axis=1)
        rng = np.random.default_rng(S.derived_seed(c["key"], -1, -1, "gt") % (2 ** 63))
        n = S.N_GT_POINTS
        fi = rng.choice(len(tri), n, p=area / area.sum())
        r1, r2 = rng.random(n), rng.random(n)
        s = np.sqrt(r1)
        p = (1 - s)[:, None] * tri[fi, 0] + (s * (1 - r2))[:, None] * tri[fi, 1] + (s * r2)[:, None] * tri[fi, 2]
        self.pts = S_OUT * np.stack([p[:, 0], p[:, 2], -p[:, 1]], 1)               # runtime y-up
        self.eps_rt, self.vis = S.EPS_MODEL * c["k_I"], c["visibility_ratio"]
        self.covered = np.zeros(n, bool)
        self.tree = cKDTree(self.pts)

    def near(self, obs):
        from scipy.spatial import cKDTree
        if len(obs) == 0:
            return np.zeros(len(self.pts), bool)
        d, _ = cKDTree(np.asarray(obs, np.float64)).query(self.pts, k=1, distance_upper_bound=self.eps_rt, workers=-1)
        return np.isfinite(d)

    def update(self, obs):
        self.covered |= self.near(obs)
        return self.value()

    def value(self):
        return float(min(1.0, self.covered.mean() / self.vis))


def mode_prep():
    import t5_diagnostic as TD
    rows_all, states, support = {}, [], {}
    for scene in DEV5:
        c = card(scene)
        for k in range(5):
            traj = f"OUT_{scene}_S{k}"
            od = os.path.join(S.W, "preprocessing", "outdoor_traj", traj)
            os.makedirs(od, exist_ok=True)
            cov = OutCoverage(c)
            rows = []
            for t in range(51):
                src = TD.svc_path(scene, k, t)
                dst = os.path.join(od, f"svc_t{t}.npz")
                if not os.path.islink(dst):
                    os.symlink(src, dst)
                rec = S.MC.load_service_record(dst)
                rows.append({"t": t, "cov": cov.update(rec["sample_points"])})
            json.dump({"scene": {"key": c["key"]}, "start_pano": k, "rows": rows, "source": os.path.dirname(TD.svc_path(scene, k, 0))},
                      open(os.path.join(od, "traj.json"), "w"), indent=1)
            rows_all[traj] = rows
            for nm, lo, hi in TS.STRATA:
                t = next((r["t"] for r in rows if lo <= r["cov"] < hi), None)
                if t is None:
                    support.setdefault(nm, {"missing": []})["missing"].append(traj)
                    continue
                states.append({"id": f"{traj}_t{t}", "traj": traj, "level_key": c["key"], "house": f"OUT_{scene}", "start": k,
                               "t": t, "stratum": nm, "cov": rows[t]["cov"]})
            print(traj, "cov", [round(r["cov"], 3) for r in rows[::10]], flush=True)
    sup = {nm: {"states": sum(s["stratum"] == nm for s in states), "scenes": len({s["house"] for s in states if s["stratum"] == nm}),
                "missing": support.get(nm, {}).get("missing", [])} for nm, _, _ in TS.STRATA}
    json.dump({"rule": "protocol_v1.2_outdoor.md §3", "states": states, "support": sup, "n_states": len(states)},
              open(os.path.join(S.W, "manifests", "outdoor_states.json"), "w"), indent=1)
    print(json.dumps({k: {kk: (len(vv) if isinstance(vv, list) else vv) for kk, vv in v.items()} for k, v in sup.items()}), len(states), "states")


def mode_label(part):
    import torch
    import t11_labels as TL
    import t5_diagnostic as TD
    from pytorch3d.structures import Meshes
    sys.path.insert(0, "/bsuscratch/jiahuizhang/projects/occupancy/MAGICIAN_clean_9521d69")
    from macarons.utility.macarons_utils import load_scene
    dev = torch.device("cuda")
    sts = sorted(json.load(open(os.path.join(S.W, "manifests", "outdoor_states.json")))["states"], key=lambda s: s["house"])
    pi, pn = (int(x) for x in part.split("/"))
    out = os.path.join(OUTD, "labels")
    os.makedirs(out, exist_ok=True)
    ctx = {}
    for st in sts[pi::pn]:
        if os.path.exists(os.path.join(out, f"{st['id']}.json")):
            continue
        scene = st["house"][4:]
        if ctx.get("scene") != scene:
            c = json.load(open(os.path.join(S.W, "preprocessing", "scenes", f"OUT_{scene}.json")))
            obj = [f for f in os.listdir(os.path.join(S.MACARONS, scene)) if f.endswith(".obj")][0]
            mm = load_scene(os.path.join(S.MACARONS, scene, obj), 10.0, dev, mirror=False, mirrored_axis=None)
            sen = S.Sensor(mm.verts_packed().cpu().numpy(), mm.faces_packed().cpu().numpy(), dev)
            sen.mesh = mm
            v, f = S.mesh_zup(f"OUT_{scene}")
            vrt = S_OUT * np.stack([v[:, 0], v[:, 2], -v[:, 1]], 1)
            ren = TL.Renderer(Meshes(verts=[torch.as_tensor(vrt, dtype=torch.float32, device=dev)],
                                     faces=[torch.as_tensor(f, dtype=torch.int64, device=dev)]), dev)
            ctx = {"scene": scene, "card": c, "sen": sen, "ren": ren}
        c, sen, ren = ctx["card"], ctx["sen"], ctx["ren"]
        td = os.path.join(S.W, "preprocessing", "outdoor_traj", st["traj"])
        tr = json.load(open(os.path.join(td, "traj.json")))
        cov = OutCoverage(c)
        for i in range(st["t"] + 1):
            cv = cov.update(S.MC.load_service_record(os.path.join(td, f"svc_t{i}.npz"))["sample_points"])
        assert abs(cv - tr["rows"][st["t"]]["cov"]) < 1e-12
        rec = S.MC.load_service_record(os.path.join(td, f"svc_t{st['t']}.npz"))
        base = cov.covered.copy()
        gains = []
        for j in range(len(rec["legal_R"])):
            R = torch.tensor(rec["legal_R"][j][None], dtype=torch.float32, device=dev)
            T = torch.tensor(rec["legal_T"][j][None], dtype=torch.float32, device=dev)
            fr = sen.render(R, T)
            pts, _, _, _, _ = sen.sample(fr, S.MC.derived_seed(scene, st["start"], st["t"] + 1, "sample"))
            cov.covered = base.copy()
            gains.append(100.0 * (cov.update(pts.cpu().numpy()) - cv))
        n = len(rec["legal_R"])
        tstar = ren.tstar(rec["legal_R"], rec["legal_T"], np.zeros((n, 2)), c["k_I"]).astype(np.float32) if n else np.zeros((0, 7296), np.float32)
        np.savez_compressed(os.path.join(out, f"{st['id']}.npz"), tstar=tstar, gain=np.asarray(gains), legal_idx=rec["legal_idx"])
        json.dump({"state": st, "cov_t": cv, "gain": gains, "k_rt": c["k_I"]}, open(os.path.join(out, f"{st['id']}.json"), "w"),
                  indent=1, default=float)
        print(st["id"], "gains", [round(g, 2) for g in gains], flush=True)


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("mode", choices=["prep", "label"])
    ap.add_argument("--part", default="0/1")
    a = ap.parse_args()
    mode_prep() if a.mode == "prep" else mode_label(a.part)
