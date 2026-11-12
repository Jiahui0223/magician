"""FisherRF reproduction, Stage F1 -- planner child (fisherrf env; planner side, no GT).

The scorer and the online map are the OFFICIAL FisherRF-active-mapping code (commit 8b196cd22b1f, imported read-only from
the clone, not one line changed):
  map       models/SLAM/gaussian.py GaussianSLAM: init (first frame), track_rgbd (GT pose of the record = executed pose,
            add_new_gaussians, `mapping.num_iters` = 60 Adam iterations, keyframe), configured by the official
            configs/mp3d_gaussian_FR_eccv.yaml + the overrides of PROTOCOL_ADAPTATION.md §5 (listed in the hello line)
  score     GaussianSLAM.pose_eval(c2w of the legal candidates) = sum H_cand * (H_train + 0.1)^-1, H from the modified
            rasterizer with backward_power = 2 (gaussian.py:1312-1335, 1463-1527)
  pick      argmax; ties -> seeded uniform choice among the maxima (D5, seed sha256("fisherrf|scene|start|t")); empty legal
            list -> hover (pick None)
Adaptation decisions D1-D11 of PROTOCOL_ADAPTATION.md §9 (user decisions 2026-11-11). open3d is the stub in ./stubs (D8).

Input per step (JSON lines of m_t6_sim.PlannerClient: {t, svc} -> {t, pick, ...}; {cmd: end} -> {bye}):
  the whitelisted service record svc_<scene>_s<k>_t<t>.npz (m_common validator) and its RGB side file
  svc_<scene>_s<k>_t<t>_rgb.npz {t, rgb uint8 (256, 456, 3), R, T} written by the simulator wrapper (D9); t, R, T must
  equal the record. Nothing else is read: an open() guard records every path and refuses simulator / GT / evaluation
  files (GT mesh, dataset folders, the simulator's episodes/ ckpt/ frames/ folders, coverage metrics).
Own log: --log (one JSON line per step: timing with torch.cuda.synchronize, Gaussian counts, candidates, scores, ties,
peak GPU memory, render-vs-observation check, pose self-checks).
"""
import argparse
import builtins
import hashlib
import importlib.util
import io
import json
import math
import os
import random
import sys
import time

sys.dont_write_bytecode = True
HERE = os.path.dirname(os.path.abspath(__file__))
REPO = os.path.abspath(os.path.join(HERE, "..", "..", "..", ".."))
AM = "/bsuhome/jiahuizhang/scratch/projects/occupancy/FisherRF/FisherRF-active-mapping"
AM_COMMIT = "8b196cd22b1f4030f205778ad14b23fbd796c3b7"
AM_YAML = os.path.join(AM, "configs", "mp3d_gaussian_FR_eccv.yaml")
STUBS = os.path.join(HERE, "stubs")
M_COMMON = os.path.join(REPO, "experiments", "exp3b_macaronspp_transfer", "m_common.py")

H_IMG, W_IMG = 256, 456
FX = (min(H_IMG, W_IMG) / 2.0) / math.tan(math.radians(30.0))   # 221.70250 = PyTorch3D fov 60 on the short side (D6)
CX, CY = (W_IMG - 1) / 2.0, (H_IMG - 1) / 2.0                                                        # 227.5, 127.5 (D6)
SENSOR_RANGE = 70.0                                                                                  # runtime units (D11)
F_FLIP = (-1.0, -1.0, 1.0)            # PyTorch3D view (+x left, +y up, +z fwd) -> OpenCV (+x right, +y down, +z fwd)


def seed_of(s):
    return int(hashlib.sha256(s.encode()).hexdigest()[:15], 16)


class Guard:
    """Records every path opened through builtins.open / io.open / os.open; refuses simulator / GT / evaluation files."""
    SUBSTR = ("/Macarons++/", "/MAGICIAN/data/", "MAGICIAN_clean", "magician_lmdb", "occupied_pose", "/matterport3d/",
              "/mp3d_magician/", "/preprocessing/coverage/", "/preprocessing/mesh_cache/", "/metrics/", "/replay_frames/",
              "/cov_tmp/")

    def __init__(self, svc_dir):
        sim_root, od = os.path.dirname(os.path.dirname(svc_dir)), os.path.dirname(svc_dir)
        self.sim_dirs = [os.path.join(sim_root, x) + "/" for x in ("episodes", "ckpt", "frames")]
        self.sim_files = (os.path.join(od, "episode.json"), os.path.join(od, "planner.jsonl"))   # indoor t23 layout
        self.paths, self.denied = [], []

    def forbidden(self, p):
        a = os.path.abspath(p)
        return (any(x in a for x in self.SUBSTR) or a.endswith((".obj", ".ply", ".glb", ".house"))
                or any(a.startswith(d) for d in self.sim_dirs) or a in self.sim_files)

    def install(self):
        self._b, self._o = builtins.open, os.open
        g = self

        def check(p):
            if isinstance(p, int):
                return
            p = os.fspath(p) if isinstance(p, (str, bytes, os.PathLike)) else repr(p)
            p = p.decode() if isinstance(p, bytes) else p
            g.paths.append(p)
            if g.forbidden(p):
                g.denied.append(p)
                raise PermissionError(f"planner guard: forbidden open {p}")

        def b_open(file, *a, **k):
            check(file)
            return g._b(file, *a, **k)

        def o_open(path, *a, **k):
            check(path)
            return g._o(path, *a, **k)
        builtins.open, io.open, os.open = b_open, b_open, o_open
        return self

    def uninstall(self):
        builtins.open, io.open, os.open = self._b, self._b, self._o


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--scene", required=True)
    ap.add_argument("--start", type=int, required=True)
    ap.add_argument("--svc-dir", required=True)
    ap.add_argument("--log", required=True)
    ap.add_argument("--workdir", required=True, help="GaussianSLAM workdir (official mapping PNGs go here)")
    ap.add_argument("--resume-t", type=int, default=0)
    ap.add_argument("--svc-name", default="svc_{scene}_s{start}_t{t}.npz", help="record file name (indoor t23: svc_t{t}.npz)")
    ap.add_argument("--scene-key", default=None, help="expected record scene_key (default <scene>/<start>)")
    ap.add_argument("--scale", type=float, default=7.0, help="D4: FisherRF unit = runtime / s")
    ap.add_argument("--map-iters", type=int, default=60, help="D2 (a): official mapping.num_iters")
    ap.add_argument("--init-pose", choices=["tester", "docstring"], default="tester",
                    help="first frame: 'tester' = slam.init(rgb, depth, w2c) exactly as official tester.py:243; "
                         "'docstring' = slam.init(rgb, depth, c2w) as GaussianSLAM.init documents (gaussian.py:446)")
    ap.add_argument("--verify-every", type=int, default=0,
                    help=">0: every k steps recompute the EIG of the candidates from compute_H_train / compute_Hessian "
                         "and compare with pose_eval (smoke check; costs one extra H_train)")
    a = ap.parse_args()

    proto = os.fdopen(os.dup(1), "w")                 # protocol channel; everything else printed goes to stderr
    os.dup2(2, 1)
    sys.stdout = sys.stderr
    assert os.environ.get("SLURM_JOB_QOS") == "kd-high", "set SLURM_JOB_QOS=kd-high (R5: disables ClusterStateManager)"

    sys.path.insert(0, AM)
    sys.path.insert(0, STUBS)
    import numpy as np
    import torch
    import open3d
    assert os.path.abspath(open3d.__file__).startswith(STUBS + "/"), open3d.__file__
    spec = importlib.util.spec_from_file_location("m_common", M_COMMON)
    MC = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(MC)
    from configs.base_config import get_cfg_defaults
    import models.SLAM.gaussian as G
    from models.SLAM.utils.slam_helpers import transformed_params2rendervar, transformed_params2depthplussilhouette

    dev = torch.device("cuda:0")
    torch.cuda.set_device(dev)
    s_unit = float(a.scale)
    tag = f"{a.scene}_s{a.start}"
    seed = seed_of(f"fisherrf|{a.scene}|{a.start}")
    np.random.seed(seed % 2 ** 32)
    random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)

    cfg = get_cfg_defaults()
    cfg.merge_from_file(AM_YAML)
    over = {"SLAM.Dataset.Calibration.fx": FX, "SLAM.Dataset.Calibration.fy": FX, "SLAM.Dataset.Calibration.cx": CX,
            "SLAM.Dataset.Calibration.cy": CY, "SLAM.Dataset.Calibration.width": W_IMG,
            "SLAM.Dataset.Calibration.height": H_IMG, "map_every": 1, "keyframe_every": 1,
            "mapping.num_iters": int(a.map_iters), "workdir": os.path.abspath(a.workdir), "run_name": tag,
            "use_wandb": False, "report_iter_progress": False}
    for k, v in over.items():                         # direct assignment (yacs merge_from_list refuses int -> float)
        node = cfg
        *path, leaf = k.split(".")
        for p in path:
            node = node[p]
        assert leaf in node, k
        node[leaf] = v
    slam = G.GaussianSLAM(cfg)
    assert not slam.sm.on_cluster

    TIM = {}

    def timed(name, fn):
        def w(*x, **y):
            torch.cuda.synchronize()
            t0 = time.perf_counter()
            r = fn(*x, **y)
            torch.cuda.synchronize()
            TIM[name] = TIM.get(name, 0.0) + time.perf_counter() - t0
            return r
        return w
    # timing wrappers only (same call, same arguments, same result); instance attributes shadow the methods
    slam.compute_H_train = timed("h_train_s", slam.compute_H_train)
    slam.visualize_frame = timed("map_png_s", slam.visualize_frame)

    def to_w2c(R, T):
        R = np.asarray(R, np.float64).reshape(3, 3)
        T = np.asarray(T, np.float64).reshape(3)
        Fm = np.diag(F_FLIP)
        w = np.eye(4)
        w[:3, :3] = Fm @ R.T
        w[:3, 3] = Fm @ T / s_unit
        return w

    @torch.no_grad()
    def render_check(w2c_t, color_chw, depth_1hw):
        """Render the current map at the observed pose (no side effects on slam) and compare with the observation."""
        pts = slam.params["means3D"]
        tp = (w2c_t @ torch.cat([pts, torch.ones_like(pts[:, :1])], 1).T).T[:, :3]
        im, _, _ = G.Renderer(raster_settings=slam.cam)(**transformed_params2rendervar(slam.params, tp))
        ds, _, _ = G.Renderer(raster_settings=slam.cam)(**transformed_params2depthplussilhouette(slam.params, slam.first_frame_w2c, tp))
        m = depth_1hw[0] > 0
        if int(m.sum()) == 0:
            return {"valid_px": 0}
        mse = ((im - color_chw) ** 2)[:, m].mean().item()
        return {"valid_px": int(m.sum()), "psnr_valid": float(-10 * np.log10(max(mse, 1e-12))),
                "depth_l1_valid_runtime": float((ds[0] - depth_1hw[0]).abs()[m].mean().item() * s_unit),
                "depth_rel_err_valid": float(((ds[0] - depth_1hw[0]).abs()[m] / depth_1hw[0][m]).mean().item()),
                "silhouette_mean_valid": float(ds[1][m].mean().item()),
                "silhouette_ge_0.5_frac_valid": float((ds[1][m] > 0.5).float().mean().item())}

    def map_stats():
        p = slam.params
        ls = p["log_scales"]
        return {"n_gauss": int(p["means3D"].shape[0]),
                "n_logscale_neg_inf": int(torch.isneginf(ls).any(1).sum().item()),
                "n_nonfinite_means": int((~torch.isfinite(p["means3D"])).any(1).sum().item()),
                "n_nonfinite_other": int(sum((~torch.isfinite(p[k])).sum().item() for k in ("rgb_colors", "logit_opacities", "unnorm_rotations"))),
                "n_keyframes": len(slam.keyframe_list)}

    logf = open(a.log, "a")
    svc_dir = os.path.abspath(a.svc_dir)

    @torch.no_grad()      # the official calling context: tester.py:227 runs the whole loop under @torch.no_grad();
    def step(t, svc_path, score=True):   # get_loss / compute_Hessian re-enable grad themselves (@torch.enable_grad)
        TIM.clear()
        torch.cuda.reset_peak_memory_stats(dev)
        torch.cuda.synchronize()
        t_all = time.perf_counter()
        assert os.path.dirname(os.path.abspath(svc_path)) == svc_dir, svc_path
        rec = MC.load_service_record(svc_path)
        if int(rec["t"]) != t or rec["scene_key"] != (a.scene_key or f"{a.scene}/{a.start}"):
            raise RuntimeError(f"record mismatch: t {int(rec['t'])} vs {t}, key {rec['scene_key']}")
        side = svc_path[:-len(".npz")] + "_rgb.npz"
        z = np.load(side, allow_pickle=False)
        rgb = z["rgb"]
        if int(z["t"]) != t or not np.array_equal(z["R"], rec["R"]) or not np.array_equal(z["T"], rec["T"]):
            raise RuntimeError(f"RGB side file does not match the record at t={t}")
        if rgb.shape != (H_IMG, W_IMG, 3) or rgb.dtype != np.uint8:
            raise RuntimeError(f"RGB side file shape / dtype {rgb.shape} {rgb.dtype}")
        zb, mk = rec["zbuf"].astype(np.float64), rec["mask"].astype(bool)
        valid = mk & (zb > 0) & (zb < SENSOR_RANGE)
        depth = np.where(valid, zb / s_unit, 0.0).astype(np.float32)                    # D11: out of range -> 0
        w2c = to_w2c(rec["R"], rec["T"])
        c2w = np.linalg.inv(w2c)
        pose_err = float(np.abs(c2w[:3, 3] * s_unit - np.asarray(rec["cam_center"], np.float64)).max())
        if pose_err > 1e-4:
            raise RuntimeError(f"pose self-check failed at t={t}: {pose_err}")
        color_t = torch.from_numpy(rgb).to(dev)                                          # (H, W, 3) uint8 as Habitat
        depth_t = torch.from_numpy(depth).to(dev).reshape(1, H_IMG, W_IMG)
        w2c_t = torch.from_numpy(w2c).float().to(dev)
        torch.cuda.synchronize()
        t_load = time.perf_counter() - t_all
        info = {"t": t, "rgb_mean": float(rgb.mean()), "rgb_float_minmax": [float(z["rgb_float_min"]), float(z["rgb_float_max"])],
                "depth_valid_frac": float(valid.mean()), "mask_frac": float(mk.mean()),
                "depth_out_of_range_frac": float((mk & (zb >= SENSOR_RANGE)).mean()), "pose_selfcheck_err": pose_err}
        before = map_stats() if slam.initialize else {"n_gauss": 0, "n_keyframes": 0, "n_logscale_neg_inf": 0, "n_nonfinite_means": 0, "n_nonfinite_other": 0}
        t0 = time.perf_counter()
        has_depth = bool(valid.any())
        info["skipped_no_valid_depth"] = not has_depth                                  # D13: no valid pixel -> no map update
        if not has_depth:
            pass
        elif not slam.initialize:                                                       # D10: init, then track the same frame
            arg = w2c_t if a.init_pose == "tester" else torch.linalg.inv(w2c_t)
            slam.init(color_t, depth_t, arg)
            n0 = int(slam.params["means3D"].shape[0])
            with torch.no_grad():         # the same official get_pointcloud with the record's true w2c = correct placement
                ref = G.get_pointcloud(color_t.permute(2, 0, 1).float() / 255, depth_t, slam.intrinsics, w2c_t,
                                       mask=(depth_t > 10 * slam.cell_size).reshape(-1), compute_mean_sq_dist=False,
                                       downsample=cfg["downsample_pcd"])
                d = (slam.params["means3D"][:n0] - ref[:, :3]).norm(dim=1) * s_unit
                st = map_stats()
                info["init"] = {"pose_arg": a.init_pose, "n_init_gauss": n0, "n_init_logscale_neg_inf": st["n_logscale_neg_inf"],
                                "init_pos_err_vs_true_pose_runtime": {"median": float(d.median().item()), "max": float(d.max().item())}}
        if has_depth:
            slam.track_rgbd(color_t, depth_t, w2c_t)
        torch.cuda.synchronize()
        t_map = time.perf_counter() - t0
        after = map_stats() if slam.initialize else {"n_gauss": 0, "n_keyframes": 0, "n_logscale_neg_inf": 0, "n_nonfinite_means": 0, "n_nonfinite_other": 0}
        t0 = time.perf_counter()
        info["render_check"] = render_check(w2c_t, color_t.permute(2, 0, 1).float() / 255, depth_t) if slam.initialize else {"valid_px": 0}
        torch.cuda.synchronize()
        t_diag = time.perf_counter() - t0
        n = int(len(rec["legal_idx"]))
        pick, fallback, scores, ties, t_score, verify = None, None, [], 0, 0.0, None
        cand_err = 0.0
        if score and n == 0:
            fallback = "hover_empty_legal"
        elif score and not slam.initialize:                                             # D13: no map yet -> seeded random legal
            pick = int(np.random.default_rng(seed_of(f"fisherrf|nomap|{a.scene}|{a.start}|{t}")).integers(n))
            fallback = "no_map_seeded_random"
        elif score:
            c2ws = []
            for j in range(n):
                cw = np.linalg.inv(to_w2c(rec["legal_R"][j], rec["legal_T"][j]))
                cand_err = max(cand_err, float(np.abs(cw[:3, 3] * s_unit - rec["legal_center"][j]).max()))
                c2ws.append(torch.from_numpy(cw).float().to(dev))
            if cand_err > 1e-4:
                raise RuntimeError(f"candidate pose self-check failed at t={t}: {cand_err}")
            torch.cuda.synchronize()
            t0 = time.perf_counter()
            sc, _ = slam.pose_eval(torch.stack(c2ws))                                    # official FisherRF EIG
            torch.cuda.synchronize()
            t_score = time.perf_counter() - t0
            scores = [float(x) for x in sc.cpu().numpy().astype(np.float64)]
            if not all(np.isfinite(scores)):
                raise RuntimeError(f"non-finite FisherRF scores at t={t}: {scores}")
            s_np = np.asarray(scores)
            best = np.flatnonzero(s_np == s_np.max())
            ties = int(len(best) - 1)
            if len(best) == 1:
                pick = int(best[0])
            else:                                                                        # D5
                pick = int(np.random.default_rng(seed_of(f"fisherrf|{a.scene}|{a.start}|{t}")).choice(best))
                fallback = f"tie_{len(best)}"
            if a.verify_every > 0 and t % a.verify_every == 0:
                tv = time.perf_counter()
                Htr = G.GaussianSLAM.compute_H_train(slam)                               # class method (bypasses timer)
                rec_sc, hmin = [], float(Htr.min().item())
                for cw in c2ws:
                    Hc = slam.compute_Hessian(torch.linalg.inv(cw), return_points=True, random_gaussian_params=False)
                    hmin = min(hmin, float(Hc.min().item()))
                    rec_sc.append(float(torch.sum(Hc * torch.reciprocal(Htr + 0.1)).item()))
                rel = [abs(x - y) / max(abs(y), 1e-30) for x, y in zip(rec_sc, scores)]
                torch.cuda.synchronize()
                verify = {"recomputed": rec_sc, "max_rel_diff": max(rel), "H_min": hmin, "H_shape": list(Htr.shape),
                          "H_train_nonzero_rows": int((Htr.abs().sum(1) > 0).sum().item()),
                          "argmax_same": int(np.argmax(rec_sc)) == int(np.argmax(scores)), "verify_s": time.perf_counter() - tv}
        torch.cuda.synchronize()
        t_total = time.perf_counter() - t_all
        h_tr = TIM.get("h_train_s", 0.0)
        timing = {"load_s": t_load, "map_update_s": t_map, "map_png_s": TIM.get("map_png_s", 0.0), "render_check_s": t_diag,
                  "score_s": t_score, "score_h_train_s": h_tr, "score_candidates_s": t_score - h_tr if t_score else 0.0,
                  "core_s": t_load + t_map + t_score,          # official path only (no render check / verify)
                  "total_s": t_total}
        mem = {"peak_alloc_gb": torch.cuda.max_memory_allocated(dev) / 2 ** 30, "peak_reserved_gb": torch.cuda.max_memory_reserved(dev) / 2 ** 30}
        row = dict(info, scored=score, n_legal=n, pick=pick, fallback=fallback, scores=scores, ties=ties,
                   map_before=before, map_after=after, n_new_gauss=after["n_gauss"] - before["n_gauss"], timing=timing,
                   mem=mem, cand_pose_selfcheck_err=cand_err, verify=verify)
        logf.write(json.dumps(row, default=float) + "\n")
        logf.flush()
        return row

    for t in range(a.resume_t):                       # resume: re-map the past records (not bitwise, risk R1)
        step(t, os.path.join(svc_dir, a.svc_name.format(scene=a.scene, start=a.start, t=t)), score=False)
    g = Guard(svc_dir).install()
    hello = {"ready": True, "resume_t": a.resume_t,
             "info": {"method": "FisherRF (official FisherRF-active-mapping core)", "am_commit": AM_COMMIT, "yaml": AM_YAML,
                      "overrides": over, "scale_s": s_unit, "init_pose": a.init_pose, "seed": seed, "lambda": 0.1,
                      "open3d": "stub " + open3d.__file__, "torch": torch.__version__, "gpu": torch.cuda.get_device_name(dev),
                      "cuda_visible_devices": os.environ.get("CUDA_VISIBLE_DEVICES")}}
    proto.write(json.dumps(hello, default=float) + "\n")
    proto.flush()
    tot = 0.0
    for line in sys.stdin:
        q = json.loads(line)
        if q.get("cmd") == "end" or q.get("bye"):          # outdoor m_t6_sim / indoor t23
            break
        t = int(q["t"])
        try:
            r = step(t, q["svc"])
        except Exception as e:                        # reported to the simulator, which raises (episode invalid)
            import traceback
            traceback.print_exc()
            proto.write(json.dumps({"t": t, "error": f"{type(e).__name__}: {e}"}) + "\n")
            proto.flush()
            continue
        tot += r["timing"]["total_s"]
        proto.write(json.dumps({"t": t, "pick": r["pick"], "fallback": r["fallback"], "decision_s": r["timing"]["total_s"],
                                "timing": dict(r["timing"], decision_s=r["timing"]["total_s"]), "scores": r["scores"], "ties": r["ties"],
                                "diag": {"n_gauss": r["map_after"]["n_gauss"], "n_new_gauss": r["n_new_gauss"],
                                         "n_keyframes": r["map_after"]["n_keyframes"], "mem": r["mem"],
                                         "render_check": r["render_check"]},
                                "guard_denied": list(g.denied)}, default=float) + "\n")
        proto.flush()
    g.uninstall()
    proto.write(json.dumps({"bye": True, "decision_s_total": tot, "guard_denied": list(g.denied),
                            "n_paths_opened": len(g.paths), "paths_opened_outside_svc": sorted({p for p in g.paths if not os.path.abspath(p).startswith(svc_dir + "/")})[:50],
                            "peak_mem_gb_process": torch.cuda.max_memory_allocated(dev) / 2 ** 30,
                            "n_gauss_final": int(slam.params["means3D"].shape[0])}, default=float) + "\n")
    proto.flush()


if __name__ == "__main__":
    main()
