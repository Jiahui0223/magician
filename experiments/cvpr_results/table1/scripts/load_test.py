"""Table-1 dataset audit: light load test of the frozen M-protocol simulator (m_t6_sim.EpisodeSim) on every scene.

Run with the magician env on ONE GPU (the caller sets CUDA_VISIBLE_DEVICES):
  env -u LD_PRELOAD -u PYTHONPATH /bsuscratch/jiahuizhang/envs/magician/bin/python load_test.py worker SCENE TMP_ROOT OUT.json
  env -u LD_PRELOAD -u PYTHONPATH /bsuscratch/jiahuizhang/envs/magician/bin/python load_test.py all TMP_ROOT OUT.json
Worker (one fresh process per scene; nothing of EXP3b / EXP5 / MAGICIAN is modified, only imported exactly as
task2/scripts/t25_out_cov.py imports it):
  EpisodeSim(scene, 0, TMP_ROOT/<scene>)  -> observe(0) once -> legal() once
  records init / observe times, number of GT points (as t25_out_cov: points of the non-empty GT cells), eps, vis,
  coverage at t = 0, the t = 0 pose key and the number of legal actions; the 5 official starts
  settings.camera.start_positions[k] mapped through the simulator camera's pose space (pose_space[key] = [X, elev, azim])
  and compared with X_cam_history[0] / V_cam_history[0] of magician_lmdb (read from lmdb_summary.json);
  on failure the exception and traceback.
`all` runs the 15 workers sequentially and deletes TMP_ROOT/<scene> after each worker.
"""
import json
import os
import shutil
import subprocess
import sys
import time
import traceback

HERE = os.path.dirname(os.path.abspath(__file__))
REPO = "/bsuscratch/jiahuizhang/projects/occupancy/seen2scene"
T2_SCRIPTS = os.path.join(REPO, "experiments", "exp5_future_view", "task2", "scripts")
LMDB_SUMMARY = os.path.join(os.path.dirname(HERE), "lmdb_summary.json")
SCENES = ["alhambra", "bannerman", "barts", "bridge", "colosseum", "dunnottar", "eiffel", "fushimi", "liberty",
          "neuschwanstein", "pantheon", "pisa", "redeemer", "sestino_museum", "sofia_church"]


def worker(scene, tmp_root, out):
    res = {"scene": scene, "ok": False, "stage": "import", "host": os.uname()[1],
           "cuda_visible_devices": os.environ.get("CUDA_VISIBLE_DEVICES")}
    t0 = time.time()
    try:
        sys.path.insert(0, T2_SCRIPTS)
        import t2_common as C                                       # same import chain as t25_out_cov.py
        sys.path.insert(0, os.path.join(C.EXP5, "task1", "scripts"))
        import v2_common as LC
        sys.path.insert(1, LC.EXP3B)
        import m_t6_sim as T6S
        import m_t6_common as T6
        np, torch = T6S.np, T6S.torch
        res["import_s"] = time.time() - t0
        res["stage"] = "EpisodeSim"
        t1 = time.time()
        sim = T6S.EpisodeSim(scene, 0, os.path.join(tmp_root, scene))
        torch.cuda.synchronize()
        res["init_s"] = time.time() - t1
        gt = sim.gt_scene
        cells = [c for c in gt.cells if len(gt.cells[c].cell_pts) > 0]
        res["n_gt_points"] = int(sum(len(gt.cells[c].cell_pts) for c in cells))
        res["n_gt_cells_nonempty"] = len(cells)
        res["eps"] = float(sim.eps)
        res["vis"] = float(sim.vis)
        res["mesh_n_verts"] = int(sim.mesh.verts_list()[0].shape[0])
        res["mesh_n_faces"] = int(sim.mesh.faces_list()[0].shape[0])
        res["texture_type"] = type(sim.mesh.textures).__name__
        res["pose_space_size"] = len(sim.camera.pose_space)
        sc = sim.settings.camera
        res["camera_grid"] = [int(sc.pose_l), int(sc.pose_w), int(sc.pose_h), int(sc.pose_n_elev), int(sc.pose_n_azim)]
        res["start_positions"] = sc.start_positions.cpu().numpy().astype(int).tolist()
        res["key_at_init"] = sim.key()
        res["init_key_equals_settings_start0"] = sim.key() == T6.key_of(sc.start_positions[0].cpu().numpy())
        res["stage"] = "observe"
        t2 = time.time()
        ob = sim.observe(0)
        torch.cuda.synchronize()
        res["observe0_s"] = time.time() - t2
        res["cov0"] = float(ob["cov"])
        res["n_samples0"] = int(len(ob["sample_flat"]))
        res["stage"] = "legal"
        t3 = time.time()
        legal, flags = sim.legal()
        res["legal0_s"] = time.time() - t3
        res["n_legal0"] = len(legal)
        res["n_valid_neighbours0"] = len(flags)
        res["n_collide0"] = int(sum(1 for f in flags if f["collides"]))
        res["stage"] = "starts"
        lm = json.load(open(LMDB_SUMMARY))["rows"] if os.path.exists(LMDB_SUMMARY) else {}
        starts = []
        for k in range(len(res["start_positions"])):
            key = T6.key_of(res["start_positions"][k])
            pose = sim.camera.pose_space[key].double().cpu().numpy()
            row = {"k": k, "key": key, "X": pose[:3].tolist(), "elev": float(pose[3]), "azim": float(pose[4])}
            L = lm.get(f"{scene}/{k}")
            if L is not None:
                X0, V0 = np.asarray(L["X0"]), np.asarray(L["V0"])
                row["lmdb_X0"], row["lmdb_V0"] = X0.tolist(), V0.tolist()
                row["max_abs_dX"] = float(np.abs(pose[:3] - X0).max())
                row["abs_d_elev"] = float(abs(pose[3] - V0[0]))
                row["abs_d_azim_mod360"] = float(abs(((pose[4] - V0[1]) + 180.0) % 360.0 - 180.0))
                row["matches_lmdb"] = bool(row["max_abs_dX"] < 1e-4 and row["abs_d_elev"] < 1e-3 and
                                           row["abs_d_azim_mod360"] < 1e-3)
            starts.append(row)
        res["starts"] = starts
        res["peak_mem_gb"] = torch.cuda.max_memory_allocated() / 2 ** 30
        res["ok"] = True
        res["stage"] = "done"
    except BaseException as e:  # noqa: BLE001
        res["error"] = f"{type(e).__name__}: {e}"
        res["traceback"] = traceback.format_exc()
    res["total_s"] = time.time() - t0
    json.dump(res, open(out, "w"), indent=1, default=float)
    print(json.dumps({k: res.get(k) for k in ("scene", "ok", "stage", "init_s", "observe0_s", "n_gt_points", "eps",
                                              "vis", "n_legal0", "cov0", "error")}, default=float), flush=True)


def run_all(tmp_root, out):
    os.makedirs(tmp_root, exist_ok=True)
    env = {k: v for k, v in os.environ.items() if k not in ("LD_PRELOAD", "PYTHONPATH")}
    results = {}
    for s in SCENES:
        o = os.path.join(tmp_root, f"_res_{s}.json")
        t = time.time()
        p = subprocess.run([sys.executable, os.path.abspath(__file__), "worker", s, tmp_root, o], env=env,
                           capture_output=True, text=True, timeout=3600)
        r = json.load(open(o)) if os.path.exists(o) else {"scene": s, "ok": False, "error": "no result file"}
        r["exit_code"] = p.returncode
        r["wall_s"] = time.time() - t
        r["stderr_tail"] = p.stderr[-3000:]
        results[s] = r
        shutil.rmtree(os.path.join(tmp_root, s), ignore_errors=True)
        print(p.stdout.strip(), flush=True)
    json.dump({"created": time.ctime(), "python": sys.executable, "results": results}, open(out, "w"), indent=1,
              default=float)


if __name__ == "__main__":
    if sys.argv[1] == "worker":
        worker(sys.argv[2], sys.argv[3], sys.argv[4])
    else:
        run_all(sys.argv[2], sys.argv[3])
