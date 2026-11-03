"""Table-1 dataset audit, DIAGNOSTIC ONLY (not the frozen M protocol): would sofia_church load if the stray
`torch.tensor()` of MAGICIAN load_scene's multi-texture branch (clean worktree macarons_utils.py:494) were skipped?

In this process only, m_magician_replay.load_scene (the name EpisodeSim calls) is replaced by a copy of MAGICIAN's
load_scene without that one line; no file is modified. Then EpisodeSim(sofia_church, 0) -> observe(0) -> legal().
Run with the magician env on GPU 3:
  env -u LD_PRELOAD -u PYTHONPATH /bsuscratch/jiahuizhang/envs/magician/bin/python sofia_diag.py TMP_ROOT OUT.json
"""
import json
import os
import shutil
import sys
import time
import traceback

REPO = "/bsuscratch/jiahuizhang/projects/occupancy/seen2scene"
T2_SCRIPTS = os.path.join(REPO, "experiments", "exp5_future_view", "task2", "scripts")


def main():
    tmp, out = sys.argv[1], sys.argv[2]
    res = {"scene": "sofia_church", "diagnostic": "load_scene without the stray torch.tensor() (in-process patch)",
           "ok": False}
    t0 = time.time()
    try:
        sys.path.insert(0, T2_SCRIPTS)
        import t2_common as C
        sys.path.insert(0, os.path.join(C.EXP5, "task1", "scripts"))
        import v2_common as LC
        sys.path.insert(1, LC.EXP3B)
        import m_t6_sim as T6S
        import m_t6_common as T6
        RP, torch = T6S.RP, T6S.torch
        import macarons.utility.macarons_utils as MU

        def load_scene_no_stray(mesh_path, scene_scale_factor, device, mirror=False, mirrored_axis=None,
                                texture_wrap="repeat", texture_atlas_size=32):
            if MU.count_map_kd(mesh_path) <= 1:
                return MU.load_scene(mesh_path, scene_scale_factor, device, mirror, mirrored_axis)
            verts, faces, aux = MU.load_obj(mesh_path, device=device, load_textures=True, create_texture_atlas=True,
                                            texture_atlas_size=texture_atlas_size, texture_wrap=texture_wrap)
            verts = verts * scene_scale_factor
            return MU.Meshes(verts=[verts], faces=[faces.verts_idx], textures=MU.TexturesAtlas(atlas=[aux.texture_atlas]))

        RP.load_scene = load_scene_no_stray
        t1 = time.time()
        sim = T6S.EpisodeSim("sofia_church", 0, os.path.join(tmp, "sofia_diag"))
        torch.cuda.synchronize()
        res["init_s"] = time.time() - t1
        gt = sim.gt_scene
        cells = [c for c in gt.cells if len(gt.cells[c].cell_pts) > 0]
        res["n_gt_points"] = int(sum(len(gt.cells[c].cell_pts) for c in cells))
        res["eps"], res["vis"] = float(sim.eps), float(sim.vis)
        res["texture_type"] = type(sim.mesh.textures).__name__
        res["mesh_n_faces"] = int(sim.mesh.faces_list()[0].shape[0])
        res["key_at_init"] = sim.key()
        res["init_key_equals_settings_start0"] = sim.key() == T6.key_of(sim.settings.camera.start_positions[0].cpu().numpy())
        ob = sim.observe(0)
        res["cov0"], res["n_samples0"] = float(ob["cov"]), int(len(ob["sample_flat"]))
        legal, flags = sim.legal()
        res["n_legal0"], res["n_collide0"] = len(legal), int(sum(f["collides"] for f in flags))
        sc = sim.settings.camera
        res["starts"] = []
        for k in range(5):
            key = T6.key_of(sc.start_positions[k].cpu().numpy())
            p = sim.camera.pose_space[key].double().cpu().numpy().tolist()
            res["starts"].append({"k": k, "key": key, "X": p[:3], "elev": p[3], "azim": p[4]})
        res["peak_mem_gb"] = torch.cuda.max_memory_allocated() / 2 ** 30
        res["ok"] = True
    except BaseException as e:  # noqa: BLE001
        res["error"] = f"{type(e).__name__}: {e}"
        res["traceback"] = traceback.format_exc()
    res["total_s"] = time.time() - t0
    shutil.rmtree(os.path.join(tmp, "sofia_diag"), ignore_errors=True)
    json.dump(res, open(out, "w"), indent=1, default=float)
    print(json.dumps({k: v for k, v in res.items() if k not in ("traceback", "starts")}, default=float))


if __name__ == "__main__":
    main()
