"""Geometry / ROI audit A1 (magician env, 1 GPU, read-only): per scene, the simulator mesh and the OFFICIAL coverage target, built
with the frozen simulator code path (exp3b m_t6_sim.EpisodeSim.__init__: MAGICIAN clean 9521d69 load_params / Settings / load_scene /
setup_test_scene with the frozen "gt" seed of start 0).

Per scene -> work_dirs/cvpr_results/geometry_roi_audit/a1/<scene>.npz (GT target points, mesh sample) and results/a1_gt_target.json:
  mesh      : OBJ path, sha256, n verts / faces, vertex bounds (runtime = OBJ x 10, y-up), total area
  ROI box   : settings.json scene x_min / x_max x 10 (the box of gt_scene / covered_scene), coarse cell grid, visibility_ratio
  target    : faces with ALL 3 vertices inside the box (get_scene_gt_surface) -> area; faces straddling the box; faces fully outside;
              area-weighted share of the whole mesh surface inside the box (2 M area-weighted samples, by point);
              GT target points after fill_cells (count, bounds, per-coarse-cell counts, cells at the 1000-point capacity)
  proxy     : 'ground-like' surface = upward faces (n_y > 0.9) whose centroid is in the lowest 10 % of the box height (NO semantic
              labels exist; geometric proxy only) inside / outside the box
  cameras   : camera box (settings camera x_min / x_max x 10), start positions
Nothing is written to any existing experiment directory."""
import glob
import hashlib
import json
import os
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
AUD = os.path.dirname(HERE)
REPO = os.path.dirname(os.path.dirname(os.path.dirname(AUD)))
sys.path.insert(0, os.path.join(REPO, "experiments", "exp3b_macaronspp_transfer"))
import m_magician_replay as RP  # noqa: E402
import m_common as MC  # noqa: E402

np, torch, MP, P = RP.np, RP.torch, RP.MP, RP.P
OUT = os.path.join(REPO, "work_dirs", "cvpr_results", "geometry_roi_audit", "a1")
ALL14 = ["bannerman", "dunnottar", "fushimi", "liberty", "pisa", "redeemer", "barts", "colosseum", "eiffel", "pantheon", "sestino_museum",
         "alhambra", "bridge", "neuschwanstein"]


def sha256(p):
    h = hashlib.sha256()
    with open(p, "rb") as f:
        for b in iter(lambda: f.read(1 << 22), b""):
            h.update(b)
    return h.hexdigest()


def face_stats(verts, faces, lo, hi):
    v = verts.double()
    inside_v = ((v >= lo) & (v <= hi)).all(-1)                      # Scene.get_pts_in_bounding_box (>=, <=)
    fi = inside_v[faces]                                            # [F, 3]
    n_in = fi.sum(-1)
    a, b, c = v[faces[:, 0]], v[faces[:, 1]], v[faces[:, 2]]
    cr = torch.linalg.cross(b - a, c - a)
    area = 0.5 * cr.norm(dim=-1)
    nrm = cr / cr.norm(dim=-1, keepdim=True).clamp(min=1e-30)
    cen = (a + b + c) / 3.0
    return n_in, area, nrm, cen


def main():
    os.makedirs(OUT, exist_ok=True)
    dev = torch.device("cuda")
    scenes = sys.argv[1:] or ALL14
    params = RP.load_params(os.path.join(P.MAGICIAN_CLEAN_ROOT, "configs", "macarons", "macarons_default_training_config.json"))
    res = {}
    for s in scenes:
        sdir = P.scene_dir(s)
        sj = json.load(open(os.path.join(sdir, "settings.json")))
        settings = RP.Settings(sj, dev, params.scene_scale_factor)
        objs = sorted(glob.glob(os.path.join(sdir, "*.obj")))
        assert len(objs) == 1, objs
        mesh = RP.load_scene(objs[0], params.scene_scale_factor, dev, mirror=False, mirrored_axis=None)
        verts, faces = mesh.verts_list()[0], mesh.faces_list()[0]
        lo, hi = settings.scene.x_min.double().view(3), settings.scene.x_max.double().view(3)
        n_in, area, nrm, cen = face_stats(verts, faces, lo, hi)
        A = float(area.sum())
        full, strad, outf = n_in == 3, (n_in > 0) & (n_in < 3), n_in == 0
        # area-weighted share of the whole mesh surface that lies inside the box (by point; independent of the face rule)
        g = torch.Generator(device="cpu").manual_seed(0)
        fidx = torch.multinomial((area / area.sum()).cpu(), 2_000_000, replacement=True, generator=g).to(dev)
        u = torch.rand(2_000_000, 2, generator=g).to(dev).double()
        su = u[:, 0].sqrt()
        w0, w1, w2 = 1 - su, su * (1 - u[:, 1]), su * u[:, 1]
        vd = verts.double()
        pts = w0[:, None] * vd[faces[fidx, 0]] + w1[:, None] * vd[faces[fidx, 1]] + w2[:, None] * vd[faces[fidx, 2]]
        p_in = ((pts >= lo) & (pts <= hi)).all(-1)
        p_full = full[fidx]
        # ground-like proxy
        hbox = float(hi[1] - lo[1])
        ground = (nrm[:, 1] > 0.9) & (cen[:, 1] < lo[1] + 0.10 * hbox)
        # official GT target (frozen seed of start 0)
        torch.manual_seed(MC.derived_seed(s, 0, -1, "gt"))
        gt, covered, surface, proxy = MP.setup_test_scene(params, mesh, settings, False, dev, mirrored_axis=None, test_resolution=RP.TEST_RESOLUTION)
        cells = {k: int(len(c.cell_pts)) for k, c in gt.cells.items()}
        gpts = torch.vstack([c.cell_pts for c in gt.cells.values()]).double()
        del covered, surface, proxy
        sel = torch.randperm(len(pts), generator=g)[:300_000].to(dev)
        np.savez_compressed(os.path.join(OUT, f"{s}.npz"), gt_points=gpts.cpu().numpy().astype(np.float32), box_lo=lo.cpu().numpy(), box_hi=hi.cpu().numpy(),
                            mesh_sample=pts[sel].cpu().numpy().astype(np.float32), mesh_sample_in_box=p_in[sel].cpu().numpy(),
                            mesh_sample_target_face=p_full[sel].cpu().numpy(), mesh_sample_ground=ground[fidx][sel].cpu().numpy())
        cam = sj["camera"]
        res[s] = {
            "mesh": {"obj": objs[0], "sha256": sha256(objs[0]), "n_verts": int(len(verts)), "n_faces": int(len(faces)), "scale": float(params.scene_scale_factor),
                     "bounds_lo": verts.min(0)[0].tolist(), "bounds_hi": verts.max(0)[0].tolist(), "area": A},
            "roi_box": {"lo": lo.tolist(), "hi": hi.tolist(), "extent": (hi - lo).tolist(), "coarse_cells": [int(sj["scene"]["grid_l"]), int(sj["scene"]["grid_w"]), int(sj["scene"]["grid_h"])],
                        "visibility_ratio": float(settings.scene.visibility_ratio), "cell_resolution_rt": RP.TEST_RESOLUTION * params.scene_scale_factor,
                        "eps_rt": 2 * RP.TEST_RESOLUTION * params.scene_scale_factor, "cell_capacity": int(params.surface_cell_capacity),
                        "n_gt_surface_points_sampled": int(params.n_gt_surface_points)},
            "faces": {"n_all3_inside": int(full.sum()), "n_straddling": int(strad.sum()), "n_outside": int(outf.sum()),
                      "area_all3_inside": float(area[full].sum()), "area_straddling": float(area[strad].sum()), "area_outside": float(area[outf].sum())},
            "area_share": {"target_faces_of_mesh": float(area[full].sum() / A), "straddling_of_mesh": float(area[strad].sum() / A), "outside_of_mesh": float(area[outf].sum() / A),
                           "mesh_surface_inside_box_by_point": float(p_in.double().mean()),
                           "inside_box_but_not_target_by_point": float((p_in & ~p_full).double().mean()),
                           "target_face_points_inside_box": float((p_in & p_full).double().sum() / p_full.double().sum().clamp(min=1))},
            "ground_proxy": {"definition": "faces with n_y > 0.9 and centroid in the lowest 10 % of the box height (no semantic labels)",
                             "share_of_target_area": float(area[full & ground].sum() / area[full].sum().clamp(min=1e-30)),
                             "share_of_non_target_area": float(area[~full & ground].sum() / area[~full].sum().clamp(min=1e-30)),
                             "non_target_area_below_box_top_share": float(area[~full & (cen[:, 1] <= hi[1])].sum() / area[~full].sum().clamp(min=1e-30))},
            "gt_target": {"n_points": int(len(gpts)), "bounds_lo": gpts.min(0)[0].tolist(), "bounds_hi": gpts.max(0)[0].tolist(),
                          "all_inside_box": bool(((gpts >= lo) & (gpts <= hi)).all()), "n_cells": len(cells), "n_cells_nonempty": sum(v > 0 for v in cells.values()),
                          "n_cells_at_capacity": sum(v >= int(params.surface_cell_capacity) for v in cells.values()), "cell_counts": cells},
            "camera": {"box_lo": [10.0 * x for x in cam["x_min"]], "box_hi": [10.0 * x for x in cam["x_max"]],
                       "grid": [cam["pose_l"], cam["pose_w"], cam["pose_h"], cam.get("pose_n_theta", cam.get("pose_n_elev")), cam["pose_n_azim"]],
                       "start_positions": cam["start_positions"]},
        }
        r = res[s]
        print(f"{s:15s} mesh {r['mesh']['n_faces']} faces area {A:.0f} | box extent {[round(x, 1) for x in r['roi_box']['extent']]} | target-face area {r['area_share']['target_faces_of_mesh']:.3f} "
              f"straddle {r['area_share']['straddling_of_mesh']:.4f} outside {r['area_share']['outside_of_mesh']:.3f} | inside-box by point {r['area_share']['mesh_surface_inside_box_by_point']:.3f} "
              f"| GT pts {r['gt_target']['n_points']} cells at cap {r['gt_target']['n_cells_at_capacity']}/{r['gt_target']['n_cells_nonempty']} all in box {r['gt_target']['all_inside_box']}", flush=True)
        del mesh, gt, pts
        torch.cuda.empty_cache()
    p = os.path.join(AUD, "results", "a1_gt_target.json")
    old = json.load(open(p)) if os.path.exists(p) else {}
    old.update(res)
    json.dump(old, open(p, "w"), indent=1)


if __name__ == "__main__":
    main()
