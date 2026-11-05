"""Geometry / ROI audit K1 (magician env, 1 GPU, read-only): REAL renders of colosseum with the simulator's own renderer
(MAGICIAN clean 9521d69 load_scene + get_rgb_renderer: textured mesh, ambient light 0.85, FoV 60, zfar 750, contrast 1.2 -- the
settings of the closed-loop simulator), at 2x the simulator resolution, and the per-pixel geometry needed to label them.

Renders:  bird's-eye top-down (whole mesh), top-down zoom (camera action space), oblique bird's-eye;
          the candidate camera views listed in results/colosseum_real_views.json (poses = legal_R / legal_T of the stored service
          records of the Table 1 D1 episodes, i.e. exactly the poses the planner scored).
Per pixel (from the rasteriser fragments: face index + barycentric coordinates -> exact world point of the visible surface):
  class 0 no surface | 1 visible surface INSIDE the ROI box (coverage target) | 2 surface OUTSIDE the ROI that hides the ROI (the
  ray would enter the ROI box behind it) | 3 surface outside the ROI, ray does not enter the ROI behind it
Also: ROI box / camera-box edges and camera positions projected into every image. Output work_dirs/.../geometry_roi_audit/k/render_*.npz."""
import glob
import json
import os
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
AUD = os.path.dirname(HERE)
REPO = os.path.dirname(os.path.dirname(os.path.dirname(AUD)))
sys.path.insert(0, os.path.join(REPO, "experiments", "exp3b_macaronspp_transfer"))
import m_magician_replay as RP  # noqa: E402

np, torch, P = RP.np, RP.torch, RP.P
from macarons.utility.macarons_utils import get_rgb_renderer, adjust_contrast, transpose_channels  # noqa: E402
from pytorch3d.renderer import FoVPerspectiveCameras, look_at_view_transform  # noqa: E402

OUT = os.path.join(REPO, "work_dirs", "cvpr_results", "geometry_roi_audit", "k")
EP = os.path.join(REPO, "work_dirs", "cvpr_results", "table1", "episodes")
POSE = os.path.join(REPO, "work_dirs", "exp6_v3", "H", "pose_tables")
SCENE = "colosseum"
EDGES = [(0, 1), (0, 2), (0, 4), (1, 3), (1, 5), (2, 3), (2, 6), (3, 7), (4, 5), (4, 6), (5, 7), (6, 7)]


def corners(lo, hi):
    return np.array([[x, y, z] for x in (lo[0], hi[0]) for y in (lo[1], hi[1]) for z in (lo[2], hi[2])], np.float64)


def main():
    os.makedirs(OUT, exist_ok=True)
    dev = torch.device("cuda")
    params = RP.load_params(os.path.join(P.MAGICIAN_CLEAN_ROOT, "configs", "macarons", "macarons_default_training_config.json"))
    sdir = P.scene_dir(SCENE)
    sj = json.load(open(os.path.join(sdir, "settings.json")))
    obj = sorted(glob.glob(os.path.join(sdir, "*.obj")))[0]
    mesh = RP.load_scene(obj, params.scene_scale_factor, dev, mirror=False, mirrored_axis=None)
    verts, faces = mesh.verts_packed(), mesh.faces_packed()
    lo = 10.0 * np.asarray(sj["scene"]["x_min"], np.float64)
    hi = 10.0 * np.asarray(sj["scene"]["x_max"], np.float64)
    clo = 10.0 * np.asarray(sj["camera"]["x_min"], np.float64)
    chi = 10.0 * np.asarray(sj["camera"]["x_max"], np.float64)
    contrast = float(sj["camera"]["contrast_factor"])
    C_all = np.unique(np.load(os.path.join(POSE, f"{SCENE}.npz"))["C"], axis=0)
    lo_t, hi_t = torch.as_tensor(lo, device=dev), torch.as_tensor(hi, device=dev)

    def project(cam, pts, H, W):
        p = torch.as_tensor(np.asarray(pts, np.float64), dtype=torch.float32, device=dev)
        zv = cam.get_world_to_view_transform().transform_points(p)[:, 2]
        s = cam.transform_points_screen(p, image_size=((H, W),))[:, :2]
        return s.cpu().numpy(), (zv > 0.05).cpu().numpy()

    def lines(cam, cs, H, W, n=160):
        out = []
        for a, b in EDGES:
            pts = cs[a][None] + (cs[b] - cs[a])[None] * np.linspace(0, 1, n)[:, None]
            s, ok = project(cam, pts, H, W)
            s[~ok] = np.nan
            out.append(s)
        return np.stack(out)

    def render(name, R, T, H, W, extra=None):
        cam = FoVPerspectiveCameras(R=R, T=T, zfar=params.zfar, device=dev)
        rnd = get_rgb_renderer(H, W, params.ambient_light_intensity, cam, dev, max_faces_per_bin=600000)
        with torch.no_grad():
            images, frag = rnd(mesh, cameras=cam)
            rgb = transpose_channels(adjust_contrast(transpose_channels(images[..., :3], True), contrast), False)[0].clamp(0, 1)
            p2f = frag.pix_to_face[0, ..., 0]
            bary = frag.bary_coords[0, ..., 0, :].double()
            hit = p2f >= 0
            tri = verts[faces[p2f.clamp(min=0)]].double()                    # [H, W, 3, 3]
            world = (bary[..., None] * tri).sum(-2)
            c = cam.get_camera_center()[0].double()
            d = world - c
            L = d.norm(dim=-1)
            d = d / L.clamp(min=1e-9)[..., None]
            inside = hit & ((world >= lo_t) & (world <= hi_t)).all(-1)
            safe = torch.where(d.abs() < 1e-12, torch.full_like(d, 1e-12), d)
            ta, tb = (lo_t - c) / safe, (hi_t - c) / safe
            t_in = torch.minimum(ta, tb).amax(-1).clamp(min=0.0)
            t_out = torch.maximum(ta, tb).amin(-1)
            enters = t_out > t_in                                             # the ray meets the ROI box somewhere in front of the camera
            hides = hit & ~inside & enters & (t_in > L)                       # an outside surface in front of the ROI entry
            cls = torch.zeros_like(p2f, dtype=torch.uint8)
            cls[inside] = 1
            cls[hides] = 2
            cls[hit & ~inside & ~hides] = 3
            # rays without a visible surface that cross the ROI are also 'through the ROI'; rays over the ROI: all pixels whose ray meets the box
            ndir = None
        cam_in = bool(((c >= lo_t) & (c <= hi_t)).all())
        st = {"pixels": int(H * W), "no_surface": float((cls == 0).double().mean()), "roi_target_visible": float((cls == 1).double().mean()),
              "outside_hides_roi": float((cls == 2).double().mean()), "outside_other": float((cls == 3).double().mean()), "camera_inside_roi": cam_in,
              "of_rays_meeting_the_roi": {"see_target": float((cls == 1).double().sum() / max(1.0, float((enters & hit).double().sum()))),
                                          "blocked_by_outside_surface": float((cls == 2).double().sum() / max(1.0, float((enters & hit).double().sum()))),
                                          "pass_through_and_hit_outside": float(((cls == 3) & enters).double().sum() / max(1.0, float((enters & hit).double().sum())))}}
        box = lines(cam, corners(lo, hi), H, W)
        cbox = lines(cam, corners(clo, chi), H, W)
        cams, cok = project(cam, C_all, H, W)
        foot = np.unique(np.stack([C_all[:, 0], np.zeros(len(C_all)), C_all[:, 2]], 1), axis=0)      # horizontal footprint of the action space (y = 0)
        fpt, fok = project(cam, foot, H, W)
        vc = np.asarray(view_centres, np.float64).reshape(-1, 3)
        vpt, vok = project(cam, np.concatenate([vc, np.stack([vc[:, 0], np.zeros(len(vc)), vc[:, 2]], 1)]), H, W) if len(vc) else (np.zeros((0, 2)), np.zeros(0, bool))
        np.savez_compressed(os.path.join(OUT, f"render_{name}.npz"), rgb=(rgb.cpu().numpy() * 255).round().astype(np.uint8), cls=cls.cpu().numpy(),
                            depth=torch.where(hit, L, torch.zeros_like(L)).float().cpu().numpy(), center=c.cpu().numpy(), R=R.cpu().numpy(), T=T.cpu().numpy(),
                            roi_lines=box, cam_box_lines=cbox, cam_pts=cams, cam_pts_ok=cok, cam_pts_in_roi=((C_all >= lo) & (C_all <= hi)).all(-1),
                            foot_pts=fpt, foot_ok=fok, foot_in_roi_footprint=((foot[:, [0, 2]] >= lo[[0, 2]]) & (foot[:, [0, 2]] <= hi[[0, 2]])).all(-1),
                            n_heights_in_roi=int(((np.unique(C_all[:, 1]) >= lo[1]) & (np.unique(C_all[:, 1]) <= hi[1])).sum()), n_heights=int(len(np.unique(C_all[:, 1]))),
                            view_pts=vpt, view_ok=vok,
                            stats=json.dumps(st), extra=json.dumps(extra or {}))
        print(name, "centre", np.round(c.cpu().numpy(), 1), {k: (round(v, 3) if isinstance(v, float) else v) for k, v in st.items() if not isinstance(v, dict)},
              {k: round(v, 3) for k, v in st["of_rays_meeting_the_roi"].items()}, flush=True)
        del rnd, images, frag
        torch.cuda.empty_cache()

    views = json.load(open(os.path.join(AUD, "results", "colosseum_real_views.json")))["views"]
    view_centres = []
    for w in views:
        r = json.load(open(os.path.join(EP, "V3-D1-H3-NORISK", f"{SCENE}_s{w['start']}.json")))
        e = json.load(open(os.path.join(REPO, r["episode_file"])))
        root = os.path.dirname(os.path.dirname(os.path.join(REPO, r["episode_file"])))
        z = np.load(os.path.join(root, "svc", e["tag"], f"svc_{SCENE}_s{w['start']}_t{w['t']}.npz"))
        view_centres.append(np.asarray(z["legal_center"])[w["j"]])
    # ---- bird's-eye views (2 x resolution of the report figures; same renderer settings)
    mlo, mhi = verts.min(0)[0].cpu().numpy(), verts.max(0)[0].cpu().numpy()
    mc = 0.5 * (mlo + mhi)
    rc = 0.5 * (lo + hi)
    H2, W2 = 900, 1600
    for name, eye, at, up in (("top_whole_mesh", (mc[0], 205.0, mc[2]), (mc[0], 0.0, mc[2]), (0.0, 0.0, -1.0)),
                              ("top_camera_region", (rc[0], 150.0, rc[2]), (rc[0], 0.0, rc[2]), (0.0, 0.0, -1.0)),
                              ("oblique_from_north", (rc[0], 70.0, rc[2] + 140.0), (rc[0], 4.0, rc[2]), (0.0, 1.0, 0.0)),
                              ("oblique_from_east", (rc[0] + 150.0, 70.0, rc[2]), (rc[0], 4.0, rc[2]), (0.0, 1.0, 0.0))):
        R, T = look_at_view_transform(eye=(eye,), at=(at,), up=(up,), device=dev)
        render(name, R, T, H2, W2, {"eye": list(map(float, eye)), "at": list(map(float, at)), "mesh_lo": mlo.tolist(), "mesh_hi": mhi.tolist(), "roi_lo": lo.tolist(), "roi_hi": hi.tolist(),
                                    "cam_box_lo": clo.tolist(), "cam_box_hi": chi.tolist()})
    # ---- the candidate views the planner scored
    for w in views:
        k, t, j = w["start"], w["t"], w["j"]
        r = json.load(open(os.path.join(EP, "V3-D1-H3-NORISK", f"{SCENE}_s{k}.json")))
        e = json.load(open(os.path.join(REPO, r["episode_file"])))
        root = os.path.dirname(os.path.dirname(os.path.join(REPO, r["episode_file"])))
        z = np.load(os.path.join(root, "svc", e["tag"], f"svc_{SCENE}_s{k}_t{t}.npz"))
        R = torch.as_tensor(z["legal_R"][j][None], dtype=torch.float32, device=dev)
        T = torch.as_tensor(z["legal_T"][j][None], dtype=torch.float32, device=dev)
        render(f"view_s{k}_t{t}_j{j}", R, T, 512, 912, {**w, "legal_idx": np.asarray(z["legal_idx"]).reshape(-1, 5)[j].tolist(),
                                                         "legal_center": np.asarray(z["legal_center"])[j].tolist()})


if __name__ == "__main__":
    main()
