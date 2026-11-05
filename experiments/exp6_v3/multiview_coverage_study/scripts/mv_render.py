"""Multi-View Coverage Reasoning Study -- REAL renders of the candidate cameras of the case states (magician env, 1 GPU, read-only):
the simulator's own renderer (MAGICIAN clean 9521d69: textured mesh, ambient 0.85, FoV 60, zfar 750, contrast of the scene) at the
simulator resolution 256 x 456, poses = the pose table entries of the keys listed in results/cases.json (identical to the stored
legal_R / legal_T). Output work_dirs/exp6_v3/multiview_coverage_study/cases/rgb_<tag>_t<t>.npz (rgb per key)."""
import glob
import json
import os
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
MV = os.path.dirname(HERE)
R = os.path.dirname(os.path.dirname(os.path.dirname(MV)))
sys.path.insert(0, os.path.join(R, "experiments", "exp3b_macaronspp_transfer"))
import m_magician_replay as RP  # noqa: E402

np, torch, P = RP.np, RP.torch, RP.P
from macarons.utility.macarons_utils import get_rgb_renderer, adjust_contrast, transpose_channels  # noqa: E402
from pytorch3d.renderer import FoVPerspectiveCameras  # noqa: E402

OUT = os.path.join(R, "work_dirs", "exp6_v3", "multiview_coverage_study", "cases")


def main():
    dev = torch.device("cuda")
    params = RP.load_params(os.path.join(P.MAGICIAN_CLEAN_ROOT, "configs", "macarons", "macarons_default_training_config.json"))
    cases = json.load(open(os.path.join(MV, "results", "cases.json")))["cases"]
    meshes = {}
    for c in cases:
        s = c["scene"]
        if s not in meshes:
            sdir = P.scene_dir(s)
            meshes = {s: RP.load_scene(sorted(glob.glob(os.path.join(sdir, "*.obj")))[0], params.scene_scale_factor, dev, mirror=False, mirrored_axis=None)}
            contrast = float(json.load(open(os.path.join(sdir, "settings.json")))["camera"]["contrast_factor"])
            z = np.load(os.path.join(R, "work_dirs", "exp6_v3", "H", "pose_tables", f"{s}.npz"))
            lut = {tuple(int(x) for x in r): j for j, r in enumerate(z["idx"])}
        out = {}
        for key in c["render_keys"]:
            j = lut[tuple(key)]
            Rm = torch.as_tensor(z["R"][j][None], dtype=torch.float32, device=dev)
            Tm = torch.as_tensor(z["T"][j][None], dtype=torch.float32, device=dev)
            cam = FoVPerspectiveCameras(R=Rm, T=Tm, zfar=params.zfar, device=dev)
            rnd = get_rgb_renderer(256, 456, params.ambient_light_intensity, cam, dev, max_faces_per_bin=600000)
            with torch.no_grad():
                img, _ = rnd(meshes[s], cameras=cam)
                rgb = transpose_channels(adjust_contrast(transpose_channels(img[..., :3], True), contrast), False)[0].clamp(0, 1)
            out["rgb_" + "_".join(map(str, key))] = (rgb.cpu().numpy() * 255).round().astype(np.uint8)
            del rnd, img
            torch.cuda.empty_cache()
        np.savez_compressed(os.path.join(OUT, f"rgb_{c['tag']}_t{c['t']}.npz"), **out)
        print("rendered", c["tag"], c["t"], len(out), flush=True)


if __name__ == "__main__":
    main()
