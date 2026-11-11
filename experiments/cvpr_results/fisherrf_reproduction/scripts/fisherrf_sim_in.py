"""FisherRF reproduction, Stage F1 -- MP3D indoor closed loop, simulator side (magician env; GT side).

= task2/scripts/t23_in_sim.py main() run UNCHANGED (T0.12 grid simulator through t12_sim / mp3d_sim: grid, start snapping,
official grid neighbours, GT single-segment legality, frozen depth sensor, 10 % sampling seeds, C_obs evaluator,
service-record format), with these in-memory additions of this wrapper only (no frozen file is edited):
  1. its --group option also accepts "FisherRF"; t23.child is replaced by the fisherrf_planner.py child (fisherrf env);
  2. textured RGB (decision D1 (a): NEW simulator-side rendering, the MP3D sensor renders depth only): TexturedRGB loads
     the scan's matterport_mesh OBJ with its own textures (PyTorch3D load_obj geometry, which must equal the sensor's cached
     mesh exactly, + per-face TexturesAtlas built with PyTorch3D's make_material_atlas, the load_obj(create_texture_atlas)
     routine, one texture at a time), and renders with the SAME FoVPerspectiveCameras (R, T, zfar 750) and the SAME
     RasterizationSettings object as mp3d_sim.Sensor; shading = the MAGICIAN outdoor renderer analogue (SoftPhongShader +
     AmbientLights 0.85, no contrast change). t12_sim.service_record is wrapped to write svc_t<t>_rgb.npz
     {t, rgb uint8, R, T, rgb_float_min/max, zbuf check vs the sensor frame} next to each record (D9).
Usage: python fisherrf_sim_in.py --scan S --level L --start K --moves N --out DIR [--init-pose tester|docstring]
       [--verify-every K] [--atlas 4]
"""
import argparse
import glob
import json
import os
import subprocess
import sys
import time
import types

HERE = os.path.dirname(os.path.abspath(__file__))
X = os.path.dirname(os.path.dirname(os.path.dirname(HERE)))           # experiments/
sys.path.insert(0, os.path.join(X, "exp5_future_view", "task2", "scripts"))
import t23_in_sim as T23  # noqa: E402

S, C, T12 = T23.S, T23.C, T23.T12
np = T23.np
FISHERRF_PYTHON = "/bsuscratch/jiahuizhang/envs/fisherrf/bin/python"
GROUP = "FisherRF"
CTX = {"zcheck": []}


class TexturedRGB:
    def __init__(self, scan, dev, atlas_size=4, ambient=0.85):
        import torch
        from PIL import Image
        from pytorch3d.io import load_obj
        from pytorch3d.io.mtl_io import make_material_atlas
        from pytorch3d.renderer import AmbientLights, MeshRasterizer, MeshRendererWithFragments, SoftPhongShader, TexturesAtlas
        from pytorch3d.structures import Meshes
        t0 = time.time()
        self.torch, self.dev = torch, dev
        obj = glob.glob(os.path.join(S.MP, scan, "matterport_mesh", "*", "*.obj"))
        assert len(obj) == 1, obj
        verts, faces, aux = load_obj(obj[0], load_textures=False)
        names = []                                     # usemtl names in first-appearance order = load_obj material_names
        with open(obj[0]) as f:
            for line in f:
                if line.startswith("usemtl"):
                    n = line.split()[1]
                    if n not in names:
                        names.append(n)
        v_cache, f_cache = S.mesh_zup(scan)            # the sensor's geometry (trimesh, process=False; splits UV seams)
        fi = faces.verts_idx.numpy()
        # The mesh we render IS the sensor's (v_cache, f_cache); load_obj only supplies per-face UVs / materials. trimesh
        # splits UV seams and regroups faces by material, so sensor face i is matched to its OBJ face by centroid
        # (KD-tree) and every triangle corner is verified.
        from scipy.spatial import cKDTree
        tri_s = v_cache[f_cache]                                       # (F, 3, 3) float64
        tri_o = verts.numpy().astype(np.float64)[fi]
        tree = cKDTree(tri_o.mean(1))
        dist, m = tree.query(tri_s.mean(1))
        bad = np.flatnonzero(np.abs(tri_s - tri_o[m]).max((1, 2)) > 1e-5)        # duplicate centroids in the OBJ
        if len(bad):
            _, cand = tree.query(tri_s[bad].mean(1), k=32)
            err = np.abs(tri_s[bad][:, None] - tri_o[cand]).max((2, 3))           # (n_bad, 32)
            m[bad] = cand[np.arange(len(bad)), err.argmin(1)]
        n_bad_first = int(len(bad))
        tri_err = float(np.abs(tri_s - tri_o[m]).max()) if len(fi) == len(f_cache) else None
        self.geom = {"n_verts_sensor": int(len(v_cache)), "n_verts_obj": int(len(verts)), "n_faces": int(len(f_cache)),
                     "n_faces_obj": int(len(fi)), "face_match_bijective": bool(len(np.unique(m)) == len(m)),
                     "n_faces_resolved_by_knn": n_bad_first,
                     "triangle_corner_maxabs_m": tri_err,
                     "n_materials": len(names), "n_faces_no_material": int((faces.materials_idx < 0).sum())}
        if tri_err is None or tri_err > 1e-5:
            raise RuntimeError(f"textured mesh geometry differs from the sensor mesh: {self.geom}")
        m = torch.as_tensor(m, dtype=torch.int64)
        fuv = aux.verts_uvs[faces.textures_idx[m]]
        if (fuv > 1).any() or (fuv < 0).any():        # load_obj default texture_wrap = "repeat"
            fuv = fuv % 1
        fuv, mi = fuv.to(dev), faces.materials_idx[m].to(dev)
        R = int(atlas_size)
        atlas = torch.ones((len(fi), R, R, 3), dtype=torch.float32, device=dev)
        d = os.path.dirname(obj[0])
        for k, n in enumerate(names):
            sel = mi == k
            if not bool(sel.any()):
                continue
            img = torch.from_numpy(np.asarray(Image.open(os.path.join(d, n)).convert("RGB"))).to(dev).float() / 255.0
            atlas[sel] = make_material_atlas(torch.flip(img, [0]), fuv[sel], R)
        vrt = torch.as_tensor(S.zup_m_to_rt(v_cache), dtype=torch.float32, device=dev)      # exactly as mp3d_sim.Sensor
        self.mesh = Meshes(verts=[vrt], faces=[torch.as_tensor(f_cache, dtype=torch.int64, device=dev)],
                           textures=TexturesAtlas(atlas=[atlas]))
        self.ambient, self.atlas_size = float(ambient), R
        self.lights = AmbientLights(ambient_color=((ambient,) * 3,), device=dev)
        self.MeshRasterizer, self.MeshRendererWithFragments, self.SoftPhongShader = MeshRasterizer, MeshRendererWithFragments, SoftPhongShader
        self.renderer = None
        self.build_s = time.time() - t0

    def render(self, fov, raster_settings):
        if self.renderer is None:
            self.renderer = self.MeshRendererWithFragments(
                rasterizer=self.MeshRasterizer(raster_settings=raster_settings),
                shader=self.SoftPhongShader(device=self.dev, cameras=fov, lights=self.lights))
        with self.torch.no_grad():
            img, frag = self.renderer(self.mesh, cameras=fov, lights=self.lights)
        return img[0, ..., :3], frag.zbuf[0, ..., 0]


_service_record = T12.service_record


def service_record(scene_key, start, t, fr, pts, flat, pose, hist, legal, g, sen):
    rec = _service_record(scene_key, start, t, fr, pts, flat, pose, hist, legal, g, sen)
    torch = CTX["tex"].torch
    rgb, zb = CTX["tex"].render(fr["fov"], sen.rast.raster_settings)
    z0 = fr["zbuf"][0, ..., 0]
    hit_eq = bool(torch.equal(z0 > -1, zb > -1))
    both = (z0 > -1) & (zb > -1)
    zdiff = float((z0[both] - zb[both]).abs().max()) if bool(both.any()) else 0.0
    lo, hi = float(rgb.min()), float(rgb.max())
    u8 = (rgb.clamp(0, 1) * 255.0).round().to(torch.uint8).cpu().numpy()
    side = os.path.join(CTX["od"], "svc", f"svc_t{int(t)}_rgb.npz")
    np.savez_compressed(side, t=np.asarray(int(t)), rgb=u8, R=rec["R"], T=rec["T"], rgb_float_min=np.asarray(lo),
                        rgb_float_max=np.asarray(hi), zbuf_hit_mask_equal=np.asarray(hit_eq), zbuf_maxabs_diff=np.asarray(zdiff))
    CTX["zcheck"].append({"t": int(t), "hit_mask_equal": hit_eq, "zbuf_maxabs_diff": zdiff, "rgb_min": lo, "rgb_max": hi,
                          "rgb_mean_u8": float(u8.mean()), "hit_frac": float((z0 > -1).float().mean())})
    return rec


def child(group, key, tag, od, cfg):
    CTX["od"] = od
    start = tag.rsplit("_S", 1)[1]
    env = {k: v for k, v in os.environ.items() if k not in ("LD_PRELOAD", "PYTHONPATH", "PYTHONHOME")}
    env.update(PYTHONDONTWRITEBYTECODE="1", PYTHONNOUSERSITE="1", SLURM_JOB_QOS="kd-high",
               CONDA_PREFIX=os.path.dirname(os.path.dirname(FISHERRF_PYTHON)))
    cmd = [FISHERRF_PYTHON, "-B", "-s", os.path.join(HERE, "fisherrf_planner.py"), "--scene", key, "--start", start,
           "--svc-dir", os.path.join(od, "svc"), "--svc-name", "svc_t{t}.npz", "--scene-key", f"{key}/{start}",
           "--log", os.path.join(od, "planner_fisherrf.steps.jsonl"), "--workdir", os.path.join(od, "planner_work"),
           "--init-pose", CTX["init_pose"], "--verify-every", CTX["verify_every"]]
    p = subprocess.Popen(cmd, stdin=subprocess.PIPE, stdout=subprocess.PIPE, stderr=open(os.path.join(od, "planner.err"), "w"),
                         text=True, env=env)
    hello = json.loads(p.stdout.readline())
    assert hello["ready"], hello
    hello.setdefault("info", {}).update({"textured_rgb": {"geometry": CTX["tex"].geom, "atlas_size": CTX["tex"].atlas_size,
                                                          "ambient": CTX["tex"].ambient, "build_s": CTX["tex"].build_s}})
    return p, hello


class _AP(argparse.ArgumentParser):
    def add_argument(self, *a, **k):
        if a and a[0] == "--group":
            k["choices"] = list(k["choices"]) + [GROUP]
        return super().add_argument(*a, **k)


if __name__ == "__main__":
    import torch
    opts = {"--init-pose": "tester", "--verify-every": "0", "--atlas": "4"}
    for flag, default in list(opts.items()):
        if flag in sys.argv:
            i = sys.argv.index(flag)
            opts[flag] = sys.argv[i + 1]
            del sys.argv[i:i + 2]
    CTX["init_pose"], CTX["verify_every"] = opts["--init-pose"], opts["--verify-every"]
    scan = sys.argv[sys.argv.index("--scan") + 1]
    CTX["tex"] = TexturedRGB(scan, torch.device("cuda"), atlas_size=int(opts["--atlas"]))
    print("textured RGB ready", json.dumps(CTX["tex"].geom), "build %.1f s" % CTX["tex"].build_s, flush=True)
    T12.service_record = service_record
    T23.child = child
    T23.argparse = types.SimpleNamespace(ArgumentParser=_AP)
    sys.argv += ["--group", GROUP]
    T23.main()
    z = CTX["zcheck"]
    summ = {"n": len(z), "all_hit_mask_equal": all(x["hit_mask_equal"] for x in z),
            "zbuf_maxabs_diff_max": max(x["zbuf_maxabs_diff"] for x in z), "frames": z}
    json.dump(summ, open(os.path.join(CTX["od"], "rgb_zbuf_check.json"), "w"), indent=1)
    print("RGB / zbuf check:", json.dumps({k: v for k, v in summ.items() if k != "frames"}), flush=True)
