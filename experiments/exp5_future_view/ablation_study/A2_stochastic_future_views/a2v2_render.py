"""EXP5 Ablation A2 figures v2 -- RGB + depth renders of the figure cases (magician env, 1 GPU; evaluation side).

For every case state of results/cases.json: the current camera (record t) and the candidates drawn in the figures (the
picks of V2M / V2O and the oracle best), each rendered at 456 x 256 with the frozen sensor geometry:
  MP3D        the textured matterport_mesh (PyTorch3D texture atlas, ambient light only = raw texture colours) with
              the frozen Sensor camera (FoVPerspectiveCameras(R, T, zfar 750), same rasteriser size)
  Macarons++  the frozen EpisodeSim (MAGICIAN camera.update_camera + capture_image), pose keys from the stored episode
Check: the rendered depth of the current camera equals the stored service-record zbuf (MP3D: max |diff| reported;
Macarons++: exact). Output W/cases_v2/<dataset>_<sid>.npz
"""
import glob
import json
import os
import shutil
import sys
import time

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.join(os.path.dirname(HERE), "common"))
import ab_common as A  # noqa: E402

import numpy as np  # noqa: E402

OUT = os.path.join(A.W, "cases_v2")


def figure_candidates(dataset, st):
    """Candidates drawn for a case: the picks of V2M / V2O under the online choice rule (as computed by A1, results CSV)
    and the oracle best of the candidate set."""
    import csv
    rows = [r for r in csv.DictReader(open(os.path.join(os.path.dirname(HERE), "A1_prediction_scoring", "results",
                                                         f"a1_states_{dataset}.csv"))) if r["sid"] == st["sid"]]
    picks = {r["scorer"]: int(r["pick"]) for r in rows if r["scorer"] in ("V2M", "V2O", "GTD") and r["pick"] not in ("", "None")}
    o = np.load(os.path.join(A.state_dir(dataset, st["sid"]), "oracle.npz"))
    C = np.asarray(st["cand"])
    best = int(C[np.argmax(o["gain"][C])])
    return best, picks


def indoor(cases, dev):
    import torch
    from pytorch3d.io import load_objs_as_meshes
    from pytorch3d.renderer import (AmbientLights, BlendParams, FoVPerspectiveCameras, HardPhongShader, MeshRasterizer,
                                    MeshRenderer, RasterizationSettings)
    from pytorch3d.structures import Meshes
    sys.path.insert(0, os.path.join(A.EXP5, "task0", "scripts"))
    sys.path.insert(0, os.path.join(A.EXP5, "task2", "scripts"))
    import mp3d_sim as S
    import t2_common as C2
    man = {s["sid"]: s for s in A.load_manifest("indoor")["states"]}
    by_scan = {}
    for nm, c in cases.items():
        if c["sid"]:
            by_scan.setdefault(man[c["sid"]]["scan"], []).append(c["sid"])
    for scan, sids in sorted(by_scan.items()):
        C2.assert_mesh_allowed(scan, "eval")
        obj = glob.glob(os.path.join(S.MP, scan, "matterport_mesh", "*", "*.obj"))
        assert len(obj) == 1, obj
        tb = time.time()
        m = load_objs_as_meshes(obj, device=dev, load_textures=True, create_texture_atlas=True, texture_atlas_size=6)
        v = m.verts_packed()
        v_rt = S_RT_of(S, v)
        mesh = Meshes(verts=[v_rt], faces=[m.faces_packed()], textures=m.textures)
        rs = RasterizationSettings(image_size=(S.H_IMG, S.W_IMG), blur_radius=0.0, faces_per_pixel=1,
                                   max_faces_per_bin=max(200000, int(m.faces_packed().shape[0])))
        print(scan, "mesh", int(m.faces_packed().shape[0]), "faces, %.0fs" % (time.time() - tb), flush=True)
        for sid in sids:
            st = man[sid]
            p = os.path.join(OUT, f"indoor_{sid}.npz")
            if os.path.exists(p):
                continue
            src = st["sources"][0]["source"]
            ep = {"tag": st["tag"], "scene": st["scene"], "start": st["start"]}
            rec = np.load(A.svc_path("indoor", src, ep, st["t"]))
            best, picks = figure_candidates("indoor", st)
            js = sorted({best, *[x for x in picks.values() if x is not None]})
            poses = [("cur", rec["R"], rec["T"])] + [(f"c{j}", rec["legal_R"][j], rec["legal_T"][j]) for j in js]
            save = {"cands": np.asarray(js), "best": best, **{f"pick_{k}": (-1 if x is None else x) for k, x in picks.items()}}
            for nm, R, T in poses:
                cam = FoVPerspectiveCameras(R=torch.tensor(R[None], dtype=torch.float32, device=dev),
                                            T=torch.tensor(T[None], dtype=torch.float32, device=dev), zfar=S.ZFAR, device=dev)
                ren = MeshRenderer(MeshRasterizer(cameras=cam, raster_settings=rs),
                                   HardPhongShader(device=dev, cameras=cam, lights=AmbientLights(device=dev),
                                                   blend_params=BlendParams(background_color=(1.0, 1.0, 1.0))))
                with torch.no_grad():
                    frag = ren.rasterizer(mesh, cameras=cam)
                    img = ren.shader(frag, mesh, cameras=cam)
                save[f"rgb_{nm}"] = (img[0, ..., :3].clamp(0, 1).cpu().numpy() * 255).astype(np.uint8)
                save[f"z_{nm}"] = frag.zbuf[0, ..., 0].cpu().numpy().astype(np.float32)
            zs = rec["zbuf"]
            both = (zs > -1) & (save["z_cur"] > -1)
            save["check_cur_zbuf_maxdiff"] = float(np.abs(zs[both] - save["z_cur"][both]).max()) if both.any() else 0.0
            save["check_cur_mask_agree"] = float(((zs > -1) == (save["z_cur"] > -1)).mean())
            os.makedirs(OUT, exist_ok=True)
            np.savez_compressed(p, **save)
            print("indoor", sid, "cands", js, "zbuf check max|d| %.2e, mask agree %.4f" % (save["check_cur_zbuf_maxdiff"],
                                                                                   save["check_cur_mask_agree"]), flush=True)
        del mesh, m
        torch.cuda.empty_cache()


def S_RT_of(S, v):
    """z-up metres (matterport_mesh) -> runtime frame, as mp3d_sim.zup_m_to_rt (torch)."""
    import torch
    return S.S_RT * torch.stack([v[:, 0], v[:, 2], -v[:, 1]], -1)


def outdoor(cases):
    sys.path.insert(0, os.path.join(A.EXP5, "task1", "scripts"))
    import v2_common as LC
    sys.path.insert(1, LC.EXP3B)
    import m_t6_sim as T6S
    import m_magician_replay as RP
    man = {s["sid"]: s for s in A.load_manifest("outdoor")["states"]}
    for nm, c in cases.items():
        if not c["sid"]:
            continue
        sid = c["sid"]
        st = man[sid]
        p = os.path.join(OUT, f"outdoor_{sid}.npz")
        if os.path.exists(p):
            continue
        assert st["scene"] in A.SPLIT_OF
        src = st["sources"][0]["source"]
        ep = json.load(open(A.outdoor_episode_json(src, st)))
        step = ep["steps"][st["t"]]
        best, picks = figure_candidates("outdoor", st)
        js = sorted({best, *[x for x in picks.values() if x is not None]})
        fr = os.path.join(A.W, "tmp_frames", f"v2fig_{sid}_{os.getpid()}")
        shutil.rmtree(fr, ignore_errors=True)
        sim = T6S.EpisodeSim(st["scene"], st["start"], fr)
        save = {"cands": np.asarray(js), "best": best, **{f"pick_{k}": (-1 if x is None else x) for k, x in picks.items()}}
        for nm2, key in [("cur", step["pose_key"])] + [(f"c{j}", step["legal_keys"][j]) for j in js]:
            sim.camera.update_camera(sim.camera.get_idx_from_key(key), interpolation_step=1)
            sim.camera.capture_image(sim.mesh)
            f = RP.MP.load_current_frame_perfect_depth(sim.camera, sim.dev)
            save[f"rgb_{nm2}"] = (f["rgb"][0].clamp(0, 1).cpu().numpy() * 255).astype(np.uint8)
            save[f"z_{nm2}"] = f["zbuf"][0, ..., 0].cpu().numpy().astype(np.float32)
        rec = np.load(A.svc_path("outdoor", src, st, st["t"]))
        save["check_cur_zbuf_maxdiff"] = float(np.abs(rec["zbuf"] - save["z_cur"]).max())
        save["check_cur_mask_agree"] = float(((rec["zbuf"] > -1) == (save["z_cur"] > -1)).mean())
        os.makedirs(OUT, exist_ok=True)
        np.savez_compressed(p, **save)
        shutil.rmtree(fr, ignore_errors=True)
        print("outdoor", sid, "cands", js, "zbuf check max|d| %.2e" % save["check_cur_zbuf_maxdiff"], flush=True)
        del sim


def main():
    import torch
    cases = json.load(open(os.path.join(HERE, "results", "cases.json")))
    which = sys.argv[1] if len(sys.argv) > 1 else "both"
    if which in ("indoor", "both"):
        indoor(cases["indoor"], torch.device("cuda"))
    if which in ("outdoor", "both"):
        outdoor(cases["outdoor"])


if __name__ == "__main__":
    main()
