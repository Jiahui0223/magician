"""EXP5 Task 0.3 -- the indoor Sensor (mp3d_sim.Sensor) must reproduce the frozen outdoor sensor bitwise: re-render
archived B0 (abl_1 A2) service records of barts with the same R, T on MAGICIAN's own load_scene mesh and compare
zbuf, mask and the 10 % sample points (derived "sample" seed). Output W/preprocessing/sensor_equiv.json."""
import json
import os
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
import mp3d_sim as S  # noqa: E402

import numpy as np  # noqa: E402
import torch  # noqa: E402

sys.path.insert(0, "/bsuscratch/jiahuizhang/projects/occupancy/MAGICIAN_clean_9521d69")
from macarons.utility.macarons_utils import load_scene  # noqa: E402

B0 = os.path.join(S.REPO, "work_dirs/exp4_multiworld_belief/ablation_study/abl_1/T1C_FRindep_DistP1_Greedy_R10_K4/dev100/svc/A2_barts_s0_r0")
dev = torch.device("cuda")
obj = [f for f in os.listdir("/bsuscratch/jiahuizhang/projects/occupancy/MAGICIAN/data/Macarons++/barts") if f.endswith(".obj")][0]
mesh = load_scene(os.path.join("/bsuscratch/jiahuizhang/projects/occupancy/MAGICIAN/data/Macarons++/barts", obj), 10.0, dev,
                  mirror=False, mirrored_axis=None)
sen = S.Sensor(mesh.verts_packed().cpu().numpy(), mesh.faces_packed().cpu().numpy(), dev)
sen.mesh = mesh                       # MAGICIAN's own Meshes object (identical vertices / faces)
rows = []
for t in (0, 5, 37, 99):
    rec = S.MC.load_service_record(os.path.join(B0, f"svc_barts_s0_t{t}.npz"))
    R = torch.tensor(rec["R"][None], dtype=torch.float32, device=dev)
    T = torch.tensor(rec["T"][None], dtype=torch.float32, device=dev)
    fr = sen.render(R, T)
    pts, flat, _, _, _ = sen.sample(fr, S.MC.derived_seed("barts", 0, t, "sample"))
    z = fr["zbuf"][0, ..., 0].cpu().numpy().astype(np.float32)
    rows.append({"t": t, "zbuf_equal": bool(np.array_equal(z, rec["zbuf"])), "mask_equal": bool(np.array_equal(fr["mask"][0, ..., 0].cpu().numpy(), rec["mask"])),
                 "sample_flat_equal": bool(np.array_equal(flat.numpy(), rec["sample_pixel_flat"])),
                 "sample_points_equal": bool(np.array_equal(pts.cpu().numpy().astype(np.float32), rec["sample_points"])),
                 "max_abs_zbuf": float(np.abs(z - rec["zbuf"]).max()), "n_points": int(len(pts)),
                 "cam_center_equal": bool(np.allclose(fr["fov"].get_camera_center()[0].cpu().numpy(), rec["cam_center"], atol=0))})
    print(rows[-1], flush=True)
out = {"rows": rows, "pass": all(r["zbuf_equal"] and r["mask_equal"] and r["sample_flat_equal"] and r["sample_points_equal"] for r in rows)}
os.makedirs(os.path.join(S.W, "preprocessing"), exist_ok=True)
json.dump(out, open(os.path.join(S.W, "preprocessing", "sensor_equiv.json"), "w"), indent=1)
print("PASS" if out["pass"] else "FAIL")
