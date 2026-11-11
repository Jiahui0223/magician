"""Summarise UAVScenes sampleinfos_interpolated.json (camera <-> 3D map calibration).

python -I inspect_poses.py <sampleinfos.json> <zip_listing_CAM_LIDAR.tsv> <run> <out_dir>
Writes poses_summary.json, poses_interval5.csv and a trajectory plot.
"""
import json
import os
import sys

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

js, listing, run, out = sys.argv[1:5]
os.makedirs(out, exist_ok=True)
S = json.load(open(js))
cam_files = set()
for line in open(listing):
    p = line.split("\t")[0]
    if f"interval5_{run}/interval5_CAM/" in p and p.endswith(".jpg"):
        cam_files.add(os.path.basename(p))

T = np.array([s["T4x4"] for s in S], dtype=np.float64)
K = np.array([s["P3x3"] for s in S], dtype=np.float64)
names = [s["OriginalImageName"] for s in S]
in5 = np.array([n in cam_files for n in names])
R = T[:, :3, :3]
t = T[:, :3, 3]
detR = np.linalg.det(R)
orth = np.abs(np.einsum("nij,nkj->nik", R, R) - np.eye(3)).max(axis=(1, 2))
zaxis = R[:, :, 2]  # optical axis in world if T is cam->world (OpenCV)
tilt = np.degrees(np.arccos(np.clip(-zaxis[:, 2], -1, 1)))  # 0 = nadir
dist = {k: sorted(set(map(lambda s: round(float(s[k]), 6), S))) for k in ["K1", "K2", "K3", "P1", "P2"]}
summ = {
    "n_entries": len(S),
    "n_interval5_images_in_zip": len(cam_files),
    "n_entries_matching_interval5": int(in5.sum()),
    "keys": sorted(S[0].keys()),
    "width_height": sorted(set((s["Width"], s["Height"]) for s in S)),
    "K_unique": np.unique(K.reshape(len(K), -1), axis=0).tolist(),
    "dist_unique": dist,
    "det_R_minmax": [float(detR.min()), float(detR.max())],
    "orthonormality_err_max": float(orth.max()),
    "cam_center_min": t.min(0).tolist(),
    "cam_center_max": t.max(0).tolist(),
    "optical_axis_tilt_from_nadir_deg_pct": np.percentile(tilt, [0, 5, 50, 95, 100]).tolist(),
    "step_len_m_interval5_median": float(np.median(np.linalg.norm(np.diff(t[in5], axis=0), axis=1))),
    "path_len_m": float(np.linalg.norm(np.diff(t, axis=0), axis=1).sum()),
}
json.dump(summ, open(os.path.join(out, "poses_summary.json"), "w"), indent=1)
print(json.dumps(summ, indent=1))
with open(os.path.join(out, "poses_interval5.csv"), "w") as f:
    f.write("idx,image,x,y,z,tilt_deg\n")
    for i in np.where(in5)[0]:
        f.write(f"{i},{names[i]},{t[i,0]:.3f},{t[i,1]:.3f},{t[i,2]:.3f},{tilt[i]:.2f}\n")
fig, ax = plt.subplots(1, 2, figsize=(12, 5))
ax[0].plot(t[:, 0], t[:, 1], lw=0.5)
ax[0].scatter(t[in5, 0], t[in5, 1], s=1, c="r")
ax[0].set_aspect("equal"); ax[0].set_xlabel("x (E, m)"); ax[0].set_ylabel("y (N, m)")
ax[0].set_title(f"{run} camera centres (red = interval5)")
ax[1].plot(t[:, 2]); ax[1].set_xlabel("entry"); ax[1].set_ylabel("z (U, m)")
ax2 = ax[1].twinx(); ax2.plot(tilt, "g", lw=0.5); ax2.set_ylabel("tilt from nadir (deg)", color="g")
plt.tight_layout(); plt.savefig(os.path.join(out, "trajectory.png"), dpi=110)
