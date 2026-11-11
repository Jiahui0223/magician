"""Units / frame check: RTK GNSS (lat, lon, alt) -> local ENU at the mesh SRS origin vs. camera centres
of sampleinfos_interpolated.json (T4x4 translation).

python -I check_units_rtk.py <sampleinfos.json> <rtk_positions_raw.csv> <metadata.xml> <out_json>
"""
import json
import re
import sys

import numpy as np

js, rtk_csv, meta, out = sys.argv[1:5]
txt = open(meta).read()
lat0, lon0 = map(float, re.search(r"<SRS>ENU:([-\d.]+),([-\d.]+)</SRS>", txt).groups())
h0 = float(re.search(r"<SRSOrigin>([-\d.]+),([-\d.]+),([-\d.]+)</SRSOrigin>", txt).group(3))

a, f = 6378137.0, 1 / 298.257223563
e2 = f * (2 - f)


def ecef(lat, lon, h):
    la, lo = np.radians(lat), np.radians(lon)
    N = a / np.sqrt(1 - e2 * np.sin(la) ** 2)
    return np.stack([(N + h) * np.cos(la) * np.cos(lo), (N + h) * np.cos(la) * np.sin(lo), (N * (1 - e2) + h) * np.sin(la)], -1)


def enu(lat, lon, h):
    d = ecef(lat, lon, h) - ecef(lat0, lon0, h0)
    la, lo = np.radians(lat0), np.radians(lon0)
    Rm = np.array([[-np.sin(lo), np.cos(lo), 0],
                   [-np.sin(la) * np.cos(lo), -np.sin(la) * np.sin(lo), np.cos(la)],
                   [np.cos(la) * np.cos(lo), np.cos(la) * np.sin(lo), np.sin(la)]])
    return d @ Rm.T


rows = [l.strip().split(",") for l in open(rtk_csv)][1:]
ts = np.array([float(r[1]) for r in rows])
llh = np.array([[float(r[2]), float(r[3]), float(r[4])] for r in rows])
S = json.load(open(js))
tc = np.array([float(s["OriginalImageName"][:-4]) for s in S])
C = np.array([np.array(s["T4x4"])[:3, 3] for s in S])
m = (tc >= ts.min()) & (tc <= ts.max())
li = np.stack([np.interp(tc[m], ts, llh[:, k]) for k in range(3)], 1)
E = enu(li[:, 0], li[:, 1], li[:, 2])
dlt = C[m] - E
res = {"srs_origin_lat_lon_h": [lat0, lon0, h0], "n_compared": int(m.sum()),
       "rtk_rate_hz": float(1 / np.median(np.diff(ts))),
       "cam_minus_rtkENU_mean_m": dlt.mean(0).tolist(), "cam_minus_rtkENU_std_m": dlt.std(0).tolist(),
       "horizontal_err_after_mean_removal_m_pct_50_95": np.percentile(np.linalg.norm((dlt - dlt.mean(0))[:, :2], axis=1), [50, 95]).tolist(),
       "path_length_ratio_cam_over_rtk": float(np.linalg.norm(np.diff(C[m], axis=0), axis=1).sum() / np.linalg.norm(np.diff(E, axis=0), axis=1).sum())}
json.dump(res, open(out, "w"), indent=1)
print(json.dumps(res, indent=1))
