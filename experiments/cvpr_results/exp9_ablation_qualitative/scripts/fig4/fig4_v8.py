"""EXP9 Fig. 4 v8 (user 2026-11-16): v7 final WITHOUT the first scene -- only Colosseum s3 t0 and MP3D QUCTc6BB5sX L1 S34 t40 (each a main
row + its x3 zoom row), same data, zoom windows (v5 = v6 rule), colours and v5-3b layout (fig4_v5_layouts.composite, unchanged).
Output fig/fig4/fig4_v8/fig4_v8.{pdf,png,svg}. CPU only (seen2scene env, CUDA_VISIBLE_DEVICES="")."""
import csv
import os
import sys

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
import fig4_v5 as F5  # noqa: E402
import fig4_v5_layouts as LY  # noqa: E402

FS, F3 = F5.FS, F5.F3
OUT = FS.REPO + "experiments/cvpr_results/fig/fig4/fig4_v8/"
LY.META = {"pdf": {"CreationDate": None, "Creator": "EXP9 fig4_v8.py"}, "png": {"Software": "EXP9 fig4_v8.py"},
           "svg": {"Date": None, "Creator": "EXP9 fig4_v8.py"}}


def main():
    os.makedirs(OUT, exist_ok=True)
    v5win = {d["row"]: (int(d["win_row0"]), int(d["win_col0"])) for d in csv.DictReader(open(F5.OUT + "zoom_windows.csv"))}
    rows = []
    for sname, tag, case, byrow, label in F5.ROWS[1:]:
        r = F3.load_row(sname, tag, case)
        best, _, _, _ = F5.pick_window(r, F5.roi_inside(r, np.load(r["npz"])))
        assert (best[2], best[3]) == v5win[byrow], (byrow, best, v5win[byrow])
        r.update({"win": (best[2], best[3]), "label": label, "byrow": byrow})
        rows.append(r)
    h = LY.composite(rows, OUT + "fig4_v8", LY.VERSIONS["v5-3b"])
    print(f"fig4_v8 6.875 x {h:.2f} in", [(r["label"], r["win"]) for r in rows])


if __name__ == "__main__":
    main()
