"""EXP9 Fig. 4 v7 FINAL (user 2026-11-16): row 1 = Neuschwanstein s3 t0 (Unseen-6 scene; pool of fig4_v7.py), rows 2-3 unchanged from v5
(Colosseum s3 t0, MP3D QUCTc6BB5sX L1 S34 t40 with their v5 zoom windows); layout = v5-3b (fig4_v5_layouts.py, imported unchanged:
connectors in the box's own style, one-line legend, 6.875 in wide).
Row 1 window = the v6 rule (v5 rule + >= 30 % of the window's rays hit a real surface; fig4_v6.best_window), read from ranking_v7.csv
and recomputed here (asserted equal).  Rows 2-3 satisfy the v6 rule with their v5 windows (asserted in fig4_v6.py; re-asserted here).
Row 1 colour image = work_dirs/.../fig4_v7/render/outdoor_neuschwanstein_s3_t0.npz (fig4_v4_render.run_case, all v4 guards passed).
Outputs fig/fig4/fig4_v7_final/: fig4_v7_final.{pdf,png,svg}, panels/row1_neuschwanstein/ (single / box / zoom / hires panels as v5),
colorbar_row1.{pdf,png,svg}, zoom_windows.csv.  CPU only (seen2scene env, CUDA_VISIBLE_DEVICES="")."""
import csv
import os
import sys

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
import fig4_v5 as F5  # noqa: E402
import fig4_v5_layouts as LY  # noqa: E402
import fig4_v6 as V6  # noqa: E402
import fig4_v7 as V7  # noqa: E402
import fig4_v4_sources as V4  # noqa: E402

FS, F3 = F5.FS, F5.F3
OUT = FS.REPO + "experiments/cvpr_results/fig/fig4/fig4_v7_final/"
CASE = "outdoor_neuschwanstein_s3_t0"
LY.META = {k: dict(v, Creator="EXP9 fig4_v7_final.py") if k != "png" else {"Software": "EXP9 fig4_v7_final.py"} for k, v in LY.META.items()}


def row1():
    e = next(r for r in csv.DictReader(open(V7.OUT + "ranking_v7.csv")) if r["case"] == CASE)
    d = {"domain": "outdoor", "group": "neuschwanstein", "start": 3, "t": 0, "case": CASE, "pick": int(e["pick"])}
    _, A, inb, code = V6.state_arrays(d)
    b, _ = V6.best_window(A["unknown"], A["gt_valid"], inb, code, V6.GT_MIN)
    assert (b[1], b[2]) == (int(e["win_row0"]), int(e["win_col0"])), (b, e["win_row0"], e["win_col0"])
    z = np.load(V7.W7 + f"{CASE}.npz")
    code5, share = F3.disagreement_codes(A["hyp_valid"], A["unknown"])
    assert np.array_equal(code5, code)
    return {"meta": {"domain": "outdoor", "depth_unit": A["unit"], "case": CASE, "scene": "neuschwanstein"}, "cur_rgb": z["cur_rgb"],
            "cand_rgb": z["cand_rgb"], "unk": A["unknown"], "gt": A["gt_depth"], "gtv": A["gt_valid"], "hyp": A["hyp_depth"],
            "hypv": A["hyp_valid"], "vmin": A["vmin"], "vmax": A["vmax"], "code": code, "dis_share": share, "win": (b[1], b[2]),
            "label": "Neuschwanstein", "byrow": "row1_neuschwanstein", "win_score": b[0]}


def colorbars(r, d):
    """Row-1 colour bars in the three v4 variants (fig4_v4_sources.colorbar, fig4_v4_colorbars_hires.draw, fig4_v4_colorbars_paste.draw), same
    parameters; the paste variant uses the font size the original 3 rows were fitted with (so it matches rows 2-3)."""
    import fig4_v4_colorbars_hires as CH
    import fig4_v4_colorbars_paste as CP
    plt = F5.plt
    os.makedirs(d, exist_ok=True)
    vmin, vmax, dom, unit = r["vmin"], r["vmax"], "outdoor", r["meta"]["depth_unit"]
    V4.colorbar(d + "colorbar_depth", vmin, vmax, unit)
    for suffix, rt in (("", False), ("_round", True)):
        fig, _ = CH.draw(vmin, vmax, dom, unit, rt)
        for ext in ("png", "pdf", "svg"):
            fig.savefig(d + f"colorbar_depth_hires{suffix}.{ext}", dpi=CH.DPI, facecolor="white", metadata=CH.META[ext])
        plt.close(fig)
    orig = {x: CP.load(x) for x in CP.ROWS}
    fs = None
    for cand in np.arange(8.0, 4.49, -0.25):
        ok = True
        for vmin_, vmax_, dom_, unit_ in orig.values():
            f_, _ = CP.draw(vmin_, vmax_, dom_, unit_, cand)
            ok &= CP.fits(f_)[0]
            plt.close(f_)
        if ok:
            fs = float(cand)
            break
    fig, _ = CP.draw(vmin, vmax, dom, unit, fs)
    assert CP.fits(fig)[0], "row-1 paste colour bar does not fit at the rows' font size"
    for ext in ("png", "pdf", "svg"):
        fig.savefig(d + f"{CP.NAME}.{ext}", dpi=CP.DPI, facecolor="white", metadata=CP.META[ext])
    plt.close(fig)
    print("colour bars", f"{vmin:g}-{vmax:g} {unit}", "paste font", fs)


def main():
    os.makedirs(OUT, exist_ok=True)
    v5win = {d["row"]: (int(d["win_row0"]), int(d["win_col0"])) for d in csv.DictReader(open(F5.OUT + "zoom_windows.csv"))}
    rows = [row1()]
    for sname, tag, case, byrow, label in F5.ROWS[1:]:                       # rows 2-3 exactly as v5
        r = F3.load_row(sname, tag, case)
        z = np.load(r["npz"])
        inb = F5.roi_inside(r, z)
        best, _, _, _ = F5.pick_window(r, inb)
        assert (best[2], best[3]) == v5win[byrow], (byrow, best, v5win[byrow])
        b6, _ = V6.best_window(r["unk"], r["gtv"], inb, r["code"], V6.GT_MIN)
        assert (b6[1], b6[2]) == v5win[byrow], ("v6 rule changes", byrow)
        r.update({"win": (best[2], best[3]), "label": label, "byrow": byrow, "win_score": best[0]})
        rows.append(r)
    h = LY.composite(rows, OUT + "fig4_v7_final", LY.VERSIONS["v5-3b"])
    cbd = OUT + "colorbars/row1_neuschwanstein/"
    colorbars(rows[0], cbd)
    F5.BYROW = OUT + "colorbars/"                                        # export_panels copies the row's colorbar_depth* files from BYROW/<row>
    F5.export_panels(rows[0], OUT + "panels/row1_neuschwanstein/")
    with open(OUT + "zoom_windows.csv", "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["row", "case", "win_row0", "win_col0", "win_h", "win_w", "disagreeing_unknown_rays", "win_gt_hit_share", "win_unknown_share", "rule"])
        for r in rows:
            y, x = r["win"]
            sl = (slice(y, y + F5.WIN_H), slice(x, x + F5.WIN_W))
            w.writerow([r["byrow"], r["meta"].get("case", ""), y, x, F5.WIN_H, F5.WIN_W, r["win_score"], f"{r['gtv'][sl].mean():.4f}",
                        f"{r['unk'][sl].mean():.4f}", "v6 (= v5 + >= 30 % real surface)" + (" ; v5 window unchanged" if r["byrow"] != "row1_neuschwanstein" else "")])
    print(f"fig4_v7_final 6.875 x {h:.2f} in", [(r["label"], r["win"], r["win_score"]) for r in rows])


if __name__ == "__main__":
    main()
