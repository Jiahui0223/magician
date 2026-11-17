"""EXP9 Fig. 6 v4 -- contact sheets of every unseen candidate (magician env, PyTorch3D, ONE GPU per process).
Usage:  CUDA_VISIBLE_DEVICES=g $MAGPY -I -B f6v4_render.py --group outdoorA|outdoorB|indoor

Rendering = the v3 pipeline unchanged (f6v3_render.py, which imports f6v2_render.py and renormalises barycentrics): real textured
mesh restricted to the evaluator's GT domain D; rows 2-3 show the nearest surface point observed by the method (per-GT-point masks of
f6v4_masks.py, |x - g(x)| <= 2 eps); one orthographic camera per column; tiles 128; K = 48 layers; 3x supersampling; 400 dpi;
panel height = the v3 main-figure panel height (1.508 in), so any panel can be dropped into the final figure unchanged.
Changes for v4 (user 2026-11-16: "show all six unseen scenes and unseen indoor levels; adjust the viewpoints"):
  1. Outdoor view grid: azimuth {0, 45, ..., 315} x elevation {20, 35, 50} deg (v3: elevation 35 only).  Auto view = the v3 score
     (probe pixels whose visible surface point is observed by EXACTLY ONE of the two methods -- symmetric, not biased to either
     method) maximised over the grid; ties -> elevation 35, then smallest azimuth.  The v3 choice (best azimuth at 35 deg) is
     recorded too.
  2. Viewpoint sheet per outdoor scene: GT render at all 24 grid views (mesh D is the same for every start of a scene), labelled
     'az/el', with the starts whose auto view it is -> the user can override a view by its label.
  3. Indoor: v3 / v2 rule unchanged (elevation 60, back-face culling, faces above min(2 m, 0.8 h) removed, azimuth = most pixels
     observed by either method).
Groups (one GPU each): outdoorA = alhambra, neuschwanstein, eiffel; outdoorB = bridge, barts, sestino_museum; indoor = HO-12 x 2 starts.
Outputs (fig/fig6/fig6_v4/): panels/<case>__{gt,mag,ours}.png, sheets/<group sheet>.{pdf,png}, viewpoints/<scene>_viewpoints.png,
views/<case>.json (view grid scores + chosen view), checks/<case>__render_stats.json."""
import argparse
import csv
import json
import os
import sys
import time

import numpy as np
import torch

REPO = "/bsuscratch/jiahuizhang/projects/occupancy/seen2scene/"
EXP9 = REPO + "experiments/cvpr_results/exp9_ablation_qualitative/"
sys.path.insert(0, EXP9 + "scripts/fig6_v2")
sys.path.insert(0, EXP9 + "scripts/fig6_v3")
import f6v3_render as V3R  # noqa: E402  (patches R2.select_layer with barycentric renormalisation)

R2 = V3R.R2
plt = R2.plt
V4 = REPO + "work_dirs/cvpr_results/exp9_ablation_qualitative/fig6_v4/"
OUT = REPO + "experiments/cvpr_results/fig/fig6/fig6_v4/"
R2.WD, R2.OUT = V4, OUT
R2.NAMES.update({"alhambra": "Alhambra Palace", "neuschwanstein": "Neuschwanstein Castle", "eiffel": "Eiffel Tower",
                 "bridge": "Manhattan Bridge", "barts": "Barts", "sestino_museum": "Sestino Museum"})
GROUPS = {"outdoorA": ["alhambra", "neuschwanstein", "eiffel"], "outdoorB": ["bridge", "barts", "sestino_museum"],
          "interior": ["barts", "sestino_museum"]}       # interior-type Macarons++ scenes, re-rendered with back-face culling
R2.T_IN = 0.33                              # two-line column titles (scene / start + view)
INDOOR_SHEETS = {"indoor_1": ["2azQ1b91cZZ", "QUCTc6BB5sX"], "indoor_2": ["TbHJrupSAjP", "Z6MFQCViBuw"], "indoor_3": ["oLBMNvg9in8"]}
EL_GRID = (20.0, 35.0, 50.0)
PH = 1.5084207906781038                     # v3 main panel height (in): panels drop into the final figure unchanged
THUMB_H = 300                               # viewpoint-sheet thumbnail height (px)
_title = R2.title_of
R2.title_of = lambda r: r.get("_title") or _title(r)

# Deviation (2026-11-16, logged in the README): the v2 tile sanity check compares, for ~2000 sampled mesh points, each tile camera's
# projection (+ tile offset) with the full-frame projection.  For points far outside a tile the tile camera's NDC coordinates are huge
# and float32 extrapolation error grows; on indoor_TbHJrupSAjP_L2_S86 this gave 0.117 px over all points but 0.0002 px over points that
# project inside the tile.  Only in-tile points are rasterised by a tile, so v4 evaluates the same threshold (0.05 render px) on the
# sampled points inside the tile (+- 2 px) and records the all-points value as well.  Everything else in render_column is unchanged.
import inspect  # noqa: E402

_SRC = inspect.getsource(R2.render_column)
_OLD = ("    tile_err = 0.0\n"
        "    for x0, y0, tw, th in tiles(W, H, TILE)[::7]:\n"
        "        st = tile_camera(cam, W, H, x0, y0, tw, th, dev).transform_points_screen(smp, image_size=((th, tw),))[:, :2]\n"
        "        tile_err = max(tile_err, float((st + torch.tensor([float(x0), float(y0)], device=dev) - sf).abs().max()))\n")
_NEW = ("    tile_err, tile_err_all = 0.0, 0.0\n"
        "    for x0, y0, tw, th in tiles(W, H, TILE)[::7]:\n"
        "        st = tile_camera(cam, W, H, x0, y0, tw, th, dev).transform_points_screen(smp, image_size=((th, tw),))[:, :2]\n"
        "        e = (st + torch.tensor([float(x0), float(y0)], device=dev) - sf).abs().max(-1).values\n"
        "        ins = (sf[:, 0] >= x0 - 2) & (sf[:, 0] < x0 + tw + 2) & (sf[:, 1] >= y0 - 2) & (sf[:, 1] < y0 + th + 2)\n"
        "        tile_err_all = max(tile_err_all, float(e.max()))\n"
        "        if bool(ins.any()):\n"
        "            tile_err = max(tile_err, float(e[ins].max()))\n")
_OLD2 = 'camp["tiling_check"] = {"tile_px": TILE, "max_tile_projection_offset_px": tile_err,'
_NEW2 = ('camp["tiling_check"] = {"tile_px": TILE, "max_tile_projection_offset_px": tile_err, "rule_v4": "in-tile sampled points", '
         '"max_tile_projection_offset_px_all_points": tile_err_all,')
assert _OLD in _SRC and _OLD2 in _SRC
exec(compile(_SRC.replace(_OLD, _NEW).replace(_OLD2, _NEW2), R2.__file__, "exec"), R2.__dict__)


def vkey(az, el):
    return f"{az:g}/{el:g}"


def plan_outdoor_grid(r, dev):
    case = R2.Case(r, dev)
    P = case.fv.reshape(-1, 3)
    center = (0.5 * (P.min(0).values + P.max(0).values)).double().cpu().numpy()
    scores = {}
    for el in EL_GRID:
        for az in R2.AZ_CANDS:
            asp, _ = R2.natural_aspect(P, center, az, el, dev)
            wp = int(round(np.sqrt(R2.PROBE_AREA / asp)))
            hp = int(round(wp * asp))
            cam, _ = R2.fit_camera(P, center, az, el, wp, hp, dev)
            scores[vkey(az, el)] = {**V3R.probe_counts(case, cam, wp, hp), "probe_px": [wp, hp]}
    rank = lambda k: (scores[k][V3R.AZ_RULE], float(k.split("/")[1]) == 35.0, -float(k.split("/")[0]))  # noqa: E731
    best = max(scores, key=rank)
    v3best = max([k for k in scores if k.endswith("/35")], key=rank)
    az, el = (float(x) for x in best.split("/"))
    asp, asp_raw = R2.natural_aspect(P, center, az, el, dev)
    del case
    torch.cuda.empty_cache()
    return {"center": center, "az": az, "el": el, "scores": scores, "aspect": asp, "aspect_unclamped": asp_raw, "v3_rule_view": v3best,
            "azimuth_rule": "v4: argmax over az {0..315 step 45} x el {20, 35, 50} of #probe pixels whose visible surface point is observed "
                            "by exactly one method; ties -> el 35, then smallest az"}


def gt_thumb(case, center, az, el, h_px):
    P = case.fv.reshape(-1, 3)
    asp, _ = R2.natural_aspect(P, center, az, el, case.dev)
    H = h_px * R2.SS
    W = int(round(H / asp))
    cam, _ = R2.fit_camera(P, center, az, el, W, H, case.dev)
    acc = torch.ones((H, W, 3), dtype=torch.float32)
    for x0, y0, tw, th in R2.tiles(W, H, R2.TILE):
        frag = R2.rasterize(case.mesh, R2.tile_camera(cam, W, H, x0, y0, tw, th, case.dev), th, tw, R2.CULL[case.domain], 1)
        f1, hit, _ = R2.select_layer(frag, frag.pix_to_face[0] >= 0)
        acc[y0:y0 + th, x0:x0 + tw] = R2.colour(case.mesh, f1, hit).cpu()
    return R2.downsample(acc.numpy().astype(np.float64), R2.SS)


def viewpoint_sheet(scene, r0, views_of_scene, dev):
    case = R2.Case(r0, dev)
    P = case.fv.reshape(-1, 3)
    center = (0.5 * (P.min(0).values + P.max(0).values)).double().cpu().numpy()
    chosen = {}
    for n, v in views_of_scene.items():
        chosen.setdefault(vkey(v["az"], v["el"]), []).append("s" + n.rsplit("_s", 1)[1])
    fig, axs = plt.subplots(len(EL_GRID), len(R2.AZ_CANDS), figsize=(2.0 * len(R2.AZ_CANDS), 2.1 * len(EL_GRID)))
    for i, el in enumerate(EL_GRID):
        for j, az in enumerate(R2.AZ_CANDS):
            ax = axs[i, j]
            ax.imshow(gt_thumb(case, center, az, el, THUMB_H), interpolation="lanczos")
            ax.set_axis_off()
            k = vkey(az, el)
            lab = f"az {az:g} / el {el:g}" + (f"  <- auto: {', '.join(chosen[k])}" if k in chosen else "")
            ax.set_title(lab, fontsize=8, color="#B33" if k in chosen else "#222", fontweight="bold" if k in chosen else "normal")
    fig.suptitle(f"{R2.NAMES[scene]} -- GT at every candidate view (az = azimuth, el = elevation, deg); red = auto view of the listed starts",
                 fontsize=10)
    fig.tight_layout()
    os.makedirs(OUT + "viewpoints", exist_ok=True)
    fig.savefig(OUT + f"viewpoints/{scene}_viewpoints.png", dpi=150)
    plt.close(fig)
    del case
    torch.cuda.empty_cache()


def render_cases(names, cases, views, dev):
    imgs, pws = {}, {}
    for n in names:
        w = PH / views[n]["aspect"]
        pws[n] = w
        torch.cuda.reset_peak_memory_stats()
        prep, im, camp, stats = R2.render_column(cases[n], dev, int(round(w * R2.DPI)), int(round(PH * R2.DPI)), views[n])
        imgs[n] = im
        for row, img in im.items():
            R2.save_panel(img, OUT + f"panels/{n}__{row}.png")
        rec = {k: v for k, v in views[n].items() if k != "center"}
        rec.update({"center": [float(c) for c in views[n]["center"]], "camera": {k: camp[k] for k in camp if k not in ("azimuth_scores",)},
                    "pixel_share_of_gt_pixels": {k: stats[k]["pixel_share_of_gt_pixels"] for k in stats}, "mesh_sha256": prep["mesh_sha256"]})
        json.dump(rec, open(OUT + f"views/{n}.json", "w"), indent=1, default=float)
        print(f"render {n:34s} view {views[n]['az']:g}/{views[n]['el']:g} | px share MAG {stats['mag']['pixel_share_of_gt_pixels']:.3f} "
              f"Ours {stats['ours']['pixel_share_of_gt_pixels']:.3f} | tiles {camp['tiling_check']['share_px_other_face_tile_vs_half_tile']:.5f}"
              f" | GPU {camp['gpu_peak_gb']:.2f} GB | {camp['wall_s']:.0f}s", flush=True)
    return imgs, pws


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--group", required=True, choices=["outdoorA", "outdoorB", "interior", "indoor"])
    ap.add_argument("--redraw", action="store_true", help="rebuild the sheets from the saved panels / views (no rendering)")
    a = ap.parse_args()
    global OUT
    if a.group == "interior":                   # v4 option 4: barts / sestino_museum are interiors (hall, museum); without culling the
        OUT = OUT + "interior_cull/"            # near walls / roof hide the room and unobserved walls reveal it -> render them like
        R2.OUT = OUT                            # the MP3D levels (back faces culled, cutaway view); same rule for both methods
        R2.CULL["outdoor"] = True
    t0 = time.time()
    for d in ("panels", "sheets", "views", "checks", "viewpoints"):
        os.makedirs(OUT + d, exist_ok=True)
    dev = torch.device("cuda")
    allc = {R2.case_name(r): r for r in csv.DictReader(open(V4 + "v4_cases.csv"))}
    mm = {(x["case"], x["method"]): x for x in csv.DictReader(open(V4 + "f6v4_masks_manifest.csv"))}
    covs = {n: {"mag": float(mm[(n, "MAGICIAN (MAG-B)")]["official_cov20"]), "ours": float(mm[(n, "Ours (V3-T0 + D1)")]["official_cov20"])}
            for n in allc}
    if a.group == "indoor":
        sheets = {k: sorted([n for n, r in allc.items() if r["domain"] == "indoor" and r["scene"].rsplit("_L", 1)[0] in v],
                            key=lambda n: (allc[n]["scene"], int(allc[n]["start"]))) for k, v in INDOOR_SHEETS.items()}
        assert sum(len(v) for v in sheets.values()) == 24, sheets
    else:
        sheets = {f"outdoor_{s}" + ("_cull" if a.group == "interior" else ""): sorted([n for n, r in allc.items() if r["domain"] == "outdoor" and r["scene"] == s],
                                         key=lambda n: int(allc[n]["start"])) for s in GROUPS[a.group]}
    if a.redraw:
        from PIL import Image
        for sheet, names in sheets.items():
            views = {n: json.load(open(OUT + f"views/{n}.json")) for n in names}
            imgs = {n: {row: np.asarray(Image.open(OUT + f"panels/{n}__{row}.png").convert("RGB"), np.float64) / 255.0
                        for row, _ in R2.ROWS} for n in names}
            pws = {n: PH / views[n]["aspect"] for n in names}
            rows = {}
            for n in names:
                r = dict(allc[n])
                st = f"s{r['start']}" if r["domain"] == "outdoor" else f"S{r['start']}"
                r["_title"] = f"{_title(r)}\n{st} · {views[n]['az']:g}°/{views[n]['el']:g}°"
                rows[n] = r
            V3R.draw(names, rows, imgs, views, pws, PH, covs, OUT + f"sheets/{sheet}")
            print("redraw", sheet, flush=True)
        return
    for sheet, names in sheets.items():
        views = {}
        for n in names:
            r = allc[n]
            if r["domain"] == "outdoor":
                views[n] = plan_outdoor_grid(r, dev)
            else:
                views[n] = R2.plan_column(r, dev)
                views[n]["azimuth_rule"] = "v2/v3 indoor (unchanged): argmax #GT-render pixels observed by MAGICIAN or Ours, el 60"
            print(f"plan {n:34s} view {views[n]['az']:g}/{views[n]['el']:g} aspect {views[n]['aspect']:.3f}"
                  + (f" (v3 rule: {views[n]['v3_rule_view']})" if "v3_rule_view" in views[n] else ""), flush=True)
        if names and allc[names[0]]["domain"] == "outdoor":
            viewpoint_sheet(allc[names[0]]["scene"], allc[names[0]], views, dev)
        imgs, pws = render_cases(names, allc, views, dev)
        rows = {}
        for n in names:
            r = dict(allc[n])
            st = f"s{r['start']}" if r["domain"] == "outdoor" else f"S{r['start']}"
            r["_title"] = f"{_title(r)}\n{st} · {views[n]['az']:g}°/{views[n]['el']:g}°"
            rows[n] = r
        V3R.draw(names, rows, imgs, views, pws, PH, covs, OUT + f"sheets/{sheet}")
        print("sheet", sheet, f"{time.time() - t0:.0f}s", flush=True)
    print("done", a.group, f"{time.time() - t0:.0f}s", flush=True)


if __name__ == "__main__":
    main()
