"""EXP9 Fig. 6 v4 -- overview images (CPU): stack the per-scene / per-house contact sheets vertically (white gap), left-aligned, at
half resolution, so all candidates can be browsed in one image.  Output fig/fig6/fig6_v4/ALL_unseen6_outdoor.png, ALL_HO12_indoor.png."""
from PIL import Image

D = "/bsuscratch/jiahuizhang/projects/occupancy/seen2scene/experiments/cvpr_results/fig/fig6/fig6_v4/"
SETS = {"ALL_unseen6_outdoor": [f"outdoor_{s}" for s in ("alhambra", "neuschwanstein", "eiffel", "bridge", "barts", "sestino_museum")]
        + ["../interior_cull/sheets/outdoor_barts_cull", "../interior_cull/sheets/outdoor_sestino_museum_cull"],
        "ALL_HO12_indoor": ["indoor_1", "indoor_2", "indoor_3"]}
SCALE, GAP = 0.5, 60
Image.MAX_IMAGE_PIXELS = None

for out, names in SETS.items():
    ims = []
    for n in names:
        im = Image.open(D + f"sheets/{n}.png").convert("RGB")
        ims.append(im.resize((int(im.width * SCALE), int(im.height * SCALE)), Image.LANCZOS))
    W = max(i.width for i in ims)
    H = sum(i.height for i in ims) + GAP * (len(ims) - 1)
    canvas = Image.new("RGB", (W, H), "white")
    y = 0
    for i in ims:
        canvas.paste(i, (0, y))
        y += i.height + GAP
    canvas.save(D + out + ".png", optimize=True)
    print(out, canvas.size)
