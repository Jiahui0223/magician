"""
Compare all trajectories of a scene in ONE montage: each row = one trajectory,
sampled evenly across its steps, labeled with traj index + max coverage (from LMDB).
Output: work_dir/vis/<scene>_compare.png

Usage: python compare_trajectories.py --scene pantheon
"""
import os, glob, argparse, pickle
import lmdb
from PIL import Image, ImageDraw

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))  # repo root (this file lives in jiahui/)
OUT  = os.path.join(ROOT, "work_dir", "vis")
LMDB = os.path.join(ROOT, "results", "scene_exploration", "magician_lmdb")


def max_cov(scene, traj):
    try:
        env = lmdb.open(LMDB, readonly=True, lock=False)
        with env.begin() as t:
            raw = t.get(f"{scene}/{traj}".encode())
        env.close()
        if raw:
            cov = pickle.loads(raw).get("coverage")
            if cov:
                return max(cov[:100])
    except Exception:
        pass
    return None


def row(scene, traj, gpu, ncols, cell_w):
    d = os.path.join(ROOT, "data", "Macarons++", scene,
                     f"test_memory_{gpu}", "training", str(traj), "imgs")
    fs = [f for f in glob.glob(os.path.join(d, "*.png")) if os.path.basename(f)[:-4].isdigit()]
    fs.sort(key=lambda p: int(os.path.basename(p)[:-4]))
    if not fs:
        return None
    idx = [round(i * (len(fs) - 1) / (ncols - 1)) for i in range(ncols)]
    ims = []
    for j in idx:
        im = Image.open(fs[j]).convert("RGB").resize((cell_w, int(Image.open(fs[j]).height * cell_w / Image.open(fs[j]).width)))
        ImageDraw.Draw(im).text((4, 4), f"#{os.path.basename(fs[j])[:-4]}", fill=(255, 255, 0))
        ims.append(im)
    return ims


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--scene", required=True)
    ap.add_argument("--trajs", type=int, nargs="+", default=[0, 1, 2, 3, 4])
    ap.add_argument("--gpu", type=int, default=0)
    ap.add_argument("--ncols", type=int, default=8, help="sampled frames per trajectory")
    ap.add_argument("--cell_w", type=int, default=200)
    args = ap.parse_args()
    os.makedirs(OUT, exist_ok=True)

    label_w = 130
    rows = []
    for tr in args.trajs:
        ims = row(args.scene, tr, args.gpu, args.ncols, args.cell_w)
        if ims is None:
            print(f"[skip] traj{tr}: no imgs"); continue
        rows.append((tr, ims))
    if not rows:
        print("no data"); return

    cw, ch = rows[0][1][0].size
    W = label_w + args.ncols * cw
    H = len(rows) * ch
    sheet = Image.new("RGB", (W, H), (15, 15, 15))
    drw = ImageDraw.Draw(sheet)
    for r, (tr, ims) in enumerate(rows):
        y = r * ch
        mc = max_cov(args.scene, tr)
        drw.text((8, y + ch // 2 - 8), f"traj {tr}\ncov={mc:.2f}" if mc is not None else f"traj {tr}",
                 fill=(0, 255, 0))
        for c, im in enumerate(ims):
            sheet.paste(im, (label_w + c * cw, y))
    out = os.path.join(OUT, f"{args.scene}_compare.png")
    sheet.save(out)
    print(f"-> {out}  ({len(rows)} trajectories x {args.ncols} frames)")


if __name__ == "__main__":
    main()
