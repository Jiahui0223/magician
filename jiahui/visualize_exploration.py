"""
Visualize MAGICIAN step-by-step exploration renders.
Reads the per-step RGB PNGs cached under test_memory_<gpu>/training/<traj>/imgs/
and writes, per (scene, traj), to work_dir/vis/:
  - <scene>_traj<t>.gif        : animated exploration
  - <scene>_traj<t>_grid.png   : contact sheet (every Nth step)

Usage:
  python visualize_exploration.py --scene redeemer --traj 0
  python visualize_exploration.py --scene pantheon  --traj 2 --gpu 0
"""
import os, glob, argparse
import imageio.v2 as imageio
from PIL import Image, ImageDraw

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))  # repo root (this file lives in jiahui/)
OUT  = os.path.join(ROOT, "work_dir", "vis")


def step_pngs(scene, traj, gpu):
    d = os.path.join(ROOT, "data", "Macarons++", scene,
                     f"test_memory_{gpu}", "training", str(traj), "imgs")
    fs = glob.glob(os.path.join(d, "*.png"))
    fs = [f for f in fs if os.path.basename(f)[:-4].isdigit()]
    fs.sort(key=lambda p: int(os.path.basename(p)[:-4]))
    return fs


def make_gif(pngs, out_path, max_w=360, fps=12):
    frames = []
    for p in pngs:
        im = Image.open(p).convert("RGB")
        if im.width > max_w:
            im = im.resize((max_w, int(im.height * max_w / im.width)))
        frames.append(im)
    imageio.mimsave(out_path, frames, duration=1.0 / fps, loop=0)
    return len(frames)


def make_grid(pngs, out_path, every=10, cols=6, cell_w=240):
    sel = pngs[::every]
    if pngs[-1] not in sel:
        sel.append(pngs[-1])
    ims = []
    for p in sel:
        im = Image.open(p).convert("RGB")
        im = im.resize((cell_w, int(im.height * cell_w / im.width)))
        d = ImageDraw.Draw(im)
        d.text((5, 5), f"step {os.path.basename(p)[:-4]}", fill=(255, 255, 0))
        ims.append(im)
    cw, ch = ims[0].size
    rows = (len(ims) + cols - 1) // cols
    sheet = Image.new("RGB", (cols * cw, rows * ch), (20, 20, 20))
    for i, im in enumerate(ims):
        sheet.paste(im, ((i % cols) * cw, (i // cols) * ch))
    sheet.save(out_path)
    return len(ims)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--scene", required=True)
    ap.add_argument("--traj", type=int, default=0)
    ap.add_argument("--gpu", type=int, default=0, help="test_memory_<gpu>")
    ap.add_argument("--every", type=int, default=10, help="grid: every Nth step (use 1 for ALL steps)")
    ap.add_argument("--cols", type=int, default=10, help="grid columns")
    args = ap.parse_args()

    os.makedirs(OUT, exist_ok=True)
    pngs = step_pngs(args.scene, args.traj, args.gpu)
    if not pngs:
        print(f"[!] no PNGs for {args.scene}/traj{args.traj} (test_memory_{args.gpu})")
        return
    tag = f"{args.scene}_traj{args.traj}"
    g = make_gif(pngs, os.path.join(OUT, tag + ".gif"))
    n = make_grid(pngs, os.path.join(OUT, tag + "_grid.png"), every=args.every, cols=args.cols)
    print(f"{tag}: {len(pngs)} steps -> {tag}.gif ({g} frames), {tag}_grid.png ({n} tiles)")


if __name__ == "__main__":
    main()
