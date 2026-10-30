"""EXP5 Task 0 / T0.9 -- one official MAGICIAN run (released-code MP3D protocol; magician_comparison protocol v1.1 §8).
The official entry point test_magician_planning.py is executed UNCHANGED (runpy, same process) from the official clone
e697009 with our config copies (configs/t09_official, absolute paths). One (scene, repeat) per process.
  - per-run data working copy MC/official_reproduction/<run>/data/<scene>/ = symlinks to the raw scene files, so the
    tester's test_memory_0/ frames never land in the raw data directory;
  - LMDB output MC/official_reproduction/<run>/lmdb; run record run.json (wall s, peak GPU memory, GPU, config sha256).
Usage (magician env, env -u LD_PRELOAD, CUDA_VISIBLE_DEVICES = one GPU):
  python t09_run.py --scene GdvgFV5R1Z5 --rep 0 --mode formal|smoke
"""
import argparse
import hashlib
import json
import os
import runpy
import socket
import sys
import time

HERE = os.path.dirname(os.path.abspath(__file__))
REPO = "/bsuscratch/jiahuizhang/projects/occupancy/seen2scene"
MC = os.path.join(REPO, "work_dirs", "exp5_future_view", "task0", "magician_comparison")
EXT = os.path.join(MC, "external", "MAGICIAN")
CFG = os.path.join(REPO, "experiments", "exp5_future_view", "task0", "magician_comparison", "configs", "t09_official")
RAW = "/bsuhome/jiahuizhang/scratch/data/mp3d_magician/mp3d_magician"
SCENES = ("GdvgFV5R1Z5", "gZ6f7yhEvPG", "HxpKQynjfin", "pLe4wQe7qrG", "YmJkqBEsHnH")


def sha(p):
    return hashlib.sha256(open(p, "rb").read()).hexdigest()


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--scene", required=True, choices=SCENES)
    ap.add_argument("--rep", type=int, required=True)
    ap.add_argument("--mode", choices=["formal", "smoke"], required=True)
    a = ap.parse_args()
    run = f"{a.mode}_{a.scene}_rep{a.rep}"
    od = os.path.join(MC, "official_reproduction", run)
    if os.path.exists(os.path.join(od, "run.json")):
        print("done already", run)
        return
    dd = os.path.join(od, "data", a.scene)
    os.makedirs(dd, exist_ok=True)
    for f in os.listdir(os.path.join(RAW, a.scene)):
        dst = os.path.join(dd, f)
        if not os.path.lexists(dst):
            os.symlink(os.path.join(RAW, a.scene, f), dst)
    tmpl = json.load(open(os.path.join(CFG, "test_mp3d_template.json")))
    tmpl.update(dataset_path=os.path.join(od, "data"), test_scenes=[a.scene], lmdb_dir_name=os.path.join(od, "lmdb"),
                params_name=os.path.join(CFG, f"macarons_mp3d_{a.mode}.json"))
    tc = os.path.join(od, "test_config.json")
    json.dump(tmpl, open(tc, "w"), indent=2)
    import torch
    t0 = time.time()
    os.chdir(EXT)
    sys.path.insert(0, EXT)
    sys.argv = ["test_magician_planning.py", "-c", tc]
    runpy.run_path(os.path.join(EXT, "test_magician_planning.py"), run_name="__main__")
    rec = {"run": run, "scene": a.scene, "rep": a.rep, "mode": a.mode, "wall_s": time.time() - t0,
           "peak_gpu_mem_gb": torch.cuda.max_memory_allocated() / 2 ** 30, "gpu": torch.cuda.get_device_name(0),
           "host": socket.gethostname(), "time": time.strftime("%Y-%m-%d %H:%M:%S"),
           "test_config_sha256": sha(tc), "macarons_config_sha256": sha(tmpl["params_name"]),
           "official_commit": "e697009b3f9360adfe65e8db1e5c27421158e388",
           "weights_sha256": {k: sha(os.path.join(EXT, "weights", "macarons", k)) for k in ("pretrained_macarons.pth", "trained_macarons.pth")}}
    json.dump(rec, open(os.path.join(od, "run.json"), "w"), indent=1)
    print("RUN DONE", json.dumps(rec), flush=True)


if __name__ == "__main__":
    main()
