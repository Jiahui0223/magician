"""EXP6 V4 -- paths and config loading (seen2scene env; importable from the magician env for paths only)."""
import hashlib
import json
import os

HERE = os.path.dirname(os.path.abspath(__file__))
E4 = os.path.dirname(HERE)
EXPS = os.path.dirname(E4)
REPO = os.path.dirname(EXPS)
EXP5 = os.path.join(EXPS, "exp5_future_view")
E31 = os.path.join(EXPS, "exp6_v3.1")
CFG = os.path.join(E4, "configs")
W4 = os.path.join(REPO, "work_dirs", "exp6_v4")
W31 = os.path.join(REPO, "work_dirs", "exp6_v3_1")
OUTD = os.path.join(REPO, "work_dirs", "exp5_future_view", "task3_full_depth_p1", "e2", "out_views_d100")
SEEN2SCENE_PYTHON = "/bsuscratch/jiahuizhang/envs/seen2scene/bin/python"
MAGICIAN_PYTHON = "/bsuscratch/jiahuizhang/envs/magician/bin/python"


def load_cfg(name):
    return json.load(open(os.path.join(CFG, name)))


def sha256_file(p):
    h = hashlib.sha256()
    with open(p, "rb") as f:
        for b in iter(lambda: f.read(1 << 20), b""):
            h.update(b)
    return h.hexdigest()
