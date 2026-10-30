"""EXP5 Task 2 -- shared constants and helpers (numpy only; importable from the magician and the seen2scene env).

Splits (your 10-26 decisions, T2.0_audit.md §7):
  EVAL_LEVELS   the 5 T0.12 MP3D val levels x 2 starts (final indoor evaluation; their HOUSES are never read for
                training or checkpoint selection)
  VALSEL_HOUSES the other 6 MP3D val houses (indoor checkpoint-selection data)
  TRAIN_HOUSES  the 61 MP3D official train houses (indoor training data)
Indoor V2 planning region: n_win = 5 windows (overlap 0.2) around the camera, i.e. side (1 + 4 x 0.8) x height x 0.99.
"""
import hashlib
import json
import os

import numpy as np

REPO = "/bsuscratch/jiahuizhang/projects/occupancy/seen2scene"
EXP5 = os.path.join(REPO, "experiments", "exp5_future_view")
T2 = os.path.join(EXP5, "task2")
W2 = os.path.join(REPO, "work_dirs", "exp5_future_view", "task2")
W0 = os.path.join(REPO, "work_dirs", "exp5_future_view", "task0")
MP = "/bsuhome/jiahuizhang/scratch/data/matterport3d/v1/scans"
SPLIT_DIR = os.path.join(W0, "download_logs", "official_raw")
S_RT = 7.0
N_WIN = 5
MIN_PANOS = 12
MOVES_INDOOR = 150

EVAL_LEVELS = [("Z6MFQCViBuw", 0, 6), ("Z6MFQCViBuw", 0, 17), ("oLBMNvg9in8", 1, 36), ("oLBMNvg9in8", 1, 78),
               ("QUCTc6BB5sX", 1, 34), ("QUCTc6BB5sX", 1, 135), ("2azQ1b91cZZ", 0, 60), ("2azQ1b91cZZ", 0, 179),
               ("TbHJrupSAjP", 1, 112), ("TbHJrupSAjP", 1, 114)]
EVAL_HOUSES = sorted({s for s, _, _ in EVAL_LEVELS})


def split(name):
    return [x.strip() for x in open(os.path.join(SPLIT_DIR, f"scenes_{name}.txt")) if x.strip()]


TRAIN_HOUSES = split("train")
VALSEL_HOUSES = [h for h in split("val") if h not in EVAL_HOUSES]
TEST_HOUSES = split("test")


def purpose_of(scan):
    """Which GT reads a house allows: 'train' (labels + training trajectories), 'valsel' (selection labels), 'eval'
    (simulator / evaluator only), 'sealed' (MP3D test: never read)."""
    if scan in TRAIN_HOUSES:
        return "train"
    if scan in VALSEL_HOUSES:
        return "valsel"
    if scan in EVAL_HOUSES:
        return "eval"
    return "sealed"


def assert_mesh_allowed(scan, purpose):
    p = purpose_of(scan)
    if p != purpose:
        raise RuntimeError(f"GT GUARD: house {scan} is '{p}', requested for '{purpose}'")


def seed_of(key):
    return int(hashlib.sha256(key.encode()).hexdigest()[:15], 16)


def zup_m_to_rt(p):
    p = np.asarray(p, np.float64)
    return S_RT * np.stack([p[..., 0], p[..., 2], -p[..., 1]], -1)


def rt_to_zup_m(p):
    p = np.asarray(p, np.float64) / S_RT
    return np.stack([p[..., 0], -p[..., 2], p[..., 1]], -1)


def region_box_m(card, center_rt, n_win=N_WIN):
    """mp3d_frame.region_box_m with n_win windows (Task 2: n_win = 5)."""
    box = np.asarray(card["box_m"])
    c = rt_to_zup_m(center_rt)[:2]
    half = 0.5 * (1 + (n_win - 1) * 0.8) * card["height_m"] * 0.99
    lo, hi = np.maximum(c - half, box[0, :2]), np.minimum(c + half, box[1, :2])
    return np.array([[lo[0], lo[1], box[0, 2]], [hi[0], hi[1], box[1, 2]]])


def card_path(key):
    """Task 2 cards (train / valsel levels) in W2; the eval levels keep their Task 0 cards."""
    p = os.path.join(W2, "mp3d", "scenes", f"{key}.json")
    return p if os.path.exists(p) else os.path.join(W0, "preprocessing", "scenes", f"{key}.json")


def load_card(key):
    return json.load(open(card_path(key)))
