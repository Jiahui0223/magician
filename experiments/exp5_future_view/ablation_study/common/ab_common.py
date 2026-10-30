"""EXP5 Ablation A1 / A2 -- shared constants and helpers (imported by both envs; numpy / json only at import time).

Fixed states come from the stored Task 2 closed-loop service records (never re-simulated for the planner side); the
planner-side state of a stored episode at step t is determined by its executed pose history, so the state id is the
sha256 of (dataset, scene / level, start, executed pose indices 0..t).
"""
import hashlib
import json
import os
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
ABL = os.path.dirname(HERE)
EXP5 = os.path.dirname(ABL)
REPO = os.path.dirname(os.path.dirname(EXP5))
W = os.path.join(REPO, "work_dirs", "exp5_future_view", "ablation_study")
T2 = os.path.join(EXP5, "task2")
T2W = os.path.join(REPO, "work_dirs", "exp5_future_view", "task2")
CL_IN = os.path.join(T2W, "eval", "indoor", "closed_loop")
CL_OUT = os.path.join(T2W, "eval", "outdoor", "closed_loop")

_CFG_IN = json.load(open(os.path.join(T2, "configs", "eval_indoor_v1.json")))
_CFG_OUT = json.load(open(os.path.join(T2, "configs", "eval_outdoor_v1.json")))
MODELS = {g: {"path": _CFG_IN["groups"][g]["checkpoint"], "sha256": _CFG_IN["groups"][g]["sha256"]} for g in ("V2O", "V2M")}
assert all(MODELS[g]["sha256"] == _CFG_OUT["groups"][g]["sha256"] for g in MODELS)
SOURCES = ("V2O", "V2M", "MAG-G", "MAG-B")
EP_IN = [(s, int(lv), int(st)) for s, lv, st in _CFG_IN["episodes"]]
SCENES_OUT = {"test": list(_CFG_OUT["scenes"]["test"]), "dev": list(_CFG_OUT["scenes"]["dev"])}
STARTS_OUT = list(_CFG_OUT["starts"])
SPLIT_OF = {s: sp for sp, v in SCENES_OUT.items() for s in v}

# state times (protocol §2.2): the brief's times (all refresh steps, encoder input current) + the same times + 5
# (mid-block, encoder input 5 moves stale, as in the closed loop)
T_PRIMARY = {"indoor": (0, 20, 50, 100, 140), "outdoor": (0, 10, 20, 50, 80)}
T_STATES = {d: tuple(sorted(set(v) | {t + 5 for t in v})) for d, v in T_PRIMARY.items()}
TIE_TOL = 1e-4          # oracle gains (coverage fraction) closer than this count as tied (protocol §3.4)
K_MAIN = 4
K_ABL = (1, 4, 8, 16)
N_SEEDSETS = 4


def level_key(scan, lv):
    return f"{scan}_L{lv}"


def ep_tag_in(scan, lv, start):
    return f"{level_key(scan, lv)}_S{start}"


def episodes(dataset):
    if dataset == "indoor":
        return [{"dataset": "indoor", "scene": level_key(s, lv), "scan": s, "level": lv, "start": st,
                 "tag": ep_tag_in(s, lv, st), "split": "mp3d_val"} for s, lv, st in EP_IN]
    return [{"dataset": "outdoor", "scene": s, "start": k, "tag": f"{s}_s{k}", "split": SPLIT_OF[s]}
            for sp in ("test", "dev") for s in SCENES_OUT[sp] for k in STARTS_OUT]


def svc_path(dataset, source, ep, t):
    if dataset == "indoor":
        return os.path.join(CL_IN, source, ep["tag"], "svc", f"svc_t{t}.npz")
    tg = f"{source}_{ep['scene']}_s{ep['start']}_r0"
    return os.path.join(CL_OUT, source, "svc", tg, f"svc_{ep['scene']}_s{ep['start']}_t{t}.npz")


def outdoor_episode_json(source, ep):
    return os.path.join(CL_OUT, source, "episodes", f"ep_{source}_{ep['scene']}_s{ep['start']}_r0.json")


def state_id(dataset, scene, start, hist):
    h = hashlib.sha256(f"{dataset}|{scene}|{start}|".encode())
    h.update(json.dumps([[int(x) for x in p] for p in hist]).encode())
    return h.hexdigest()[:16]


def sha_seed(key):
    return int(hashlib.sha256(key.encode()).hexdigest()[:15], 16)


def candidate_set(dataset, rec):
    """Indices (into the stored legal list) of the candidates the Task 2 V2 choice rule ranks: indoor = MAGICIAN's
    prefer-unvisited set (v2nv_planner.nv_choice), outdoor = the legal list itself (the M-protocol environment applies
    get_valid_neighbors before the GT segment test)."""
    legal = [tuple(int(x) for x in r) for r in rec["legal_idx"]]
    if dataset == "outdoor":
        return list(range(len(legal)))
    hist = {tuple(int(x) for x in r) for r in rec["executed_pose_idx_history"][:-1]}
    return [j for j, p in enumerate(legal) if p not in hist] or list(range(len(legal)))


def state_dir(dataset, sid):
    return os.path.join(W, "states", dataset, sid)


def load_manifest(dataset):
    return json.load(open(os.path.join(W, "manifests", f"states_{dataset}.json")))


def add_paths():
    """Planner-side import paths (task1 / task0 scripts, EXP3b, task2) in the order of the Task 2 planners."""
    for p in (os.path.join(EXP5, "task0", "magician_comparison", "adapters"), os.path.join(EXP5, "task0", "scripts"),
              os.path.join(EXP5, "task1", "scripts"), os.path.join(EXP5, "task2", "scripts"), HERE):
        if p not in sys.path:
            sys.path.insert(0, p)
