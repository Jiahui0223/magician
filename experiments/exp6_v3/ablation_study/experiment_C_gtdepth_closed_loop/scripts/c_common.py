"""EXP6 V3 Ablation Experiment C -- shared, environment-free definitions (numpy + stdlib; imported by the magician-env
simulator side, the seen2scene-env planner children and the CPU analysis).

Groups
  C1  GT depth + DistP1 : the frozen Task 2 / Task 3 B2 planner with the K = 4 predicted future depths replaced by the
                          ONE true first-hit depth t* of every legal candidate (64 x 114 scoring rays, jitter 0); the
                          sparse 10 % P1 band_dil (B2 scoring map), the same DistP1 (scoring.new_set_from_depth with the
                          distance weight min(1, (z / z0)^2)), the same choice rule (indoor v2nv_planner.nv_choice,
                          outdoor m_score.choose), the same candidates, moves, records and coverage.
  C2  GT one-step gain   : the same loop; the score of a candidate = its TRUE one-step coverage gain under the frozen
                          evaluator of the dataset (MP3D: C_obs rule of T0.12-C / EXP5 S1; Macarons++: the monotone
                          coverage of t25_out_cov = the primary outdoor metric; the official MAGICIAN gain is logged),
                          then the same choice rule. Candidate-set one-step oracle greedy, not a trajectory optimum.
GT side channel: at step t the simulator writes <episode>/cq/cq_t<t>.npz = {t, legal_idx, tstar (C1) | gain (C2)} for
the CURRENT legal list only, and sends its path with the service record. The planner reads nothing else. The
simulator's own state (coverage, history, pose, RNG-independent legality) is advanced by the executed pick only; the GT
queries are stateless (asserted by a state hash before / after every query).
"""
import hashlib
import json
import os

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
EXPC = os.path.dirname(HERE)                                   # experiment_C_gtdepth_closed_loop
ABL = os.path.dirname(EXPC)                                    # ablation_study
E6 = os.path.dirname(ABL)                                      # exp6_v3
REPO = os.path.dirname(os.path.dirname(E6))
EXP5 = os.path.join(REPO, "experiments", "exp5_future_view")
T2 = os.path.join(EXP5, "task2")
T3S = os.path.join(EXP5, "task3_full_depth_p1", "scripts")
WC = os.path.join(REPO, "work_dirs", "exp6_v3", "ablation_study", "experiment_C")
WEVAL = os.path.join(WC, "eval")
WG0 = os.path.join(WC, "g0")
W6E = os.path.join(REPO, "work_dirs", "exp6_v3", "eval")                       # T0 / T1 (+ symlinked B0 / B2 / E2D)
T2E = os.path.join(REPO, "work_dirs", "exp5_future_view", "task2", "eval")     # MAG-G / MAG-B (Task 2)
W5A = os.path.join(REPO, "work_dirs", "exp5_future_view", "ablation_study")    # EXP5 A1 / A2 fixed states (oracle.npz)
WA = os.path.join(REPO, "work_dirs", "exp6_v3", "ablation_study")              # Experiment A cache
S2E = "/bsuscratch/jiahuizhang/envs/seen2scene"
MGE = "/bsuscratch/jiahuizhang/envs/magician"

GROUPS = ("C1", "C2")
LABEL = {"C1": "C1 GT depth + DistP1 (K=1)", "C2": "C2 GT one-step gain greedy"}
CFG_IN = json.load(open(os.path.join(T2, "configs", "eval_indoor_v1.json")))
CFG_OUT = json.load(open(os.path.join(T2, "configs", "eval_outdoor_v1.json")))
MOVES_IN = int(CFG_IN["budget"]["moves"])
MOVES_OUT = int(CFG_OUT["budget"]["moves"])
EPISODES_IN = [(s, int(lv), int(k)) for s, lv, k in CFG_IN["episodes"]]
SCENES_OUT = {"test": list(CFG_OUT["scenes"]["test"]), "dev": list(CFG_OUT["scenes"]["dev"])}
STARTS_OUT = [int(x) for x in CFG_OUT["starts"]]
SPLIT_OF = {s: sp for sp, v in SCENES_OUT.items() for s in v}
N_WIN = 5                      # Task 2 planning region (t2_common.N_WIN); used only for the region diagnostic
TIE_TOL = 1e-4                 # oracle-gain tie tolerance (EXP5 protocol §3.3)
CQ_KEYS = {"C1": ("t", "legal_idx", "tstar"), "C2": ("t", "legal_idx", "gain")}


def level_key(scan, lv):
    return f"{scan}_L{lv}"


def tag_in(scan, lv, start):
    return f"{level_key(scan, lv)}_S{start}"


def tag_out(grp, scene, start):
    return f"{grp}_{scene}_s{start}_r0"


def indoor_dir(grp, tag, root=None):
    return os.path.join(root or os.path.join(WEVAL, "closed_loop", "indoor"), grp, tag)


def outdoor_root(grp, root=None):
    return os.path.join(root or os.path.join(WEVAL, "closed_loop", "outdoor"), grp)


def outdoor_metrics_dir(grp, root=None):
    return os.path.join(root or os.path.join(WEVAL, "metrics", "outdoor"), grp)


def cq_name(t):
    return f"cq_t{int(t)}.npz"


def write_cq(path, grp, t, legal_idx, **arrays):
    """Atomic write of the GT side-channel file; only the keys of CQ_KEYS[grp] are allowed."""
    d = {"t": np.asarray(int(t)), "legal_idx": np.asarray(legal_idx, np.int64).reshape(-1, 5)}
    d.update(arrays)
    assert set(d) == set(CQ_KEYS[grp]), (sorted(d), CQ_KEYS[grp])
    tmp = path + f".tmp{os.getpid()}.npz"
    np.savez(tmp, **d)
    os.replace(tmp, path)
    return path


def read_cq(path, grp, t, legal_idx):
    z = np.load(path, allow_pickle=False)
    assert set(z.files) == set(CQ_KEYS[grp]), (sorted(z.files), grp)
    assert int(z["t"]) == int(t), (int(z["t"]), t)
    assert np.array_equal(np.asarray(z["legal_idx"]), np.asarray(legal_idx, np.int64).reshape(-1, 5)), "cq legal_idx != record"
    return {k: z[k] for k in z.files}


def sha_arrays(*arrays):
    h = hashlib.sha256()
    for a in arrays:
        a = np.ascontiguousarray(np.asarray(a))
        h.update(str(a.dtype).encode() + str(a.shape).encode() + a.tobytes())
    return h.hexdigest()


def argmax_lowest(x):
    s = np.asarray(x, np.float64)
    return int(np.flatnonzero(s == s.max())[0])


def oracle_summary(gain, cand, pick, tol=TIE_TOL):
    """Per-step one-step oracle summary of a pick inside a candidate set (gains as coverage fraction)."""
    g = np.asarray(gain, np.float64)
    C = [int(c) for c in cand]
    gc = g[C]
    gmax = float(gc.max()) if len(gc) else 0.0
    best = int(C[int(np.flatnonzero(gc == gc.max())[0])]) if len(gc) else None
    gp = float(g[pick]) if pick is not None else None
    return {"gstar": gmax, "best": best, "gain_pick": gp,
            "regret": (gmax - gp) if gp is not None else None,
            "top1": (gp >= gmax - tol) if gp is not None else None, "n_cand": len(C),
            "n_best": int((gc >= gmax - tol).sum()) if len(gc) else 0}


def region_box_xy(card, center_rt, n_win=N_WIN):
    """Task 2 planning region (n_win windows around the refresh camera), horizontal box in z-up metres."""
    box = np.asarray(card["box_m"])
    c = np.asarray(center_rt, np.float64) / card["s_rt"]
    c = np.array([c[0], -c[2]])
    half = 0.5 * (1 + (n_win - 1) * 0.8) * card["height_m"] * 0.99
    return np.maximum(c - half, box[0, :2]), np.minimum(c + half, box[1, :2])
