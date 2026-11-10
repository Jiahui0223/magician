"""CVPR overnight tables / figures -- shared library (CPU only; no closed loop, no GT).

Single source of truth = the canonical per-episode records work_dirs/cvpr_results/table1/episodes/<method>/<scene>_s<k>.json
(written by table1/scripts/import_episodes.py only after the official-coverage replay t25_out_cov.py reproduced the stored
curve bitwise). Nothing here smooths, cummax-es or imputes a curve: a method x scene needs all 5 official starts to count as a
complete scene; incomplete scenes are reported with n_completed / n_expected and are not used in scene-equal means.

Metrics (frozen): C[0..100] = MAGICIAN official coverage; AUC@T = trapezoid(C[0..T], dx = 1) / T; Cov@T = C[T]; Final = C[100];
scene value = mean over the 5 starts; group value = unweighted mean over the scenes of the group (scene-equal).
Uncertainty: paired scene-level bootstrap (resampling unit = scene, 10,000 draws, seed 0, percentile 95 % interval)."""
import csv
import glob
import hashlib
import json
import os

import numpy as np
import yaml

REPO = "/bsuscratch/jiahuizhang/projects/occupancy/seen2scene"
CV = os.path.join(REPO, "experiments", "cvpr_results")
T1 = os.path.join(CV, "table1")
EPI = os.path.join(REPO, "work_dirs", "cvpr_results", "table1", "episodes")
T1W = os.path.join(REPO, "work_dirs", "cvpr_results", "table1", "g3")
OV = os.path.join(REPO, "work_dirs", "cvpr_results", "overnight_20261110")
FIG, MAN, REP = os.path.join(CV, "fig"), os.path.join(CV, "manifest"), os.path.join(CV, "report")
N_START, MOVES = 5, 100
PROTOCOL_ID = "table1-protocol-v1.0"
PROTOCOL_DOC = ("frozen M-protocol EpisodeSim (exp3b m_t6_sim), Macarons++ official settings grid and the 5 official start poses, "
                "C0 + 100 moves = 101 coverage values, legal actions = MAGICIAN get_valid_neighbors (unvisited first) + GT segment test in the "
                "simulator only, MAGICIAN official coverage (eps = 1.0 runtime units), replay-verified by task2 t25_out_cov.py")
T_PAPER = (20, 50, 100)                                   # pre-registered paper columns (plan section 3): AUC@20 / AUC@50 / AUC@100 / Cov@20 / Final
T_FULL = (20, 34, 45, 50, 60, 78, 80, 90, 100)
T0_SHA = "7ed0ddd72bb91d1e8c6563f3192201c24a49fe862b49d2387ea8770915145ab6"
T1_SHA = "639c9d3ae3f3c069dc0399dc4cdd44102be2bf20e70d666a6a529c6affc62bdb"
MAC_SHA = "34c5247f13bedda08bb57e7e7e7a5baba53af6da7faf3d5f6bb57a27c2dbb59c"
NOISE_KEY = "exp5-V2|{scene}|s{start}|rep0|t{t_r}|k{k}"   # task1/configs/closed_loop_v1.json seed_key; t_r = last R10 refresh step
_trapz = getattr(np, "trapezoid", None) or np.trapz

# frozen scene order = MAGICIAN paper Tab. 6 / 7 order without St. Sofia (paper_tables/config.yaml scene_order)
SCENE_ORDER = ["dunnottar", "colosseum", "bannerman", "pantheon", "redeemer", "liberty", "pisa", "fushimi", "alhambra", "neuschwanstein",
               "eiffel", "bridge", "barts", "sestino_museum"]

# method table: id -> display / role / planner facts. `table1` = row of Table 1 (plan order); status BLOCKED = no runnable implementation.
METHODS = {
    "RANDOM": dict(display="Random Walk", table1=1, kind="baseline", source="re-evaluated, adapted", K="", depth="", score="uniform over the simulator legal list (seeded, seed 0)", ckpt=""),
    "SCONE": dict(display="SCONE", table1=2, kind="baseline", source="BLOCKED", K="", depth="", score="", ckpt="", blocked="no scene-level SCONE planner for Macarons++ in the repositories (only the ShapeNet object demo: MACARONS/demos/scone_demo.py, configs/scone, weights/scone); MAGICIAN's SCONE numbers exist only as paper-reported values"),
    "MACARONS": dict(display="MACARONS", table1=3, kind="baseline", source="re-evaluated, adapted", K="", depth="1", score="MACARONS coverage-gain NBV over the simulator legal list (EXP3b adapter-repaired tester, seed 0)", ckpt=MAC_SHA),
    "FISHERRF": dict(display="FisherRF", table1=4, kind="baseline", source="BLOCKED", K="", depth="", score="", ckpt="", blocked="no FisherRF code, weights or cached episodes in the repositories; it needs an online 3D Gaussian-splatting model per step and has no adapter for the frozen simulator"),
    "MAG-B": dict(display="MAGICIAN (beam 10x10)", table1=5, kind="baseline", source="re-evaluated", K="", depth="beam 10x10", score="MAGICIAN official core e697009 (beam search over the simulator legal list)", ckpt=MAC_SHA),
    "V3-T0-D1": dict(display="Ours (V3-T0 + D1)", table1=6, kind="ours", source="re-evaluated", K=4, depth=2, score="value_cov two-step union (D1, lambda 0)", ckpt=T0_SHA),
    "V3-T0-K4D1": dict(display="V3-T0, K = 4, depth 1", table1=0, kind="table3", source="re-evaluated", K=4, depth=1, score="value_cov one-step (mean of K samples)", ckpt=T0_SHA),
    "V3-T0-K1D2": dict(display="V3-T0, K = 1, depth 2", table1=0, kind="table3", source="re-evaluated", K=1, depth=2, score="value_cov two-step union, single sample (noise map 0)", ckpt=T0_SHA),
    "V3-T0-K1D1": dict(display="V3-T0, K = 1, depth 1", table1=0, kind="table3", source="re-evaluated", K=1, depth=1, score="value_cov one-step, single sample (noise map 0)", ckpt=T0_SHA),
    "MAG-G": dict(display="MAGICIAN (greedy)", table1=0, kind="reference", source="re-evaluated", K="", depth="beam 1x1", score="MAGICIAN official core e697009, beam 1x1", ckpt=MAC_SHA),
    "V3-D1-H3-NORISK": dict(display="V3-T1 + D1 (cross-attention control)", table1=0, kind="reference", source="re-evaluated", K=4, depth=2, score="value_cov two-step union (D1, lambda 0)", ckpt=T1_SHA),
}
for _d in (2, 1):
    for _j in (1, 2, 3):
        METHODS[f"V3-T0-K1D{_d}-N{_j}"] = dict(display=f"V3-T0, K = 1, depth {_d}, noise map {_j}", table1=0, kind="table3_noise", source="re-evaluated", K=1, depth=_d,
                                              score=f"value_cov {'two-step union' if _d == 2 else 'one-step'}, single sample (noise map {_j})", ckpt=T0_SHA, noise=_j)
TABLE1 = [m for m, d in sorted(METHODS.items(), key=lambda x: x[1]["table1"]) if d["table1"]]
BLOCKED = [m for m, d in METHODS.items() if d.get("blocked")]


def registries():
    sr = yaml.safe_load(open(os.path.join(T1, "SCENE_REGISTRY.yaml")))
    sc = {s["local_id"]: s for s in sr["scenes"]}
    assert set(SCENE_ORDER) == {k for k, v in sc.items() if (v.get("runnable") or {}).get("overall")}, "scene list differs from the registry"
    return sr, sc


_SR, SCENES = registries()
SPLIT = {s: SCENES[s]["split"] for s in SCENE_ORDER}
NAME = {s: SCENES[s]["official_name"] for s in SCENE_ORDER}
GROUPS = {"Train-6": [s for s in SCENE_ORDER if SPLIT[s] == "train"], "Dev-5": [s for s in SCENE_ORDER if SPLIT[s] == "dev"],
          "Test-3": [s for s in SCENE_ORDER if SPLIT[s] == "test"]}
GROUPS["Dev+Test-8"] = GROUPS["Dev-5"] + GROUPS["Test-3"]
GROUPS["All-14"] = list(SCENE_ORDER)
assert [len(GROUPS[g]) for g in ("Train-6", "Dev-5", "Test-3", "Dev+Test-8", "All-14")] == [6, 5, 3, 8, 14]


def auc(c, T):
    c = np.asarray(c, np.float64)
    return float(_trapz(c[:T + 1], dx=1.0) / T)


def cov(c, T):
    return float(c[T])


def metric(c, name):
    """name: 'AUC@T' | 'Cov@T' | 'Final'."""
    if name == "Final":
        return float(c[MOVES])
    k, T = name.split("@")
    return auc(c, int(T)) if k == "AUC" else cov(c, int(T))


def sha_curve(c):
    return hashlib.sha256(np.asarray(c, np.float64).tobytes()).hexdigest()


def sha_file(p):
    h = hashlib.sha256()
    with open(p, "rb") as f:
        for b in iter(lambda: f.read(1 << 20), b""):
            h.update(b)
    return h.hexdigest()


def load(method):
    """-> {(scene, start): canonical record} (no validation here; see check_record)."""
    out = {}
    for f in sorted(glob.glob(os.path.join(EPI, method, "*.json"))):
        r = json.load(open(f))
        r["_path"] = os.path.relpath(f, REPO)
        out[(r["scene"], int(r["start"]))] = r
    return out


_MAN = None


def load_manifest(method):
    """COMPLETE rows of manifest/episodes.csv -> {(scene, start): record}. This file is the immutable intermediate every table and
    figure is generated from (build_manifest.py writes it after validating each canonical record against its episode file and its
    official-coverage replay); curves are stored with repr(float), so the round trip is exact."""
    global _MAN
    if _MAN is None:
        _MAN = {}
        with open(os.path.join(MAN, "episodes.csv"), newline="") as f:
            for r in csv.DictReader(f):
                if r["run_status"] == "COMPLETE":
                    _MAN.setdefault(r["method_id"], {})[(r["scene_id"], int(r["start_id"]))] = {
                        "model": r["method_id"], "scene": r["scene_id"], "start": int(r["start_id"]), "split": r["split"],
                        "coverage_official": [float(r[f"c{t}"]) for t in range(MOVES + 1)], "episode_file": r["trajectory_path"], "metrics_file": r["metrics_path"],
                        "ckpt_sha256": r["checkpoint_sha"], "_path": r["source_path"], "executed_keys": [None] * (MOVES + 1)}
    return dict(_MAN.get(method, {}))


def check_record(r, deep=True):
    """Returns (ok, list of problems). deep: also open the episode / replay metrics files."""
    bad = []
    c = np.asarray(r["coverage_official"], np.float64)
    if c.shape != (MOVES + 1,):
        bad.append(f"curve length {c.shape}")
    if not np.all(np.isfinite(c)):
        bad.append("non-finite coverage")
    elif c.min() < 0 or c.max() > 1.0 + 1e-9:
        bad.append(f"coverage outside [0, 1]: {c.min():.4f}..{c.max():.4f}")
    if len(r.get("executed_keys", [])) != MOVES + 1:
        bad.append("executed_keys != 101")
    if r["scene"] not in SPLIT or not 0 <= int(r["start"]) < N_START:
        bad.append("scene / start outside the frozen grid")
    elif r["split"] != SPLIT[r["scene"]]:
        bad.append("split label differs from the registry")
    info = {"replay_equal": None, "replay_hash": None}
    if deep:
        try:
            ep = json.load(open(os.path.join(REPO, r["episode_file"])))
            mt = json.load(open(os.path.join(REPO, r["metrics_file"])))
            if ep["coverage"] != r["coverage_official"]:
                bad.append("episode file curve != canonical curve")
            if ep["executed_keys"] != r["executed_keys"]:
                bad.append("episode executed_keys != canonical")
            if ep.get("moves") != MOVES or ep["scene"] != r["scene"] or int(ep["start"]) != int(r["start"]):
                bad.append("episode header mismatch")
            eq = bool(mt.get("official_equal_stored")) and mt["cov_official"] == r["coverage_official"]
            info.update(replay_equal=eq, replay_hash=sha_curve(mt["cov_official"]), ep=ep)
            if not eq:
                bad.append("official replay != stored curve")
        except Exception as e:                                     # missing source file etc.
            bad.append(f"source files: {type(e).__name__}: {e}")
    return not bad, bad, info


def scene_table(E, names):
    """E {(scene, start): record}; -> ({scene: {metric: mean over the 5 starts}} for complete scenes, n_episodes)."""
    out = {}
    for s in SCENE_ORDER:
        ks = [(s, k) for k in range(N_START) if (s, k) in E]
        if len(ks) == N_START:
            out[s] = {n: float(np.mean([metric(E[k]["coverage_official"], n) for k in ks])) for n in names}
    return out


def group_mean(tab, scenes, name):
    done = [s for s in scenes if s in tab]
    return (float(np.mean([tab[s][name] for s in done])) if done else None), len(done)


def boot_ci(v, n=10000, seed=0):
    v = np.asarray(v, np.float64)
    if len(v) == 0:
        return None, None, None
    rng = np.random.default_rng(seed)
    b = v[rng.integers(0, len(v), (n, len(v)))].mean(1)
    return float(v.mean()), float(np.percentile(b, 2.5)), float(np.percentile(b, 97.5))


def paired(tab_a, tab_b, scenes, name):
    """Scene-paired difference a - b over the scenes complete in both; (mean, lo, hi, n_scenes, wins, losses)."""
    ss = [s for s in scenes if s in tab_a and s in tab_b]
    d = [tab_a[s][name] - tab_b[s][name] for s in ss]
    m, lo, hi = boot_ci(d)
    return m, lo, hi, len(ss), int(sum(x > 0 for x in d)), int(sum(x < 0 for x in d))


def mean_curve(E, scenes):
    """Scene-equal mean curve [101] over complete scenes (None if no complete scene) and the per-scene mean curves."""
    per = {s: np.mean([E[(s, k)]["coverage_official"] for k in range(N_START)], 0) for s in scenes if all((s, k) in E for k in range(N_START))}
    return (np.mean(list(per.values()), 0) if per else None), per


def write_csv(path, header, rows):
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path, "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(header)
        w.writerows(rows)


def fmt(x, nd=3, sign=False):
    return "—" if x is None else (f"{x:+.{nd}f}" if sign else f"{x:.{nd}f}")
