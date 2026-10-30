"""EXP5 Task 1.4 closed loop -- environment-free definitions shared by the simulator (magician env), the V2 planner child
(seen2scene env), the orchestrator and the analysis (protocol §5, configs/closed_loop_v1.json)."""
import json
import os

HERE = os.path.dirname(os.path.abspath(__file__))
T1 = os.path.dirname(HERE)
REPO = os.path.abspath(os.path.join(T1, "..", "..", ".."))
EXP3B = os.path.join(REPO, "experiments", "exp3b_macaronspp_transfer")
EXP4 = os.path.join(REPO, "experiments", "exp4_multiworld_belief")
ABL1_COMMON = os.path.join(EXP4, "ablation_study", "abl_1", "common")
T3_SCRIPTS = os.path.join(EXP4, "task3", "scripts")
TASK1_SCRIPTS = os.path.join(EXP4, "task1", "scripts")
WORK = os.path.join(REPO, "work_dirs", "exp5_future_view", "task1")
LOOP = os.path.join(WORK, "closed_loop")
B0_ROOT = os.path.join(REPO, "work_dirs", "exp4_multiworld_belief", "ablation_study", "abl_1",
                       "T1C_FRindep_DistP1_Greedy_R10_K4", "dev100")
SEEN2SCENE_PYTHON = "/bsuscratch/jiahuizhang/envs/seen2scene/bin/python"
MAGICIAN_PYTHON = "/bsuscratch/jiahuizhang/envs/magician/bin/python"
CFG = json.load(open(os.path.join(T1, "configs", "closed_loop_v1.json")))
ARMS = CFG["arms"]
RUN_ORDER = CFG["run_order"]
MOVES = CFG["moves"]
SCENES = CFG["scenes"]
STARTS = CFG["starts"]
REPS = CFG["reps"]
R10 = CFG["refresh_R10"]
SMOKE_MOVES = CFG["smoke"]["moves"]
SMOKE_EPISODES = [tuple(x) for x in CFG["smoke"]["episodes"]]
SMOKE_INTERRUPT_AFTER = CFG["smoke"]["interrupt_after"]
EXIT_INTERRUPT = 75
Z0 = {"barts": 15.1142, "pantheon": 38.2155, "colosseum": 36.6305}   # MAGICIAN scene_scale / 2 (abl_1, frozen)


def exp_root(eid, smoke=False):
    return os.path.join(LOOP, "smoke" if smoke else ARMS[eid]["name"])


def tag(eid, scene, start, rep):
    return f"{eid}_{scene}_s{start}_r{rep}"


def dirs(root, tg):
    return {"episode": os.path.join(root, "episodes", f"ep_{tg}.json"), "svc": os.path.join(root, "svc", tg),
            "plog": os.path.join(root, "planner_logs"), "ckpt": os.path.join(root, "ckpt", f"{tg}.jsonl"),
            "frames": os.path.join(root, "frames", tg)}


def refresh_time(eid, t):
    """t_r: the latest refresh <= t (R10: 0, 10, ..., 90; native: every step)."""
    return t if ARMS[eid]["refresh"] == "native" else 10 * (t // 10)


def seed_key(scene, start, rep, t_r, k):
    return CFG["seed_key"].format(scene=scene, start=start, rep=rep, t_r=t_r, k=k)


def load_json(p):
    with open(p) as f:
        return json.load(f)
