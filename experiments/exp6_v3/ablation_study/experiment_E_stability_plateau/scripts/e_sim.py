"""EXP6 V3 / ablation E -- one Macarons++ closed loop with a chosen planner-noise replicate `rep` and/or from a branch root
(magician env). The frozen simulator (task1 v2_sim.run via task2 t24_out_sim: M protocol, official coverage, per-step
checkpoint + bitwise resume) and the frozen planner children are used unchanged:
  --family h -> experiments/exp6_v3/H/scripts/sim_out_h.Planner      (H-v1 policies h0 / h3; env H_POLICY)
  --family d -> experiments/exp6_v3/ablation_study/experiment_D_h3_lookahead/scripts/sim_out_d.Planner (D-v1 d1; env D_POLICY)
`rep` only changes the planner noise key v2_common.seed_key(scene, start, rep, t_r, k) and the episode tag (..._r<rep>); the GT
observation seeds (m_common.derived_seed(scene, start, t, ...)) and the fallback seed do not depend on it. rep 0 = the Table 1 run.
A branch root prepared by e_branch.py holds the first T steps of a source episode (simulator checkpoint lines, service
records, planner step log); v2_sim then replays them (bitwise check against the archived service records) and the planner
resumes at T (records 0..T-1 re-integrated, the last R10 refresh recomputed) -> the continuation policy acts from step T on.
Usage: python e_sim.py --family h|d --policy P --v3-ckpt PATH --scene S --start K --rep N --root DIR [--moves 100]"""
import argparse
import importlib.util
import os
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
X = os.path.dirname(os.path.dirname(os.path.dirname(os.path.dirname(HERE))))           # experiments/
MOD = {"h": os.path.join(X, "exp6_v3", "H", "scripts", "sim_out_h.py"),
       "d": os.path.join(X, "exp6_v3", "ablation_study", "experiment_D_h3_lookahead", "scripts", "sim_out_d.py")}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--family", required=True, choices=("h", "d"))
    ap.add_argument("--policy", required=True)
    ap.add_argument("--v3-ckpt", required=True)
    ap.add_argument("--scene", required=True)
    ap.add_argument("--start", type=int, required=True)
    ap.add_argument("--rep", type=int, required=True)
    ap.add_argument("--root", required=True)
    ap.add_argument("--moves", type=int, default=100)
    a = ap.parse_args()
    os.environ["V3_CKPT"] = os.path.abspath(a.v3_ckpt)
    os.environ["H_POLICY" if a.family == "h" else "D_POLICY"] = a.policy
    fam = a.family
    spec = importlib.util.spec_from_file_location(f"sim_out_{fam}", MOD[fam])
    m = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(m)                       # defines Planner; its __main__ block is not executed
    m.VS.Planner = m.Planner
    ns = argparse.Namespace(exp="V2M", group="V2M", scene=a.scene, start=a.start, rep=a.rep, root=os.path.abspath(a.root),
                            moves=a.moves, planner_gpu=None)
    assert a.moves == m.CFG["budget"]["moves"], "budget must stay the frozen 100 moves"
    m.VS.run(ns)


if __name__ == "__main__":
    main()
