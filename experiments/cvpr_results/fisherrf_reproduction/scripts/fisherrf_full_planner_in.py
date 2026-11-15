"""FisherRF-full (frozen variant f1c4) on MP3D-150 indoors -- planner-side thin wrapper (fisherrf env; planner side, no GT).

Runs fisherrf_full_planner.main() UNCHANGED (file not edited; the outdoor path stays bitwise identical). The only indoor
difference is where the planner's settings dict comes from. Outdoors fisherrf_full_planner reads
<DATA>/<scene>/settings.json (camera grid + scene box of Macarons++). Indoors there is no settings.json: the T0.12 camera grid
is a pure function of the level's scene card (exp5 task0 grid_env.grid_def; level box, floor, ceiling; no GT geometry). This
wrapper builds the SAME settings dict the MAGICIAN indoor planner of EXP7 / T0.12 uses
(task0/magician_comparison/adapters/magician_planner_grid.py:51-53):
    scene  = station_env.settings_dict(card)["scene"]     (card level box, official settings unit = runtime / 10)
    camera = grid_env.settings_camera(grid_def(card))     (n = card grid, n_theta 5 -> elev -60..60, n_azim 8, same pose formula)
writes it to <svc-dir>/../planner_settings/<scene>/settings.json, and points fisherrf_full_planner.DATA there (module constant
override, read once before the guard is installed, exactly like the outdoor settings.json). PoseGrid then uses the indoor
pose_l / pose_w / pose_h, n_theta = 5, n_azim = 8; the per-step grid-vs-simulator R / T check (rt_max_err <= 1e-3) of the
frozen planner verifies that the grid equals the simulator's T0.12 grid on every legal candidate.
Everything else is the frozen planner: --scene = the level key (record scene_key = <level key>/<start>), record file name is
whatever the simulator sends (svc_t<t>.npz; resume is unsupported so no name pattern is needed), FRF_FRONTIER_MODE /
FRF_REACH_FILTER from the environment (the f1c4 task sets magician / 1; asserted here).
Usage: same arguments as fisherrf_full_planner.py (no extra flags).
"""
import json
import os
import sys

sys.dont_write_bytecode = True
HERE = os.path.dirname(os.path.abspath(__file__))
REPO = os.path.abspath(os.path.join(HERE, "..", "..", "..", ".."))
sys.path.insert(0, HERE)
ADAPTERS = os.path.join(REPO, "experiments", "exp5_future_view", "task0", "magician_comparison", "adapters")
CARDS = os.path.join(REPO, "work_dirs", "exp5_future_view", "task0", "preprocessing", "scenes")


def indoor_settings(level_key):
    sys.path.insert(0, ADAPTERS)
    import grid_env as GE
    import station_env as SE
    sys.path.remove(ADAPTERS)
    card = json.load(open(os.path.join(CARDS, f"{level_key}.json")))
    grid = GE.grid_def(card)
    return {"scene": SE.settings_dict(card)["scene"], "camera": GE.settings_camera(grid),
            "_source": {"card": os.path.join(CARDS, f"{level_key}.json"), "rule": "magician_planner_grid.py:51-53",
                        "grid_n": [int(x) for x in grid["n"]], "step_m": [float(x) for x in grid["step_m"]]}}


if __name__ == "__main__":
    assert os.environ.get("FRF_FRONTIER_MODE") == "magician" and os.environ.get("FRF_REACH_FILTER") == "1", \
        "indoor FisherRF-full is the frozen f1c4 variant: FRF_FRONTIER_MODE=magician FRF_REACH_FILTER=1"
    scene = sys.argv[sys.argv.index("--scene") + 1]
    svc_dir = os.path.abspath(sys.argv[sys.argv.index("--svc-dir") + 1])
    data = os.path.join(os.path.dirname(svc_dir), "planner_settings")
    os.makedirs(os.path.join(data, scene), exist_ok=True)
    with open(os.path.join(data, scene, "settings.json"), "w") as f:
        json.dump(indoor_settings(scene), f, indent=1)
    import fisherrf_full_planner as FFP
    FFP.DATA = data
    FFP.main()
