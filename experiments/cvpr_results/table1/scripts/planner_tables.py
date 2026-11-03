"""Table-1 dataset audit: planner-side per-scene tables (seen2scene env, CPU only, no GT read).

For each of the 15 scenes:
  Z0      frozen value in task1/scripts/v2_common.Z0 (Dev-3 only) and the documented rule
          abl_common.z0_of(scene) = sum((x_max - x_min) * 10) / 3 / 2 from settings.json "scene" (the rule the Task 2
          outdoor planner child t24_v2_out_planner.py:46 patches in for every evaluation scene)
  em2     entry of work_dirs/exp3b_macaronspp_transfer/task1_protocol/em2_windows.json (k_M, windows) and, if missing,
          the documented rule k_M = (y extent x 10) / 2.816 (marked derived)
  geometry  t5_diagnostic.scene_geometry(scene) (SceneFrame + grid dims + domain + official windows), exactly the call
          of v2_planner.py:112; the exception if it fails
Usage: env -u LD_PRELOAD -u PYTHONPATH /bsuscratch/jiahuizhang/envs/seen2scene/bin/python planner_tables.py OUT.json
"""
import json
import os
import sys
import traceback

REPO = "/bsuscratch/jiahuizhang/projects/occupancy/seen2scene"
EXP3B = os.path.join(REPO, "experiments", "exp3b_macaronspp_transfer")
ABL1 = os.path.join(REPO, "experiments", "exp4_multiworld_belief", "ablation_study", "abl_1", "common")
T1S = os.path.join(REPO, "experiments", "exp5_future_view", "task1", "scripts")
for p in (REPO, EXP3B, ABL1, T1S):
    sys.path.insert(0, p)
os.environ.setdefault("CUDA_VISIBLE_DEVICES", "")
SCENES = ["alhambra", "bannerman", "barts", "bridge", "colosseum", "dunnottar", "eiffel", "fushimi", "liberty",
          "neuschwanstein", "pantheon", "pisa", "redeemer", "sestino_museum", "sofia_church"]
DATA = "/bsuscratch/jiahuizhang/projects/occupancy/MAGICIAN/data/Macarons++"
EM2 = os.path.join(REPO, "work_dirs", "exp3b_macaronspp_transfer", "task1_protocol", "em2_windows.json")
EM2_OFF = os.path.join(REPO, "work_dirs", "exp3b_macaronspp_transfer", "task1_protocol",
                       "em2_windows_official_positions.json")


def main():
    import abl_common as AC
    import v2_common as LC
    em2 = json.load(open(EM2))["scenes"]
    em2o = json.load(open(EM2_OFF))["scenes"]
    out = {}
    for s in SCENES:
        st = json.load(open(os.path.join(DATA, s, "settings.json")))
        sc = st["scene"]
        r = {"scene": s}
        ext = [(b - a) * 10.0 for a, b in zip(sc["x_min"], sc["x_max"])]
        r["scene_bbox_runtime_min"] = [10.0 * x for x in sc["x_min"]]
        r["scene_bbox_runtime_max"] = [10.0 * x for x in sc["x_max"]]
        r["scene_extent_runtime_yup"] = ext
        r["z0_frozen"] = LC.Z0.get(s)
        try:
            r["z0_rule"] = AC.z0_of(s)
        except Exception as e:  # noqa: BLE001
            r["z0_rule_error"] = repr(e)
        r["em2_present"] = s in em2
        r["em2_official_positions_present"] = s in em2o
        r["k_M_rule"] = ext[1] / 2.816
        if s in em2:
            r["k_M_em2"] = em2[s]["k_M"]
            r["n_windows_em2"] = em2[s]["n_windows"]
        if s in em2o:
            r["n_windows_official"] = em2o[s]["n_windows"]
            r["dims_grid_official"] = em2o[s]["dims_grid"]
        try:
            import t5_diagnostic as TD
            fr, dims, dom, win = TD.scene_geometry(s)
            r["scene_geometry_ok"] = True
            r["geometry_dims"] = [int(x) for x in dims]
            r["geometry_n_windows"] = int(win["n"])
            r["geometry_k_M"] = float(fr.k_M)
        except Exception as e:  # noqa: BLE001
            r["scene_geometry_ok"] = False
            r["scene_geometry_error"] = f"{type(e).__name__}: {e}"
            r["scene_geometry_traceback"] = traceback.format_exc()[-1500:]
        out[s] = r
        print(s, "z0 frozen", r["z0_frozen"], "rule %.4f" % r.get("z0_rule", float("nan")), "| em2", r["em2_present"],
              "| geometry", r["scene_geometry_ok"], r.get("geometry_dims"), r.get("scene_geometry_error", ""), flush=True)
    json.dump(out, open(sys.argv[1], "w"), indent=1)


if __name__ == "__main__":
    main()
