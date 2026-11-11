"""EXP7 / UAVScenes -- D1 pose tables of UAV-v0 tiles: the frozen experiments/exp6_v3/H/scripts/build_pose_tables.py run unchanged
(EpisodeSim camera, settings grid only, macarons get_camera_RT) with its output folder redirected to work_dirs/cvpr_results/exp7/
uav_pose_tables. Needs MACARONS_PLUSPLUS_ROOT = the UAV root. Usage: python uav_build_pose_tables.py --scenes uav_<...> [...]"""
import os
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
import uav_hooks as UH  # noqa: E402

assert os.path.abspath(os.environ.get("MACARONS_PLUSPLUS_ROOT", "")) == UH.UAV_ROOT
sys.path.insert(0, "/bsuscratch/jiahuizhang/projects/occupancy/seen2scene/experiments/exp6_v3/H/scripts")
import build_pose_tables as BPT  # noqa: E402

BPT.OUT = UH.UAV_POSE_TABLES
os.makedirs(BPT.OUT, exist_ok=True)
if __name__ == "__main__":
    BPT.main()
