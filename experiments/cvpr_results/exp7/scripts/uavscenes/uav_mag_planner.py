"""EXP7 / UAVScenes -- MAGICIAN beam planner child (magician env, run with python -I): task2 t24_mag_out_planner.py unchanged except its
hard-coded settings folder DATA, redirected to the UAV-v0 tile root (the tile settings.json = scene / camera blocks, as the official
Macarons++ settings). Official MAGICIAN core (e697009 adapter), beam 10 x 10, official neighbours / validity / occupancy-point segment
test for deeper layers; layer 0 = the simulator's legal list."""
import os
import sys

R = "/bsuscratch/jiahuizhang/projects/occupancy/seen2scene"
sys.path.insert(0, os.path.join(R, "experiments", "exp5_future_view", "task2", "scripts"))
import t24_mag_out_planner as T24M  # noqa: E402

T24M.DATA = os.path.join(R, "work_dirs", "cvpr_results", "exp7", "uav_macpp")
if __name__ == "__main__":
    T24M.main()
