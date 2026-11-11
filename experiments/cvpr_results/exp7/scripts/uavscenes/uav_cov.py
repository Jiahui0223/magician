"""EXP7 / UAVScenes -- official coverage replay of one UAV-v0 episode (magician env, 1 GPU): task2 t25_out_cov.py run unchanged (the
frozen EpisodeSim replays the executed poses; the MAGICIAN-style official coverage must equal the stored curve bitwise, else STOP;
plus the monotone coverage), exactly as CVPR Table 1 (t1_out_cov.py), with MACARONS_PLUSPLUS_ROOT = the UAV tile root and the
replay frames / outputs under work_dirs/cvpr_results/exp7. Usage: python uav_cov.py --method OURS|RANDOM --scene uav_<...> --start K"""
import os
import sys

R = "/bsuscratch/jiahuizhang/projects/occupancy/seen2scene"
W7 = os.path.join(R, "work_dirs", "cvpr_results", "exp7")
assert os.path.abspath(os.environ.get("MACARONS_PLUSPLUS_ROOT", "")) == os.path.join(W7, "uav_macpp")
sys.path.insert(0, os.path.join(R, "experiments", "exp5_future_view", "task2", "scripts"))
import t25_out_cov as T25  # noqa: E402

if __name__ == "__main__":
    g = lambda k: sys.argv[sys.argv.index(k) + 1]  # noqa: E731
    m, s, k = g("--method"), g("--scene"), g("--start")
    group = {"OURS": "V2M", "RANDOM": "random"}[m]
    T25.C.W2 = os.path.join(W7, "cov_tmp", m)
    sys.argv = [sys.argv[0], "--group", group, "--tag", f"{group}_{s}_s{k}_r0", "--root", os.path.join(W7, "episodes", "uavscenes", "raw", m, group),
                "--out", os.path.join(W7, "episodes", "uavscenes", "metrics", m)]
    T25.main()
