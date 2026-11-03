"""EXP Density A/B -- Macarons++ coverage of a closed-loop config (magician env, 1 GPU; evaluation side): task2
t25_out_cov.py unchanged (official coverage must equal the stored curve, else STOP; monotone coverage), with the replay
frames in a per-config temp dir (no race between configs). Usage: python da_out_cov.py --cfg CFG --scene S --start K"""
import os
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
REPO = os.path.dirname(os.path.dirname(os.path.dirname(HERE)))
sys.path.insert(0, os.path.join(REPO, "experiments", "exp5_future_view", "task2", "scripts"))
import t25_out_cov as T25  # noqa: E402

G3 = os.environ.get("DA_G3_ROOT", os.path.join(REPO, "work_dirs", "exp_density_ab", "g3"))

if __name__ == "__main__":
    cfg = sys.argv[sys.argv.index("--cfg") + 1]
    s = sys.argv[sys.argv.index("--scene") + 1]
    k = sys.argv[sys.argv.index("--start") + 1]
    T25.C.W2 = os.path.join(G3, "cov_tmp", cfg)       # replay_frames -> <this>/eval/outdoor/replay_frames/V2M_<tag>
    sys.argv = [sys.argv[0], "--group", "V2M", "--tag", f"V2M_{s}_s{k}_r0",
                "--root", os.path.join(G3, "closed_loop", "outdoor", cfg, "V2M"), "--out", os.path.join(G3, "metrics", "outdoor", cfg)]
    T25.main()
