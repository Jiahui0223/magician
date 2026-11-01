"""EXP6 V4 -- MP3D V2-planner child with a V4 checkpoint: task2/scripts/t23_v2_in_planner.py run unchanged under
p1_full.patch() (T3_P1_MODE=b2: D100 input map + sparse DistP1 map). The checkpoint (env V4_CKPT) replaces the group's
--ckpt / --sha256; v2_plan_core.Core builds M.V2Model(ck["model_cfg"]) -> the ray_xattn flag of the checkpoint."""
import hashlib
import os
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
EXP5 = os.path.join(os.path.dirname(os.path.dirname(HERE)), "exp5_future_view")
sys.path.insert(0, os.path.join(EXP5, "task2", "scripts"))
import t23_v2_in_planner as P23  # noqa: E402  (sets the planner import paths)

sys.path.insert(0, os.path.join(EXP5, "task3_full_depth_p1", "scripts"))
import p1_full as PF  # noqa: E402


def ckpt_override():
    ck = os.environ["V4_CKPT"]
    sha = hashlib.sha256(open(ck, "rb").read()).hexdigest()
    for flag, val in (("--ckpt", ck), ("--sha256", sha)):
        i = sys.argv.index(flag)
        sys.argv[i + 1] = val
    return ck, sha


if __name__ == "__main__":
    sys.stderr.write(f"V4_CKPT={ckpt_override()}\n")
    sys.stderr.write(f"T3_P1_MODE={PF.patch()}\n")
    P23.main()
