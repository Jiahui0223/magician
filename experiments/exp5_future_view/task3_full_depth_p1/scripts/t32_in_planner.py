"""EXP5 Task 3 / E1 -- MP3D V2 planner child: the Task 2 child (task2/scripts/t23_v2_in_planner.py) run unchanged, with
m_p1.P1Fusion replaced by the FullDepth-P1 fusion when T3_P1_MODE=full (p1_full.patch; B1). T3_P1_MODE=sparse = B0."""
import os
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
EXP5 = os.path.dirname(os.path.dirname(HERE))
sys.path.insert(0, os.path.join(EXP5, "task2", "scripts"))
import t23_v2_in_planner as P23  # noqa: E402  (sets the planner import paths: t06_planner, m_p1, ...)

sys.path.insert(0, HERE)
import p1_full as PF  # noqa: E402

def ckpt_override():
    """E2 pilot: env T3_CKPT = a fine-tuned EMA checkpoint replaces the group's --ckpt / --sha256 (sha256 recomputed and
    checked by the frozen core as usual). Unset (B0 / B1 / B2): arguments untouched."""
    ck = os.environ.get("T3_CKPT")
    if not ck:
        return None
    import hashlib
    sha = hashlib.sha256(open(ck, "rb").read()).hexdigest()
    for flag, val in (("--ckpt", ck), ("--sha256", sha)):
        i = sys.argv.index(flag)
        sys.argv[i + 1] = val
    return ck


if __name__ == "__main__":
    sys.stderr.write(f"T3_CKPT={ckpt_override()}\n")
    sys.stderr.write(f"T3_P1_MODE={PF.patch()}\n")
    P23.main()
