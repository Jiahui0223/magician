"""EXP5 Task 3 / E1.2 -- roots of the two state strata (S_B0: Task 2 V2M = B0 trajectories, ablation study files, read
only; S_B2: Task 3 B2 closed-loop trajectories, manifests / oracles written under W3/e12/abroot_B2). set_stratum() points
the ablation helpers (ab_common: W, CL_IN, CL_OUT) at the stratum; call it before anything reads them."""
import os
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
EXP5 = os.path.dirname(os.path.dirname(os.path.dirname(HERE)))
sys.path.insert(0, os.path.join(EXP5, "ablation_study", "common"))
import ab_common as A  # noqa: E402

W3 = os.path.join(A.REPO, "work_dirs", "exp5_future_view", "task3_full_depth_p1")
E12 = os.path.join(W3, "e12")
ROOT_B2 = os.path.join(E12, "abroot_B2")


_ORIG = (A.W, A.CL_IN, A.CL_OUT)


def set_stratum(stratum):
    A.W, A.CL_IN, A.CL_OUT = _ORIG
    if stratum == "B2":
        A.W = ROOT_B2
        A.CL_IN = os.path.join(W3, "closed_loop", "indoor", "B2")
        A.CL_OUT = os.path.join(W3, "closed_loop", "outdoor", "B2")
    elif stratum != "B0":
        raise ValueError(stratum)
    return A
