"""Shared fixtures of the EXP6 V5 tests (seen2scene env, CPU unless a test asks for CUDA)."""
import json
import os
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
REPO = os.path.dirname(os.path.dirname(os.path.dirname(HERE)))
sys.path.insert(0, REPO)
sys.path.insert(0, os.path.join(REPO, "experiments", "exp6_v5", "scripts"))
import torch  # noqa: E402

from seen2scene.futureview import model as M  # noqa: E402
from seen2scene.futureview import v5_model as V5  # noqa: E402

CFG = os.path.join(REPO, "experiments", "exp6_v5", "configs", "model_v5.json")
V4CK = os.path.join(REPO, "work_dirs", "exp6_v4", "checkpoints", "final.pt")


def v5_cfg():
    return json.load(open(CFG))


def tiny_state(X=16, Y=20, Z=32, seed=0):
    g = torch.Generator().manual_seed(seed)
    coarse = torch.zeros(3, X, Y, Z, dtype=torch.uint8)
    coarse[2] = 255                                    # in domain
    coarse[0, :, :, : Z // 2] = (torch.rand(X, Y, Z // 2, generator=g) * 200).to(torch.uint8)
    coarse[1, X // 2:, :, : Z // 2] = 40
    return coarse


def run(tests):
    bad = 0
    for t in tests:
        try:
            t()
            print(f"PASS {t.__name__}")
        except Exception as e:                     # noqa: BLE001
            bad += 1
            print(f"FAIL {t.__name__}: {type(e).__name__}: {e}")
    print("ALL PASS" if bad == 0 else f"{bad} FAILED")
    return bad
