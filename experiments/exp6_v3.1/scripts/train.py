"""EXP6 V3.1 -- Outdoor-only scratch training of the V3-T0 architecture (entry point).
torchrun --standalone --nproc_per_node 4 experiments/exp6_v3.1/scripts/train.py --out <run dir> [--smoke overfit|short|profile]
"""
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import trainer_outdoor  # noqa: E402

if __name__ == "__main__":
    trainer_outdoor.main()
