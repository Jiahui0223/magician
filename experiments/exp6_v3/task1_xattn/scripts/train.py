"""EXP6 V3 Task 1 -- scratch training of V3-VAE-XAttn (T1): the shared trainer with --arm t1 (same configs as T0;
the only difference is xattn.enabled). It never loads T0 weights.
torchrun --standalone --nproc_per_node 4 experiments/exp6_v3/task1_xattn/scripts/train.py --out <run dir> [--smoke ...]
"""
import sys

from seen2scene.v3 import trainer

if __name__ == "__main__":
    sys.argv[1:1] = ["--arm", "t1"]
    trainer.main()
