"""EXP6 V3 Task 0 -- scratch training of V3-VAE (T0): the shared trainer with --arm t0.
torchrun --standalone --nproc_per_node 4 experiments/exp6_v3/task0_vae/scripts/train.py --out <run dir> [--smoke overfit|short|profile]
"""
import sys

from seen2scene.v3 import trainer

if __name__ == "__main__":
    sys.argv[1:1] = ["--arm", "t0"]
    trainer.main()
