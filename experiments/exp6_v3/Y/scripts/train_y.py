"""EXP6 V3 / Y -- continuation of V3-T1 from its real 40K training state (seen2scene env; torchrun, 4 ranks).
The frozen shared V3 trainer (seen2scene.v3.trainer, arm t1) is run UNCHANGED except for two module-level hooks:
  CFG_DIR -> experiments/exp6_v3/Y/configs (train_y.json; its model entry points to the original model_v3.json)
  lr_at   -> continuation schedule: step < cont_start + cont_warmup: cont_peak * (step - cont_start + 1) / cont_warmup;
             then cont_peak * 0.5 (1 + cos(pi (step - cont_start - cont_warmup) / (steps - cont_start - cont_warmup)))
The run directory must contain the parent ckpt_last.pt copy (train_y.sh seeds it, sha256 checked); the trainer's own resume
restores model / EMA / AdamW / step and re-seeds the per-rank generators from (seed, rank, step) (its documented rule).
torchrun --standalone --nproc_per_node 4 train_y.py --out DIR [--override k=v ...]"""
import math
import os
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
from seen2scene.v3 import trainer  # noqa: E402

trainer.CFG_DIR = os.path.join(os.path.dirname(HERE), "configs")


def lr_at(step, cfg):
    s0, w, peak, total = cfg["cont_start"], cfg["cont_warmup"], cfg["cont_peak"], cfg["steps"]
    if step < s0 + w:
        return peak * (step - s0 + 1) / w
    p = (step - s0 - w) / max(1, total - s0 - w)
    return peak * 0.5 * (1.0 + math.cos(math.pi * min(1.0, p)))


trainer.lr_at = lr_at

if __name__ == "__main__":
    sys.argv[1:1] = ["--arm", "t1", "--config", "train_y.json"]
    trainer.main()
