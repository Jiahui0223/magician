"""EXP6 V3 / CT -- CT-M continuation of V3-T1 from its real 40K training state (seen2scene env; torchrun, 4 ranks).
The shared V3 trainer (seen2scene.v3.trainer, arm t1) runs UNCHANGED except for module-level hooks (the Y mechanism):
  CFG_DIR -> this experiment's configs (train_ct.json; its model entry points to the original model_v3.json)
  lr_at   -> linear warmup from cont_start over cont_warmup to cont_peak, then cosine to 0 at steps
The run directory holds a copy of the parent ckpt_last.pt (train_ct.sh checks its sha256); the trainer's resume restores
model / EMA / AdamW / step and re-seeds the per-rank generators from (seed, rank, step).
torchrun ... train_ct.py --out DIR [--config train_ct.json] [--override k=v ...]"""
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
    if "--config" not in sys.argv:
        sys.argv[1:1] = ["--config", "train_ct.json"]
    sys.argv[1:1] = ["--arm", "t1"]
    trainer.main()
