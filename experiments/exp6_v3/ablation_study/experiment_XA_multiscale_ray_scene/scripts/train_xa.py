"""Experiment XA -- scratch training of one XA variant with the UNCHANGED V3 trainer (seen2scene.v3.trainer; seen2scene env, torchrun, 4 ranks).
Module-level hooks only (the mechanism of the Y / CT continuation scripts):
  CFG_DIR    -> this experiment's configs (train_<id>.json = the V3-T1 recipe train_v3.json, early stopping off; model_<id>.json)
  model_cfg  -> the complete XA model config (no T0 / T1 arm switch)
  make_model -> seen2scene.v3.xa.xa_model.build under the same seed call (the shared modules get the V3-T1 initial values)
Data, sampling stream, losses, optimiser, schedule, EMA, validation and checkpoints are the trainer's (the T1 run).
torchrun ... train_xa.py --id xa_s2d --out DIR [--smoke overfit|short|profile] [--override k=v ...]"""
import json
import os
import sys

import torch

HERE = os.path.dirname(os.path.abspath(__file__))
from seen2scene.v3 import trainer  # noqa: E402
from seen2scene.v3.xa import xa_model as XM  # noqa: E402

trainer.CFG_DIR = os.path.join(os.path.dirname(HERE), "configs")


def model_cfg(arm, cfg):
    return json.load(open(os.path.join(trainer.CFG_DIR, cfg["model"])))


def make_model(mcfg, seed, dev):
    torch.manual_seed(seed)
    return XM.build(mcfg).to(dev)


trainer.model_cfg = model_cfg
trainer.make_model = make_model

if __name__ == "__main__":
    i = sys.argv.index("--id")
    vid = sys.argv[i + 1]
    del sys.argv[i:i + 2]
    sys.argv[1:1] = ["--arm", "t1", "--config", f"train_{vid}.json"]
    trainer.main()
