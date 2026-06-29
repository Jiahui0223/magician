"""
Top-level launcher for MACARONS self-supervised training (stage 3).

The repo ships the trainer as a module (macarons/trainers/train_macarons.py) but no
top-level runner, so this mirrors test_magician_planning.py for the training side.

  Smoke test (1 GPU):   python train_macarons_run.py -c smoke_test_config.json
  Full run   (1 GPU):   python train_macarons_run.py -c macarons_default_training_config.json
  Full run   (4 GPU):   set "ddp": true + "WORLD_SIZE": 4 in the config, then run the same command
                        (mp.spawn launches one process per rank).

Config is looked up under configs/macarons/ (or pass an absolute/relative path).
"""
import argparse
import os

import torch

# --- PyTorch 2.6 compatibility shim ---
# This codebase predates PyTorch 2.6, where torch.load's default flipped to
# weights_only=True. Its memory files (surface/occupancy/depth .pt) embed numpy
# scalars and fail to load under the new default with:
#   UnpicklingError: Unsupported global: GLOBAL numpy.core.multiarray.scalar
# These are our own trusted, locally-generated files, so restore the pre-2.6
# behavior for the whole training run. (Set at module level so it also applies
# in mp.spawn child processes, which re-import this module.)
_orig_torch_load = torch.load
def _torch_load_compat(*a, **k):
    k.setdefault("weights_only", False)
    return _orig_torch_load(*a, **k)
torch.load = _torch_load_compat

# This launcher lives in jiahui/; the repo root (with the `macarons` package and
# configs/) is one level up. Put it on sys.path and use it for path lookups so the
# script works when invoked as `python jiahui/train_macarons_run.py` from repo root.
import sys
repo_root = os.path.abspath(os.path.join(os.path.dirname(__file__), os.pardir))
sys.path.insert(0, repo_root)

from macarons.utility.macarons_utils import load_params
from macarons.trainers.train_macarons import run_training

dir_path = repo_root
train_configs_dir = os.path.join(dir_path, "configs", "macarons")


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Launch MACARONS self-supervised training.')
    parser.add_argument('-c', '--config', type=str, default="smoke_test_config.json",
                        help='config file name under configs/macarons/ (or a path).')
    parser.add_argument('--epochs', type=int, default=None,
                        help='override params.epochs (e.g. 1 to validate, leave unset for the config value).')
    args = parser.parse_args()

    config_path = args.config if os.path.isfile(args.config) \
        else os.path.join(train_configs_dir, args.config)
    params = load_params(config_path)
    print(f"Loaded training config: {config_path}")

    if args.epochs is not None:
        params.epochs = args.epochs
        print(f"Overriding epochs -> {args.epochs}")

    if getattr(params, "ddp", False):
        import torch.multiprocessing as mp
        world_size = params.WORLD_SIZE
        print(f"DDP training on {world_size} GPUs.")
        mp.spawn(run_training, args=(params,), nprocs=world_size)
    else:
        run_training(params=params)
