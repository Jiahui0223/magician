"""Experiment XB / Task C -- scratch training of V3-T1 on a REDUCED outdoor scene set with the UNCHANGED V3 trainer (seen2scene env, torchrun, 4 ranks).
Hooks (module globals only, the Y / CT / XA mechanism): trainer.CFG_DIR -> this experiment's configs; seen2scene.v3.data.items -> the outdoor TRAINING list is
restricted to cfg["outdoor_scenes"] (S-c, MP3D train / valsel lists unchanged). Model = the trainer's own T1 (arm t1, model_v3.json).
torchrun ... train_xb.py --config train_t1_red3.json --out DIR [--smoke overfit|short|profile] [--override k=v ...]"""
import json
import os
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
from seen2scene.v3 import data as D  # noqa: E402
from seen2scene.v3 import trainer  # noqa: E402

trainer.CFG_DIR = os.path.join(os.path.dirname(HERE), "configs")
_cfg = json.load(open(os.path.join(trainer.CFG_DIR, sys.argv[sys.argv.index("--config") + 1])))
SCENES = set(_cfg["outdoor_scenes"])
_items = D.items


def items(domain, purpose):
    out = _items(domain, purpose)
    if domain == "outdoor" and purpose == "train":
        kept = [x for x in out if x[0] in SCENES]
        if os.environ.get("RANK", "0") == "0":
            print(f"XB outdoor train scenes {sorted(SCENES)}: {len(kept)} of {len(out)} trajectories kept; S-c unchanged", flush=True)
        return kept
    return out


D.items = items

if __name__ == "__main__":
    sys.argv[1:1] = ["--arm", "t1"]
    trainer.main()
