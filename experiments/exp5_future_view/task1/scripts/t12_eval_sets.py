"""EXP5 Task 1.2 -- deterministic FM loss and arclength sample metrics (t12_train.eval_fm / eval_samples) of one
checkpoint on fixed view subsets, for the LOSO decomposition (protocol §3.6):
  pisa_train_sub : pisa TRAINING trajectories of the main model (fit, scene and starts seen)
  pisa_sub       : pisa S-c trajectories (main: scene seen, starts unseen; LOSO: scene unseen)
  sc_sub         : all S-c trajectories, per-scene quota (as in training)
  train_sub_main : the main model's train_sub
Output W/eval_sets/<name>.json
"""
import argparse
import json
import os
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
import v2_frozen as VF  # noqa: E402
import v2_data as VD  # noqa: E402
import t12_train as TR  # noqa: E402

import torch  # noqa: E402

from seen2scene.futureview import model as M  # noqa: E402


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--ckpt", required=True)
    ap.add_argument("--name", required=True)
    ap.add_argument("--sets", default="pisa_train_sub,pisa_sub,sc_sub,train_sub_main")
    a = ap.parse_args()
    VF.verify_frozen()
    dev = torch.device("cuda")
    ck = torch.load(a.ckpt, map_location=dev)
    model = M.V2Model(ck["model_cfg"]).to(dev)
    model.load_state_dict(ck["ema"])
    model.eval()
    cfg = VF.load_cfg("train_main.json")
    train_main, held = VD.load_split("train6_minus_Sc")
    res = {"ckpt": a.ckpt, "ckpt_sha256": VF.sha256_file(a.ckpt), "step": ck["step"]}
    for nm in a.sets.split(","):
        if nm == "pisa_train_sub":
            d = VD.ViewData([x for x in train_main if x[0] == "pisa"])
            st = d.fixed_states("exp5-V2|eval|pisa_train_sub", 2048)
        elif nm == "pisa_sub":
            d = VD.ViewData([x for x in held if x[0] == "pisa"])
            st = d.fixed_states("exp5-V2|eval|pisa_sub", 2048)
        elif nm == "sc_sub":
            d = VD.ViewData(held)
            st = d.fixed_states("exp5-V2|eval|sc_sub", cfg["eval_sets"]["sc_sub"], per_scene=True)
        else:
            d = VD.ViewData(train_main)
            st = d.fixed_states("exp5-V2|eval|train_sub|train6_minus_Sc", cfg["eval_sets"]["train_sub"])
        res[nm] = {"views": int(sum(len(x[3]) for x in st)), "fm": TR.eval_fm(model, d, st, nm, dev, cfg["eval_tau"]),
                   "samples": TR.eval_samples(model, d, st, nm, dev)}
        print(nm, json.dumps(res[nm]), flush=True)
        del d
    od = os.path.join(VF.WORK, "eval_sets")
    os.makedirs(od, exist_ok=True)
    json.dump(res, open(os.path.join(od, f"{a.name}.json"), "w"), indent=1)


if __name__ == "__main__":
    main()
