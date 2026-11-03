"""EXP Unified Outdoor -- fixed-state G1 / G2 of one trained model under the density settings of UNIFIED_PROTOCOL §5
(seen2scene env, 1 GPU per process). Runs exp_density_ab/scripts/da_fixed.py (unchanged replay, regions, noise keys, state
sets VALSEL 90 + Dev-3 275, G1 ray sets, scoring masks) with its checkpoint list replaced by ONE model, so every state
yields: S10|S10, S10|D100, D100|S10, D100|D100 (inference | scoring); the frozen-protocol settings are D100|S10, D100|D100
and the input-density control S10|S10. V2-architecture checkpoints (V2, V4, V4C) load through the frozen planner core
(task1 v2_plan_core.Core: V2Model(model_cfg) + EMA). No historical-reference checks (new models); the band_dil double-path
check of da_fixed stays active.
Output work_dirs/exp_unified_outdoor/eval_fixed/<name>/<set>/<stratum>/<tag>.jsonl
Usage: python uo_fixed.py --name NAME --ckpt PATH --set valsel|dev3 --stratum B0|B2 --part i/n
"""
import argparse
import json
import os
import sys
import time

HERE = os.path.dirname(os.path.abspath(__file__))
REPO = os.path.dirname(os.path.dirname(os.path.dirname(HERE)))
sys.path.insert(0, os.path.join(REPO, "experiments", "exp_density_ab", "scripts"))
import da_fixed as DA  # noqa: E402

OUT = os.path.join(REPO, "work_dirs", "exp_unified_outdoor", "eval_fixed")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--name", required=True)
    ap.add_argument("--ckpt", required=True)
    ap.add_argument("--set", choices=["valsel", "dev3"], required=True)
    ap.add_argument("--stratum", choices=["B0", "B2"], default="B0")
    ap.add_argument("--part", default="0/1")
    a = ap.parse_args()
    X, EF = DA.X, DA.EF
    X.setup(a.stratum)
    sys.path.insert(0, os.path.join(DA.EP.EXP5, "task2", "scripts"))
    torch = X.torch
    torch.use_deterministic_algorithms(True, warn_only=True)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    dev = torch.device("cuda")
    X.S2.load_cores(dev)                       # sets up the frozen import paths exactly as da_fixed
    DA.CKPTS = (a.name,)
    cores = {a.name: X.PC.Core(os.path.abspath(a.ckpt), dev)}
    meta = {"name": a.name, "ckpt": os.path.relpath(os.path.abspath(a.ckpt), REPO), "sha256": cores[a.name].ckpt_sha}
    pi, pn = map(int, a.part.split("/"))
    od = os.path.join(OUT, a.name, a.set, a.stratum)
    os.makedirs(od, exist_ok=True)
    json.dump(meta, open(os.path.join(OUT, a.name, "model.json"), "w"), indent=1)
    for tag, sts in EF.states(a.set, a.stratum)[pi::pn]:
        po = os.path.join(od, f"{tag}.jsonl")
        if os.path.exists(po):
            continue
        tb = time.time()
        rows = DA.run_episode(a.set, a.stratum, tag, sts, cores, dev, {}, {})
        with open(po + ".tmp", "w") as f:
            for r in rows:
                r["model"] = meta
                f.write(json.dumps(r, default=float) + "\n")
        os.replace(po + ".tmp", po)
        print(a.name, a.set, a.stratum, tag, len(rows), "states %.0fs" % (time.time() - tb), "peak %.1f GB" % rows[0]["episode_peak_gb"], flush=True)
        torch.cuda.empty_cache()


if __name__ == "__main__":
    main()
