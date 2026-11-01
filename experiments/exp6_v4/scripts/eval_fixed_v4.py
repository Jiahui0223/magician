"""EXP6 V4 / V3.1 -- G1 / G2 on the development fixed states (seen2scene env, 1 GPU per process): exp6_v3/shared/
eval_fixed.py run_episode unchanged with the cores B2 (frozen V2-Mix on the D100 coarse grid, replay check against the
E2 rows), V31 (V3.1 final.pt, V3 predictor on the live D100 fusion) and V4 (V4 final.pt through the V2 planner core:
encoder on the D100 coarse grid, as B2). Same states, candidates, noise and sparse DistP1 scoring as E2 / V3.
Output work_dirs/exp6_v4/eval_fixed/<set>_<tag>/<stratum>/<tag>.jsonl.
Usage: python eval_fixed_v4.py --set valsel|dev3 --stratum B0|B2 [--v31 CKPT] [--v4 CKPT] [--tag s_final] [--part i/n]
(the arms may be evaluated in separate runs / tags: same states, candidates and noise; analyze_g12_v4 merges by sid)"""
import argparse
import glob
import json
import os
import sys
import time

from seen2scene import PROJECT_ROOT

REPO = str(PROJECT_ROOT)
sys.path.insert(0, os.path.join(REPO, "experiments", "exp6_v3", "shared"))
import eval_fixed as EF  # noqa: E402

X, E2F = EF.X, EF.E2F
W = os.path.join(REPO, "work_dirs", "exp6_v4", "eval_fixed")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--set", choices=["valsel", "dev3"], required=True)
    ap.add_argument("--stratum", choices=["B0", "B2"], default="B0")
    ap.add_argument("--v31", default=None)
    ap.add_argument("--v4", default=None)
    ap.add_argument("--tag", default="s_final")
    ap.add_argument("--part", default="0/1")
    a = ap.parse_args()
    X.setup(a.stratum)
    sys.path.insert(0, os.path.join(REPO, "experiments", "exp5_future_view", "task2", "scripts"))
    torch = X.torch
    torch.use_deterministic_algorithms(True, warn_only=True)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    dev = torch.device("cuda")
    v2 = X.S2.load_cores(dev)["V2M"]
    assert a.v31 or a.v4, "give --v31 and/or --v4"
    cores = {"B2": EF.V2Core(v2)}
    if a.v31:
        cores["V31"] = EF.V3Core(a.v31, dev)
    if a.v4:
        v4core = X.PC.Core(a.v4, dev)                    # V2Model(ck["model_cfg"]) -> ray_xattn enabled by the checkpoint
        assert v4core.model.ray_xattn is not None, "V4 checkpoint without ray_xattn"
        cores["V4"] = EF.V2Core(v4core)
    ref = {}
    for f in glob.glob(os.path.join(E2F.EP.W, "fixed", f"{a.set}_s5000", a.stratum, "*.jsonl")):
        for line in open(f):
            r = json.loads(line)
            ref[r["sid"]] = r["scores"]["B2"]
    pi, pn = map(int, a.part.split("/"))
    od = os.path.join(W, f"{a.set}_{a.tag}", a.stratum)
    os.makedirs(od, exist_ok=True)
    json.dump({"v31": a.v31, "v31_sha256": cores["V31"].ckpt_sha if "V31" in cores else None, "v4": a.v4, "v4_sha256": cores["V4"].ckpt_sha if "V4" in cores else None, "v2m_sha256": v2.ckpt_sha},
              open(os.path.join(od, f"models_{pi}.json"), "w"), indent=1)
    for tag, sts in E2F.states(a.set, a.stratum)[pi::pn]:
        po = os.path.join(od, f"{tag}.jsonl")
        if os.path.exists(po):
            continue
        tb = time.time()
        rows = EF.run_episode(a.set, a.stratum, tag, sts, cores, dev, ref)
        with open(po + ".tmp", "w") as f:
            for r in rows:
                f.write(json.dumps(r, default=float) + "\n")
        os.replace(po + ".tmp", po)
        print(a.set, a.stratum, tag, len(rows), "%.0fs" % (time.time() - tb), flush=True)
        torch.cuda.empty_cache()


if __name__ == "__main__":
    main()
