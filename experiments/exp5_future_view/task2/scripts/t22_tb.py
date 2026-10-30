"""EXP5 Task 2 -- TensorBoard view of the V2-Mix training logs (seen2scene env, CPU; read-only on the logs).
Converts <run>/log.jsonl (train loss, lr, grad norm, rays per domain, s/step) and <run>/eval.jsonl (FM loss and sample
metrics of in_val / out_sc / in_train / out_train, selection score) into TensorBoard events in <run>/tb/ (rebuilt on every
call) and writes <run>/curves.png. Usage: python t22_tb.py [--run DIR]
"""
import argparse
import json
import os
import shutil

import numpy as np


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--run", default="/bsuscratch/jiahuizhang/projects/occupancy/seen2scene/work_dirs/exp5_future_view/task2/training/mix_v1")
    a = ap.parse_args()
    from torch.utils.tensorboard import SummaryWriter
    L = [json.loads(x) for x in open(os.path.join(a.run, "log.jsonl"))]
    E = [json.loads(x) for x in open(os.path.join(a.run, "eval.jsonl"))]
    tb = os.path.join(a.run, "tb")
    shutil.rmtree(tb, ignore_errors=True)
    w = SummaryWriter(tb)
    for r in L:
        for k in ("loss", "lr", "grad_norm", "s_per_step", "rays_indoor", "rays_outdoor"):
            w.add_scalar(f"train/{k}", r[k], r["step"])
    e0 = E[0]
    for e in E:
        for k, v in e.items():
            if k.startswith("fm_"):
                w.add_scalar(f"eval_fm/{k[3:]}", v, e["step"])
            elif k.startswith("samp_"):
                for kk, vv in v.items():
                    if vv is not None:
                        w.add_scalar(f"eval_{kk}/{k[5:]}", vv, e["step"])
        sc = (e["samp_in_val"]["fair_es_arclength_vox"] / e0["samp_in_val"]["fair_es_arclength_vox"]
              + e["samp_out_sc"]["fair_es_arclength_vox"] / e0["samp_out_sc"]["fair_es_arclength_vox"])
        w.add_scalar("selection/score (lower is better)", sc, e["step"])
    w.close()
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    fig, ax = plt.subplots(1, 3, figsize=(15, 4))
    st = np.array([r["step"] for r in L])
    lo = np.array([r["loss"] for r in L])
    ax[0].plot(st, lo, lw=0.8, color="#bbb", label="mean of 50 steps")
    k = 10
    if len(lo) > k:
        ax[0].plot(st[k - 1:], np.convolve(lo, np.ones(k) / k, "valid"), color="#eb6834", label=f"moving average ({k} x 50 steps)")
    ax[0].set_title("training loss (rectified-flow MSE, mixed batch)")
    ax[0].set_xlabel("step")
    ax[0].legend(fontsize=8)
    es = [e["step"] for e in E]
    for key, col, lab in (("in_val", "#eb6834", "indoor selection set (MP3D val, 6 houses)"), ("out_sc", "#2a78d6", "outdoor S-c (Macarons++)")):
        ax[1].plot(es, [e[f"samp_{key}"]["fair_es_arclength_vox"] for e in E], "o-", color=col, label=lab)
        ax[2].plot(es, [e[f"fm_{key}"] for e in E], "o-", color=col, label=lab)
    ax[1].set_title("fair ES (voxels along the unknown arc; lower = better)")
    ax[2].set_title("deterministic FM loss on the fixed eval sets")
    for x in ax[1:]:
        x.set_xlabel("step")
        x.legend(fontsize=8)
    for x in ax:
        x.grid(alpha=0.3)
    fig.tight_layout()
    fig.savefig(os.path.join(a.run, "curves.png"), dpi=110)
    print("tb:", tb, "| png:", os.path.join(a.run, "curves.png"), "| last step", L[-1]["step"], "| evals", es)


if __name__ == "__main__":
    main()
