"""EXP5 Task 3 / E2 pilot -- training wrapper (seen2scene env; torchrun --nproc_per_node 4 e2_train.py ARM --out DIR [--smoke ...]).

Runs the FROZEN Task 2 trainer (task2/scripts/t22_train_mix.py: model, loss, sampler, DDP, AdamW + warmup / cosine, EMA,
evaluation) unchanged. Only redirected:
  configs   C.T2 -> e2_pilot (configs/e2_<arm>_v1.json: init = V2-Mix EMA@17.5k, 5k steps, eval / ckpt every 1k)
  data      ARM d100: MP3D shards from e2/w2_d100 (C.W2), Macarons++ shards from e2/out_views_d100 (ViewData root);
            ARM s10: the Task 2 / T1.1 shards themselves (no redirection)
The sampler (numpy RNG [seed, rank]) only sees the trajectory lists, states and view rows, which are identical in the two
data sets (checked by e2_build check-*), so both arms draw identical sample indices.
"""
import os
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
import e2_paths as EP  # noqa: E402

sys.path.insert(0, os.path.join(EP.EXP5, "task2", "scripts"))
import t2_common as C  # noqa: E402

ARM = sys.argv.pop(1)
assert ARM in ("d100", "s10"), ARM
C.T2 = EP.E2
import t22_data as TD  # noqa: E402

if ARM == "d100":
    C.W2 = EP.W2D
    _Compact = TD.CompactViewData

    class CompactD100(_Compact):
        def __init__(self, items, root=None, log=print):
            super().__init__(items, root=EP.OUTD, log=log)
    TD.CompactViewData = CompactD100
import t22_train_mix as TM  # noqa: E402

def guard():
    """D100 arm: the trajectory lists must equal the sparse arm's (else the samplers would diverge)."""
    import json
    jobs = json.load(open(os.path.join(EP.T2W, "mp3d", "manifests", "traj_jobs.json")))["jobs"]
    miss = [j for j in jobs if j["purpose"] in ("train", "valsel")
            and os.path.exists(os.path.join(EP.T2W, "mp3d", "views", j["key"], f"{j['traj']}.npz"))
            and not os.path.exists(os.path.join(EP.W2D, "mp3d", "views", j["key"], f"{j['traj']}.npz"))]
    import v2_frozen as VF
    o_miss = [(s_, f) for s_ in os.listdir(os.path.join(VF.WORK, "data", "views")) for f in os.listdir(os.path.join(VF.WORK, "data", "views", s_))
              if f.endswith(".npz") and not os.path.exists(os.path.join(EP.OUTD, s_, f))]
    if miss or o_miss:
        raise SystemExit(f"STOP: D100 data incomplete: MP3D missing {len(miss)}, Macarons++ missing {len(o_miss)}")


if __name__ == "__main__":
    if ARM == "d100" and "--smoke" not in sys.argv:
        guard()
    if "--config" not in sys.argv:
        sys.argv += ["--config", f"e2_{ARM}_v1.json"]
    TM.main()
