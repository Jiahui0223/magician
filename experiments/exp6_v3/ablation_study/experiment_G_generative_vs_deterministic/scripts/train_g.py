"""Experiment G -- training entry of D0 with the UNCHANGED V3 trainer (protocol G-v1, reports/G0_PROTOCOL.md; seen2scene env, torchrun).

torchrun --standalone --nproc_per_node 4 train_g.py --head det|flow --out DIR [trainer args: --smoke ..., --override k=v ...]

--head flow  NOTHING is patched: exactly experiments/exp6_v3/task0_vae/scripts/train.py (seen2scene.v3.trainer.main with --arm t0, the
             frozen train_v3.json / model_v3.json). Used only to prove that this entry reproduces the V3-T0 run (flag-off check).
--head det   D0. Module-global hooks only (the XB / Y / CT mechanism), installed before trainer.main():
               trainer.model_cfg      -> the T0 model config + {"head": "det"} (V3Model ignores the key; recorded in every checkpoint)
               losses.fm_terms        -> d0_head.det_terms        (same masks, same global 2 x N normalisation in the trainer)
               eval_dev.per_view_loss -> d0_head.det_per_view_loss (eval "fm_*" = the D0 objective on the same fixed subsets)
               v3.model.sample        -> d0_head.det_sample        (eval "samp_*" = the same sample metrics, K = 4 identical copies)
             Data, seed, world size, batch, optimiser, LR / KL schedules, EMA, eval cadence, checkpoints, selection / early-stop rule,
             and the per-step eps ~ N(0, 1) / tau ~ U(0, 1) draws (unused by D0, kept so the torch RNG stream is the T0 one) are untouched.
"""
import json
import os
import sys
import time

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
import d0_head as DH  # noqa: E402
from seen2scene.v3 import eval_dev as E  # noqa: E402
from seen2scene.v3 import losses as L  # noqa: E402
from seen2scene.v3 import model as V3  # noqa: E402
from seen2scene.v3 import trainer  # noqa: E402


def install_det():
    orig_model_cfg = trainer.model_cfg

    def model_cfg(arm, cfg):
        mcfg = orig_model_cfg(arm, cfg)
        assert arm == "t0" and not mcfg["xattn"]["enabled"], (arm, mcfg["xattn"])
        mcfg[DH.HEAD_KEY] = DH.DET
        return mcfg
    trainer.model_cfg = model_cfg
    L.fm_terms = DH.det_terms
    E.per_view_loss = DH.det_per_view_loss
    V3.sample = DH.det_sample
    return ["trainer.model_cfg (+head det)", "losses.fm_terms -> det_terms", "eval_dev.per_view_loss -> det_per_view_loss",
            "v3.model.sample -> det_sample"]


if __name__ == "__main__":
    i = sys.argv.index("--head")
    head = sys.argv[i + 1]
    del sys.argv[i:i + 2]
    assert head in ("flow", "det"), head
    patches = install_det() if head == "det" else []
    out = sys.argv[sys.argv.index("--out") + 1]
    if os.environ.get("RANK", "0") == "0":
        os.makedirs(out, exist_ok=True)
        with open(os.path.join(out, f"g_manifest_{int(time.time())}.json"), "w") as f:
            json.dump({"protocol": "G-v1", "head": head, "patches": patches, "argv": sys.argv, "code_sha256": DH.code_hashes(),
                       "host": os.uname()[1], "cuda_visible": os.environ.get("CUDA_VISIBLE_DEVICES"), "time": time.ctime()}, f, indent=1)
    sys.argv[1:1] = ["--arm", "t0"]
    trainer.main()
