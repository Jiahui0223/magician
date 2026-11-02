"""EXP6 V5.2 -- multi-surface belief metrics on the evaluation-side mesh GT (seen2scene env, CPU): the V5 script
exp6_v5/scripts/eval_belief_metrics.py run unchanged (same queries: every GT surface + 64 log-uniform depths, Delta 8 VS,
threshold 0.5; recall of the first / second+ surfaces by gap and depth, grid precision / recall / Brier / ECE, false
surfaces behind the first) with the V5.2 checkpoint: only the model class is swapped (it uses model.dec.surface_logits,
the V5.2 surface head). Usage: python eval_belief_metrics_v52.py --ckpt V52_CKPT --dirs D1 [D2 ...] --out OUT.json"""
import os
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
REPO = os.path.dirname(os.path.dirname(os.path.dirname(HERE)))
sys.path.insert(0, REPO)
sys.path.insert(0, os.path.join(REPO, "experiments", "exp6_v5", "scripts"))
import eval_belief_metrics as EBM  # noqa: E402

from seen2scene.futureview import v52_model as V52  # noqa: E402

EBM.V5.V5Model = V52.V52Model

if __name__ == "__main__":
    EBM.main()
