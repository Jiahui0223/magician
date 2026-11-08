"""Experiment XB / Task B2 -- decision counterfactuals (oracle upper bounds) of V3-T1, XA-SR, XA-MI through the unchanged CT tool ct_cf.py
(pred / no_fp / no_fn / hit_oracle / depth_oracle / perfect / perfect_all; GT only to build the counterfactual samples; same states, noise and candidates).
xb_fixed is imported first: it patches the CT paths (output under experiment_XB_bottleneck_audit) and the core loader (XA models through the XA loader).
Checkpoint names: T1@none | XA_SR@none | XA_MI@none. Usage: python xb_cf.py --set indoor|explore|confirm --group G --ckpt NAME [--only TAG]"""
import os
import sys

R = "/bsuscratch/jiahuizhang/projects/occupancy/seen2scene"
sys.path.insert(0, os.path.join(R, "experiments/exp6_v3/ablation_study/experiment_XB_bottleneck_audit/scripts"))
import xb_fixed  # noqa: E402,F401
sys.path.insert(0, os.path.join(R, "experiments/exp6_v3/ablation_study/experiment_CT_continued_training/scripts"))
import ct_cf as CF  # noqa: E402

if __name__ == "__main__":
    CF.main()
