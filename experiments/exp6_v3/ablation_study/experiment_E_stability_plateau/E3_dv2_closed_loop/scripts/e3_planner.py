"""E3 D-v2 planner child: the frozen D-v1 chain (d_out_planner.py: t24_v2_out_planner under p1_full.patch() b2, planner_v3.install(),
d_policy.install(D_POLICY)) + dv2.install(scene) with env E3_DV2 = off | full | first (config env E3_DV2_CONFIG).
off: the D-v2 grid is still fed but value_cov is the frozen function -> D1 bitwise."""
import os
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
X = os.path.dirname(os.path.dirname(os.path.dirname(os.path.dirname(os.path.dirname(HERE)))))
EXP5 = os.path.join(X, "exp5_future_view")
sys.path.insert(0, os.path.join(EXP5, "task2", "scripts"))
import t24_v2_out_planner as P24  # noqa: E402

sys.path.insert(0, os.path.join(EXP5, "task3_full_depth_p1", "scripts"))
sys.path.insert(0, os.path.join(X, "exp6_v3", "shared"))
sys.path.insert(0, os.path.join(X, "exp6_v3", "ablation_study", "experiment_D_h3_lookahead", "scripts"))
sys.path.insert(0, HERE)
import p1_full as PF  # noqa: E402
import planner_v3 as PV3  # noqa: E402
import d_policy as DP  # noqa: E402

if __name__ == "__main__":
    sys.stderr.write(f"V3_CKPT={PV3.ckpt_override()}\n")
    sys.stderr.write(f"T3_P1_MODE={PF.patch()}\n")
    PV3.install()
    DP.install(os.environ["D_POLICY"])
    import torch
    import yaml
    import dv2 as DV2
    mode = os.environ["E3_DV2"]
    assert mode in ("off", "full", "first"), mode
    cfg = yaml.safe_load(open(os.environ["E3_DV2_CONFIG"]))
    arg = lambda k: sys.argv[sys.argv.index(k) + 1]  # noqa: E731
    DV2.install(arg("--scene"), torch.device("cuda"), cfg["dv2"])
    DV2.STATE["mode"] = mode
    DV2.install_step_log(os.path.join(arg("--root"), "planner_logs", f"dv2_{arg('--exp')}_{arg('--scene')}_s{arg('--start')}_r{arg('--rep')}.jsonl"))
    sys.stderr.write("E3_DV2 " + str({"mode": mode, "version": cfg["version"], **{k: DV2.CFG[k] for k in ("cell", "step", "margin", "min_t", "support", "footprint")}}) + "\n")
    P24.main()
