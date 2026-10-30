"""EXP5 Task 1 -- glue to the frozen EXP3b / EXP4 code (protocol §1, §6): import paths, the frozen fingerprints (checked
at the start of every formal run: any mismatch -> INVALID), the run-time code fingerprint of the files a process
actually imported, and the planner-side ray / projection helpers built on the frozen SceneCaster.
Importable from the seen2scene env (torch helpers lazily) and from the magician env (paths / fingerprints only).
"""
import hashlib
import json
import os
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
T1 = os.path.dirname(HERE)
EXP5 = os.path.dirname(T1)
REPO = os.path.abspath(os.path.join(EXP5, "..", ".."))
EXP3B = os.path.join(REPO, "experiments", "exp3b_macaronspp_transfer")
EXP4 = os.path.join(REPO, "experiments", "exp4_multiworld_belief")
ABL1_COMMON = os.path.join(EXP4, "ablation_study", "abl_1", "common")
ABL2_COMMON = os.path.join(EXP4, "ablation_study", "abl_2", "common")
TASK0_SCRIPTS = os.path.join(EXP4, "task0", "scripts")
TASK1_SCRIPTS = os.path.join(EXP4, "task1", "scripts")
T3_SCRIPTS = os.path.join(EXP4, "task3", "scripts")
DATA_V1 = os.path.join(REPO, "data", "macaronspp_belief", "v1")
WORK = os.path.join(REPO, "work_dirs", "exp5_future_view", "task1")
SEEN2SCENE_PYTHON = "/bsuscratch/jiahuizhang/envs/seen2scene/bin/python"
MAGICIAN_PYTHON = "/bsuscratch/jiahuizhang/envs/magician/bin/python"
for _p in (HERE, EXP3B, ABL1_COMMON, TASK0_SCRIPTS, TASK1_SCRIPTS, T3_SCRIPTS, REPO):
    if _p not in sys.path:
        sys.path.insert(1, _p)

FROZEN_SHA256 = {
    "experiments/exp3b_macaronspp_transfer/m_common.py": "06b4f682560d44012c35bbad9c0af2360058719a9afb53e82f9d2af452837cc3",
    "experiments/exp3b_macaronspp_transfer/m_p1.py": "d3c8df9199e9d14360ee199a36f44acd1dcf728f9bd8d310435fc8fbf5925fc8",
    "experiments/exp3b_macaronspp_transfer/m_score.py": "b47be8a6b01af582743e928994a5d4860fc7a1ab901106b98740a20d9e7a5e2e",
    "experiments/exp3b_macaronspp_transfer/t5_diagnostic.py": "a13b4561fef167001b4511e278dfbe9f9fa8bec0b1496e656e38a74c7a877ea1",
    "experiments/exp3b_macaronspp_transfer/m_world.py": "5ff0b863dedd32370418c58c7e184070ee041b8b711b75beae4ed62cc72a07b2",
    "experiments/exp3b_macaronspp_transfer/m_t6_sim.py": "9d7a5b342176ceaa4fcf479745c6161b88f5e518a94319fe667b8a3dd4e71489",
    "experiments/exp3b_macaronspp_transfer/m_t6_common.py": "acb4c55d8601ba282c94d9f3e8b6c2bc3946973bcec92da0f58b948f54067312",
    "experiments/exp3b_macaronspp_transfer/m_t7_common.py": "8a8d7e5567b0aacdd3a8df64d03e8b1ff9c44008fe93d3e4728263d1e1619612",
    "experiments/exp3b_macaronspp_transfer/m_t7_planner.py": "77134ad6118ddb0e440848a8ae1cd1de1534f1587fc749237a5a18e4c896de9c",
    "experiments/exp3b_macaronspp_transfer/m_gain_eval.py": "6cebf5a4f8b2c609cb3da5987e43c79cd6079f78ac0727a771a39781efda34b3",
    "experiments/exp3b_macaronspp_transfer/m_magician_replay.py": "6c3f3ee450080db8f08736723241681fb6c08ee0a1e8131d4d0826f13cd4c8e0",
    "experiments/exp4_multiworld_belief/ablation_study/abl_1/common/abl_score.py": "9f917eb1cfb54f114d982866a1b1024f7992fad54ff8a44a0ed3cfe47201f539",
    "experiments/exp4_multiworld_belief/ablation_study/abl_1/common/abl_planner.py": "af8452e2ab7ed6a2dbe1ced52486392afa052a6e5b9baa41007a9d8bb4db9eb6",
    "experiments/exp4_multiworld_belief/ablation_study/abl_1/common/abl_common.py": "fee28ad0d3e42cd0cad8f2e53e84a87eaf02aee6e07efa9f30c1ad4afd883e02",
    "experiments/exp4_multiworld_belief/ablation_study/abl_1/common/abl_cov_replay.py": "0ac660ae35ee5de35cc322e7ef4796781fb50d9d3c35a37f8a1ebcc89adf0ddc",
    "experiments/exp4_multiworld_belief/ablation_study/abl_1/common/abl_analyze.py": "6c81ef87c77967f78d8d52d8ea53cbb411ecf16fb5d689cf8952bc2c14fcb8d3",
    "experiments/exp4_multiworld_belief/ablation_study/abl_1/common/abl_fixed_state.py": "188ff0eb5568277c61f2873c0086fb20386f9073de63cf464928c9c3aa9e165f",
    "experiments/exp4_multiworld_belief/ablation_study/abl_2/common/abl2_sim.py": "92571175b1229bbdd42147431aac1439bf59ee14048dafa45feeae683224cca1",
    "experiments/exp4_multiworld_belief/ablation_study/abl_2/common/abl2_planner.py": "6413fee513719806a750c642f583e67ab059133b0271716f585c222de92251f2",
    "experiments/exp4_multiworld_belief/ablation_study/abl_2/common/abl2_run.py": "c824aa1357daa8c0c6e83f4b2df17e0d7ff30a00017f3d16552a5b25dcaf08da",
    "experiments/exp4_multiworld_belief/ablation_study/abl_2/common/abl2_common.py": "9fc07391290e6144befa54a5ddff30e63c2990e44f93206362bb6216bf1da4d9",
    "experiments/exp4_multiworld_belief/task1/scripts/g1_common.py": "b3216a1e04c55eaf0c9e50ce96c3ce9f160214a93d8bdde5e1e7199c55155369",
    "experiments/exp4_multiworld_belief/task3/scripts/t3_loop_oracle_diag.py": "ac9e9c0f1841c97ed6a3785a3b6018cd9a3b6eacf6af00fe3d24fecf4b00ba62",
    "experiments/exp4_multiworld_belief/task0/scripts/t0_sim.py": "f7b3139b1533b888ba8a5ec4798dfdc3c0eb59a874a144d3237ea7a9b133a4b4",
    "experiments/exp4_multiworld_belief/task0/scripts/t0_common.py": "8db817d80e08a94b8cba684517933006771982649deb6b9ac9a7c77ca4548a5c",
    "experiments/exp4_multiworld_belief/task0/scripts/t0_state_io.py": "c60c20fa372a2258968a803183bbb11c27a39e4bbbcf9e7c78d2d57469818fed",
}


class InvalidRun(RuntimeError):
    pass


def sha256_file(p):
    h = hashlib.sha256()
    with open(p, "rb") as f:
        for b in iter(lambda: f.read(1 << 20), b""):
            h.update(b)
    return h.hexdigest()


def verify_frozen():
    bad = {k: sha256_file(os.path.join(REPO, k)) for k in FROZEN_SHA256}
    bad = {k: v for k, v in bad.items() if v != FROZEN_SHA256[k]}
    if bad:
        raise InvalidRun(f"INVALID: frozen files changed: {sorted(bad)}")
    return True


def code_fingerprint(extra=()):
    """sha256 of every imported module file under the repo (+ extra files: configs, checkpoints)."""
    out = {}
    for m in list(sys.modules.values()):
        f = getattr(m, "__file__", None)
        if f and os.path.abspath(f).startswith(REPO + "/") and f.endswith(".py"):
            out[os.path.relpath(os.path.abspath(f), REPO)] = sha256_file(f)
    for f in extra:
        out[os.path.relpath(os.path.abspath(f), REPO)] = sha256_file(f)
    return dict(sorted(out.items()))


def load_cfg(name):
    return json.load(open(os.path.join(T1, "configs", name)))


def frozen_values():
    return json.load(open(os.path.join(T1, "frozen_values.json")))


# ------------------------------------------------------------------------------------------- planner-side geometry
def view_rays(caster, R, T, jitter=(0.0, 0.0)):
    """abl_score.pose_rays for the scoring grid shifted by jitter (jitter (0, 0): bitwise equal, test T-R1)."""
    import numpy as np
    import torch
    import m_common as MC
    from seen2scene.futureview import rays as RY
    u, v = RY.grid_uv(jitter)
    d_y, dz, c = MC.pixel_rays(np.asarray(R, np.float64), np.asarray(T, np.float64), u, v)
    o = torch.as_tensor(caster.f.runtime_to_model(c.reshape(1, 3))[0], dtype=caster.dt, device=caster.dev)
    d = torch.as_tensor(MC.to_z_up(d_y), dtype=caster.dt, device=caster.dev)
    tcap = torch.as_tensor(MC.SENSOR_RANGE / dz / caster.f.k_M, dtype=caster.dt, device=caster.dev)
    return o, d, tcap, torch.as_tensor(dz, dtype=torch.float64, device=caster.dev)


class StateGeometry:
    """Frozen SceneCaster of one P1 state + its observation-only cast volume and structure."""

    def __init__(self, fr, dims, dom, fusion, device):
        import torch
        import m_common as MC
        import m_score as MS
        self.caster = MS.SceneCaster(fr, dims, dom, fusion.input_volume(), device=device)
        self.vol_obs = self.caster.cast_volume(torch.full(self.caster.obs_surf.shape, MC.TRUNC, device=self.caster.dev))
        self.struct = fusion.structure()
        self.domain = self.caster.domain
        self.ext = self.caster.ext

    def project(self, rays):
        import torch
        from seen2scene.futureview import geometry as G
        o, d, tcap, dz = rays
        out = self.caster.cast(self.vol_obs, o, d, tcap)
        tk = torch.where(out["hit"], out["t"], torch.full_like(out["t"], float("inf")))
        return G.project(o, d, tcap, self.ext, tk, self.struct, self.domain)
