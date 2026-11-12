"""FisherRF reproduction, Stage F1 -- Macarons++ closed loop, simulator side (magician env; GT side).

= task2/scripts/t24_out_sim.py -> task1/scripts/v2_sim.run -> exp3b m_t6_sim.EpisodeSim, all run UNCHANGED (frozen M
protocol: official settings grid, official neighbours -> get_valid_neighbors -> GT segment test, hover at dead ends,
seeded observation, MAGICIAN-style coverage, per-step checkpoint), with two in-memory additions of this wrapper only:
  1. group "FisherRF" is registered in t24's CFG / v2_common.ARMS (no config file is edited);
  2. RGB side file (decisions D1 (a) + D9): EpisodeSim.service_record is wrapped to keep the frame of the record, and
     Planner.ask writes svc_<s>_s<k>_t<t>_rgb.npz = {t, rgb uint8 (256, 456, 3), R, T, rgb_float_min/max} next to the
     record before forwarding the request. The RGB is ob["frame"]["rgb"], i.e. the MAGICIAN capture_image render
     (SoftPhong, ambient light, adjust_contrast) of the SAME frame file as the record's zbuf / mask. The whitelisted
     record itself (m_common.SERVICE_KEYS_*) is unchanged.
Planner child: fisherrf_planner.py in the fisherrf env (/bsuscratch/jiahuizhang/envs/fisherrf).
Usage: python fisherrf_sim_out.py --scene S --start K [--moves N] --root DIR [--init-pose tester|docstring]
       [--verify-every K] [--planner fisherrf_planner.py|fisherrf_full_planner.py]   (CUDA_VISIBLE_DEVICES: GPU of both)
"""
import os
import subprocess
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
X = os.path.dirname(os.path.dirname(os.path.dirname(HERE)))           # experiments/
sys.path.insert(0, os.path.join(X, "exp5_future_view", "task2", "scripts"))
import t24_out_sim as T24  # noqa: E402

LC, C, CFG, T6, T6S, VS = T24.LC, T24.C, T24.CFG, T24.T6, T24.T6S, T24.VS
np, torch = T6S.np, T6S.torch
FISHERRF_PYTHON = "/bsuscratch/jiahuizhang/envs/fisherrf/bin/python"
GROUP = "FisherRF"
CFG["groups"][GROUP] = {"name": "FisherRF official core (FisherRF-active-mapping 8b196cd) on the legal list"}
LC.ARMS[GROUP] = {"name": "cvpr_fisherrf", "score": "fisherrf_eig", "K": None, "refresh": "native"}
if GROUP not in LC.RUN_ORDER:
    LC.RUN_ORDER.append(GROUP)

_STASH = {}
_service_record = T6S.EpisodeSim.service_record


def service_record(self, t, ob, legal):
    rec = _service_record(self, t, ob, legal)
    _STASH.clear()
    _STASH.update(t=int(t), rgb=ob["frame"]["rgb"], R=rec["R"], T=rec["T"])
    return rec


T6S.EpisodeSim.service_record = service_record


def write_rgb_side(svc_path, t):
    if _STASH.get("t") != int(t):
        raise RuntimeError(f"RGB stash is for t={_STASH.get('t')}, request t={t}")
    x = _STASH["rgb"][0, ..., :3].float()                       # (H, W, 3) in [0, 1] (adjust_contrast clamps)
    lo, hi = float(x.min()), float(x.max())
    if hi > 1.0 + 1e-4 or lo < -1e-4:
        raise RuntimeError(f"unexpected RGB range [{lo}, {hi}]")
    u8 = (x.clamp(0, 1) * 255.0).round().to(torch.uint8).cpu().numpy()
    side = svc_path[:-len(".npz")] + "_rgb.npz"
    np.savez_compressed(side, t=np.asarray(int(t)), rgb=u8, R=_STASH["R"], T=_STASH["T"],
                        rgb_float_min=np.asarray(lo), rgb_float_max=np.asarray(hi))
    return side


class Planner(T6S.PlannerClient):
    def __init__(self, a, resume_t, err_path):
        tg = LC.tag(a.exp, a.scene, a.start, a.rep)
        D = LC.dirs(a.root, tg)
        cmd = [FISHERRF_PYTHON, "-B", "-s", os.path.join(HERE, os.environ["FRF_PLANNER"]), "--scene", a.scene,
               "--start", str(a.start), "--svc-dir", D["svc"], "--log", os.path.join(D["plog"], f"pl_{tg}.steps.jsonl"),
               "--workdir", os.path.join(a.root, "planner_work"), "--resume-t", str(resume_t),
               "--init-pose", os.environ["FRF_INIT_POSE"], "--verify-every", os.environ["FRF_VERIFY_EVERY"]]
        env = {k: v for k, v in os.environ.items() if k not in ("LD_PRELOAD", "PYTHONPATH", "PYTHONHOME")}
        env.update(PYTHONDONTWRITEBYTECODE="1", PYTHONNOUSERSITE="1", SLURM_JOB_QOS="kd-high",
                   CONDA_PREFIX=os.path.dirname(os.path.dirname(FISHERRF_PYTHON)))
        if a.planner_gpu is not None:
            env["CUDA_VISIBLE_DEVICES"] = str(a.planner_gpu)
        self.err = open(err_path, "a")
        self.p = subprocess.Popen(cmd, stdin=subprocess.PIPE, stdout=subprocess.PIPE, stderr=self.err, text=True,
                                  bufsize=1, cwd=LC.REPO, env=env)
        self.hello = self._read()
        if not self.hello.get("ready"):
            raise RuntimeError(f"planner did not start: {self.hello}")

    def ask(self, t, svc_path):
        write_rgb_side(svc_path, t)
        return super().ask(t, svc_path)


if __name__ == "__main__":
    opts = {"--init-pose": ("FRF_INIT_POSE", "tester"), "--verify-every": ("FRF_VERIFY_EVERY", "0"),
            "--planner": ("FRF_PLANNER", "fisherrf_planner.py")}      # F1b: --planner fisherrf_full_planner.py
    for flag, (env, default) in opts.items():
        os.environ[env] = default
        if flag in sys.argv:
            i = sys.argv.index(flag)
            os.environ[env] = sys.argv[i + 1]
            del sys.argv[i:i + 2]
    sys.argv += ["--group", GROUP]
    VS.Planner = Planner
    T24.main()
