"""EXP6 V3 Ablation Experiment C -- Macarons++ closed loop with GT side channel (magician env; simulator / GT side).

task2/scripts/t24_out_sim.py -> task1/scripts/v2_sim.run UNCHANGED (official settings grid, legal list, hover, seeded
observation, official coverage, per-step checkpoint, bitwise resume, 100 moves); two plug-ins:
  EpisodeSimC  = the frozen T6 EpisodeSim + the monotone coverage of t25_out_cov maintained online (GT points, eps,
                 visibility ratio; used by C2's gains and for the online == replay check) + the last legal list cached;
  PlannerC     = the Task 2 planner client spawning c_out_planner.py; ask(t, svc) first writes the GT query of the
                 current legal list (C1: t*; C2: true gains) to <root>/cq/<tag>/cq_t<t>.npz (simulator state hash must be
                 identical before / after), sends {"t", "svc", "cq"}, then (after the answer, never sent to the child)
                 attaches the diagnostics c_diag: true gains of all candidates (C1 computes them now, C2 reuses), oracle
                 best / regret of the pick, remaining work and its reachable share, online monotone coverage.
Checks at every step: the executed candidate's precomputed sample / official coverage equal the next observation and
its monotone gain equals the online increment (|d| <= 1e-12); monotone coverage never decreases.
Groups: C1 | C2 | V2M (B2 replay for the G0 test: child = Task 3 B2 planner t32_out_planner.py, T3_P1_MODE=b2).
Usage: python c_out_sim.py --group C1 --scene S --start K [--moves N] [--root DIR] [--no-diag]
"""
import json
import os
import subprocess
import sys
import time

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
import c_common as CC  # noqa: E402

sys.path.insert(0, os.path.join(CC.EXP5, "task2", "scripts"))
import t24_out_sim as T24  # noqa: E402

LC, C, CFG, T6, T6S, VS = T24.LC, T24.C, T24.CFG, T24.T6, T24.T6S, T24.VS
np, torch = T6S.np, T6S.torch
for _g in CC.GROUPS:
    LC.ARMS[_g] = {"name": f"expC_{_g}", "score": "gtd" if _g == "C1" else "gtgain", "K": 1, "refresh": "R10"}
    if _g not in LC.RUN_ORDER:
        LC.RUN_ORDER.append(_g)
STATE = {"sim": None, "diag": True}
_EpisodeSim = T6S.EpisodeSim


class EpisodeSimC(_EpisodeSim):
    def __init__(self, scene, start, frames_root, **kw):
        super().__init__(scene, start, frames_root, **kw)
        from scipy.spatial import cKDTree
        cells = [c for c in self.gt_scene.cells if len(self.gt_scene.cells[c].cell_pts) > 0]
        self.G = torch.cat([self.gt_scene.cells[c].cell_pts for c in cells]).double().cpu().numpy()
        self.tree = cKDTree(self.G)
        self.covered_mono = np.zeros(len(self.G), bool)
        self.mono, self.off, self._legal = [], [], None
        STATE["sim"] = self

    def mono_new(self, pts):
        new = np.zeros(len(self.G), bool)
        if len(pts):
            for idx in self.tree.query_ball_point(pts, r=self.eps * (1 - 1e-12), workers=16):
                new[idx] = True
        return new & ~self.covered_mono

    def observe(self, t):
        ob = super().observe(t)
        pts = ob["part_pc"].double().cpu().numpy()
        if len(pts):
            for idx in self.tree.query_ball_point(pts, r=self.eps * (1 - 1e-12), workers=16):
                self.covered_mono[idx] = True
        m = float(self.covered_mono.sum()) / len(self.G) / self.vis
        if self.mono and m < self.mono[-1] - 1e-15:
            raise RuntimeError(f"STOP: monotone coverage decreased at t={t}")
        self.mono.append(m)
        self.off.append(ob["cov"])
        self.last_pts = ob["part_pc"].cpu().numpy().astype(np.float32)
        return ob

    def legal(self):
        legal, flags = super().legal()
        self._legal = legal
        return legal, flags


class PlannerC(T6S.PlannerClient):
    def __init__(self, a, resume_t, err_path):
        from c_gt import OutdoorGT
        cmd = [LC.SEEN2SCENE_PYTHON, os.path.join(HERE, "c_out_planner.py"), "--exp", a.exp, "--scene", a.scene,
               "--start", str(a.start), "--rep", str(a.rep), "--root", a.root, "--resume-t", str(resume_t)]
        env = T6.seen2scene_env(os.environ)
        env["PYTORCH_CUDA_ALLOC_CONF"] = "expandable_segments:True"
        if a.planner_gpu is not None:
            env["CUDA_VISIBLE_DEVICES"] = str(a.planner_gpu)
        self.a, self.grp = a, a.exp
        self.tg = LC.tag(a.exp, a.scene, a.start, a.rep)
        self.cq_dir = os.path.join(a.root, "cq", self.tg)
        os.makedirs(self.cq_dir, exist_ok=True)
        self.err = open(err_path, "a")
        self.p = subprocess.Popen(cmd, stdin=subprocess.PIPE, stdout=subprocess.PIPE, stderr=self.err, text=True, bufsize=1, cwd=LC.REPO, env=env)
        self.hello = self._read()
        if not self.hello.get("ready"):
            raise RuntimeError(f"planner did not start: {self.hello}")
        self.sim = STATE["sim"]
        self.gt = OutdoorGT(self.sim, a.scene, self.sim.dev)
        self.expect = None
        self.checks = {"executed_sample_equal": 0, "executed_official_equal": 0, "executed_mono_equal": 0, "state_hash_equal": 0, "n": 0}

    def ask(self, t, svc_path):
        sim, gt = self.sim, self.gt
        legal = sim._legal
        keys = [T6.key_of(x[0]) for x in legal]
        legal_idx = np.asarray([x[0] for x in legal], np.int64).reshape(-1, 5)
        legal_R, legal_T = [x[1] for x in legal], [x[2] for x in legal]
        cov_t, mono_t = sim.off[-1], sim.mono[-1]
        self._check_executed(t)                   # executed candidate of the previous step vs this observation
        tq = time.time()
        h0 = gt.state_hash()
        cq_path = os.path.join(self.cq_dir, CC.cq_name(t))
        gm = go = cov_as = pts = None
        if self.grp == "C1":
            CC.write_cq(cq_path, "C1", t, legal_idx, tstar=gt.tstar(legal_R, legal_T))
        else:
            gm, go, cov_as, pts = gt.gains(t, keys, cov_t)
            CC.write_cq(cq_path, "C2", t, legal_idx, gain=gm)
        if gt.state_hash() != h0:
            raise RuntimeError(f"STOP: GT query changed the simulator state at t={t}")
        self.checks["state_hash_equal"] += 1
        gt_s = time.time() - tq
        self.p.stdin.write(json.dumps({"t": int(t), "svc": svc_path, "cq": cq_path}) + "\n")
        self.p.stdin.flush()
        ans = self._read()
        if "error" in ans:
            raise RuntimeError(f"planner error at t={t}: {ans['error']}")
        pick = ans["pick"]
        if pick is not None and not (0 <= pick < len(keys)):
            raise RuntimeError(f"INVALID: illegal pick {pick} of {len(keys)} at t={t}")
        diag = {"gt_query_s": gt_s, "mono_cov_t": mono_t, "official_cov_t": cov_t, "n_legal": len(keys)}
        if STATE["diag"] and len(keys):
            td = time.time()
            if gm is None:
                gm, go, cov_as, pts = gt.gains(t, keys, cov_t)
            if gt.state_hash() != h0:
                raise RuntimeError(f"STOP: diagnostic GT query changed the simulator state at t={t}")
            diag.update({"gain": gm.tolist(), "gain_official": go.tolist(), **CC.oracle_summary(gm, list(range(len(keys))), pick),
                         **gt.reach_diag([x[3] for x in legal]), "diag_s": time.time() - td})
        self.expect = (pts[pick], cov_as[pick], float(gm[pick]), t) if (gm is not None and pick is not None) else None
        self.checks["n"] += 1
        ans["c_diag"] = diag
        return ans

    def _check_executed(self, t):
        """The candidate executed at the previous step: its precomputed sample / official coverage / monotone gain must
        equal the observation that followed (sim.observe of this step has already run)."""
        if self.expect is None:
            return
        sim = self.sim
        pts_e, cov_e, gm_e, t_prev = self.expect
        eq_s = bool(np.array_equal(sim.last_pts, pts_e))
        eq_o = bool(sim.off[-1] == cov_e)
        eq_m = bool(abs((sim.mono[-1] - sim.mono[-2]) - gm_e) <= 1e-12)
        self.checks["executed_sample_equal"] += eq_s
        self.checks["executed_official_equal"] += eq_o
        self.checks["executed_mono_equal"] += eq_m
        self.expect = None
        if not (eq_s and eq_o and eq_m):
            raise RuntimeError(f"STOP: executed candidate of t={t_prev} differs from the next observation (sample {eq_s}, official {eq_o}, mono {eq_m})")

    def close(self):
        self._check_executed(None)                 # the last move (the final observation is already in)
        bye = super().close()
        bye["checks"] = self.checks
        return bye


def main():
    import argparse
    ap = argparse.ArgumentParser()
    ap.add_argument("--group", required=True, choices=list(CC.GROUPS) + ["V2M"])
    ap.add_argument("--scene", required=True)
    ap.add_argument("--start", type=int, required=True)
    ap.add_argument("--moves", type=int, default=CFG["budget"]["moves"])
    ap.add_argument("--root", default=None)
    ap.add_argument("--planner-gpu", type=int, default=None)
    ap.add_argument("--no-diag", action="store_true")
    a = ap.parse_args()
    a.exp, a.rep = a.group, 0
    a.root = a.root or CC.outdoor_root(a.group)
    STATE["diag"] = not a.no_diag
    T6S.EpisodeSim = EpisodeSimC
    if a.group == "V2M":                       # G0 baseline replay: Task 3 B2 planner on the modified simulator
        sys.path.insert(0, CC.T3S)
        import t33_out_sim as T33
        os.environ["T3_P1_MODE"] = "b2"
        VS.Planner = T33.Planner
    else:
        VS.Planner = PlannerC
    VS.run(a)


if __name__ == "__main__":
    main()
