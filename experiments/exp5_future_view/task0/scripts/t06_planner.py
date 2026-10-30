"""EXP5 Task 0.6 -- indoor closed-loop planner child (seen2scene env; planner side). JSON lines on stdin / stdout:
{"t": t, "svc": path} -> {"pick": i, "scores": [...], "fallback": bool, "timing": {...}}.

Methods (all DistP1 + Greedy, R10, the level z0):
  frozen / C : at t in {0, 10, ..., 90} region frame around the current camera (mp3d_frame), P1 replay of records
               0..t, K = 4 worlds (frozen E-M2 world model, key "T0CL|<tag>", seed_world(key, 0, t_r, k)); every step:
               P1 updated in the refresh's region frame, SceneCaster on the current P1, abl_score.PoseScorer DistP1
               over the legal list (B0 rule).
  V2K4 / V2K1: at refresh: region frame + P1 replay, V2 encoder on the coarse grid, noise seeds
               exp5-V2|task0cl|<tag>|t<t_r>|k<k>; every step: V2 Geometry on the current P1 in the refresh's region
               frame, v2_plan_core.score_state -> DistP1 (K4 mean / K1 sample 0).
Choice: argmax, ties -> lowest index; all scores 0 -> seeded uniform fallback (sha256 "exp5-T0CL|<tag>|<t>").
A LeakGuard (EXP3b guard + MP3D paths: dataset, coverage / mesh caches, labels, traj.json, .house) is installed
before the first record is read.
"""
import argparse
import hashlib
import json
import os
import sys
import time

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
import mp3d_frame as F  # noqa: E402

import numpy as np  # noqa: E402
import torch  # noqa: E402

sys.path.insert(0, os.path.join(F.REPO, "experiments", "exp4_multiworld_belief", "task1", "scripts"))
sys.path.insert(0, os.path.join(F.REPO, "experiments", "exp4_multiworld_belief", "ablation_study", "abl_1", "common"))
sys.path.insert(0, os.path.join(F.REPO, "experiments", "exp5_future_view", "task1", "scripts"))
import m_common as MC  # noqa: E402
import m_p1  # noqa: E402
import t5_diagnostic as TD  # noqa: E402


class Guard(TD.LeakGuard):
    @staticmethod
    def forbidden(p):
        if TD.LeakGuard.forbidden(p):
            return True
        a = os.path.abspath(p) if isinstance(p, str) else ""
        return any(x in a for x in ("/matterport3d/", "/preprocessing/coverage/", "/preprocessing/mesh_cache/", "/labels/",
                                    "/eval3d/")) or a.endswith(("traj.json", ".house", "episode.json"))


def fallback(tag, t, n):
    rng = np.random.default_rng(int(hashlib.sha256(f"exp5-T0CL|{tag}|{t}".encode()).hexdigest()[:15], 16))
    return int(rng.integers(n))


def choose(scores, tag, t):
    s = np.asarray(scores, float)
    if len(s) == 0:
        return None, False
    if s.max() <= 0:
        return fallback(tag, t, len(s)), True
    return int(np.flatnonzero(s == s.max())[0]), False


class Planner:
    def __init__(self, method, card_key, tag, dev):
        self.m, self.card, self.tag, self.dev = method, F.load_card(card_key), tag, dev
        self.z0 = F.z0_of_card(self.card)
        self.recs, self.fu, self.region = [], None, None
        if method in ("frozen", "C"):
            import t04_gen as TG
            self.wm, _ = TG.build_wm(method)
        else:
            import v2_frozen as VF
            import v2_plan_core as PC
            VF.verify_frozen()
            torch.backends.cudnn.deterministic = True
            torch.backends.cudnn.benchmark = False
            fv = VF.frozen_values()
            self.core = PC.Core(os.path.join(VF.REPO, fv["checkpoint"]["path"]), dev, expect_sha=fv["checkpoint"]["sha256"])
            self.K = 4 if method == "V2K4" else 1

    def refresh(self, t):
        import m_world as MW
        import m_t7_planner as T7P
        fr, dims, dom, win, box = F.region_geometry(self.card, self.recs[-1]["cam_center"])
        self.region = (fr, dims, dom, win, box)
        self.fu = m_p1.P1Fusion(fr, dims, dom, device=self.dev)
        for r in self.recs:
            self.fu.integrate_record(r)
        info = {"t": t, "n_windows": win["n"], "box_m": box.tolist()}
        tg = time.time()
        if self.m in ("frozen", "C"):
            key = f"T0CL|{self.tag}|t{t}"
            MW.register_scene(self.wm, key, fr)
            ws, gi = T7P.generate_refresh(self.wm, self.fu, key, 4, t, dims, self.dev)
            self.worlds = ws
            info.update({"availability": gi["availability"], "world_sha256": [w["sha256"] for w in gi["worlds"]]})
        else:
            import v2_plan_core as PC
            from seen2scene.futureview import predictor as PR
            from seen2scene.futureview import rays as RY
            geo = PC.Geometry.from_fusion(fr, dims, dom, self.fu, self.dev)
            self.h = self.core.pred.encode(geo.coarse())
            self.eps = PR.noise(self.dev, self.K, lambda k: RY.sha_seed(f"exp5-V2|task0cl|{self.tag}|t{t}|k{k}"))
        info["refresh_s"] = time.time() - tg
        return info

    def step(self, t, rec):
        import abl_score as AS
        import m_score as MS
        self.recs.append(rec)
        tm, info = {}, None
        t0 = time.time()
        if t % 10 == 0:
            info = self.refresh(t)
        else:
            self.fu.integrate_record(rec)
        fr, dims, dom, win, box = self.region
        if self.m in ("frozen", "C"):
            if not self.worlds:
                scores = [0.0] * len(rec["legal_R"])
            else:
                caster = MS.SceneCaster(fr, dims, dom, self.fu.input_volume(), device=self.dev)
                scorer = AS.PoseScorer(caster, [w.to(self.dev) for w in self.worlds], z0=self.z0)
                scores = [scorer.one_step(str(j), rec["legal_R"][j], rec["legal_T"][j]) for j in range(len(rec["legal_R"]))]
                del scorer, caster
        else:
            import v2_plan_core as PC
            geo = PC.Geometry.from_fusion(fr, dims, dom, self.fu, self.dev)
            out, tm, _ = PC.score_state(self.core, geo, self.h, rec["legal_R"], rec["legal_T"], self.eps, self.z0)
            scores = out["distp1"]
        pick, fb = choose(scores, self.tag, t)
        tm["decision_s"] = time.time() - t0
        return {"t": t, "pick": pick, "scores": [float(x) for x in scores], "fallback": fb, "timing": tm, "refresh": info,
                "peak_mem_gb": torch.cuda.max_memory_allocated(self.dev) / 2 ** 30}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--method", choices=["frozen", "C", "V2K4", "V2K1"], required=True)
    ap.add_argument("--card", required=True)
    ap.add_argument("--tag", required=True)
    a = ap.parse_args()
    # protocol channel = a private copy of fd 1; fd 1 itself -> stderr, so library prints (world-model progress, etc.)
    # can never corrupt the JSON-lines protocol
    proto = os.fdopen(os.dup(1), "w")
    os.dup2(2, 1)
    sys.stdout = sys.stderr
    dev = torch.device("cuda")
    pl = Planner(a.method, a.card, a.tag, dev)
    g = Guard().install()
    proto.write(json.dumps({"ready": True}) + "\n")
    proto.flush()
    for line in sys.stdin:
        q = json.loads(line)
        if q.get("bye"):
            break
        rec = MC.load_service_record(q["svc"])
        ans = pl.step(int(q["t"]), rec)
        ans["guard_denied"] = list(g.denied)
        proto.write(json.dumps(ans) + "\n")
        proto.flush()
    g.uninstall()


if __name__ == "__main__":
    main()
