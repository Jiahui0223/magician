"""EXP Temporal Refresh -- encoder-only refresh period R for the frozen closed-loop planners (REFRESH_PROTOCOL.md §2).

Only the 3D-encoder features h are recomputed more often. Everything that the historical R10 refresh ALSO did stays on
the R10 schedule:
  indoor  (task0 t06_planner.Planner): at t % 10 == 0 the original refresh() (region re-centred on the camera, P1 rebuilt
          from records 0..t, encode, noise key t) runs unchanged; at t % R == 0, t % 10 != 0 the encoder is re-run on the
          CURRENT map (the P1 of the R10 region with records 0..t integrated, i.e. the geometry the step scores with);
          noise and region are untouched.
  outdoor (task1 v2_planner.main): refresh_time(t) = R * (t // R) (the planner then calls its refresh(t, geo) on the current
          scene-grid geometry), and the noise seed key uses t_r = 10 * (t // 10), so the K noise maps are bitwise the
          R10 ones at every step.
R = 10 installs nothing (bitwise the historical code). P1 integration every step, candidates, DistP1, K, Euler steps
and choice rules are untouched."""
import os
import time


def patch_indoor(R):
    if R == 10:
        return "R10 (no patch)"
    import torch
    import t06_planner as T06

    def step(self, t, rec):
        import v2_plan_core as PC
        assert self.m in ("V2K4", "V2K1"), self.m
        self.recs.append(rec)
        tm, info = {}, None
        t0 = time.time()
        if t % 10 == 0:
            info = self.refresh(t)
        else:
            self.fu.integrate_record(rec)
        fr, dims, dom, win, box = self.region
        geo = PC.Geometry.from_fusion(fr, dims, dom, self.fu, self.dev)
        if t % 10 != 0 and t % R == 0:
            torch.cuda.synchronize()
            te = time.time()
            self.h = self.core.pred.encode(geo.coarse())
            torch.cuda.synchronize()
            info = {"t": t, "encoder_only": True, "encode_s": time.time() - te}
        out, tm, _ = PC.score_state(self.core, geo, self.h, rec["legal_R"], rec["legal_T"], self.eps, self.z0)
        scores = out["distp1"]
        pick, fb = T06.choose(scores, self.tag, t)
        tm["decision_s"] = time.time() - t0
        return {"t": t, "pick": pick, "scores": [float(x) for x in scores], "fallback": fb, "timing": tm, "refresh": info,
                "peak_mem_gb": torch.cuda.max_memory_allocated(self.dev) / 2 ** 30}
    T06.Planner.step = step
    return f"R{R} indoor encoder-only"


def patch_outdoor(R):
    if R == 10:
        return "R10 (no patch)"
    import v2_common as LC
    orig_key = LC.seed_key
    LC.refresh_time = lambda eid, t: R * (t // R)
    LC.seed_key = lambda scene, start, rep, t_r, k: orig_key(scene, start, rep, 10 * (t_r // 10), k)
    return f"R{R} outdoor encoder-only (noise key t_r -> 10*(t_r//10))"


def period():
    R = int(os.environ.get("TR_REFRESH", "10"))
    assert R in (10, 5, 2, 1), R
    return R
