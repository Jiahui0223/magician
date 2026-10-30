"""EXP5 Task 1.3 S-TIME (seen2scene env; ONE process per GPU, nothing else on that GPU; protocol §4.7).

States: Dev-3 x start 0 x t in {0, 10, ..., 90} of the B0 (abl_1 A2) closed-loop trajectories (P1 replayed from B0's
service records; legal list of record t). Every part is timed with cuda.synchronize before and after; 1 warm-up, then 3
repeats, median reported.
  b0   : T1-C@2000 world model (frozen E-M2 configuration, no world cache): generate_refresh with K = 4 (per-world
         seconds -> B2 = world 0 alone), cast-volume build, DistP1 PoseScorer over the legal list with the 4 worlds
         (B0) and with world 0 (B2); regenerated world sha256 compared with B0's archived refresh record
  v2   : V2 checkpoint: encoder, candidate projection, conditioning, K = 4 / K = 1 sampling, decode + DistP1
Output W/timing/<mode>_<scene>.json
"""
import argparse
import json
import os
import statistics
import sys
import time

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
import v2_frozen as VF  # noqa: E402
import v2_common as LC  # noqa: E402

import numpy as np  # noqa: E402
import torch  # noqa: E402

import m_common as MC  # noqa: E402
import m_p1  # noqa: E402
import t5_diagnostic as TD  # noqa: E402

TIMES = list(range(0, 100, 10))
REPEATS = 3


def sync():
    torch.cuda.synchronize()
    return time.time()


def b0_states(scene):
    tg = f"A2_{scene}_s0_r0"
    svc = os.path.join(LC.B0_ROOT, "svc", tg)
    pl = json.load(open(os.path.join(LC.B0_ROOT, "planner_logs", f"pl_{tg}.json")))
    return svc, pl


def replay_to(fu, svc, scene, t0, t1):
    for t in range(t0, t1 + 1):
        fu.integrate_record(MC.load_service_record(os.path.join(svc, f"svc_{scene}_s0_t{t}.npz")))
    return MC.load_service_record(os.path.join(svc, f"svc_{scene}_s0_t{t1}.npz"))


def build_wm(fr, key):
    """= abl_1 abl_planner.build_wm (frozen T1-C@2000 E-M2 world model, same checks) with this experiment's run root."""
    import abl_common as AC
    import g1_common as GC
    import m_world as MW
    import t1_eval as E
    from seen2scene.active.world_model import EXP2_CKPT
    E.RUN_ROOT = os.path.relpath(os.path.join(VF.WORK, "timing", "config_runs"), VF.REPO)
    if E.sha_file(GC.CKPT_C2000) != AC.CKPT_SHA:
        raise VF.InvalidRun("INVALID: T1-C@2000 checkpoint sha256 differs from the frozen record")
    wm, info = E.build_outdoor_wm("C", GC.CKPT_C2000)
    cfg = wm.config_summary()
    ok = (cfg["task_class"] == "LargeScaleCompletion" and abs(cfg["overlap"] - 0.2) < 1e-12 and cfg["cpu_offload"] is True
          and cfg["drop_bbox"] is True and cfg["world_seeding"] is True and cfg["num_inference_steps"] == 10
          and all(x in cfg["argv"] for x in EXP2_CKPT))
    if not ok:
        raise VF.InvalidRun(f"INVALID: world-model configuration differs from E-M2: {cfg}")
    MW.register_scene(wm, key, fr)
    return wm, cfg


def mode_b0(scene, out):
    import abl_score as SC
    import m_score as MS
    import m_t7_planner as T7P
    dev = torch.device("cuda")
    fr, dims, dom, win = TD.scene_geometry(scene)
    key = MC.world_key(scene, 0)
    tb = time.time()
    wm, info = build_wm(fr, key)
    build_s = time.time() - tb
    svc, pl = b0_states(scene)
    arch = {r["t"]: r for r in pl["refreshes"]}
    fu = m_p1.P1Fusion(fr, dims, dom, device=dev)
    rows, last = [], -1
    z0 = LC.Z0[scene]
    for t in TIMES:
        rec = replay_to(fu, svc, scene, last + 1, t)
        last = t
        if int((fu.structure() == MC.BAND).sum()) == 0:
            rows.append({"t": t, "availability": "A2 (no BAND)"})
            continue
        first = not any("generation_repeats" in x for x in rows)
        nrep = REPEATS + 1 if first else REPEATS
        gens = []
        for r in range(nrep):
            torch.cuda.reset_peak_memory_stats(dev)
            t0 = sync()
            ws, gi = T7P.generate_refresh(wm, fu, key, 4, t, dims, dev)
            t1 = sync()
            gens.append({"refresh_s": t1 - t0, "per_world_s": [x["seconds"] for x in gi["worlds"]],
                         "peak_mem_gb": torch.cuda.max_memory_allocated(dev) / 2 ** 30,
                         "sha256": [x["sha256"] for x in gi["worlds"]], "availability": gi["availability"]})
        a_sha = [w["sha256"] for w in arch.get(t, {}).get("worlds", [])]
        worlds = [w.to(dev) for w in ws]
        keys = list(range(len(rec["legal_R"])))
        acts = list(zip([str(i) for i in keys], rec["legal_R"], rec["legal_T"]))
        sc_t = {"b0": [], "b2": [], "cast_volume_b0": [], "cast_volume_b2": []}
        for r in range(nrep):
            for nm, wl in (("b0", worlds), ("b2", worlds[:1])):
                t0 = sync()
                caster = MS.SceneCaster(fr, dims, dom, fu.input_volume(), device=dev)
                scorer = SC.PoseScorer(caster, wl, z0=z0)
                t1 = sync()
                scores = [scorer.one_step(*x) for x in acts]
                MS.choose(scores, scene, 0, t, True)
                t2 = sync()
                sc_t[f"cast_volume_{nm}"].append(t1 - t0)
                sc_t[nm].append(t2 - t1)
                del scorer, caster
        med = lambda v: statistics.median(v[1:] if first else v)   # noqa: E731  (drop the warm-up)
        rows.append({"t": t, "n_legal": len(acts), "generation_repeats": gens, "warmup_dropped": first,
                     "b0_refresh_s_median": med([g["refresh_s"] for g in gens]),
                     "b2_world0_s_median": med([g["per_world_s"][0] for g in gens]),
                     "b0_scoring_s_median": med(sc_t["b0"]), "b2_scoring_s_median": med(sc_t["b2"]),
                     "b0_setup_s_median": med(sc_t["cast_volume_b0"]), "b2_setup_s_median": med(sc_t["cast_volume_b2"]),
                     "worlds_equal_archived_B0": gens[0]["sha256"] == a_sha if a_sha else None})
        print(scene, t, {k: v for k, v in rows[-1].items() if k.endswith("median") or k.startswith("worlds")}, flush=True)
        del worlds, ws
        torch.cuda.empty_cache()
    res = {"mode": "b0", "scene": scene, "wm_build_s": build_s, "gpu": torch.cuda.get_device_name(0),
           "host": os.uname()[1], "rows": rows, "code_sha256": VF.code_fingerprint()}
    json.dump(res, open(out, "w"), indent=1, default=float)


def mode_v2(scene, out):
    import v2_plan_core as PC
    from seen2scene.futureview import predictor as PR
    from seen2scene.futureview import rays as RY
    dev = torch.device("cuda")
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    ck = VF.frozen_values()["checkpoint"]
    core = PC.Core(os.path.join(VF.REPO, ck["path"]), dev, expect_sha=ck["sha256"])
    fr, dims, dom, win = TD.scene_geometry(scene)
    svc, _ = b0_states(scene)
    fu = m_p1.P1Fusion(fr, dims, dom, device=dev)
    rows, last = [], -1
    z0 = LC.Z0[scene]
    for t in TIMES:
        rec = replay_to(fu, svc, scene, last + 1, t)
        last = t
        rr = {"t": t, "n_legal": len(rec["legal_R"])}
        for K in (4, 1):
            parts = {"geometry_s": [], "encode_s": [], "projection_s": [], "cond_s": [], "sampling_s": [],
                     "decode_score_s": [], "decision_s": [], "peak_mem_gb": []}
            for r in range(REPEATS + 1):
                torch.cuda.reset_peak_memory_stats(dev)
                t0 = sync()
                geo = PC.Geometry.from_fusion(fr, dims, dom, fu, dev)
                t1 = sync()
                h = core.pred.encode(geo.coarse())
                t2 = sync()
                eps = PR.noise(dev, K, lambda kk: RY.sha_seed(f"exp5-V2|timing|{scene}|t{t}|k{kk}"))
                out_, tm, _ = PC.score_state(core, geo, h, rec["legal_R"], rec["legal_T"], eps, z0)
                t3 = sync()
                for k_, v in (("geometry_s", t1 - t0), ("encode_s", t2 - t1), ("decision_s", t3 - t0),
                              ("peak_mem_gb", torch.cuda.max_memory_allocated(dev) / 2 ** 30)):
                    parts[k_].append(v)
                for k_ in ("projection_s", "cond_s", "sampling_s", "decode_score_s"):
                    parts[k_].append(tm[k_])
                del geo, h
            rr[f"K{K}"] = {k_: statistics.median(v[1:]) for k_, v in parts.items()}
        rows.append(rr)
        print(scene, t, rr, flush=True)
    res = {"mode": "v2", "scene": scene, "ckpt_sha256": core.ckpt_sha, "gpu": torch.cuda.get_device_name(0),
           "host": os.uname()[1], "rows": rows, "code_sha256": VF.code_fingerprint()}
    json.dump(res, open(out, "w"), indent=1, default=float)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("mode", choices=["b0", "v2"])
    ap.add_argument("--scene", required=True)
    a = ap.parse_args()
    VF.verify_frozen()
    out = os.path.join(VF.WORK, "timing", f"{a.mode}_{a.scene}.json")
    os.makedirs(os.path.dirname(out), exist_ok=True)
    assert not os.path.exists(out), out
    (mode_b0 if a.mode == "b0" else mode_v2)(a.scene, out)


if __name__ == "__main__":
    main()
