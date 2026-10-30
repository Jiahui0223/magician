"""EXP5 Task 1.4 -- V2 planner child (seen2scene env; protocol §5). Same JSON-lines protocol, service-record whitelist,
LeakGuard and per-step resume as the frozen abl_1 / abl_2 planners, with the V2 decision:

  per step t: P1 integrate the whitelisted record t -> Geometry (frozen SceneCaster) -> refresh (R10: t in {0, 10, ..,
  90}; native: every t): encoder features h of the CURRENT P1 coarse grid + K noise maps seed_key(scene, start, rep,
  t_r, k) -> projection of every legal candidate (current P1) -> K samples with shared noise -> decode -> score:
    distp1 | u_raw | dist_u | gu (r(DistP1) + lambda r(DistU)) with delta / lambda from frozen_values.json
  -> m_score.choose (argmax, ties lowest index, all-zero / no-signal -> the seeded fallback of m_common).
Resume (--resume-t T): records 0..T-1 are re-integrated; h and the noise of the last refresh are recomputed from the P1
state of that time (deterministic -> bitwise equal to an uninterrupted run).
Guard: T5-D LeakGuard + EXP4 / T6 / T7a outputs + every EXP5 output except this episode's service records and planner
logs (labels, tstar, evaluation-side results are forbidden); the checkpoint and configs are read before the guard.
"""
import argparse
import json
import os
import sys
import time

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
import v2_common as LC  # noqa: E402
import v2_frozen as VF  # noqa: E402

os.environ.setdefault("CUBLAS_WORKSPACE_CONFIG", ":4096:8")
import m_common as MC  # noqa: E402
import m_t6_common as T6  # noqa: E402
import m_t7_common as T7  # noqa: E402
from t5_diagnostic import LeakGuard  # noqa: E402

EXP4_WORK = os.path.join(LC.REPO, "work_dirs", "exp4_multiworld_belief")


class V2Guard(LeakGuard):
    def __init__(self, svc_dir, plog_dir, tg):
        super().__init__()
        self.allow = [os.path.abspath(svc_dir) + "/", os.path.join(os.path.abspath(plog_dir), f"pl_{tg}.")]

    def forbidden(self, p) -> bool:
        if LeakGuard.forbidden(p):
            return True
        if not isinstance(p, str):
            return False
        a = os.path.abspath(p)
        if a.startswith(os.path.abspath(T6.T6_ROOT) + "/") or a.startswith(os.path.abspath(T7.T7_ROOT) + "/"):
            return True
        if any(a.startswith(x) for x in self.allow):
            return False
        if a.startswith(EXP4_WORK + "/") or a.startswith(LC.WORK + "/"):
            return True
        return os.path.basename(a).startswith(("ep_", "replay_", "rep_", "diag_", "gain_", "tstar"))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--exp", required=True, choices=LC.RUN_ORDER)
    ap.add_argument("--scene", required=True)
    ap.add_argument("--start", type=int, required=True)
    ap.add_argument("--rep", type=int, required=True)
    ap.add_argument("--root", required=True)
    ap.add_argument("--resume-t", type=int, default=0)
    a = ap.parse_args()
    proto = os.fdopen(os.dup(1), "w", buffering=1)
    os.dup2(2, 1)
    sys.stdout = sys.stderr

    def send(obj):
        proto.write(json.dumps(obj, default=float) + "\n")
        proto.flush()

    t_boot = time.time()
    E = LC.ARMS[a.exp]
    s, k, rep, T = a.scene, a.start, a.rep, a.resume_t
    tg = LC.tag(a.exp, s, k, rep)
    D = LC.dirs(a.root, tg)
    svc_dir = os.path.abspath(D["svc"])
    f_steps = os.path.join(D["plog"], f"pl_{tg}.steps.jsonl")
    f_log = os.path.join(D["plog"], f"pl_{tg}.json")
    try:
        VF.verify_frozen()
        import numpy as np
        import torch
        torch.use_deterministic_algorithms(True, warn_only=True)
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False
        import m_p1
        import m_score as MS
        import t5_diagnostic as TD
        import v2_plan_core as PC
        from seen2scene.futureview import predictor as PR
        from seen2scene.futureview import rays as RY
        from seen2scene.futureview import scoring as SCO
        fv = VF.frozen_values()
        test = os.environ.get("V2_TEST_OVERRIDE")          # code tests only: never under a formal arm root
        if test:
            assert "/closed_loop/_dev_tests/" in os.path.abspath(a.root) + "/", "test override outside _dev_tests"
            fv = dict(fv, **json.load(open(test)))
        if "checkpoint" not in fv:
            raise VF.InvalidRun("INVALID: no frozen checkpoint in frozen_values.json")
        ck = fv["checkpoint"]
        need_u = E["score"] in ("u_raw", "dist_u", "gu")
        if need_u and "scoring" not in fv:
            raise VF.InvalidRun("INVALID: delta / lambda not frozen yet")
        delta = fv["scoring"]["delta_vox"] if need_u else None
        lam = fv["scoring"]["lambda"] if E["score"] == "gu" else None
        if need_u and not fv["scoring"]["u_raw_valid" if E["score"] == "u_raw" else "dist_u_valid"]:
            raise VF.InvalidRun("INVALID: this U was judged FAIL (protocol §4.4); the arm must not run")
        dev = torch.device("cuda")
        core = PC.Core(os.path.join(LC.REPO, ck["path"]), dev, expect_sha=ck["sha256"])
        cfg_sha = {n: VF.sha256_file(os.path.join(VF.T1, "configs", n)) for n in ("closed_loop_v1.json", "scoring_v1.json", "model_v1.json")}
        guard = V2Guard(svc_dir, D["plog"], tg).install()
        fr, dims, dom, win = TD.scene_geometry(s)
        fu = m_p1.P1Fusion(fr, dims, dom, device=dev)
        z0 = LC.Z0[s]
        K = E["K"]

        def read_jsonl(p, tmax):
            out = []
            if os.path.exists(p):
                for line in open(p):
                    try:
                        r = json.loads(line)
                    except json.JSONDecodeError:
                        break
                    if r["t"] < tmax:
                        out.append(r)
            return out

        if T == 0 and os.path.exists(f_steps):
            os.remove(f_steps)
        steps = read_jsonl(f_steps, T)
        with open(f_steps, "w") as f:
            for r in steps:
                f.write(json.dumps(r, default=float) + "\n")
        assert [r["t"] for r in steps] == list(range(T)), ("planner step log does not match resume point", T)

        def append(r):
            with open(f_steps, "a") as f:
                f.write(json.dumps(r, default=float) + "\n")
                f.flush()
                os.fsync(f.fileno())

        state = {"t_r": None, "h": None, "eps": None, "refreshes": []}
        torch.cuda.reset_peak_memory_stats(dev)

        def refresh(t, geo):
            tr = time.time()
            torch.cuda.synchronize()
            coarse = geo.coarse()
            state["h"] = core.pred.encode(coarse)
            state["eps"] = PR.noise(dev, K, lambda kk: RY.sha_seed(LC.seed_key(s, k, rep, t, kk)))
            state["t_r"] = t
            torch.cuda.synchronize()
            r = {"t": t, "encode_s": time.time() - tr,
                 "coarse_sha256": VF.hashlib.sha256(coarse.cpu().numpy().tobytes()).hexdigest()}
            state["refreshes"].append(r)
            return r

        resume = None
        if T > 0:
            tr0 = time.time()
            last = LC.refresh_time(a.exp, T - 1)
            for t in range(T):
                rec = MC.load_service_record(os.path.join(svc_dir, f"svc_{s}_s{k}_t{t}.npz"))
                assert int(rec["t"]) == t and str(rec["scene_key"]) == f"{s}/{k}"
                fu.integrate_record(rec)
                if t == last:
                    refresh(t, PC.Geometry.from_fusion(fr, dims, dom, fu, dev))
            resume = {"T": T, "last_refresh": last, "resume_s": time.time() - tr0}
        send({"ready": True, "boot_s": time.time() - t_boot, "resume": resume, "exp": a.exp, "rep": rep,
              "ckpt_sha256": core.ckpt_sha, "planner": E, "delta_vox": delta, "lambda": lam})
        t_expect = T
        for line in sys.stdin:
            req = json.loads(line)
            if req.get("cmd") == "end":
                break
            t = int(req["t"])
            assert t == t_expect, (t, t_expect)
            t_expect += 1
            p = os.path.abspath(req["svc"])
            assert os.path.dirname(p) == svc_dir and os.path.basename(p) == f"svc_{s}_s{k}_t{t}.npz", p
            torch.cuda.synchronize()
            tic = time.time()
            rec = MC.load_service_record(p)
            assert int(rec["t"]) == t and str(rec["scene_key"]) == f"{s}/{k}"
            fu.integrate_record(rec)
            assert fu.n_frames == t + 1, (fu.n_frames, t)
            torch.cuda.synchronize()
            t_p1 = time.time()
            geo = PC.Geometry.from_fusion(fr, dims, dom, fu, dev)
            torch.cuda.synchronize()
            t_geo = time.time()
            rinfo = None
            if LC.refresh_time(a.exp, t) == t:
                rinfo = refresh(t, geo)
            n = len(rec["legal_R"])
            step = {"t": t, "n": n, "t_r": state["t_r"]}
            sc_all, tm = {}, {}
            if n == 0:
                scores, ch = [], {"pick": None, "fallback": "no_legal_action"}
            else:
                out, tm, _ = PC.score_state(core, geo, state["h"], rec["legal_R"], rec["legal_T"], state["eps"], z0,
                                            deltas=(delta,) if need_u else ())
                sc_all = {kk: v for kk, v in out.items()}
                if E["score"] == "distp1":
                    scores = out["distp1"]
                    ch = MS.choose(scores, s, k, t, True)
                elif E["score"] == "u_raw":
                    scores = out[f"u_raw@{delta}"]
                    ch = MS.choose(scores, s, k, t, True)
                elif E["score"] == "dist_u":
                    scores = out[f"dist_u@{delta}"]
                    ch = MS.choose(scores, s, k, t, True)
                else:
                    scores = SCO.gu_scores(out["distp1"], out[f"dist_u@{delta}"], lam)
                    if max(out["distp1"]) == 0.0 and max(out[f"dist_u@{delta}"]) == 0.0:
                        ch = {"pick": MC.fallback_choice(s, k, t, n), "fallback": "all_zero"}
                    else:
                        ch = {"pick": MC.argmax_lowest(scores), "fallback": None}
                if not np.all(np.isfinite(scores)):
                    raise VF.InvalidRun(f"INVALID: non-finite scores at t={t}")
            torch.cuda.synchronize()
            dt = time.time() - tic
            step.update({"scores": scores, "choice": ch, "all_scores": sc_all, "decision_s": dt,
                         "p1_s": t_p1 - tic, "geometry_s": t_geo - t_p1, "refresh": rinfo, "timing": tm,
                         "peak_mem_gb": torch.cuda.max_memory_allocated(dev) / 2 ** 30})
            append(step)
            steps.append(step)
            send({"t": t, "pick": ch["pick"], "fallback": ch.get("fallback"), "decision_s": dt, "scores": scores})
            del geo
        log = {"exp": a.exp, "method": a.exp, "planner": E, "scene": s, "start": k, "rep": rep, "tag": tg,
               "ckpt": ck, "configs_sha256": cfg_sha, "test_override": bool(test), "delta_vox": delta, "lambda": lam, "K": K,
               "refresh": E["refresh"], "resume": resume, "boot_s": time.time() - t_boot, "steps": steps,
               "refreshes": state["refreshes"], "peak_mem_gb_this_process": torch.cuda.max_memory_allocated(dev) / 2 ** 30,
               "decision_s_total": float(sum(x["decision_s"] for x in steps))}
        guard.uninstall()
        log["guard"] = guard.summary()
        log["code_sha256"] = VF.code_fingerprint()
        assert not os.path.exists(f_log), f_log
        json.dump(log, open(f_log, "w"), indent=1, default=float)
        os.chmod(f_log, 0o444)
        send({"bye": True, "peak_mem_gb": log["peak_mem_gb_this_process"], "decision_s_total": log["decision_s_total"],
              "guard_denied": log["guard"]["denied"]})
    except Exception as e:
        import traceback
        traceback.print_exc()
        send({"error": repr(e)})
        sys.exit(1)


if __name__ == "__main__":
    main()
