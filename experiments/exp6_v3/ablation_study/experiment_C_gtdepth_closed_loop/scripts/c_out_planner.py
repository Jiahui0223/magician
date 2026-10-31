"""EXP6 V3 Ablation Experiment C -- Macarons++ planner child for C1 / C2 (seen2scene env; planner side).

= the Task 1.4 / Task 2 V2 planner child (task1/scripts/v2_planner.py: JSON-lines protocol, whitelisted service
records, per-step resume, planner log pl_<tag>.json) with the V2 model removed:
  C1  P1 = the frozen sparse m_p1.P1Fusion of records 0..t on the whole-scene grid (t5_diagnostic.scene_geometry), the
      frozen SceneCaster geometry (v2_plan_core.Geometry.from_fusion = the B2 scoring map), scores = GT-depth DistP1
      (c_scoring.gtd_scores) from the t* of the side-channel file cq_t<t>.npz, z0 = MAGICIAN scene_scale / 2;
  C2  scores = the true one-step monotone-coverage gains of the side-channel file.
Choice: m_score.choose(scores, scene, start, t, True) -- argmax, ties -> lowest index, all zero -> the seeded fallback
of m_common. Request {"t", "svc", "cq"}; answer {"t", "pick", "fallback", "decision_s", "scores"}; {"cmd": "end"} ->
bye. Guard: the V2 guard + everything under the Experiment C work root except this episode's svc / cq / planner log
is forbidden; cq opens are counted (cq_reads).
Usage: python c_out_planner.py --exp C1|C2 --scene S --start K --rep 0 --root DIR --resume-t T
"""
import argparse
import json
import os
import sys
import time

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
import c_common as CC  # noqa: E402

sys.path.insert(0, os.path.join(CC.EXP5, "task1", "scripts"))
import v2_common as LC  # noqa: E402
import v2_frozen as VF  # noqa: E402

os.environ.setdefault("CUBLAS_WORKSPACE_CONFIG", ":4096:8")
import m_common as MC  # noqa: E402
import v2_planner as VP  # noqa: E402

for _g in CC.GROUPS:
    LC.ARMS[_g] = {"name": f"expC_{_g}", "score": "gtd" if _g == "C1" else "gtgain", "K": 1, "refresh": "R10"}
    if _g not in LC.RUN_ORDER:
        LC.RUN_ORDER.append(_g)


class Guard(VP.V2Guard):
    def __init__(self, svc_dir, plog_dir, tg, cq_dir):
        super().__init__(svc_dir, plog_dir, tg)
        self.cq_dir = os.path.abspath(cq_dir) + "/"
        self.allow.append(self.cq_dir)

    def forbidden(self, p):
        if not isinstance(p, str):
            return False
        a = os.path.abspath(p)
        if any(a.startswith(x) for x in self.allow):
            return False
        if VP.V2Guard.forbidden(self, p):
            return True
        return a.startswith(CC.WC + "/") or a.startswith(CC.W6E + "/") or a.startswith(CC.T2E + "/")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--exp", required=True, choices=list(CC.GROUPS))
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
    cq_dir = os.path.join(a.root, "cq", tg)
    f_steps = os.path.join(D["plog"], f"pl_{tg}.steps.jsonl")
    f_log = os.path.join(D["plog"], f"pl_{tg}.json")
    try:
        VF.verify_frozen()
        import numpy as np
        import torch
        torch.use_deterministic_algorithms(True, warn_only=True)
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False
        import abl_common as AC
        import m_p1
        import m_score as MS
        import t5_diagnostic as TD
        import v2_plan_core as PC
        import c_scoring as CS
        dev = torch.device("cuda")
        z0 = AC.z0_of(s)
        guard = Guard(svc_dir, D["plog"], tg, cq_dir).install()
        fr, dims, dom, win = TD.scene_geometry(s)
        fu = m_p1.P1Fusion(fr, dims, dom, device=dev) if a.exp == "C1" else None

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

        torch.cuda.reset_peak_memory_stats(dev)
        resume = None
        if T > 0:
            tr0 = time.time()
            for t in range(T):
                rec = MC.load_service_record(os.path.join(svc_dir, f"svc_{s}_s{k}_t{t}.npz"))
                assert int(rec["t"]) == t and str(rec["scene_key"]) == f"{s}/{k}"
                if fu is not None:
                    fu.integrate_record(rec)
            resume = {"T": T, "resume_s": time.time() - tr0}
        send({"ready": True, "boot_s": time.time() - t_boot, "resume": resume, "exp": a.exp, "rep": rep, "planner": E, "z0": z0,
              "scoring": "GT-depth DistP1 on sparse P1 (K=1)" if a.exp == "C1" else "true one-step monotone-coverage gain",
              "gt_channel": "cq_t<t>.npz (tstar | gain of the current legal list)"})
        t_expect = T
        n_cq = 0
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
            cq = CC.read_cq(req["cq"], a.exp, t, rec["legal_idx"])
            n_cq += 1
            n = len(rec["legal_R"])
            step = {"t": t, "n": n}
            tm = {}
            if a.exp == "C1":
                fu.integrate_record(rec)
                assert fu.n_frames == t + 1, (fu.n_frames, t)
                torch.cuda.synchronize()
                t_p1 = time.time()
                geo = PC.Geometry.from_fusion(fr, dims, dom, fu, dev)
                torch.cuda.synchronize()
                t_geo = time.time()
                tm = {"p1_s": t_p1 - tic, "geometry_s": t_geo - t_p1}
                if n == 0:
                    scores, stats = [], []
                else:
                    h0 = CS.p1_hash(fu)
                    scores, stats = CS.gtd_scores(geo, rec["legal_R"], rec["legal_T"], cq["tstar"], z0, return_views=True)
                    assert CS.p1_hash(fu) == h0, "P1 changed during GT scoring"
                del geo
                step["cand_stats"] = stats
            else:
                scores = [float(x) for x in np.asarray(cq["gain"], np.float64)]
            if n == 0:
                ch = {"pick": None, "fallback": "no_legal_action"}
            else:
                ch = MS.choose(scores, s, k, t, True)
                if not np.all(np.isfinite(scores)):
                    raise VF.InvalidRun(f"INVALID: non-finite scores at t={t}")
            torch.cuda.synchronize()
            dt = time.time() - tic
            step.update({"scores": scores, "choice": ch, "decision_s": dt, "timing": tm, "cq": os.path.basename(req["cq"]),
                         "peak_mem_gb": torch.cuda.max_memory_allocated(dev) / 2 ** 30})
            append(step)
            steps.append(step)
            send({"t": t, "pick": ch["pick"], "fallback": ch.get("fallback"), "decision_s": dt, "scores": scores, "cq_reads": n_cq})
        log = {"exp": a.exp, "method": a.exp, "planner": E, "scene": s, "start": k, "rep": rep, "tag": tg, "z0": z0,
               "scoring": "gtd" if a.exp == "C1" else "gtgain", "K": 1, "resume": resume, "boot_s": time.time() - t_boot, "steps": steps,
               "peak_mem_gb_this_process": torch.cuda.max_memory_allocated(dev) / 2 ** 30,
               "decision_s_total": float(sum(x["decision_s"] for x in steps)), "cq_reads": n_cq}
        guard.uninstall()
        log["guard"] = guard.summary()
        log["code_sha256"] = VF.code_fingerprint()
        assert not os.path.exists(f_log), f_log
        json.dump(log, open(f_log, "w"), indent=1, default=float)
        os.chmod(f_log, 0o444)
        send({"bye": True, "peak_mem_gb": log["peak_mem_gb_this_process"], "decision_s_total": log["decision_s_total"],
              "guard_denied": log["guard"]["denied"], "cq_reads": n_cq})
    except Exception as e:
        import traceback
        traceback.print_exc()
        send({"error": repr(e)})
        sys.exit(1)


if __name__ == "__main__":
    main()
