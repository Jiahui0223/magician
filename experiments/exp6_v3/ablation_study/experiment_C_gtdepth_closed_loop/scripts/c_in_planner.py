"""EXP6 V3 Ablation Experiment C -- MP3D planner child for C1 / C2 (seen2scene env; planner side).

= the Task 2 V2 planner child (task2/scripts/t23_v2_in_planner.py: t06_planner region logic with n_win = 5, R10
refresh, sparse P1 of records 0..t, prefer-unvisited choice v2nv_planner.nv_choice) with the V2 model removed:
  C1  scores = GT-depth DistP1 (c_scoring.gtd_scores) of the legal candidates on the CURRENT sparse-P1 geometry
      (v2_plan_core.Geometry.from_fusion of the frozen m_p1.P1Fusion, i.e. the B2 scoring map) from the t* of the
      side-channel file cq_t<t>.npz written by the simulator for the current legal list;
  C2  scores = the true one-step gains of the side-channel file (no P1 is needed; the record is still read and
      validated through the whitelist).
Choice: nv_choice(scores, rec, "T2|<group>|<tag>", t) -- argmax over the unvisited legal poses (all legal if none),
ties -> lowest index, max <= 0 -> the seeded uniform fallback. Protocol: {"t", "svc", "cq"} -> answer with pick /
scores / fallback / timing / refresh / peak_mem_gb / guard_denied / cq_reads; {"bye": true} ends.
Guard: the Task 2 guard + everything under the Experiment C work root except this episode's svc/ and cq/ dirs is
forbidden; every open of a cq file is counted (cq_reads) so the GT channel is explicit in the log.
"""
import json
import os
import sys
import time

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
import c_common as CC  # noqa: E402

sys.path.insert(0, os.path.join(CC.EXP5, "task2", "scripts"))
import t23_v2_in_planner as P23  # noqa: E402  (patches TP.F.region_box_m to n_win = 5; imports NV, TP, VF)

import numpy as np  # noqa: E402
import torch  # noqa: E402

import c_scoring as CS  # noqa: E402

TP, NV = P23.TP, P23.NV


class Guard(P23.Guard):
    allow = ()
    cq_dir = ""

    @classmethod
    def forbidden(cls, p):
        a = os.path.abspath(p) if isinstance(p, str) else ""
        if any(a.startswith(x) for x in cls.allow):
            return False
        if P23.Guard.forbidden(p):
            return True
        return a.startswith(CC.WC + "/") or a.startswith(CC.W6E + "/") or a.startswith(CC.T2E + "/")


class Planner:
    """Region / P1 logic of t06_planner.Planner (V2 branch) without the encoder; GT-depth or GT-gain scores."""

    def __init__(self, group, card_key, tag, dev):
        self.g, self.card, self.tag, self.dev = group, TP.F.load_card(card_key), tag, dev
        self.z0 = TP.F.z0_of_card(self.card)
        self.recs, self.fu, self.region = [], None, None
        self.PC = None
        if group == "C1":
            import v2_plan_core as PC
            self.PC = PC

    def refresh(self, t):
        fr, dims, dom, win, box = TP.F.region_geometry(self.card, self.recs[-1]["cam_center"])
        self.region = (fr, dims, dom, win, box)
        tg = time.time()
        self.fu = TP.m_p1.P1Fusion(fr, dims, dom, device=self.dev)
        for r in self.recs:
            self.fu.integrate_record(r)
        return {"t": t, "n_windows": win["n"], "box_m": box.tolist(), "region_box_m": box.tolist(), "refresh_s": time.time() - tg}

    def step(self, t, rec, cq):
        self.recs.append(rec)
        t0 = time.time()
        info, tm, extra = None, {}, None
        if self.g == "C1":
            if t % 10 == 0:
                info = self.refresh(t)
            else:
                self.fu.integrate_record(rec)
            fr, dims, dom, win, box = self.region
            t1 = time.time()
            geo = self.PC.Geometry.from_fusion(fr, dims, dom, self.fu, self.dev)
            torch.cuda.synchronize()
            tm["geometry_s"] = time.time() - t1
            h0 = CS.p1_hash(self.fu)
            t2 = time.time()
            scores, extra = CS.gtd_scores(geo, rec["legal_R"], rec["legal_T"], cq["tstar"], self.z0, return_views=True)
            torch.cuda.synchronize()
            tm["gtd_score_s"] = time.time() - t2
            assert CS.p1_hash(self.fu) == h0, "P1 changed during GT scoring"
            del geo
        else:
            scores = [float(x) for x in np.asarray(cq["gain"], np.float64)]
        tm["decision_s"] = time.time() - t0
        return {"t": t, "scores": [float(x) for x in scores], "timing": tm, "refresh": info, "cand_stats": extra,
                "peak_mem_gb": torch.cuda.max_memory_allocated(self.dev) / 2 ** 30}


def main():
    import argparse
    ap = argparse.ArgumentParser()
    ap.add_argument("--group", choices=list(CC.GROUPS), required=True)
    ap.add_argument("--card", required=True)
    ap.add_argument("--tag", required=True)
    ap.add_argument("--svc-dir", required=True)
    ap.add_argument("--cq-dir", required=True)
    a = ap.parse_args()
    proto = os.fdopen(os.dup(1), "w")
    os.dup2(2, 1)
    sys.stdout = sys.stderr
    torch.use_deterministic_algorithms(True, warn_only=True)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    dev = torch.device("cuda")
    pl = Planner(a.group, a.card, a.tag, dev)
    Guard.allow = (os.path.abspath(a.svc_dir) + "/", os.path.abspath(a.cq_dir) + "/")
    Guard.cq_dir = os.path.abspath(a.cq_dir) + "/"
    g = Guard().install()
    proto.write(json.dumps({"ready": True, "info": {"group": a.group, "base": "t06_planner region / P1 (no model) + prefer-unvisited (v2nv_planner.nv_choice)",
                                                    "n_win": P23.C.N_WIN, "scoring": "GT-depth DistP1 on sparse P1 (K=1)" if a.group == "C1" else "true one-step C_obs gain",
                                                    "gt_channel": "cq_t<t>.npz (tstar | gain of the current legal list)", "z0": pl.z0}}) + "\n")
    proto.flush()
    n_cq = 0
    for line in sys.stdin:
        q = json.loads(line)
        if q.get("bye"):
            break
        rec = TP.MC.load_service_record(q["svc"])
        assert int(rec["t"]) == int(q["t"])
        cq = CC.read_cq(q["cq"], a.group, int(q["t"]), rec["legal_idx"])
        ans = pl.step(int(q["t"]), rec, cq)
        ans["pick"], ans["fallback"], ans["nv_candidates"] = NV.nv_choice(ans["scores"], rec, f"T2|{a.group}|{a.tag}", int(q["t"]))
        if ans["pick"] is not None:
            assert 0 <= ans["pick"] < len(rec["legal_idx"]) and ans["pick"] in ans["nv_candidates"]
        n_cq = sum(1 for p in g.paths if isinstance(p, str) and os.path.abspath(p).startswith(Guard.cq_dir))
        ans["cq_reads"] = n_cq
        ans["guard_denied"] = list(g.denied)
        proto.write(json.dumps(ans) + "\n")
        proto.flush()
    g.uninstall()


if __name__ == "__main__":
    main()
