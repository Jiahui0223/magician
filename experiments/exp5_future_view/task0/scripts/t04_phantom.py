"""EXP5 Task 0 -- POST-HOC diagnostic (not pre-registered; added after the T0.6 closed-loop behaviour was seen):
"phantom gain" of the world-model DistP1 on the fixed states. Planner-side quantities only (no GT): per state the
frozen P1 replay (identical to t04_gen), the saved int8 worlds (w = q / 127 * TRUNC; quantisation 0.024 voxel), the
frozen SceneCaster + abl_score.new_set over the legal candidates. phantom_share = fraction of the candidate's DistP1
weight carried by new voxels that the CURRENT P1 already marks EMPTY (observed free space): such voxels can never be
observed as surface, yet the frozen B0 rule counts them as gain. V2 is 0 by construction (decoded hits only in UNKNOWN
voxels, T-S2). Also: the share for the argmax candidate. Output W/analysis/phantom_<protocol>_<model>.json
"""
import argparse
import json
import os
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
import mp3d_frame as F  # noqa: E402
import t04_gen as TG  # noqa: E402

import numpy as np  # noqa: E402
import torch  # noqa: E402

import m_common as MC  # noqa: E402

ROOTS = {"M": ("fixed_state", "indoor_states.json", "traj"), "O": ("fixed_state_out", "outdoor_states.json", "outdoor_traj")}


def main():
    import abl_score as AS
    import m_score as MS
    ap = argparse.ArgumentParser()
    ap.add_argument("--protocol", choices=["M", "O"], required=True)
    ap.add_argument("--model", choices=["frozen", "C"], required=True)
    ap.add_argument("--part", default="0/1")
    a = ap.parse_args()
    root, sf, td = ROOTS[a.protocol]
    TG.TRAJ_DIR = td
    dev = torch.device("cuda")
    sts = json.load(open(os.path.join(F.W, "manifests", sf)))["states"]
    pi, pn = (int(x) for x in a.part.split("/"))
    rows = []
    for st in sts[pi::pn]:
        p = os.path.join(F.W, root, a.model, f"{st['id']}.npz")
        if not os.path.exists(p):
            continue
        z = np.load(p)
        if z["worlds_q"].ndim != 4:
            continue
        card = F.load_card(st["level_key"])
        rec, fr, dims, dom, win, box, fu = TG.replay(card, st["traj"], st["t"], dev)
        s = fu.structure()
        assert np.array_equal(s.cpu().numpy(), z["structure"])
        empty = (s == MC.EMPTY).reshape(-1)
        caster = MS.SceneCaster(fr, dims, dom, fu.input_volume(), device=dev)
        vols = [caster.cast_volume(torch.as_tensor(q.astype(np.float32) / 127.0 * MC.TRUNC, device=dev)) for q in z["worlds_q"]]
        z0 = F.z0_of_card(card)
        cand = []
        for j in range(len(rec["legal_R"])):
            rays = AS.pose_rays(caster, rec["legal_R"][j], rec["legal_T"][j])
            tot = ph = 0.0
            for v in vols:
                ids, wv = AS.new_set(caster, v, rays, z0)
                if len(ids):
                    tot += float(wv.sum())
                    ph += float(wv[empty[ids]].sum())
            cand.append({"distp1": tot / len(vols), "phantom": ph / len(vols)})
        sc = np.array([c["distp1"] for c in cand])
        k = int(np.argmax(sc)) if sc.max() > 0 else None
        rows.append({"id": st["id"], "stratum": st["stratum"], "house": st["house"],
                     "phantom_share_mean": float(np.mean([c["phantom"] / c["distp1"] for c in cand if c["distp1"] > 0])) if (sc > 0).any() else None,
                     "phantom_share_pick": (cand[k]["phantom"] / cand[k]["distp1"]) if k is not None else None,
                     "candidates": cand})
        print(st["id"], a.model, round(rows[-1]["phantom_share_mean"] or 0, 3), round(rows[-1]["phantom_share_pick"] or 0, 3), flush=True)
        del vols, caster, fu
        torch.cuda.empty_cache()
    od = os.path.join(F.W, "analysis")
    os.makedirs(od, exist_ok=True)
    json.dump({"doc": __doc__, "rows": rows}, open(os.path.join(od, f"phantom_{a.protocol}_{a.model}_part{pi}.json"), "w"), indent=1)


if __name__ == "__main__":
    main()
