"""EXP5 Ablation S0 -- fixed-state manifest (CPU, any env; reads stored Task 2 service records and planner logs only).

For every evaluation episode x source group (V2O, V2M, MAG-G, MAG-B) x t in T_STATES: the service record t of that
closed loop defines the state (P1 input = records 0..t, legal list of record t). Identical states (same executed pose
history, e.g. t = 0 of the four groups) are merged; every source that reached the state is listed. Also stored: the
candidate set of the Task 2 V2 choice rule, the move each source actually executed, and the online V2 scores of the
V2 sources (for the reproduction check of S2).
Output W/manifests/states_<dataset>.json
"""
import json
import os
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
import ab_common as A  # noqa: E402

import numpy as np  # noqa: E402


def online_logs(dataset, source, ep):
    if source not in ("V2O", "V2M"):
        return None
    if dataset == "indoor":
        rows = [json.loads(x) for x in open(os.path.join(A.CL_IN, source, ep["tag"], "planner.jsonl"))]
        return {r["t"]: {"scores": r["scores"], "pick": r["pick"]} for r in rows}
    tg = f"{source}_{ep['scene']}_s{ep['start']}_r0"
    pl = json.load(open(os.path.join(A.CL_OUT, source, "planner_logs", f"pl_{tg}.json")))
    return {r["t"]: {"scores": r["scores"], "pick": r["choice"]["pick"]} for r in pl["steps"]}


def executed_pick(dataset, source, ep, t):
    if dataset == "indoor":
        st = json.load(open(os.path.join(A.CL_IN, source, ep["tag"], "episode.json")))["steps"][t]
        assert st["t"] == t
        return st["pick"]
    st = json.load(open(A.outdoor_episode_json(source, ep)))["steps"][t]
    assert st["t"] == t
    return st["pick"]


def main():
    os.makedirs(os.path.join(A.W, "manifests"), exist_ok=True)
    for dataset in ("indoor", "outdoor"):
        states = {}
        raw = 0
        for ep in A.episodes(dataset):
            for src in A.SOURCES:
                logs = online_logs(dataset, src, ep)
                for t in A.T_STATES[dataset]:
                    p = A.svc_path(dataset, src, ep, t)
                    z = np.load(p)
                    assert int(z["t"]) == t, p
                    hist = z["executed_pose_idx_history"]
                    assert len(hist) == t + 1
                    sid = A.state_id(dataset, ep["scene"], ep["start"], hist)
                    raw += 1
                    rec = {k: z[k] for k in ("legal_idx", "executed_pose_idx_history")}
                    cand = A.candidate_set(dataset, rec)
                    ent = {"source": src, "svc": os.path.relpath(p, A.REPO), "executed_pick": executed_pick(dataset, src, ep, t)}
                    if logs is not None:
                        ent["online"] = logs[t]
                    if sid not in states:
                        states[sid] = {"sid": sid, "dataset": dataset, "scene": ep["scene"], "start": ep["start"], "tag": ep["tag"],
                                       "split": ep["split"], "t": t, "primary_t": t in A.T_PRIMARY[dataset],
                                       "stale": t % 10, "n_legal": int(len(z["legal_idx"])), "cand": cand,
                                       "legal_idx": z["legal_idx"].tolist(), "sources": []}
                        if dataset == "indoor":
                            states[sid].update({"scan": ep["scan"], "level": ep["level"]})
                    else:
                        s0 = states[sid]
                        assert s0["t"] == t and s0["legal_idx"] == z["legal_idx"].tolist(), sid
                    states[sid]["sources"].append(ent)
        out = sorted(states.values(), key=lambda s: (s["scene"], s["start"], s["t"], s["sources"][0]["source"]))
        man = {"dataset": dataset, "t_states": A.T_STATES[dataset], "t_primary": A.T_PRIMARY[dataset], "raw_states": raw,
               "unique_states": len(out), "states": out}
        json.dump(man, open(os.path.join(A.W, "manifests", f"states_{dataset}.json"), "w"), indent=1)
        nc = np.asarray([len(s["cand"]) for s in out])
        print(dataset, "raw", raw, "unique", len(out), "| |C| mean %.2f, |C|<=1: %d" % (nc.mean(), int((nc <= 1).sum())),
              "| merged:", sum(len(s["sources"]) > 1 for s in out))


if __name__ == "__main__":
    main()
