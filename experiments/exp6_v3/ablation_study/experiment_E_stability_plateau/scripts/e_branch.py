"""EXP6 V3 / ablation E -- prepare a branch root: the first T executed steps of a finished source episode (read-only), so that
e_sim.py continues the SAME state with another policy from step T on (state-swap diagnostic).
Copies (never moves) from the source episode root:
  ckpt/<tag>.jsonl           header + step lines t = 0..T-1 (each holds the executed action_key of that step)
  svc/<tag>/svc_*_t<t>.npz   t = 0..T-1 (the simulator replay re-creates each record and requires bitwise equality)
  planner_logs/pl_<tag>.steps.jsonl  lines t = 0..T-1 (the planner resume asserts this prefix; they are the SOURCE policy's logs)
and writes branch.json (source, T, sha256 of every copied file). Tag = V2M_<scene>_s<start>_r0 (rep 0 = source noise keys).
Usage: python e_branch.py --src-episode work_dirs/.../episodes/ep_V2M_<scene>_s<k>_r0.json --T t --root DIR"""
import argparse
import hashlib
import json
import os
import shutil


def sha(p):
    return hashlib.sha256(open(p, "rb").read()).hexdigest()


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--src-episode", required=True)
    ap.add_argument("--T", type=int, required=True)
    ap.add_argument("--root", required=True)
    a = ap.parse_args()
    ep = json.load(open(a.src_episode))
    src = os.path.dirname(os.path.dirname(os.path.abspath(a.src_episode)))
    tg, s, k, T = ep["tag"], ep["scene"], int(ep["start"]), a.T
    assert tg == f"V2M_{s}_s{k}_r0" and 1 <= T < 100, (tg, T)
    root = os.path.abspath(a.root)
    assert not os.path.exists(os.path.join(root, "episodes", f"ep_{tg}.json")), "branch already finished"
    for x in ("ckpt", "svc", "planner_logs", "episodes"):
        os.makedirs(os.path.join(root, x), exist_ok=True)
    lines = open(os.path.join(src, "ckpt", f"{tg}.jsonl")).read().splitlines()
    head, steps = json.loads(lines[0]), [json.loads(x) for x in lines[1:]]
    assert [x["t"] for x in steps[:T]] == list(range(T)) and len(steps) >= T
    for t in range(T):                                    # the stored episode and its checkpoint agree on the prefix
        assert steps[t]["action_key"] == ep["steps"][t]["action_key"] and steps[t]["pose_key"] == ep["steps"][t]["pose_key"]
    with open(os.path.join(root, "ckpt", f"{tg}.jsonl"), "w") as f:
        f.write("\n".join(lines[:1 + T]) + "\n")
    os.makedirs(os.path.join(root, "svc", tg), exist_ok=True)
    files = {}
    for t in range(T):
        n = f"svc_{s}_s{k}_t{t}.npz"
        shutil.copyfile(os.path.join(src, "svc", tg, n), os.path.join(root, "svc", tg, n))
        files[n] = sha(os.path.join(root, "svc", tg, n))
    pl = [x for x in open(os.path.join(src, "planner_logs", f"pl_{tg}.steps.jsonl")).read().splitlines() if x.strip()]
    assert [json.loads(x)["t"] for x in pl[:T]] == list(range(T))
    with open(os.path.join(root, "planner_logs", f"pl_{tg}.steps.jsonl"), "w") as f:
        f.write("\n".join(pl[:T]) + "\n")
    json.dump({"source_episode": os.path.abspath(a.src_episode), "source_root": src, "tag": tg, "T": T,
               "source_ckpt_header": head, "prefix_action_keys": [x["action_key"] for x in steps[:T]],
               "prefix_coverage": [x["cov"] for x in steps[:T]], "svc_sha256": files},
              open(os.path.join(root, "branch.json"), "w"), indent=1)
    print("branch", root, tg, "T", T)


if __name__ == "__main__":
    main()
