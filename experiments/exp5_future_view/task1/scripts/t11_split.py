"""EXP5 Task 1.1 D-SPLIT (protocol §2.5): start-level hold-out S-c per Train-6 scene (1 of the 4 A starts + 2 of the
11 B starts, seeded) and the LOSO-pisa training list. Reads only the t = 0 service records (pose_idx) and the trajectory
directory names. Appends the result to task1/frozen_values.json (never overwrites an existing key).
"""
import glob
import hashlib
import json
import os

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
T1 = os.path.dirname(HERE)
REPO = os.path.abspath(os.path.join(T1, "..", "..", ".."))
CFG = json.load(open(os.path.join(T1, "configs", "data_v1.json")))
TRAJ = os.path.join(REPO, CFG["dataset_root"], "trajectories")


def starts_of(scene):
    st = {}
    for tj in sorted(os.listdir(os.path.join(TRAJ, scene))):
        f = glob.glob(os.path.join(TRAJ, scene, tj, "records", "*_t0.npz"))
        assert len(f) == 1, (scene, tj, f)
        key = tuple(int(x) for x in np.load(f[0])["pose_idx"])
        st.setdefault(key, []).append(tj)
    return st


def main():
    out = {"rule": "protocol.md v1.0 §2.5", "seed_key": CFG["holdout_seed_key"], "scenes": {}}
    for s in CFG["scenes_train6"]:
        st = starts_of(s)
        A = sorted([k for k, v in st.items() if any("-A-" in x for x in v)], key=lambda k: min(st[k]))
        B = sorted([k for k, v in st.items() if not any("-A-" in x for x in v)], key=lambda k: min(st[k]))
        assert len(A) == 4 and len(B) == 11 and len(st) == 15, (s, len(A), len(B))
        rng = np.random.default_rng(int(hashlib.sha256(CFG["holdout_seed_key"].format(scene=s).encode()).hexdigest()[:15], 16))
        ia = rng.choice(len(A), CFG["holdout_A_starts"], replace=False)
        ib = rng.choice(len(B), CFG["holdout_B_starts"], replace=False)
        hold = [A[i] for i in sorted(ia)] + [B[i] for i in sorted(ib)]
        held = sorted(x for k in hold for x in st[k])
        allt = sorted(x for v in st.values() for x in v)
        out["scenes"][s] = {"heldout_starts": [list(k) for k in hold], "heldout_trajs": held,
                            "train_trajs": [x for x in allt if x not in held], "n_trajs": len(allt)}
        print(s, "held-out starts", hold, "->", len(held), "trajs:", held)
    out["loso"] = {"scene": CFG["loso_scene"], "train_scenes": [s for s in CFG["scenes_train6"] if s != CFG["loso_scene"]]}
    fv_p = os.path.join(T1, "frozen_values.json")
    fv = json.load(open(fv_p))
    assert "split" not in fv, "split already frozen"
    fv["split"] = out
    tmp = fv_p + ".tmp"
    json.dump(fv, open(tmp, "w"), indent=1)
    os.replace(tmp, fv_p)
    n_h = sum(len(v["heldout_trajs"]) for v in out["scenes"].values())
    print(f"held-out trajectories {n_h} / 180 ({n_h / 180:.1%})")


if __name__ == "__main__":
    main()
