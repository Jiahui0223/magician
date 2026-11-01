"""Freeze the first-stage configs / code of V4 and V3.1: sha256 of every config and trainer file + the model modules into
experiments/exp6_v4/configs/FROZEN_SHA256.json and experiments/exp6_v3.1/configs/FROZEN_SHA256.json (append-only
records with the git SHA and time). Usage: python freeze.py"""
import hashlib, json, os, subprocess, time

R = "/bsuscratch/jiahuizhang/projects/occupancy/seen2scene"
FILES = {"v4": ["experiments/exp6_v4/configs/model_v4.json", "experiments/exp6_v4/configs/outdoor_train.json", "experiments/exp6_v4/scripts/train_v4.py",
                "experiments/exp6_v4/scripts/v4_paths.py", "experiments/exp6_v4/FROZEN_PROTOCOL.md", "seen2scene/futureview/model.py", "seen2scene/futureview/geometry.py",
                "seen2scene/futureview/rays.py", "seen2scene/futureview/predictor.py", "seen2scene/futureview/scoring.py",
                "experiments/exp5_future_view/task1/scripts/v2_data.py", "experiments/exp5_future_view/task1/scripts/t12_train.py",
                "experiments/exp5_future_view/task2/scripts/t22_data.py", "experiments/exp5_future_view/task1/configs/data_v1.json",
                "experiments/exp5_future_view/task1/frozen_values.json"],
         "v31": ["experiments/exp6_v3.1/configs/model_v3_1.json", "experiments/exp6_v3.1/configs/outdoor_train.json", "experiments/exp6_v3.1/scripts/trainer_outdoor.py",
                 "experiments/exp6_v3.1/scripts/train.py", "experiments/exp6_v3.1/FROZEN_PROTOCOL.md", "seen2scene/v3/model.py", "seen2scene/v3/vae.py", "seen2scene/v3/data.py",
                 "seen2scene/v3/losses.py", "seen2scene/v3/trainer.py", "seen2scene/v3/partial_tsdf.py", "seen2scene/v3/flow_unet.py", "seen2scene/v3/eval_dev.py",
                 "seen2scene/futureview/model.py"]}


def sha(p):
    return hashlib.sha256(open(os.path.join(R, p), "rb").read()).hexdigest()


git = subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=R).decode().strip()
dirty = bool(subprocess.check_output(["git", "status", "--porcelain", "--", "seen2scene", "experiments/exp6_v4", "experiments/exp6_v3.1"], cwd=R).decode().strip())
for arm, files in FILES.items():
    out = os.path.join(R, "experiments", "exp6_v4" if arm == "v4" else "exp6_v3.1", "configs", "FROZEN_SHA256.json")
    rec = {"frozen_at": time.strftime("%F %T"), "git_sha": git, "git_dirty_in_scope": dirty, "files": {f: sha(f) for f in files}}
    old = json.load(open(out)) if os.path.exists(out) else {"records": []}
    old["records"].append(rec)
    json.dump(old, open(out, "w"), indent=1)
    print(arm, out, rec["frozen_at"], git[:8], "dirty" if dirty else "clean")
