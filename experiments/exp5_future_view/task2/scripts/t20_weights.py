"""EXP5 Task 2 / T2.0 -- checkpoint archive under work_dirs/weights/ (your 10-26 decision: hard-link archive + delete
the listed redundant checkpoints). CPU only.

  archive  hard-link every kept checkpoint (and its small config files) into a numbered folder
           W<group>/W<group>-<n>_<name>/; per folder MANIFEST.json (sha256, size, source path, provenance);
           work_dirs/weights/INDEX.json + README.md. Old paths stay valid (same inode), so every frozen record still
           resolves. Re-running is idempotent (existing links with the same inode are kept).
  delete   remove the redundant checkpoints listed in DELETE (smoke runs, unreferenced pilot steps, C1 intermediate
           steps, LOSO / SCALE3 intermediate EMA, the random-init file); each file is checked to be outside the archive
           (link count 1, not hard-linked anywhere else) before removal; work_dirs/weights/DELETED.json logs path, size,
           sha256 and time.
Usage: python t20_weights.py archive | delete
"""
import glob
import hashlib
import json
import os
import shutil
import sys
import time
from concurrent.futures import ThreadPoolExecutor

R = "/bsuscratch/jiahuizhang/projects/occupancy/seen2scene"
W = os.path.join(R, "work_dirs")
WT = os.path.join(W, "weights")
E4 = os.path.join(W, "exp4_multiworld_belief")
T1A = os.path.join(E4, "task1", "runs", "T1A_pilot", "2026-10-15_17-19-28-897")
T1C = os.path.join(E4, "task1", "runs", "T1C_pilot", "2026-10-15_17-19-28-898")
C1 = os.path.join(E4, "task2", "c1_pilot", "train")
V2 = os.path.join(W, "exp5_future_view", "task1", "training")
V2CFG = os.path.join(R, "experiments", "exp5_future_view", "task1", "configs")
MAGW = os.path.join(W, "exp5_future_view", "task0", "magician_comparison", "weights", "official_drive")

# folder -> (title, [(source path, name inside the folder)], provenance)
ARCHIVE = {
    "W00_external/W00-1_seen2scene_official": (
        "Seen2Scene official release: auto-encoder, generator, control (base of every EXP4 world model)",
        [(p, os.path.relpath(p, os.path.join(WT, "official"))) for p in sorted(glob.glob(os.path.join(WT, "official", "**", "*"), recursive=True))
         if os.path.isfile(p)],
        "official download (2025-12 / 2026-02 runs); legacy path work_dirs/weights/official/ kept"),
    "W00_external/W00-2_magician_official_e697009": (
        "MAGICIAN official Drive weights (macarons, resnet, scone) used with the official code e697009",
        [(p, os.path.relpath(p, MAGW)) for p in sorted(glob.glob(os.path.join(MAGW, "**", "*"), recursive=True)) if os.path.isfile(p)],
        "EXP5 T0.9: byte-identical to the official Drive files"),
    "W10_exp4_world_models/W10-1_T1A_step1000": (
        "EXP4 Task 1 T1-A (Seen2Scene control fine-tune, pilot) step 1000",
        [(os.path.join(T1A, "checkpoint", "step_001000.ckpt"), "step_001000.ckpt")] + [(os.path.join(T1A, f), f) for f in ("config.yaml", "git_state.txt", "launch.sh", "model.txt")],
        "EXP4 T1-A pilot run 2026-10-15_17-19-28-897"),
    "W10_exp4_world_models/W10-2_T1A_step2000": (
        "EXP4 Task 1 T1-A step 2000",
        [(os.path.join(T1A, "checkpoint", "step_002000.ckpt"), "step_002000.ckpt")] + [(os.path.join(T1A, f), f) for f in ("config.yaml", "git_state.txt", "launch.sh", "model.txt")],
        "EXP4 T1-A pilot run 2026-10-15_17-19-28-897"),
    "W10_exp4_world_models/W10-3_T1C_step1000": (
        "EXP4 Task 1 T1-C (Seen2Scene generator + control fine-tune, pilot) step 1000",
        [(os.path.join(T1C, "checkpoint", "step_001000.ckpt"), "step_001000.ckpt")] + [(os.path.join(T1C, f), f) for f in ("config.yaml", "git_state.txt", "launch.sh", "model.txt")],
        "EXP4 T1-C pilot run 2026-10-15_17-19-28-898; legacy hard link work_dirs/weights/ours/exp4/T1C_pilot_step1000/"),
    "W10_exp4_world_models/W10-4_T1C_step2000": (
        "EXP4 Task 1 T1-C step 2000 = the world model called 'T1-C' in EXP4 / EXP5 (B0, Task 0)",
        [(os.path.join(T1C, "checkpoint", "step_002000.ckpt"), "step_002000.ckpt")] + [(os.path.join(T1C, f), f) for f in ("config.yaml", "git_state.txt", "launch.sh", "model.txt")],
        "EXP4 T1-C pilot run 2026-10-15_17-19-28-898; legacy hard link work_dirs/weights/ours/exp4/T1C_pilot_step2000/"),
    "W11_exp4_belief_updater/W11-1_C1A_seed0_step3000": (
        "EXP4 Task 2 C1 learned belief updater (C1-A), seed 0, step 3000",
        [(os.path.join(C1, "seed0", "ckpt_03000.pt"), "ckpt_03000.pt")], "EXP4 Task 2 c1_pilot/train/seed0"),
    "W11_exp4_belief_updater/W11-2_C1A_seed1_step3000": (
        "EXP4 Task 2 C1 learned belief updater (C1-A), seed 1, step 3000",
        [(os.path.join(C1, "seed1", "ckpt_03000.pt"), "ckpt_03000.pt")], "EXP4 Task 2 c1_pilot/train/seed1"),
    "W20_exp5_v2/W20-1_V2_original_ema60k": (
        "EXP5 Task 1 V2 main model = 'V2 Original' (frozen: EMA@60k, frozen_values.json 'checkpoint'); ckpt_last = raw "
        "weights + optimiser state at 60k",
        [(os.path.join(V2, "main", "ema_060000.pt"), "ema_060000.pt"), (os.path.join(V2, "main", "ckpt_last.pt"), "ckpt_last.pt")]
        + [(os.path.join(V2CFG, f), f) for f in ("model_v1.json", "train_main.json", "data_v1.json")],
        "EXP5 T1.2 M-MAIN, Train-6 minus S-c, 60k steps"),
    "W20_exp5_v2/W20-2_V2_loso_pisa_ema60k": (
        "EXP5 Task 1 LOSO-pisa probe (diagnostic only)",
        [(os.path.join(V2, "loso_pisa", "ema_060000.pt"), "ema_060000.pt"), (os.path.join(V2, "loso_pisa", "ckpt_last.pt"), "ckpt_last.pt")]
        + [(os.path.join(V2CFG, "train_loso_pisa.json"), "train_loso_pisa.json")], "EXP5 T1.2 M-LOSO"),
    "W20_exp5_v2/W20-3_V2_scale3_ema60k": (
        "EXP5 Task 1 SCALE3 probe (diagnostic only)",
        [(os.path.join(V2, "scale3", "ema_060000.pt"), "ema_060000.pt"), (os.path.join(V2, "scale3", "ckpt_last.pt"), "ckpt_last.pt")]
        + [(os.path.join(V2CFG, "train_scale3.json"), "train_scale3.json")], "EXP5 T1.2 M-SCALE3"),
    "W20_exp5_v2/W20-4_V2_original_intermediate_ema": (
        "EXP5 Task 1 V2 main model, intermediate EMA weights 5k-55k (learning-curve record)",
        [(p, os.path.basename(p)) for p in sorted(glob.glob(os.path.join(V2, "main", "ema_0[0-5]*.pt")))], "EXP5 T1.2 M-MAIN"),
}

DELETE = (sorted(glob.glob(os.path.join(T1A, "checkpoint", "step_000500.ckpt")) + glob.glob(os.path.join(T1A, "checkpoint", "step_001500.ckpt"))
                 + glob.glob(os.path.join(T1A, "checkpoint", "vxl_0_011_last.ckpt")) + glob.glob(os.path.join(T1C, "checkpoint", "step_000500.ckpt"))
                 + glob.glob(os.path.join(T1C, "checkpoint", "step_001500.ckpt")) + glob.glob(os.path.join(T1C, "checkpoint", "vxl_0_011_last.ckpt")))
          + sorted(glob.glob(os.path.join(E4, "task1", "runs", "T1A_smoke", "*", "checkpoint", "*.ckpt"))
                   + glob.glob(os.path.join(E4, "task1", "runs", "T1C_smoke", "*", "checkpoint", "*.ckpt")))
          + sorted(p for p in glob.glob(os.path.join(C1, "seed*", "ckpt_*.pt")) if not p.endswith("ckpt_03000.pt"))
          + sorted(glob.glob(os.path.join(E4, "task2", "c1_pilot", "smoke", "train", "seed*", "ckpt_*.pt")) + glob.glob(os.path.join(E4, "task2", "c1_s1", "ckpt_*.pt")))
          + sorted(glob.glob(os.path.join(V2, "loso_pisa", "ema_0[0-5]*.pt")) + glob.glob(os.path.join(V2, "scale3", "ema_0[0-5]*.pt")))
          + [os.path.join(V2, "smoke", "random_init.pt")])


def sha256(p):
    h = hashlib.sha256()
    with open(p, "rb") as f:
        for b in iter(lambda: f.read(1 << 23), b""):
            h.update(b)
    return h.hexdigest()


def archive():
    index = {"doc": __doc__, "created": time.strftime("%Y-%m-%d %H:%M:%S"), "folders": {}}
    jobs = []
    for folder, (title, files, prov) in ARCHIVE.items():
        d = os.path.join(WT, folder)
        for src, name in files:
            if not os.path.isfile(src):
                raise SystemExit(f"missing source {src}")
            dst = os.path.join(d, name)
            os.makedirs(os.path.dirname(dst), exist_ok=True)
            if os.path.exists(dst):
                if os.stat(dst).st_ino != os.stat(src).st_ino:
                    raise SystemExit(f"{dst} exists and is not the same file as {src}")
            else:
                os.link(src, dst)
            jobs.append((folder, name, src, dst))
    with ThreadPoolExecutor(6) as ex:
        shas = list(ex.map(lambda j: sha256(j[3]), jobs))
    for (folder, name, src, dst), sh in zip(jobs, shas):
        title, _, prov = ARCHIVE[folder]
        e = index["folders"].setdefault(folder, {"title": title, "provenance": prov, "files": []})
        e["files"].append({"name": name, "sha256": sh, "size": os.path.getsize(dst), "source_path": os.path.relpath(src, R)})
    for folder, e in index["folders"].items():
        json.dump({"folder": folder, **e}, open(os.path.join(WT, folder, "MANIFEST.json"), "w"), indent=1)
    os.makedirs(os.path.join(WT, "W21_exp5_v2mix"), exist_ok=True)
    open(os.path.join(WT, "W21_exp5_v2mix", "README.md"), "w").write(
        "# W21 — EXP5 Task 2 V2-Mix\n\nReserved for the Task 2 outputs (V2-Mix-best / V2-Mix-last and intermediate EMA); filled by the "
        "Task 2 training script, each sub-folder with its own MANIFEST.json.\n")
    json.dump(index, open(os.path.join(WT, "INDEX.json"), "w"), indent=1)
    lines = ["# work_dirs/weights — checkpoint index", "",
             f"> Built {index['created']} by `experiments/exp5_future_view/task2/scripts/t20_weights.py archive`. Files are **hard links**: "
             "the old paths (recorded in frozen manifests) still point to the same data; no extra disk is used. Each folder has a "
             "`MANIFEST.json` with sha256, size and source path.", "",
             "| ID | content | files | size | key file sha256 (first 16) |", "|---|---|---|---|---|"]
    for folder, e in index["folders"].items():
        big = max(e["files"], key=lambda f: f["size"])
        lines.append(f"| `{folder}` | {e['title']} | {len(e['files'])} | {sum(f['size'] for f in e['files']) / 1e9:.2f} GB | "
                     f"`{big['name']}` {big['sha256'][:16]} |")
    lines += ["| `W21_exp5_v2mix` | EXP5 Task 2 V2-Mix (filled by Task 2) | — | — | — |", "",
              "**Numbering**: W00 external releases · W10 EXP4 world models · W11 EXP4 belief updater · W20 EXP5 V2 (Task 1) · "
              "W21 EXP5 V2-Mix (Task 2).", "",
              "**Legacy folders** `official/` and `ours/` are kept unchanged because frozen configs refer to them (same files as "
              "W00-1 and W10-3 / W10-4).", "",
              "**Not archived here**: the MAGICIAN fork's own retraining checkpoints in `/bsuscratch/jiahuizhang/projects/occupancy/"
              "MAGICIAN/weights/macarons/` (75 files, 5.4 GB; S-stage experiments outside this repository) were left untouched.",
              "", "**Deleted redundant checkpoints**: see `DELETED.json` (your 10-26 decision)."]
    open(os.path.join(WT, "README.md"), "w").write("\n".join(lines) + "\n")
    print(json.dumps({k: len(v["files"]) for k, v in index["folders"].items()}, indent=1))


def delete():
    archived = {os.stat(os.path.join(WT, f, x["name"])).st_ino for f, e in json.load(open(os.path.join(WT, "INDEX.json")))["folders"].items()
                for x in e["files"]}
    log_p = os.path.join(WT, "DELETED.json")
    log = json.load(open(log_p)) if os.path.exists(log_p) else {"decision": "user 2026-10-26: archive + delete the listed redundant checkpoints",
                                                                 "files": []}
    todo = [p for p in DELETE if os.path.isfile(p)]
    for p in todo:
        st = os.stat(p)
        if st.st_ino in archived or st.st_nlink != 1:
            raise SystemExit(f"refusing to delete {p}: archived or hard-linked elsewhere")
    with ThreadPoolExecutor(6) as ex:
        shas = list(ex.map(sha256, todo))
    for p, sh in zip(todo, shas):
        size = os.path.getsize(p)
        os.remove(p)
        log["files"].append({"path": os.path.relpath(p, R), "size": size, "sha256": sh, "deleted": time.strftime("%Y-%m-%d %H:%M:%S")})
    json.dump(log, open(log_p, "w"), indent=1)
    print(f"deleted {len(todo)} files, {sum(f['size'] for f in log['files'][-len(todo):]) / 1e9 if todo else 0:.1f} GB")


if __name__ == "__main__":
    {"archive": archive, "delete": delete}[sys.argv[1]]()
