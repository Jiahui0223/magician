"""EXP5 Task 1 -- stage freeze (protocol §6): copy the given repo files into a read-only snapshot
W/snapshots/<stage>/<repo-relative path>, write MANIFEST.sha256 there and append the stage entry (file sha256s, repo
HEAD, extra frozen artefacts) to task1/FROZEN_MANIFEST.json. Never overwrites an existing snapshot.
Usage: python t1_snapshot.py <stage> [--extra key=path ...] file [file ...]
"""
import argparse
import json
import os
import shutil
import stat
import subprocess
import sys
import time

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
import v2_frozen as VF  # noqa: E402


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("stage")
    ap.add_argument("files", nargs="+")
    ap.add_argument("--extra", action="append", default=[])
    ap.add_argument("--note", default="")
    a = ap.parse_args()
    dst = os.path.join(VF.WORK, "snapshots", a.stage)
    assert not os.path.exists(dst), f"snapshot exists: {dst}"
    files = {}
    for f in a.files:
        p = os.path.abspath(f)
        rel = os.path.relpath(p, VF.REPO)
        assert not rel.startswith(".."), p
        q = os.path.join(dst, rel)
        os.makedirs(os.path.dirname(q), exist_ok=True)
        shutil.copy2(p, q)
        files[rel] = VF.sha256_file(p)
        assert VF.sha256_file(q) == files[rel]
    with open(os.path.join(dst, "MANIFEST.sha256"), "w") as f:
        for rel, h in sorted(files.items()):
            f.write(f"{h}  {rel}\n")
    for root, _, fs in os.walk(dst):
        for x in fs:
            os.chmod(os.path.join(root, x), stat.S_IRUSR | stat.S_IRGRP | stat.S_IROTH)
    head = subprocess.run(["git", "rev-parse", "HEAD"], cwd=VF.REPO, capture_output=True, text=True).stdout.strip()
    extra = {}
    for e in a.extra:
        k, p = e.split("=", 1)
        extra[k] = {"path": os.path.relpath(os.path.abspath(p), VF.REPO), "sha256": VF.sha256_file(p)}
    mp = os.path.join(VF.T1, "FROZEN_MANIFEST.json")
    man = json.load(open(mp))
    man["stage_freezes"].append({"stage": a.stage, "time": time.strftime("%Y-%m-%d %H:%M:%S"), "snapshot":
                                 os.path.relpath(dst, VF.REPO), "files": files, "extra": extra, "repo_head": head,
                                 "frozen_files_sha256_ok": VF.verify_frozen(), "note": a.note})
    json.dump(man, open(mp + ".tmp", "w"), indent=1)
    os.replace(mp + ".tmp", mp)
    print(f"snapshot {a.stage}: {len(files)} files -> {dst}")


if __name__ == "__main__":
    main()
