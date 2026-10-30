"""EXP5 Task 0 -- verify the Matterport3D migration /bsushare -> /bsuhome/jiahuizhang/scratch/data (before the source
is deleted): (1) every <type>.zip in the destination has the sha256 recorded by download_mp_py3.py ("done" /
"exists_ok" events of all download logs) and every logged zip exists there; (2) every other file (extracted content,
.extracted.json markers, .part files) has the same relative path and size in source and destination.
Output W/download_logs/migrate_verify.json; exit code 0 only when everything matches.
"""
import glob
import hashlib
import json
import os
import sys
from concurrent.futures import ProcessPoolExecutor

SRC = "/bsushare/yzhang-shared/data/matterport3d"
DST = "/bsuhome/jiahuizhang/scratch/data/matterport3d"
LOGS = "/bsuscratch/jiahuizhang/projects/occupancy/seen2scene/work_dirs/exp5_future_view/task0/download_logs"


def sha(p):
    h = hashlib.sha256()
    with open(p, "rb") as f:
        for b in iter(lambda: f.read(1 << 24), b""):
            h.update(b)
    return p, h.hexdigest()


def listing(root):
    out = {}
    for d, _, fs in os.walk(root):
        for f in fs:
            p = os.path.join(d, f)
            out[os.path.relpath(p, root)] = os.path.getsize(p)
    return out


def main():
    logged = {}
    for p in glob.glob(os.path.join(LOGS, "*.jsonl")):
        for line in open(p):
            r = json.loads(line)
            if r.get("event") in ("done", "exists_ok") and r.get("sha256"):
                logged[os.path.relpath(r["file"], SRC)] = r["sha256"]
    src, dst = listing(SRC), listing(DST)
    missing = sorted(set(src) - set(dst))
    extra = sorted(set(dst) - set(src))
    size_diff = sorted(k for k in set(src) & set(dst) if src[k] != dst[k])
    zips = sorted(k for k in dst if k.endswith(".zip"))
    with ProcessPoolExecutor(int(sys.argv[1]) if len(sys.argv) > 1 else 8) as ex:
        got = dict((os.path.relpath(p, DST), h) for p, h in ex.map(sha, [os.path.join(DST, k) for k in zips], chunksize=1))
    zip_mismatch = sorted(k for k in zips if k in logged and logged[k] != got[k])
    zip_unlogged = sorted(k for k in zips if k not in logged)
    logged_missing = sorted(k for k in logged if k not in dst)
    res = {"n_src": len(src), "n_dst": len(dst), "bytes_src": sum(src.values()), "bytes_dst": sum(dst.values()),
           "n_zip": len(zips), "n_zip_logged": len(logged), "missing_in_dst": missing[:50], "n_missing": len(missing),
           "extra_in_dst": extra[:50], "n_extra": len(extra), "size_diff": size_diff[:50], "n_size_diff": len(size_diff),
           "zip_sha_mismatch": zip_mismatch, "zip_unlogged": zip_unlogged[:50], "logged_zip_missing_in_dst": logged_missing[:50]}
    res["pass"] = not (missing or extra or size_diff or zip_mismatch or logged_missing)
    json.dump(res, open(os.path.join(LOGS, "migrate_verify.json"), "w"), indent=1)
    print(json.dumps({k: v for k, v in res.items() if not isinstance(v, list)}, indent=1))
    sys.exit(0 if res["pass"] else 1)


if __name__ == "__main__":
    main()
