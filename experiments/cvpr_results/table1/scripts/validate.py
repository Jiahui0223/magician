"""CVPR Table 1 -- validity checks of the canonical per-episode records (CPU; PROTOCOL.md §7 / user Task 9).
For every model with registry status READY and every registry scene x start 0..4:
  present / missing; curve length 101 (C(0)..C(100)); official curve in [0, 1.05]; monotone curve non-decreasing;
  executed pose history of 101 keys when stored; checkpoint sha == registry sha; one record per (model, scene, start)
  (the file layout guarantees it; duplicates in sources are reported); Cov@100 == Final by construction.
Writes reports/MISSING_EPISODES.md and results/validation.json; exit code 1 if any record is invalid.
Usage: python validate.py
"""
import glob
import json
import os
import sys

import numpy as np

import aggregate as AG

T1 = AG.T1


def main():
    split, order, label, models = AG.load_registries()
    reg = {m["id"]: m for m in models}
    scenes = sorted(split)
    bad, missing, ok = [], {}, {}
    for m in order:
        st = str(reg[m].get("status", "")).upper()
        if not st.startswith("READY"):
            continue
        sha = reg[m].get("sha256")
        for s in scenes:
            for k in range(AG.N_START):
                f = os.path.join(AG.EPI, m, f"{s}_s{k}.json")
                if not os.path.exists(f):
                    missing.setdefault(m, []).append(f"{s}_s{k}")
                    continue
                r = json.load(open(f))
                errs = []
                co = r.get("coverage_official")
                if co is None or len(co) != AG.T_MAX + 1:
                    errs.append("official curve missing or not 101 values")
                elif min(co) < 0 or max(co) > 1.05 or not np.all(np.isfinite(co)):
                    errs.append("official curve out of range")
                cm = r.get("coverage_mono")
                if cm is not None and (len(cm) != AG.T_MAX + 1 or np.any(np.diff(cm) < -1e-12)):
                    errs.append("monotone curve invalid")
                ek = r.get("executed_keys")
                if ek is not None and len(ek) != AG.T_MAX + 1:
                    errs.append(f"executed keys {len(ek)} != 101")
                if sha and r.get("ckpt_sha256") and r["ckpt_sha256"] != sha:
                    errs.append("checkpoint sha differs from registry")
                if r["model"] != m or r["scene"] != s or int(r["start"]) != k:
                    errs.append("identity fields differ from the file name")
                if errs:
                    bad.append({"model": m, "episode": f"{s}_s{k}", "errors": errs})
                else:
                    ok[m] = ok.get(m, 0) + 1
    out = {"ok": ok, "missing": missing, "bad": bad}
    json.dump(out, open(os.path.join(T1, "results", "validation.json"), "w"), indent=1)
    L = ["# MISSING_EPISODES — 缺失与无效的 episode（自动生成，`scripts/validate.py`）", "",
         f"期望：每个 READY 模型 × {len(scenes)} 个场景 × {AG.N_START} 个起点。缺失的 episode 不用其他起点或均值填补。", "",
         "| 模型 | 有效 | 缺失 | 缺失列表 |", "|---|---|---|---|"]
    for m in order:
        st = str(reg[m].get("status", "")).upper()
        if not st.startswith("READY"):
            L.append(f"| {m} | — | — | 状态 {reg[m].get('status')}（未纳入本轮） |")
            continue
        mi = missing.get(m, [])
        L.append(f"| {m} | {ok.get(m, 0)} | {len(mi)} | {' '.join(mi[:40])}{' …' if len(mi) > 40 else ''} |")
    if bad:
        L += ["", "## 无效记录", ""] + [f"- {b['model']} {b['episode']}: {'; '.join(b['errors'])}" for b in bad]
    os.makedirs(os.path.join(T1, "reports"), exist_ok=True)
    open(os.path.join(T1, "reports", "MISSING_EPISODES.md"), "w").write("\n".join(L) + "\n")
    print("valid", ok, "missing", {m: len(v) for m, v in missing.items()}, "bad", len(bad))
    sys.exit(1 if bad else 0)


if __name__ == "__main__":
    main()
