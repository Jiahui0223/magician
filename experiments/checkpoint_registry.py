"""Unified checkpoint registry for the EXP5 V2 family (B2 / E2) and EXP6 V3 / V3.1 / V4 / V5 / V4C / V5.2 / V5.2-M.

work_dirs/checkpoints/<experiment>/... holds RELATIVE SYMLINKS to the original files: nothing is moved, copied, renamed or
deleted, so every frozen config / script / manifest that uses an old path keeps working, and no disk is duplicated.
For every entry: the link must resolve to the original (os.path.samefile), sha256 (read through the link), size, step and
model config read from the checkpoint, and a model check -- EMA weights instantiated in their architecture class and loaded
with strict=True (parameter count); raw training states: keys model / ema / opt / step and model-vs-EMA key/shape equality.
Files whose sha equals an already registered file of the same folder are not linked again (recorded as aliases).
Existing different files / links at a destination are never overwritten (error). Expected sha256 values recorded in the
experiments' own reports are cross-checked. Outputs: work_dirs/checkpoints/<experiment>/{MANIFEST.json, README.md},
work_dirs/checkpoints/README.md and the tracked index experiments/CHECKPOINTS.md.
Usage: python experiments/checkpoint_registry.py [--only exp6_v2 ...] [--dry]  (seen2scene env, CPU only)"""
import argparse
import concurrent.futures as cf
import glob
import hashlib
import json
import os
import re
import sys
import time

R = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
CKR = os.path.join(R, "work_dirs", "checkpoints")
sys.path.insert(0, R)

E2W = "work_dirs/exp5_future_view/task3_full_depth_p1/e2/training"
EXPECTED = {  # sha256 recorded in the experiments' own manifests / reports (prefixes)
    "work_dirs/weights/W20_exp5_v2/W20-1_V2_original_ema60k/ema_060000.pt": "a379c0fb9d7d80266a430ab6f48f39315fad54fb650c6c3740a999a02db079fc",
    "work_dirs/weights/W21_exp5_v2mix/W21-1_V2Mix_best/ema_017500.pt": "343f41ab8d566c52007c881b5f32375c3253dd8f1c7a0d8a906d412a165e4516",
    f"{E2W}/d100/ema_005000.pt": "61439612cb6132afbe74976b82a531c8f827802b06b4f87ad6de1a197d0a2fa5",
    "work_dirs/exp6_v3/task0_vae/checkpoints/final.pt": "7ed0ddd72bb91d1e8c6563f3192201c24a49fe862b49d2387ea8770915145ab6",
    "work_dirs/exp6_v3/task1_xattn/checkpoints/final.pt": "639c9d3ae3f3c069dc0399dc4cdd44102be2bf20e70d666a6a529c6affc62bdb",
    "work_dirs/exp6_v5/checkpoints/v5_final.pt": "5ef30ac4fd0b4a175d1ddd72d78c504c82b31b40b56da68b1d9454533918d543",
    "work_dirs/exp6_v5/checkpoints/v4c_final.pt": "c4190f1d",
    "work_dirs/exp6_v5_2/checkpoints/v52_final.pt": "4982bc66",
    "work_dirs/exp6_v5_2_scratch70k/checkpoints/best_train_holdout.pt": "a540950738cc7191",
    "work_dirs/exp6_v5_2_scratch70k/checkpoints/step_070000.pt": "e1b168945323a21c",
}


def emas(folder, prefix, pattern, skip=()):
    out = []
    for p in sorted(glob.glob(os.path.join(R, folder, pattern))):
        m = re.search(r"(\d+)\.pt$", p)
        st = int(m.group(1))
        if st in skip:
            continue
        out.append((f"intermediate/{prefix}_ema{st:06d}.pt", os.path.relpath(p, R), "ema", f"intermediate EMA @ {st}"))
    return out


def spec():
    S = {}
    S["exp6_v2"] = {
        "title": "V2 family: B2 (frozen V2-Mix, D100 input / S10 DistP1) and E2 (V2-Mix fine-tuned on D100 data)",
        "groups": {
            "B2": [("v2mix_ema17500.pt", "work_dirs/weights/W21_exp5_v2mix/W21-1_V2Mix_best/ema_017500.pt", "ema", "B2 MODEL: V2-Mix EMA@17.5k (frozen-rule selection; = Task 2 group V2M, B0 / B2 planners)"),
                   ("v2mix_ema20000.pt", "work_dirs/weights/W21_exp5_v2mix/W21-2_V2Mix_last/ema_020000.pt", "ema", "V2-Mix last EMA (not selected)"),
                   ("v2mix_raw20000.pt", "work_dirs/exp5_future_view/task2/training/mix_v1/ckpt_last.pt", "raw", "V2-Mix raw training state at 20k (model + EMA + AdamW + RNG)"),
                   ("lineage/v2orig_ema60000.pt", "work_dirs/weights/W20_exp5_v2/W20-1_V2_original_ema60k/ema_060000.pt", "ema", "V2 Original EMA@60k (Task 1; init of V2-Mix)"),
                   ("lineage/v2orig_raw60000.pt", "work_dirs/weights/W20_exp5_v2/W20-1_V2_original_ema60k/ckpt_last.pt", "raw", "V2 Original raw training state at 60k")]
                  + emas("work_dirs/weights/W21_exp5_v2mix/W21-3_V2Mix_intermediate_ema", "v2mix", "ema_*.pt", skip=(17500, 20000)),
            "E2": [("e2d100_ema5000.pt", f"{E2W}/d100/ema_005000.pt", "ema", "E2 MODEL: E2-D100 EMA@5k (V2-Mix EMA@17.5k fine-tuned 5k steps on D100 shards; B2-style planner)"),
                   ("e2d100_raw5000.pt", f"{E2W}/d100/ckpt_last.pt", "raw", "E2-D100 raw training state at 5k"),
                   ("control_e2s10/e2s10_ema5000.pt", f"{E2W}/s10/ema_005000.pt", "ema", "E2-S10 EMA@5k (control arm: same recipe on sparse S10 shards; B0-style planner)"),
                   ("control_e2s10/e2s10_raw5000.pt", f"{E2W}/s10/ckpt_last.pt", "raw", "E2-S10 raw training state at 5k")]
                  + emas(f"{E2W}/d100", "e2d100", "ema_*.pt", skip=(5000,))
                  + [(f"control_e2s10/{d}", s, k, r) for d, s, k, r in emas(f"{E2W}/s10", "e2s10", "ema_*.pt", skip=(5000,))],
        }}
    S["exp6_v3"] = {
        "title": "V3 (mixed scratch, VAE latent + ray encoder + 2D flow): T0 = V3-VAE, T1 = V3-VAE-XAttn",
        "groups": {
            "T0_vae": [("v3t0_ema40000.pt", "work_dirs/exp6_v3/task0_vae/checkpoints/final.pt", "ema", "V3-T0 final (EMA@40k)"),
                       ("v3t0_raw40000.pt", "work_dirs/exp6_v3/task0_vae/runs/main/ckpt_last.pt", "raw", "V3-T0 raw training state at 40k")]
                      + emas("work_dirs/exp6_v3/task0_vae/runs/main", "v3t0", "ema_*.pt"),
            "T1_xattn": [("v3t1_ema40000.pt", "work_dirs/exp6_v3/task1_xattn/checkpoints/final.pt", "ema", "V3-T1 final (EMA@40k)"),
                         ("v3t1_raw40000.pt", "work_dirs/exp6_v3/task1_xattn/runs/main/ckpt_last.pt", "raw", "V3-T1 raw training state at 40k")]
                        + emas("work_dirs/exp6_v3/task1_xattn/runs/main", "v3t1", "ema_*.pt")}}
    S["exp6_v3_1"] = {"title": "V3.1 (V3-T0 architecture, outdoor-only scratch)", "groups": {"V3_1": [
        ("v31_ema40000.pt", "work_dirs/exp6_v3_1/checkpoints/final.pt", "ema", "V3.1 final (EMA@40k)"),
        ("v31_raw40000.pt", "work_dirs/exp6_v3_1/runs/main/ckpt_last.pt", "raw", "V3.1 raw training state at 40k")] + emas("work_dirs/exp6_v3_1/runs/main", "v31", "ema_*.pt")}}
    S["exp6_v4"] = {"title": "V4 (V2 modules + ray-aligned cross-attention, outdoor-only scratch)", "groups": {"V4": [
        ("v4_ema40000.pt", "work_dirs/exp6_v4/checkpoints/final.pt", "ema", "V4 final (EMA@40k)"),
        ("v4_raw40000.pt", "work_dirs/exp6_v4/runs/main/ckpt_last.pt", "raw", "V4 raw training state at 40k")] + emas("work_dirs/exp6_v4/runs/main", "v4", "ema_*.pt")}}
    S["exp6_v5"] = {"title": "V5 (geometry-aware ray-to-scene memory + multi-surface belief, V4@40k warm-start) and V4C (V4 continued)", "groups": {
        "V5": [("v5_ema12000.pt", "work_dirs/exp6_v5/checkpoints/v5_final.pt", "ema", "V5-Warm12k final (EMA@12k)"),
               ("v5_raw12000.pt", "work_dirs/exp6_v5/runs/v5/ckpt_last.pt", "raw", "V5 raw training state at 12k")] + emas("work_dirs/exp6_v5/runs/v5", "v5", "ema_*.pt"),
        "V4C": [("v4c_ema12000.pt", "work_dirs/exp6_v5/checkpoints/v4c_final.pt", "ema", "V4-Continue12k final (EMA@12k; control of V5)"),
                ("v4c_raw12000.pt", "work_dirs/exp6_v5/runs/v4c/ckpt_last.pt", "raw", "V4C raw training state at 12k")] + emas("work_dirs/exp6_v5/runs/v4c", "v4c", "ema_*.pt")}}
    S["exp6_v5_2"] = {"title": "V5.2 pilot (Idea 2: ray geometry belief + conditional first-hit hazard, V5 warm-start, 2.67M)", "groups": {"V5_2": [
        ("v52_ema6000.pt", "work_dirs/exp6_v5_2/checkpoints/v52_final.pt", "ema", "V5.2 pilot final (EMA@6000)"),
        ("v52_raw6000.pt", "work_dirs/exp6_v5_2/runs/v52/ckpt_last.pt", "raw", "V5.2 raw training state at 6000")] + emas("work_dirs/exp6_v5_2/runs/v52", "v52", "ema_*.pt")}}
    S["exp6_v5_2_scratch70k"] = {"title": "V5.2-M Scratch70k (Idea 2, 27.0M, from scratch)", "groups": {"V5_2M": [
        ("v52m_best_ema50000.pt", "work_dirs/exp6_v5_2_scratch70k/checkpoints/best_train_holdout.pt", "ema", "V5.2-M best_train_holdout (EMA@50k; primary checkpoint of the run)"),
        ("v52m_ema70000.pt", "work_dirs/exp6_v5_2_scratch70k/checkpoints/step_070000.pt", "ema", "V5.2-M last70k (EMA@70k)"),
        ("v52m_raw70000.pt", "work_dirs/exp6_v5_2_scratch70k/checkpoints/last.pt", "raw", "V5.2-M raw training state at 70k")]
        + emas("work_dirs/exp6_v5_2_scratch70k/checkpoints", "v52m", "step_*.pt")}}
    return S


def sha256(p):
    h = hashlib.sha256()
    with open(p, "rb") as f:
        for b in iter(lambda: f.read(1 << 24), b""):
            h.update(b)
    return h.hexdigest()


def build_model(cfg):
    arch = cfg.get("arch")
    if arch == "v5":
        from seen2scene.futureview import v5_model as X
        return X.V5Model(cfg), "seen2scene.futureview.v5_model.V5Model"
    if arch == "v52":
        from seen2scene.futureview import v52_model as X
        return X.V52Model(cfg), "seen2scene.futureview.v52_model.V52Model"
    if arch == "v52m":
        from seen2scene.futureview import v52m_model as X
        return X.V52MModel(cfg), "seen2scene.futureview.v52m_model.V52MModel"
    if "vae" in cfg:
        from seen2scene.v3 import model as X
        return X.V3Model(cfg), "seen2scene.v3.model.V3Model"
    from seen2scene.futureview import model as X
    return X.V2Model(cfg), "seen2scene.futureview.model.V2Model" + (" (+ ray_xattn)" if cfg.get("ray_xattn") else "")


def inspect(path, kind, cfg_cache):
    import torch
    ck = torch.load(path, map_location="cpu", weights_only=False)
    info = {"keys": sorted(ck.keys()), "step": ck.get("step")}
    sd = ck.get("ema")
    if kind == "raw":
        info["has_optimizer"] = "opt" in ck
        m, e = ck.get("model"), ck.get("ema")
        info["raw_model_matches_ema_layout"] = bool(m is not None and e is not None and list(m.keys()) == list(e.keys()) and all(m[k].shape == e[k].shape for k in m))
        sd = m
        cfg = ck.get("model_cfg") or cfg_cache.get("cfg")
    else:
        cfg = ck.get("model_cfg")
    if cfg is None or sd is None:
        info["model_check"] = "no model_cfg in file (layout compared with the folder's EMA)"
        info["params"] = int(sum(v.numel() for v in (sd or {}).values()))
        return info
    cfg_cache.setdefault("cfg", cfg)
    info["model_cfg_sha256_16"] = hashlib.sha256(json.dumps(cfg, sort_keys=True).encode()).hexdigest()[:16]
    info["model_cfg_version"] = cfg.get("version")
    model, cls = build_model(cfg)
    res = model.load_state_dict(sd, strict=True)
    info["model_class"] = cls
    info["strict_load"] = "ok" if not res.missing_keys and not res.unexpected_keys else str(res)
    info["params"] = int(sum(p.numel() for p in model.parameters()))
    for k in ("arm", "lineage", "arch"):
        if k in ck:
            info[k] = ck[k] if not isinstance(ck[k], dict) else {kk: str(vv)[:200] for kk, vv in ck[k].items()}
    return info


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--only", nargs="*", default=None)
    ap.add_argument("--dry", action="store_true")
    a = ap.parse_args()
    S = spec()
    allrows = {}
    for ver, sv in S.items():
        if a.only and ver not in a.only:
            continue
        vdir = os.path.join(CKR, ver)
        entries = [(g, d, s, k, r) for g, lst in sv["groups"].items() for d, s, k, r in lst]
        for g, d, s, k, r in entries:
            assert os.path.isfile(os.path.join(R, s)), ("missing source", s)
        with cf.ThreadPoolExecutor(8) as ex:
            shas = dict(zip([s for _, _, s, _, _ in entries], ex.map(lambda e: sha256(os.path.join(R, e[2])), entries)))
        rows, seen, cfg_cache = [], {}, {}
        for g, d, s, k, r in entries:
            src = os.path.join(R, s)
            sh = shas[s]
            exp = EXPECTED.get(s)
            row = {"group": g, "new_path": os.path.join("work_dirs", "checkpoints", ver, g, d), "original_path": s, "kind": k, "role": r,
                   "sha256": sh, "size": os.path.getsize(src), "expected_sha_prefix": exp, "expected_sha_ok": (sh.startswith(exp) if exp else None)}
            if exp and not sh.startswith(exp):
                raise SystemExit(f"STOP: sha mismatch {s}: {sh} vs recorded {exp}")
            key = (g, sh)
            if key in seen:
                row["alias_of"] = seen[key]
                row["linked"] = False
                rows.append(row)
                continue
            seen[key] = row["new_path"]
            dst = os.path.join(R, row["new_path"])
            if not a.dry:
                os.makedirs(os.path.dirname(dst), exist_ok=True)
                rel = os.path.relpath(src, os.path.dirname(dst))
                if os.path.lexists(dst):
                    if not (os.path.islink(dst) and os.path.samefile(dst, src)):
                        raise SystemExit(f"STOP: {dst} exists and is not a link to {src}; not overwriting")
                else:
                    os.symlink(rel, dst)
                assert os.path.samefile(dst, src)
                row["link_target"] = rel
            row["linked"] = True
            row.update(inspect(src, k, cfg_cache.setdefault(g, {})))
            rows.append(row)
            print(ver, g, d, k, sh[:12], row.get("step"), row.get("params"), row.get("strict_load", row.get("raw_model_matches_ema_layout")), flush=True)
        allrows[ver] = {"title": sv["title"], "rows": rows}
        if not a.dry:
            json.dump({"experiment": ver, "title": sv["title"], "built": time.strftime("%F %T"), "rows": rows}, open(os.path.join(vdir, "MANIFEST.json"), "w"), indent=1)
            write_readme(vdir, ver, sv["title"], rows)
    if not a.dry and not a.only:
        write_index(allrows)


def table(rows):
    L = ["| 新路径（work_dirs/checkpoints/…） | 原路径 | 类型 | step | 参数 | sha256（前 16） | 严格加载 | 用途 |", "|---|---|---|---|---|---|---|---|"]
    for r in rows:
        if not r.get("linked"):
            L.append(f"| （未单独链接，与 `{r['alias_of'].split('checkpoints/', 1)[1]}` 字节相同） | `{r['original_path']}` | {r['kind']} | — | — | {r['sha256'][:16]} | — | {r['role']} |")
            continue
        chk = r.get("strict_load") if r["kind"] == "ema" else ("model≡EMA 布局" if r.get("raw_model_matches_ema_layout") else "layout ✗") + ("，含优化器" if r.get("has_optimizer") else "")
        L.append(f"| `{r['new_path'].split('checkpoints/', 1)[1]}` | `{r['original_path']}` | {r['kind']} | {r.get('step')} | {r.get('params', 0):,} | {r['sha256'][:16]}{' ✓' if r.get('expected_sha_ok') else ''} | {chk} | {r['role']} |")
    return L


def write_readme(vdir, ver, title, rows):
    L = [f"# work_dirs/checkpoints/{ver}", "", title, "",
         "全部是指向原文件的**相对软链接**（原文件未移动 / 复制 / 删除；旧路径仍有效）。生成：`experiments/checkpoint_registry.py`；明细 `MANIFEST.json`。"
         "✓ = sha256 与实验自身记录一致。", ""] + table(rows)
    open(os.path.join(vdir, "README.md"), "w").write("\n".join(L) + "\n")


def write_index(allrows):
    L = ["# Checkpoint 索引（V2 B2/E2 · V3 · V3.1 · V4 · V5 / V4C · V5.2 · V5.2-M）", "",
         f"由 `experiments/checkpoint_registry.py` 于 {time.strftime('%F %T')} 生成。`work_dirs/checkpoints/<实验>/` 中全部是指向原文件的相对软链接："
         "原文件未移动、复制或删除，冻结配置和旧脚本中的路径继续有效；每个链接都核对了 sha256（读穿链接）、step、模型配置，并对 EMA 权重做了严格加载（strict=True）。"
         "G0 测试运行（resume_A / resume_B、smoke、runs/g0）不登记。", ""]
    for ver, v in allrows.items():
        L += [f"## {ver}", "", v["title"], ""] + table(v["rows"]) + [""]
    txt = "\n".join(L) + "\n"
    open(os.path.join(R, "experiments", "CHECKPOINTS.md"), "w").write(txt)
    open(os.path.join(CKR, "README.md"), "w").write(txt.replace("# Checkpoint 索引", "# work_dirs/checkpoints —— 索引", 1) +
                                                     "\n另有历史目录 `2026-10-03_pre_exp3_restructure/`（未改动）。V2 之前的权重注册表见 `work_dirs/weights/README.md`（W00–W21，硬链接）。\n")


if __name__ == "__main__":
    main()
