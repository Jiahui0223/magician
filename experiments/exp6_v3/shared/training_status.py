"""Refresh one arm's training records from its run logs: the §2 / §3 tables of T{0,1}_TRAINING.md, the two P4 rows of
STATUS.md and the checkpoint size in V3_RESOURCES.md.  Usage: python training_status.py t0|t1 [run]"""
import glob, json, os, re, sys, time

R = "/bsuscratch/jiahuizhang/projects/occupancy/seen2scene"
sys.path.insert(0, R)
from seen2scene.v3 import trainer as T  # noqa: E402

arm = sys.argv[1]; run = sys.argv[2] if len(sys.argv) > 2 else "main"
task = "task0_vae" if arm == "t0" else "task1_xattn"; A = arm.upper()
W = f"{R}/work_dirs/exp6_v3/{task}/runs/{run}"; E = f"{R}/experiments/exp6_v3"
cfg = json.load(open(f"{E}/configs/train_v3.json")); steps = cfg["steps"]; EVAL_MIN = 5.5
L = [json.loads(x) for x in open(f"{W}/log.jsonl")]
ev = [json.loads(x) for x in open(f"{W}/eval.jsonl")] if os.path.exists(f"{W}/eval.jsonl") else []
sc = T.selection_score(ev); best_step = min(sc, key=sc.get) if sc else None
last = L[-1]; tail = sorted(x["s_per_step"] for x in L[-40:]); sps = tail[len(tail) // 2]   # median: the entry after an evaluation carries the eval time
rem = steps - last["step"]; evals_left = sum(1 for s in range(cfg["eval_every"], steps + 1, cfg["eval_every"]) if s > last["step"])
eta_h = rem * sps / 3600 + evals_left * EVAL_MIN / 60
eta = time.strftime("%m-%d %H:%M", time.localtime(time.time() + eta_h * 3600)); now = time.strftime("%H:%M")
es = lambda e, k: e[f"samp_{k}"]["fair_es_arclength_vox"]  # noqa: E731

# -- T*_TRAINING.md: one §2 row per call (skipped if that step is already there), §3 regenerated from eval.jsonl
row2 = (f"| {now} | {last['step']} | {last['loss']:.3f} | {last['fm']:.3f} | {last['rec']:.3f} | {last['kl']:.3f} | {sps:.2f} | "
        f"{last['mem_gb']:.1f} GB | grad-norm {last['grad_norm']:.1f}；lr {last['lr']:.2e}；β {last['beta']:.3f}；剩余 {rem} 步 ≈ {eta_h:.1f} h → {eta} |")
def erow(e):
    s = sc.get(e["step"])
    return (f"| {e['step']} | {e['fm_in_val']:.4f} | {es(e, 'in_val'):.2f} | {es(e, 'out_sc'):.2f} | {'—' if s is None else f'{s:.4f}'} | "
            f"dev FM out {e['fm_out_sc']:.4f}{'；**当前最优**' if e['step'] == best_step else ''} |")
sec3 = ("## 3. 评估点（`eval.jsonl`）与选择分数\n| 步 | dev FM (VALSEL) | ES in (VALSEL) | ES out (S-c) | 选择分数 | 备注 |\n|---|---|---|---|---|---|\n"
        + "\n".join(erow(e) for e in ev) + "\n\n")
p = f"{E}/{task}/report/{A}_TRAINING.md"; s = open(p).read()
i3 = s.index("## 3. 评估点"); i4 = s.index("## 4. 事件")
s2 = s[:i3].rstrip("\n")
if re.search(rf"^\| [^|]* \| {last['step']} \|", s2, re.M):
    s2 = re.sub(rf"^\| [^|]* \| {last['step']} \|.*$", row2, s2, count=1, flags=re.M)
else:
    s2 += "\n" + row2
open(p, "w").write(s2 + "\n\n" + sec3 + s[i4:])

# -- STATUS.md: timestamp, the ETA cell of the top-board row (6 cells after the key) and the status cell of the P4 row (3 cells)
p = f"{E}/STATUS.md"; s = open(p).read()
s = re.sub(r"\*\*最近更新：2026-\d\d-\d\d \d\d:\d\d[^*\n]*\*\*", f"**最近更新：{time.strftime('%Y-%m-%d %H:%M')}**", s, count=1)
e = ev[-1] if ev else None
top = (f" 第 {last['step']} 步 / {steps}（{100 * last['step'] / steps:.0f}%），{sps:.2f} s/步，峰值 {last['mem_gb']:.1f} GB"
       + (f"；dev FM in {e['fm_in_val']:.4f} @ {e['step']}" if e else "") + f"；预计 {eta} 结束 |")
s, n1 = re.subn(rf"^(\| P4-{A} \|(?:[^|\n]*\|){{5}})[^|\n]*\|\s*$", lambda m: m.group(1) + top, s, count=1, flags=re.M)
st = (f" 🔄 第 {last['step']} 步 / {steps}（{100 * last['step'] / steps:.0f}%），{sps:.2f} s/步，峰值 {last['mem_gb']:.1f} GB"
      + (f"；最近评估 @{e['step']}：dev FM in {e['fm_in_val']:.4f} / out {e['fm_out_sc']:.4f}，ES in {es(e, 'in_val'):.2f} / out {es(e, 'out_sc'):.2f}" if e else "")
      + (f"；选择分数 {sc[e['step']]:.4f}（最优 @{best_step}）" if e and e["step"] in sc else "")
      + f"；预计 {eta} 结束；记录 `{task}/report/{A}_TRAINING.md` |")
s, n2 = re.subn(rf"^(\| P4-{A} \|(?:[^|\n]*\|){{2}})[^|\n]*\|\s*$", lambda m: m.group(1) + st, s, count=1, flags=re.M)
open(p, "w").write(s)

# -- V3_RESOURCES.md: checkpoint sizes (first time only)
ck = sorted(glob.glob(f"{W}/checkpoints/*.pt") + glob.glob(f"{W}/*.pt"))
if ck:
    sizes = {os.path.basename(c): os.path.getsize(c) / 2 ** 20 for c in ck}
    p = f"{E}/reports/V3_RESOURCES.md"; s = open(p).read()
    if "⟨待填⟩ MB" in s:
        txt = "、".join(f"`{k}` {v:.0f} MB" for k, v in sizes.items())
        open(p, "w").write(s.replace("训练 checkpoint 每个约 ⟨待填⟩ MB", f"训练 checkpoint：{txt}"))
print(f"{A} step {last['step']} loss {last['loss']:.3f} fm {last['fm']:.3f} rec {last['rec']:.3f} kl {last['kl']:.3f} {sps:.2f} s/step "
      f"mem {last['mem_gb']:.1f} GB; evals {[x['step'] for x in ev]}; scores {{{', '.join(f'{k}: {v:.4f}' for k, v in sc.items())}}}; "
      f"ETA {eta}; STATUS rows replaced {n1}/{n2}; ckpts {ck and {os.path.basename(c): round(os.path.getsize(c) / 2 ** 20) for c in ck}}")
if e:
    print("last eval:", {k: (round(v, 4) if isinstance(v, float) else v) for k, v in e.items() if not isinstance(v, dict)},
          "| in_val", {k: round(v, 3) for k, v in e["samp_in_val"].items()}, "| out_sc", {k: round(v, 3) for k, v in e["samp_out_sc"].items()},
          "| posterior", {k: v for k, v in e.items() if "post" in k or "active" in k or "recon" in k})
