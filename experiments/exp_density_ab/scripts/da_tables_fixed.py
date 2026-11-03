"""EXP Density A/B -- markdown tables of the fixed-state results (CPU) from results/fixed_summary.json, plus a house-level
bootstrap for VALSEL (9 floors in 6 houses; the historical / primary unit is the floor = one trajectory each).
Output results/tables_fixed.md (the report tables are copied from here)."""
import json
import os

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
EXP = os.path.dirname(HERE)
S = json.load(open(os.path.join(EXP, "results", "fixed_summary.json")))
CK, DN = ("B2", "E2S", "E2D"), ("S10", "D100")
CFGS = [f"{m}|{i}|{s}" for m in CK for i in DN for s in DN]


def f(b, d=3):
    if b is None or b.get("mean") is None:
        return "—"
    return f"{b['mean']:+.{d}f} [{b['ci95'][0]:+.{d}f}, {b['ci95'][1]:+.{d}f}]"


def m(b, d=3):
    return "—" if b is None or b.get("mean") is None else f"{b['mean']:.{d}f}"


def house_boot(per_scene, n=10000, seed=0):
    h = {}
    for k, v in per_scene.items():
        h.setdefault(k.split("_")[0], []).append(v)
    g = np.asarray([np.mean(v) for v in h.values()])
    rng = np.random.default_rng(seed)
    bs = g[rng.integers(0, len(g), (n, len(g)))].mean(1)
    return {"mean": float(g.mean()), "ci95": [float(np.quantile(bs, 0.025)), float(np.quantile(bs, 0.975))], "n": len(g)}


def pair_row(P, key, d=3, house=False):
    p = P.get(key)
    if p is None:
        return "— | — | —"
    s = f"{f(p, d)} | {p['wins']}/{p['losses']}"
    e = p["episode"]
    s += f" | {f(e, d)} ({p['ep_wins']}/{p['ep_losses']})" if not house else f" | {f(house_boot(p['per_scene']), d)}"
    return s


L = []
for dom, nm in (("indoor", "MP3D VALSEL（90 个状态，9 个楼层 / episode，6 个房屋）"), ("outdoor", "Macarons++ Dev-3（275 个状态，30 个 episode，3 个场景）")):
    A = S[dom]["all"]
    M, P = A["means"], A["paired"]
    house = dom == "indoor"
    third = "房屋级 bootstrap（6 个房屋）" if house else "episode 级 bootstrap（胜/负）"
    L += [f"## {nm}", "", f"回归：{A['repro_all_ok']}/{A['n_states']} 个状态全部逐位一致（{A['n_repro_checks']} 项检查）；band_dil 双路径相等 {A['band_dil_eq']}/{A['n_states']}；oracle 最大增益 {m(A['oracle_max'])} pp", ""]
    L += ["### G2：12 组配置", "", "| 配置（ckpt\\|推理\\|评分） | Top-1 | 选中增益 pp | Spearman | 分数均值 | 0 分候选比例 | regret（附录） |", "|---|---|---|---|---|---|---|"]
    for c in CFGS:
        L.append(f"| {c} | {m(M[c + '#top1'])} | {m(M[c + '#sel_gain'])} | {m(M[c + '#rho'])} | {m(M[c + '#score_mean'], 1)} | {m(M[c + '#score_zero_frac'])} | {m(M[c + '#regret'])} |")
    L += ["", f"### G2 配对差（场景级 bootstrap 95% CI，胜/负；第三列：{third}）", "", "| 比较 | 指标 | 差值 [CI] | 胜/负 | 辅助 CI |", "|---|---|---|---|---|"]
    keys = []
    for mm in CK:
        for s in DN:
            keys.append(f"{mm}|D100|{s} - {mm}|S10|{s}")
        for i in DN:
            keys.append(f"{mm}|{i}|D100 - {mm}|{i}|S10")
    for i in DN:
        for s in DN:
            keys += [f"E2D|{i}|{s} - E2S|{i}|{s}", f"E2S|{i}|{s} - B2|{i}|{s}"]
    keys += ["E2D|D100|S10 - B2|D100|S10", "E2D|D100|S10 - E2S|S10|S10"]
    for k in keys:
        for met, d in (("sel_gain", 3), ("top1", 3), ("rho", 3)):
            L.append(f"| {k} | {met} | {pair_row(P, f'{k}|{met}', d, house)} |")
    L += ["", "### 选择改变比例（同一状态两种配置选中的候选不同）", "", "| 配置对 | 比例 |", "|---|---|"]
    for a, b in (("B2|S10|S10", "B2|D100|S10"), ("B2|D100|S10", "B2|D100|D100"), ("B2|S10|S10", "B2|S10|D100"), ("B2|S10|S10", "E2S|S10|S10"),
                 ("B2|D100|S10", "E2S|D100|S10"), ("E2S|D100|S10", "E2D|D100|S10"), ("E2D|D100|S10", "E2D|D100|D100"), ("E2S|S10|S10", "E2D|S10|S10")):
        k = f"chg#{a}#{b}" if f"chg#{a}#{b}" in M else f"chg#{b}#{a}"
        L.append(f"| {a} vs {b} | {m(M[k])} |")
    L += ["", "### G1：射线组成（占全部候选射线的比例）", "", "| 量 | 值 |", "|---|---|"]
    for k, nmx in (("unk10_frac", "S10 UNKNOWN"), ("unk100_frac", "D100 UNKNOWN"), ("unk_common_frac", "共同 UNKNOWN"), ("unk_only10_frac", "只在 S10 UNKNOWN"),
                   ("unk_only100_frac", "只在 D100 UNKNOWN"), ("only10_of_unk10", "S10 UNKNOWN 中在 D100 下已知/空的比例"), ("truth_new_frac", "真实新像素（GT 命中且离过去 10% 采样点 > eps）"),
                   ("unk10_truth_new_frac", "S10 UNKNOWN ∩ 真实新"), ("unk100_truth_new_frac", "D100 UNKNOWN ∩ 真实新")):
        L.append(f"| {nmx} | {m(M['rays#' + k], 4)} |")
    L += ["", "### G1：预测质量（每状态的比率，层级均值）", ""]
    for st, title in (("unk_common", "共同 UNKNOWN（主比较；两种推理都由模型预测）"), ("unk_only10", "只在 S10 UNKNOWN（S10 推理 = 模型；D100 推理 = 地图 t_known）"),
                      ("new_common", "共同 UNKNOWN ∩ 真实新表面"), ("unk_own", "各自地图的 UNKNOWN（历史口径，分母不同）")):
        L += [f"**{title}**", "", "| ckpt\\|推理 | 射线数/状态 | MAE（室内 m / 室外单位） | 首中/无中准确率 | 误报率 | 漏报率 | 准确首中率 |", "|---|---|---|---|---|---|---|"]
        for mm in CK:
            for n in DN:
                k = f"{mm}|{n}#{st}"
                L.append(f"| {mm}\\|{n} | {m(M[k + '#n'], 0)} | {m(M[k + '#mae'], 4)} | {m(M[k + '#hitacc'])} | {m(M.get(k + '#fp_rate'))} | {m(M[k + '#fn_rate'])} | {m(M[k + '#acc_first'])} |")
        L.append("")
    L += ["**K = 4 离散度**", "", "| ckpt\\|推理 | 相对 std | spread-误差 Spearman | UNKNOWN 射线命中分歧率 |", "|---|---|---|---|"]
    for mm in CK:
        for n in DN:
            L.append(f"| {mm}\\|{n} | {m(M[f'{mm}|{n}#spread_rel_std'], 4)} | {m(M[f'{mm}|{n}#spread_err_rho'])} | {m(M[f'{mm}|{n}#hit_dis_unk'])} |")
    L += ["", f"### G1 配对差（场景级 CI，胜/负；第三列：{third}）", "", "| 比较 | 射线集合 | 指标 | 差值 [CI] | 胜/负 | 辅助 CI |", "|---|---|---|---|---|---|"]
    for pr in ("B2|D100 - B2|S10", "E2S|D100 - E2S|S10", "E2D|D100 - E2D|S10", "E2D|D100 - E2S|D100", "E2D|S10 - E2S|S10", "E2S|S10 - B2|S10", "E2D|D100 - B2|D100"):
        for st in ("unk_common", "unk_only10", "new_common"):
            for q, d in (("mae", 4), ("hitacc", 4), ("fn_rate", 4), ("acc_first", 4)):
                L.append(f"| {pr} | {st} | {q} | {pair_row(P, f'G1:{pr}|{st}#{q}', d, house)} |")
    L += ["", "### 评分掩码（解码命中点；每状态比率）", "", "| 配置 | 被奖励比例 | 奖励中真实新 | 奖励中真实旧 | 奖励中 GT 无命中 | 奖励中准确命中 | 每候选每样本奖励射线 |", "|---|---|---|---|---|---|---|"]
    for c in CFGS:
        L.append(f"| {c} | {m(M[c + '#reward_frac'])} | {m(M[c + '#reward_truth_new_frac'])} | {m(M[c + '#reward_truth_old_frac'])} | {m(M[c + '#reward_no_gt_frac'])} | "
                 f"{m(M[c + '#reward_accurate_frac'])} | {m(M[c + '#reward_per_cand'], 0)} |")
    L += ["", "| GT 深度本身 | S10 band_dil | D100 band_dil |", "|---|---|---|",
          f"| 真实新 GT 像素被排除的比例 | {m(M['GT|10#excl_truth_new_frac'], 4)} | {m(M['GT|100#excl_truth_new_frac'], 4)} |",
          f"| GT 命中被奖励的比例 | {m(M['GT|10#reward_frac'], 4)} | {m(M['GT|100#reward_frac'], 4)} |",
          f"| GT 命中中被奖励但其实是旧表面的比例 | {m(M['GT|10#reward_truth_old_frac'], 4)} | {m(M['GT|100#reward_truth_old_frac'], 4)} |", ""]
    L += [f"### 评分掩码配对差（D100 评分 − S10 评分）", "", "| 比较 | 指标 | 差值 [CI] | 胜/负 | 辅助 CI |", "|---|---|---|---|---|"]
    for k in ("B2|D100|D100 - B2|D100|S10", "B2|S10|D100 - B2|S10|S10", "E2S|D100|D100 - E2S|D100|S10", "E2D|D100|D100 - E2D|D100|S10"):
        for q in ("reward_frac", "reward_truth_new_frac", "reward_truth_old_frac", "reward_accurate_frac", "score_zero_frac"):
            L.append(f"| {k} | {q} | {pair_row(P, f'{k}|{q}', 3, house)} |")
    L += ["", "### 按阶段（Top-1 / 选中增益 pp）", "", "| 阶段 | " + " | ".join(CFGS) + " |", "|---|" + "---|" * len(CFGS)]
    for ph in ("early", "middle", "late"):
        if ph in S[dom]:
            Q = S[dom][ph]["means"]
            L.append(f"| {ph}（{S[dom][ph]['n_states']}） | " + " | ".join(f"{m(Q[c + '#top1'])} / {m(Q[c + '#sel_gain'])}" for c in CFGS) + " |")
    L.append("")
open(os.path.join(EXP, "results", "tables_fixed.md"), "w").write("\n".join(L) + "\n")
print(len(L), "lines")
