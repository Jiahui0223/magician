"""CVPR paper-style scene-by-scene tables (display layer; MAGICIAN App. Tab. 6 / 7 style). Reads ONLY validated CVPR Table 1
outputs (results/per_scene.csv, per_model.csv, validation.json) and the MAGICIAN paper values stored in table1/SCENE_REGISTRY.yaml;
never edits them. Writes tables/{table_auc_full_scenes, table_final_full_scenes}[ _supp].{md,tex,csv} and the rendered
figures/<same>.{png,pdf}; runs consistency checks (rows, models, scenes, averages = mean of rows = Table 1 per_model All-14,
TeX column counts) and records provenance in MANIFEST.json. Usage: python make_tables.py (seen2scene env; run from anywhere)."""
import csv
import datetime
import hashlib
import json
import os
import subprocess

import numpy as np
import yaml
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402

HERE = os.path.dirname(os.path.abspath(__file__))
PT = os.path.dirname(HERE)
REPO = os.path.dirname(os.path.dirname(os.path.dirname(PT)))
CFG = yaml.safe_load(open(os.path.join(PT, "config.yaml")))
P = lambda k: os.path.join(REPO, CFG["sources"][k])  # noqa: E731
METRICS = {"auc": ("auc100", "AUC", "Table 6", "AUC@100 = trapz(C[0..100]) / 100"), "final": ("final", "Final Coverage", "Table 7", "Final = C[100]")}
CHECKS = []


def check(cond, msg):
    CHECKS.append({"ok": bool(cond), "check": msg})
    if not cond:
        raise AssertionError(msg)


def sha(p):
    return hashlib.sha256(open(p, "rb").read()).hexdigest()


def load():
    per_scene = {}
    for r in csv.DictReader(open(P("per_scene"))):
        per_scene[(r["model"], r["scene"])] = r
    per_model = {}
    for r in csv.DictReader(open(P("per_model"))):
        per_model[(r["model"], r["group"])] = r
    reg = yaml.safe_load(open(P("scene_registry")))
    paper = {it["local_id"]: it["magician_paper"] for it in reg["scenes"]}
    mreg = yaml.safe_load(open(P("model_registry")))
    status = {m["id"]: m.get("status") for m in mreg["models"]}
    val = json.load(open(P("validation")))
    return per_scene, per_model, paper, status, val


def ours_column(model, key, per_scene, per_model, status, val):
    """scene -> value for complete scenes; average over All-14 if all 14 complete (else None = partial)."""
    check(status.get(model) == "READY", f"{model}: registry status READY")
    check(not (val.get("bad") or {}).get(model), f"{model}: no invalid episodes in validation.json")
    col, complete = {}, []
    for s in CFG["scene_order"]:
        r = per_scene.get((model, s))
        if r is not None and r["complete"] == "True" and int(r["n_starts"]) == 5:
            col[s] = float(r[key])
            complete.append(s)
    s14 = [s for s in CFG["scene_order"] if s != "sofia_church"]
    full = all(s in col for s in s14)
    avg = float(np.mean([col[s] for s in s14])) if full else None
    if full:
        ref = per_model.get((model, "all14"))
        check(ref is not None and abs(float(ref[key]) - avg) < 1e-9, f"{model}: All-14 {key} mean of rows == per_model.csv ({avg:.6f})")
    return col, avg, full


def ranks(vals):
    """indices of best / second best (higher is better; ties share)."""
    v = [x for x in vals if x is not None]
    if not v:
        return set(), set()
    order = sorted(set(round(x, 3) for x in v), reverse=True)
    best = {i for i, x in enumerate(vals) if x is not None and round(x, 3) == order[0]}
    second = {i for i, x in enumerate(vals) if x is not None and len(order) > 1 and round(x, 3) == order[1]}
    return best, second


def build(metric, models, tag, per_scene, per_model, paper, status, val):
    key, mname, ptab, mdef = METRICS[metric]
    pm = CFG["paper_methods"]
    cols = {m: ours_column(m, key, per_scene, per_model, status, val) for m in models}
    rows = []
    for s in CFG["scene_order"]:
        pv = [float(paper[s][metric][m]) for m in pm]
        ov = [cols[m][0].get(s) for m in models]
        rows.append({"scene": s, "label": CFG["scene_labels"][s], "split": CFG["split_of"][s], "paper": pv, "ours": ov})
    s14 = [r for r in rows if r["scene"] != "sofia_church"]
    p14 = [float(np.mean([r["paper"][j] for r in s14])) for j in range(len(pm))]
    p15 = [float(np.mean([r["paper"][j] for r in rows])) for j in range(len(pm))]
    o14 = [cols[m][1] for m in models]
    reported15 = {"auc": [0.241, 0.534, 0.647, 0.546, 0.721], "final": [0.324, 0.670, 0.819, 0.786, 0.919]}[metric]   # paper Tab. 2 / 6 / 7 'Average' rows
    check(len(rows) == 15 and len(s14) == 14, f"{tag}: 15 scene rows, 14 in All-14")
    check(all(r["ours"][i] is not None for r in s14 for i, m in enumerate(models) if cols[m][2]), f"{tag}: complete models have all 14 scenes")
    check(all(r["ours"][i] is None for r in rows if r["scene"] == "sofia_church" for i in range(len(models))), f"{tag}: no re-evaluated value for sofia_church")
    for j in range(len(pm)):
        if not (metric == "final" and pm[j] == "fisherrf"):
            check(abs(round(p15[j], 3) - reported15[j]) <= 0.0011, f"{tag}: paper {pm[j]} 15-scene mean of rows ({p15[j]:.4f}) == reported average ({reported15[j]}) within rounding")
    return {"metric": metric, "key": key, "mname": mname, "ptab": ptab, "mdef": mdef, "models": models, "rows": rows, "p14": p14, "p15": p15,
            "reported15": reported15, "o14": o14, "complete": {m: cols[m][2] for m in models}}


def f3(x):
    return "--" if x is None else f"{x:.3f}"


def write_csv(T, path):
    pm, ms = CFG["paper_methods"], T["models"]
    with open(path, "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["scene", "scene_label", "split"] + [f"paper_reported:{CFG['paper_labels'][m]}" for m in pm] + [f"re_evaluated:{CFG['model_labels'][m]}" for m in ms])
        for r in T["rows"]:
            w.writerow([r["scene"], r["label"], r["split"]] + [f"{x:.3f}" for x in r["paper"]] + [("" if x is None else f"{x:.6f}") for x in r["ours"]])
        w.writerow(["AVERAGE_ALL14", "Average (All-14)", ""] + [f"{x:.6f}" for x in T["p14"]] + [("partial" if x is None else f"{x:.6f}") for x in T["o14"]])
        w.writerow(["AVERAGE_ALL15_REPORTED", "Average (All-15, as reported)", ""] + [f"{x:.3f}" for x in T["reported15"]] + ["n/a"] * len(ms))
        w.writerow(["AVERAGE_ALL15_RECOMPUTED", "Average (All-15, recomputed from rows)", ""] + [f"{x:.6f}" for x in T["p15"]] + ["n/a"] * len(ms))


def notes(T):
    fish = " The paper's Tab. 7 reports a FisherRF average of 0.786, whereas the mean of its 15 rows is 0.772; we show the row means." if T["metric"] == "final" else ""
    return (f"{T['mname']} of full scenes on Macarons++ (style of MAGICIAN App. {T['ptab']}). Left: values reported in the MAGICIAN paper (App. {T['ptab']}); "
            f"different runs, starts and evaluation code, shown for reference only. Right: all methods re-evaluated by us under one frozen protocol "
            f"(official MAGICIAN code, weights and evaluator; settings.json starts 0-4; 100 moves; official coverage, eps = 1.0; {T['mdef']}); "
            f"mean of 5 starts per scene, scene-equal averages. Bold / underline = best / second best within the re-evaluated group only. "
            f"St. Sofia Church cannot be loaded by the frozen evaluator (loader bug fixed upstream only in 2026-08), so all re-evaluated averages are All-14; "
            f"the paper's averages are shown both over the same 14 scenes and as reported (15 scenes).{fish}")


def write_md(T, path):
    pm, ms = CFG["paper_methods"], T["models"]
    L = [f"# {T['mname']} of full scenes (All-14 re-evaluated; MAGICIAN App. {T['ptab']} style)", "", "> " + notes(T), "",
         "| Scene | " + " | ".join(f"{CFG['paper_labels'][m]} (paper)" for m in pm) + " | " + " | ".join(CFG["model_labels"][m] for m in ms) + " |",
         "|---|" + "---:|" * (len(pm) + len(ms))]
    for r in T["rows"] + [None]:
        if r is None:
            vals, lab, pv = T["o14"], "**Average (All-14)**", T["p14"]
        else:
            vals, lab, pv = r["ours"], r["label"], r["paper"]
        b, s2 = ranks(vals)
        oc = [("**" + f3(x) + "**") if i in b else (("<u>" + f3(x) + "</u>") if i in s2 else f3(x)) for i, x in enumerate(vals)]
        if r is not None and r["scene"] == "sofia_church":
            oc = ["n/a"] * len(ms)
        L.append(f"| {lab} | " + " | ".join(f"{x:.3f}" for x in pv) + " | " + " | ".join(oc) + " |")
    L.append("| Average (All-15, as reported) | " + " | ".join(f"{x:.3f}" for x in T["reported15"]) + " | " + " | ".join("n/a" for _ in ms) + " |")
    open(path, "w").write("\n".join(L) + "\n")


def tex_escape(s):
    return s.replace("&", r"\&").replace("_", r"\_")


def write_tex(T, path, label):
    pm, ms = CFG["paper_methods"], T["models"]
    ncol = 1 + len(pm) + 1 + len(ms)
    hdr = ["Scene"] + [CFG["paper_labels"][m] for m in pm] + [""] + [tex_escape(CFG["model_labels"][m]).replace(" (re-eval.)", r"$^\dagger$") for m in ms]
    L = [r"% Auto-generated by experiments/cvpr_results/paper_tables/scripts/make_tables.py -- do not edit by hand.",
         r"% Requires \usepackage{booktabs}.",
         r"\begin{table*}[t]", r"\centering", r"\footnotesize", r"\setlength{\tabcolsep}{4pt}",
         r"\begin{tabular}{l" + "c" * len(pm) + "c" + "c" * len(ms) + "}", r"\toprule",
         rf"& \multicolumn{{{len(pm)}}}{{c}}{{Reported in MAGICIAN (App.~Tab.~{T['ptab'][-1]})}} & & \multicolumn{{{len(ms)}}}{{c}}{{Re-evaluated by us, identical protocol}} \\",
         rf"\cmidrule(lr){{2-{1 + len(pm)}}} \cmidrule(lr){{{3 + len(pm)}-{2 + len(pm) + len(ms)}}}",
         " & ".join(hdr) + r" \\", r"\midrule"]

    def cell(x, i, b, s2):
        if x is None:
            return "--"
        v = f"{x:.3f}"
        return rf"\textbf{{{v}}}" if i in b else (rf"\underline{{{v}}}" if i in s2 else v)
    for r in T["rows"]:
        b, s2 = ranks(r["ours"])
        oc = [cell(x, i, b, s2) for i, x in enumerate(r["ours"])] if r["scene"] != "sofia_church" else ["n/a"] * len(ms)
        L.append(" & ".join([tex_escape(r["label"])] + [f"{x:.3f}" for x in r["paper"]] + [""] + oc) + r" \\")
    L.append(r"\midrule")
    b, s2 = ranks(T["o14"])
    L.append(" & ".join(["Average (All-14)"] + [f"{x:.3f}" for x in T["p14"]] + [""] + [cell(x, i, b, s2) for i, x in enumerate(T["o14"])]) + r" \\")
    rep = [f"{x:.3f}" + (r"$^*$" if (T["metric"] == "final" and pm[j] == "fisherrf") else "") for j, x in enumerate(T["reported15"])]
    L.append(" & ".join(["Average (All-15, reported)"] + rep + [""] + ["n/a"] * len(ms)) + r" \\")
    L += [r"\bottomrule", r"\end{tabular}", r"\caption{" + tex_escape(notes(T)).replace("eps = 1.0", r"$\epsilon=1.0$") + r" $^\dagger$Re-evaluated MAGICIAN (beam $10\times10$).}",
          rf"\label{{{label}}}", r"\end{table*}"]
    for line in L:                                         # column-count check of every data row
        if line.endswith(r"\\") and not line.startswith(r"\cmidrule") and "multicolumn" not in line:
            check(line.count("&") == ncol - 1, f"TeX row has {ncol} columns: {line[:40]}")
    check(sum(line.count("{") for line in L) == sum(line.count("}") for line in L), f"{label}: balanced braces")
    open(path, "w").write("\n".join(L) + "\n")


def render(T, path_base):
    pm, ms = CFG["paper_methods"], T["models"]
    plt.rcParams.update({"font.family": "STIXGeneral", "mathtext.fontset": "stix"})
    hdr = ["Scene"] + [CFG["paper_labels"][m] for m in pm] + [CFG["model_labels"][m].replace(" (re-eval.)", "†") for m in ms]
    body = []
    styles = []
    for r in T["rows"] + ["avg14", "avg15"]:
        if r == "avg14":
            vals, lab, pv = T["o14"], "Average (All-14)", T["p14"]
        elif r == "avg15":
            vals, lab, pv = [None] * len(ms), "Average (All-15, reported)", T["reported15"]
        else:
            vals, lab, pv = r["ours"], r["label"], r["paper"]
        b, s2 = ranks(vals)
        sofia = r not in ("avg14", "avg15") and r["scene"] == "sofia_church"
        body.append([lab] + [f"{x:.3f}" for x in pv] + (["n/a"] * len(ms) if (sofia or r == "avg15") else [f3(x) for x in vals]))
        styles.append([None] + [None] * len(pm) + [("b" if i in b else "u" if i in s2 else None) if not (sofia or r == "avg15") else None for i in range(len(ms))])
    ncol = len(hdr)
    GAP = 0.35                                                    # visual gap between the two groups (inches)
    widths = [2.05] + [0.9] * len(pm) + [max(1.0, 0.085 * len(h) + 0.35) for h in hdr[1 + len(pm):]]
    left = [0.15]
    for j in range(1, ncol):
        left.append(left[-1] + widths[j - 1] + (GAP if j == 1 + len(pm) else 0.0))
    fig_w, row_h = left[-1] + widths[-1] + 0.15, 0.27
    nrow = len(body) + 5.0
    fig = plt.figure(figsize=(fig_w, row_h * nrow))
    ax = fig.add_axes([0, 0, 1, 1])
    ax.axis("off")
    xs = np.asarray(left) / fig_w
    xr = (np.asarray(left) + np.asarray(widths)) / fig_w
    xc = (xs + xr) / 2
    dy = 1 / nrow
    ytop = 1 - 0.6 * dy
    g1 = (xs[1], xr[len(pm)])
    g2 = (xs[1 + len(pm)], xr[ncol - 1])
    ax.text((g1[0] + g1[1]) / 2, ytop, f"Reported in MAGICIAN (App. {T['ptab']})", ha="center", va="center", fontsize=9, style="italic")
    ax.text((g2[0] + g2[1]) / 2, ytop, "Re-evaluated by us, identical protocol", ha="center", va="center", fontsize=9, style="italic")
    for g in (g1, g2):
        ax.plot([g[0] + 0.004, g[1] - 0.004], [ytop - 0.45 * dy] * 2, color="k", lw=0.6)
    y = ytop - dy
    x0, x1 = 0.01, xr[-1] + 0.005
    ax.plot([x0, x1], [y + 0.55 * dy] * 2, color="k", lw=1.0)
    for j, h in enumerate(hdr):
        ax.text(xs[j] if j == 0 else xc[j], y, h, ha="left" if j == 0 else "center", va="center", fontsize=9, weight="bold")
    ax.plot([x0, x1], [y - 0.5 * dy] * 2, color="k", lw=0.6)
    ytab = y - 0.5 * dy
    fish = T["metric"] == "final"
    for i, (row, st) in enumerate(zip(body, styles)):
        y -= dy
        if row[0].startswith("Average (All-14)"):
            ax.plot([x0, x1], [y + 0.5 * dy] * 2, color="k", lw=0.6)
        for j, v in enumerate(row):
            if fish and row[0].startswith("Average (All-15") and j == 1 + CFG["paper_methods"].index("fisherrf"):
                v = v + "*"
            kw = {"fontsize": 9, "va": "center", "ha": "left" if j == 0 else "center"}
            if st[j] == "b":
                kw["weight"] = "bold"
            ax.text(xs[j] if j == 0 else xc[j], y, v, **kw)
            if st[j] == "u":
                ax.plot([xc[j] - 0.021, xc[j] + 0.021], [y - 0.33 * dy] * 2, color="k", lw=0.6)
    ax.plot([x0, x1], [y - 0.5 * dy] * 2, color="k", lw=1.0)
    xm = (xr[len(pm)] + xs[1 + len(pm)]) / 2                        # thin rule between the groups
    ax.plot([xm, xm], [y - 0.5 * dy, ytab], color="#888888", lw=0.5)
    note = ("Left: values reported in the MAGICIAN paper (different runs / starts / evaluation code; reference only). "
            "Right: re-evaluated by us (official MAGICIAN code + evaluator, settings starts 0-4, 100 moves); "
            "† = MAGICIAN re-evaluated. Bold / underline: best / second within the right group. n/a: St. Sofia Church does not load in the frozen evaluator.")
    if fish:
        note += " *Paper-reported FisherRF average; the mean of its 15 rows is 0.772."
    ax.text(0.01, y - 0.8 * dy, note, ha="left", va="top", fontsize=6.5, wrap=True, color="#333333")
    for ext in ("png", "pdf"):
        fig.savefig(f"{path_base}.{ext}", dpi=CFG["dpi"])
    plt.close(fig)


def main():
    os.makedirs(os.path.join(PT, "tables"), exist_ok=True)
    os.makedirs(os.path.join(PT, "figures"), exist_ok=True)
    per_scene, per_model, paper, status, val = load()
    out = {}
    for metric, base in (("auc", "table_auc_full_scenes"), ("final", "table_final_full_scenes")):
        for models, suf in ((CFG["main_models"], ""), (CFG["supp_models"], "_supp")):
            T = build(metric, models, base + suf, per_scene, per_model, paper, status, val)
            name = base + suf
            write_csv(T, os.path.join(PT, "tables", name + ".csv"))
            write_md(T, os.path.join(PT, "tables", name + ".md"))
            write_tex(T, os.path.join(PT, "tables", name + ".tex"), f"tab:{name.replace('_', '-')}")
            render(T, os.path.join(PT, "figures", name))
            out[name] = {"models": models, "all14": dict(zip(models, T["o14"])), "paper_all14": dict(zip(CFG["paper_methods"], T["p14"]))}
    try:
        commit = subprocess.check_output(["git", "rev-parse", "--short", "HEAD"], cwd=REPO, text=True).strip()
    except Exception:
        commit = None
    man_p = os.path.join(PT, "MANIFEST.json")
    man = json.load(open(man_p)) if os.path.exists(man_p) else {}
    man["tables"] = {"generated": datetime.datetime.now().isoformat(timespec="seconds"), "git_head_at_generation": commit, "config_version": CFG["version"],
                     "config_sha256": sha(os.path.join(PT, "config.yaml")), "script": os.path.relpath(__file__, REPO),
                     "inputs": {k: {"path": CFG["sources"][k], "sha256": sha(P(k))} for k in ("per_scene", "per_model", "validation", "scene_registry", "model_registry")},
                     "outputs": out, "checks": CHECKS}
    json.dump(man, open(man_p, "w"), indent=1)
    print(f"tables ok: {len(CHECKS)} checks passed")


if __name__ == "__main__":
    main()
