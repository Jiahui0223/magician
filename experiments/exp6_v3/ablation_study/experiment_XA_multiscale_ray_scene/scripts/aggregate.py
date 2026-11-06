"""Experiment XA -- aggregation (CPU; re-runnable). For every variant with closed-loop outputs:
  outdoor  official MAGICIAN coverage C[0..100] of every episode (the episode's stored curve; the replay check official_equal_stored must be true):
           AUC@k = trapz(C[0..k]) / k (k = 20, 50, 100), Cov@20, Cov@50, Final = C[100]; scene = mean of its 5 starts; groups Train-6, Dev-5, Test-3,
           Unseen-8 (Dev-5 + Test-3), All-14 = equal-weight scene means; paired difference vs V3-T1 + D1 (CVPR Table 1 records V3-D1-H3-NORISK, same
           starts): per scene the mean of the same-start differences, scene bootstrap 95 % CI (10000 draws, seed 0), wins / losses. No cummax.
  indoor   C_obs[0..150] of the 6 fixed episodes (3 floors x 2 starts): AUC@20, Cov@20, AUC@100, Cov@100, AUC@150, Final; vs V3-T1 + H0 (exp6_v3 eval T1).
           Reported separately; never averaged with the outdoor coverage.
  cost     parameters, training speed / memory, planning seconds per step, peak planner memory, retries
  status   COMPLETE only with a selected checkpoint, 70 / 70 outdoor episodes of 101 values with a passed replay check and 6 / 6 indoor of 151 values
Outputs: results/<id>/{outdoor.json, indoor.json, episodes.csv, status.json, report.md, curves.png}; reports/{MASTER_TABLE.csv, PER_SCENE.csv, EARLY_CURVES.png}."""
import csv
import glob
import hashlib
import json
import os

import numpy as np
import yaml

HERE = os.path.dirname(os.path.abspath(__file__))
X = os.path.dirname(HERE)
R = os.path.dirname(os.path.dirname(os.path.dirname(os.path.dirname(X))))
W = os.path.join(R, "work_dirs", "exp6_v3", "ablation_study", "experiment_XA_multiscale_ray_scene")
TB = os.path.join(R, "work_dirs", "cvpr_results", "table1", "episodes")
T1_IN = os.path.join(R, "work_dirs", "exp6_v3", "eval", "closed_loop", "indoor", "T1", "V2M")
IDS = tuple(os.environ["XA_TEST_IDS"].split(",")) if os.environ.get("XA_TEST_IDS") else ("xa_s2d", "xa_sr", "xa_si", "xa_m2d", "xa_mr", "xa_mi")   # XA_TEST_IDS + XA_TEST_OUT: pipeline test only
INDOOR = (("Z6MFQCViBuw", 0, 6), ("Z6MFQCViBuw", 0, 17), ("oLBMNvg9in8", 1, 36), ("oLBMNvg9in8", 1, 78), ("QUCTc6BB5sX", 1, 34), ("QUCTc6BB5sX", 1, 135))
REFS = (("V3-D1-H3-NORISK", "V3-T1 + D1 (anchor)"), ("V3", "V3-T1 + H0"), ("MAG-B", "MAGICIAN Beam"), ("B0", "B0 (V2-Mix, H0)"), ("B2", "B2 (V2-Mix D100, H0)"))
reg = yaml.safe_load(open(os.path.join(R, "experiments", "cvpr_results", "table1", "SCENE_REGISTRY.yaml")))["scenes"]
items = reg.items() if isinstance(reg, dict) else [(s["local_id"], s) for s in reg]
SPLIT = {k: str(v["split"]).lower() for k, v in items}
ALL14 = sorted(k for k, v in items if (v.get("runnable") or {}).get("overall"))
GROUPS = {"Train-6": [s for s in ALL14 if SPLIT[s].startswith("train")], "Dev-5": [s for s in ALL14 if SPLIT[s].startswith("dev")], "Test-3": [s for s in ALL14 if SPLIT[s].startswith("test")]}
GROUPS["Unseen-8"] = GROUPS["Dev-5"] + GROUPS["Test-3"]
GROUPS["All-14"] = ALL14
OM = ("auc20", "cov20", "auc50", "cov50", "auc100", "final")
IM = ("auc20", "cov20", "auc100", "cov100", "auc150", "final")


def om(c):
    c = np.asarray(c, float)
    return {"auc20": float(np.trapz(c[:21]) / 20), "cov20": float(c[20]), "auc50": float(np.trapz(c[:51]) / 50), "cov50": float(c[50]), "auc100": float(np.trapz(c[:101]) / 100), "final": float(c[100])}


def im(c):
    c = np.asarray(c, float)
    return {"auc20": float(np.trapz(c[:21]) / 20), "cov20": float(c[20]), "auc100": float(np.trapz(c[:101]) / 100), "cov100": float(c[100]), "auc150": float(np.trapz(c[:151]) / 150), "final": float(c[150])}


def boot(v, n=10000, seed=0):
    v = np.asarray(v, float)
    o = {"mean": float(v.mean()), "n": len(v), "wins": int((v > 1e-12).sum()), "losses": int((v < -1e-12).sum())}
    if len(v) >= 2:
        b = v[np.random.default_rng(seed).integers(0, len(v), (n, len(v)))].mean(1)
        o["ci"] = [float(np.quantile(b, .025)), float(np.quantile(b, .975))]
    return o


def table1(model):
    out = {}
    for f in glob.glob(os.path.join(TB, model, "*.json")):
        r = json.load(open(f))
        out[(r["scene"], int(r["start"]))] = {"c": r["coverage_official"], "mono": r.get("coverage_mono"), "keys": r.get("executed_keys")}
    return out


def outdoor(vid):
    root = os.path.join(W, "eval", vid)
    out = {}
    for f in sorted(glob.glob(os.path.join(root, "outdoor", "V2M", "episodes", "ep_V2M_*_r0.json"))):
        e = json.load(open(f))
        mf = os.path.join(root, "metrics", "V2M", f"V2M_{e['scene']}_s{e['start']}_r0.json")
        m = json.load(open(mf)) if os.path.exists(mf) else None
        pl = [x["planner"] for x in e["steps"] if x.get("planner")]
        out[(e["scene"], int(e["start"]))] = {"c": e["coverage"], "mono": m["cov_mono"] if m else None, "replay_ok": bool(m and m["official_equal_stored"]), "moves": e["moves"],
                                              "n_hover": e.get("n_hover"), "sha": (e["planner_hello"][0] or {}).get("ckpt_sha256") if e.get("planner_hello") else None,
                                              "decision_s": float(np.mean([p["decision_s"] for p in pl])) if pl else None, "resumed": e.get("resume") is not None,
                                              "seconds": e.get("seconds_this_process"), "keys": e["executed_keys"]}
    return out


def indoor(vid):
    out = {}
    for scan, lv, st in INDOOR:
        f = os.path.join(W, "eval", vid, "indoor", "V2M", f"{scan}_L{lv}_S{st}", "episode.json")
        if os.path.exists(f):
            e = json.load(open(f))
            out[(f"{scan}_L{lv}", st)] = {"c": e["coverage"], "moves": e["moves"], "decision_s": float(np.mean([x["decision_s"] for x in e["steps"]])), "seconds": e.get("wall_s")}
    return out


def indoor_ref():
    out = {}
    for scan, lv, st in INDOOR:
        e = json.load(open(os.path.join(T1_IN, f"{scan}_L{lv}_S{st}", "episode.json")))
        out[(f"{scan}_L{lv}", st)] = {"c": e["coverage"]}
    return out


def group_stats(E, base=None):
    """E: {(scene, start): {"c": curve}} with complete scenes only used."""
    res = {}
    scenes = {s: [k for k in E if k[0] == s] for s in ALL14}
    full = [s for s in ALL14 if len(scenes[s]) == 5]
    per = {s: {m: float(np.mean([om(E[k]["c"])[m] for k in scenes[s]])) for m in OM} for s in full}
    for g, ss in GROUPS.items():
        done = [s for s in ss if s in per]
        if not done:
            continue
        r = {"scenes": len(done), "of": len(ss), "complete": len(done) == len(ss), "episodes": 5 * len(done)}
        for m in OM:
            v = [per[s][m] for s in done]
            r[m] = boot(v)
            if base is not None:
                d = [float(np.mean([om(E[k]["c"])[m] - om(base[k]["c"])[m] for k in scenes[s]])) for s in done if all(k in base for k in scenes[s])]
                if d:
                    r["d_" + m] = boot(d)
        res[g] = r
    return res, per


def curves_mean(E, ss):
    done = [s for s in ss if sum(k[0] == s for k in E) == 5]
    if not done:
        return None
    return np.mean([np.mean([E[(s, k)]["c"] for k in range(5)], 0) for s in done], 0)


def fmt(x, d=3, signed=False):
    if x is None:
        return "—"
    s = f"{x['mean']:+.{d}f}" if signed else f"{x['mean']:.{d}f}"
    return s + (f" [{x['ci'][0]:+.{d}f}, {x['ci'][1]:+.{d}f}]" if signed and "ci" in x else "") + (f" {x['wins']}/{x['losses']}" if signed else "")


def train_info(vid):
    run = os.path.join(W, "runs", vid)
    o = {}
    fj = os.path.join(W, "checkpoints", vid, "final.json")
    if os.path.exists(fj):
        o["final"] = {k: v for k, v in json.load(open(fj)).items() if k in ("chosen_step", "chosen_score", "sha256", "scores", "rule")}
    rj = sorted(glob.glob(os.path.join(run, "run_none_*.json")))
    if rj:
        r = json.load(open(rj[-1]))
        o.update({"params": r["params"], "common_init_sha256": r["common_init_sha256"], "git": r["git"], "config_sha256": r["config_sha256"], "model_cfg_sha256": r["model_cfg_sha256"],
                  "data_manifest_sha256": r["data_manifest_sha256"], "host": r["host"], "world_size": r["world_size"], "resumed_runs": len(rj) - 1})
    lp = os.path.join(run, "log.jsonl")
    if os.path.exists(lp):
        L = [json.loads(x) for x in open(lp)]
        if L:
            o.update({"last_step": L[-1]["step"], "s_per_step_median": float(np.median([x["s_per_step"] for x in L[1:]] or [0])), "peak_mem_gb": float(max(x["mem_gb"] for x in L)),
                      "final_loss": {k: L[-1][k] for k in ("loss", "fm", "rec", "kl")}, "curve": [{k: x[k] for k in ("step", "loss", "fm", "rec", "kl", "lr")} for x in L if x["step"] % 1000 == 0 or x["step"] == 1]})
    ep = os.path.join(run, "eval.jsonl")
    if os.path.exists(ep):
        o["validation"] = [{"step": e["step"], "fm_in": e["fm_in_val"], "fm_out": e["fm_out_sc"], "es_in": e["samp_in_val"]["fair_es_arclength_vox"], "es_out": e["samp_out_sc"]["fair_es_arclength_vox"],
                            "brier_in": e["samp_in_val"]["brier"], "brier_out": e["samp_out_sc"]["brier"], "false_wall_in": e["samp_in_val"]["false_wall"], "false_wall_out": e["samp_out_sc"]["false_wall"],
                            "hit_ratio_in": e["samp_in_val"]["hit_ratio"], "hit_ratio_out": e["samp_out_sc"]["hit_ratio"]} for e in map(json.loads, open(ep))]
    tl = os.path.join(run, "train.log")
    if os.path.exists(tl):
        txt = open(tl, errors="ignore").read()
        o["train_starts"] = txt.count("\nstart ") + txt.startswith("start ")
        o["train_exits"] = [x.split()[1] for x in txt.splitlines() if x.startswith("EXIT ")]
    return o


def main():
    base = table1("V3-D1-H3-NORISK")
    refs = {m: table1(m) for m, _ in REFS}
    iref = indoor_ref()
    master, perscene = [], []
    curves = {}
    for m, lab in REFS:
        g, per = group_stats(refs[m], base if m != "V3-D1-H3-NORISK" else None)
        for gn, r in g.items():
            master.append([m, lab, "reference (CVPR Table 1 records)", gn, r["scenes"], r["episodes"]] + [r[k]["mean"] for k in OM] + [r.get("d_" + k, {}).get("mean") for k in OM] + [None] * 12)
        curves[lab] = {gn: curves_mean(refs[m], ss) for gn, ss in GROUPS.items()}
    OUT = os.environ.get("XA_TEST_OUT") or X
    os.makedirs(os.path.join(OUT, "reports"), exist_ok=True)
    for vid in IDS:
        E, I = outdoor(vid), indoor(vid)
        if not E and not I and not os.path.exists(os.path.join(W, "runs", vid)):
            continue
        od = os.path.join(OUT, "results", vid)
        os.makedirs(od, exist_ok=True)
        tr = train_info(vid)
        g, per = group_stats(E, base)
        bad = [f"{k[0]}_s{k[1]}" for k, e in E.items() if len(e["c"]) != 101 or not e["replay_ok"] or e["moves"] != 100]
        shas = sorted({e["sha"] for e in E.values() if e["sha"]})
        st = {"id": vid, "trained": tr.get("last_step") == 40000, "selected": "final" in tr, "outdoor_episodes": len(E), "outdoor_bad": bad, "indoor_episodes": len(I),
              "indoor_bad": [f"{k[0]}_S{k[1]}" for k, e in I.items() if len(e["c"]) != 151], "planner_ckpt_sha": shas,
              "sha_matches_final": bool(shas and tr.get("final") and shas == [tr["final"]["sha256"]])}
        st["complete"] = bool(st["trained"] and st["selected"] and len(E) == 70 and not bad and len(I) == 6 and not st["indoor_bad"] and st["sha_matches_final"])
        json.dump(st, open(os.path.join(od, "status.json"), "w"), indent=1)
        cost = {"decision_s_outdoor": float(np.mean([e["decision_s"] for e in E.values() if e["decision_s"]])) if E else None,
                "decision_s_indoor": float(np.mean([e["decision_s"] for e in I.values()])) if I else None, "resumed_episodes": int(sum(e["resumed"] for e in E.values())),
                "identical_to_T1_trajectories": int(sum(e["keys"] == base.get(k, {}).get("keys") for k, e in E.items()))}
        ql = os.path.join(W, "eval", vid, "logs", "queue.log")
        if os.path.exists(ql):
            q = [x.split() for x in open(ql)]
            cost["episode_tries"] = len(q)
            cost["episode_failed_tries"] = int(sum(x[3] != "0" for x in q))
            cost["worker_hours"] = float(sum(int(x[7].rstrip("s")) for x in q if x[3] == "0") / 3600)
        json.dump({"groups": g, "per_scene": per, "cost": cost, "train": tr}, open(os.path.join(od, "outdoor.json"), "w"), indent=1, default=float)
        # indoor
        ind = {"episodes": {}, "floors": {}}
        for k, e in I.items():
            a, b = im(e["c"]), im(iref[k]["c"])
            ind["episodes"][f"{k[0]}_S{k[1]}"] = {"xa": a, "t1_h0": b, "d": {m: a[m] - b[m] for m in IM}}
        for fl in sorted({k[0] for k in I}):
            ks = [k for k in I if k[0] == fl]
            ind["floors"][fl] = {m: float(np.mean([im(I[k]["c"])[m] for k in ks])) for m in IM}
            ind["floors"][fl].update({"d_" + m: float(np.mean([im(I[k]["c"])[m] - im(iref[k]["c"])[m] for k in ks])) for m in IM})
        if len(ind["floors"]) == 3 and len(I) == 6:
            ind["mean"] = {m: float(np.mean([ind["floors"][f][m] for f in ind["floors"]])) for m in IM}
            ind["mean_t1_h0"] = {m: float(np.mean([np.mean([im(iref[k]["c"])[m] for k in I if k[0] == f]) for f in ind["floors"]])) for m in IM}
            ind["d"] = {m: boot([ind["floors"][f]["d_" + m] for f in ind["floors"]]) for m in IM}
        json.dump(ind, open(os.path.join(od, "indoor.json"), "w"), indent=1, default=float)
        with open(os.path.join(od, "episodes.csv"), "w", newline="") as f:
            w = csv.writer(f)
            w.writerow(["domain", "scene", "start", "split"] + list(OM) + ["t1d1_" + m for m in OM] + ["decision_s", "replay_ok", "resumed"] + [f"C{t}" for t in range(101)])
            for k in sorted(E):
                a, b = om(E[k]["c"]), om(base[k]["c"])
                w.writerow(["outdoor", k[0], k[1], SPLIT[k[0]]] + [a[m] for m in OM] + [b[m] for m in OM] + [E[k]["decision_s"], E[k]["replay_ok"], E[k]["resumed"]] + list(E[k]["c"]))
            for k in sorted(I):
                a, b = im(I[k]["c"]), im(iref[k]["c"])
                w.writerow(["indoor", k[0], k[1], "mp3d_val"] + [a[m] for m in IM] + [b[m] for m in IM] + [I[k]["decision_s"], "", ""] + list(I[k]["c"][:101]))
        for gn, r in g.items():
            master.append([vid, vid, "COMPLETE" if st["complete"] else "INCOMPLETE", gn, r["scenes"], r["episodes"]] + [r[k]["mean"] for k in OM] + [r.get("d_" + k, {}).get("mean") for k in OM]
                          + [x for k in OM for x in (r.get("d_" + k, {}).get("ci", [None, None]))])
        for s, p in per.items():
            b = {m: float(np.mean([om(base[(s, k)]["c"])[m] for k in range(5)])) for m in OM}
            perscene.append([vid, s, SPLIT[s]] + [p[m] for m in OM] + [p[m] - b[m] for m in OM])
        curves[vid] = {gn: curves_mean(E, ss) for gn, ss in GROUPS.items()}
        # per-model report (generated)
        L = [f"# {vid} — generated summary (`scripts/aggregate.py`)", "", f"Status: **{'COMPLETE' if st['complete'] else 'INCOMPLETE'}** — outdoor {len(E)}/70, indoor {len(I)}/6, trained {st['trained']}, selected {st['selected']}.", ""]
        if tr.get("params"):
            L += [f"- Parameters {tr['params']['total']:,} (new block {tr['params']['xattn']:,}); training {tr.get('s_per_step_median', 0):.2f} s/step, peak {tr.get('peak_mem_gb', 0):.1f} GB per rank; restarts {tr.get('resumed_runs')}.",
                  f"- Checkpoint: step {tr.get('final', {}).get('chosen_step')}, sha256 `{str(tr.get('final', {}).get('sha256'))[:16]}…` (rule: ES ratio on VALSEL + S-c)."]
        if cost["decision_s_outdoor"]:
            L += [f"- Planning: {cost['decision_s_outdoor']:.2f} s/step outdoor (D1), {cost['decision_s_indoor'] or float('nan'):.2f} s/step indoor (H0); episode tries {cost.get('episode_tries')} (failed {cost.get('episode_failed_tries')})."]
        L += ["", "## Outdoor (official coverage; scene-equal; difference vs V3-T1 + D1 with scene-bootstrap 95 % CI and scene wins/losses)", "", "| group | scenes | AUC@20 | Cov@20 | AUC@50 | AUC@100 | Final | ΔAUC@20 | ΔCov@20 | ΔAUC@100 | ΔFinal |", "|---|---|---|---|---|---|---|---|---|---|---|"]
        for gn, r in g.items():
            L.append(f"| {gn} | {r['scenes']}/{r['of']} | {fmt(r['auc20'])} | {fmt(r['cov20'])} | {fmt(r['auc50'])} | {fmt(r['auc100'])} | {fmt(r['final'])} | {fmt(r.get('d_auc20'), signed=True)} | {fmt(r.get('d_cov20'), signed=True)} | {fmt(r.get('d_auc100'), signed=True)} | {fmt(r.get('d_final'), signed=True)} |")
        L += ["", "| scene | split | AUC@20 | AUC@100 | Final | ΔAUC@20 | ΔAUC@100 | ΔFinal |", "|---|---|---|---|---|---|---|---|"]
        for row in [x for x in perscene if x[0] == vid]:
            L.append(f"| {row[1]} | {row[2]} | {row[3]:.3f} | {row[7]:.3f} | {row[8]:.3f} | {row[9]:+.3f} | {row[13]:+.3f} | {row[14]:+.3f} |")
        if ind["episodes"]:
            L += ["", "## Indoor sanity (C_obs, H0; 3 floors × 2 starts; not comparable with the outdoor coverage)", "", "| episode | AUC@20 | Cov@20 | AUC@100 | Cov@100 | AUC@150 | Final | ΔAUC@150 vs T1 | ΔFinal vs T1 |", "|---|---|---|---|---|---|---|---|---|"]
            for k, e in ind["episodes"].items():
                L.append(f"| {k} | " + " | ".join(f"{e['xa'][m]:.3f}" for m in IM) + f" | {e['d']['auc150']:+.3f} | {e['d']['final']:+.3f} |")
        open(os.path.join(od, "report.md"), "w").write("\n".join(L) + "\n")
    hdr = ["id", "label", "status", "group", "scenes", "episodes"] + list(OM) + ["d_" + m for m in OM] + [f"d_{m}_{b}" for m in OM for b in ("lo", "hi")]
    with open(os.path.join(OUT, "reports", "MASTER_TABLE.csv"), "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(hdr)
        w.writerows(master)
    with open(os.path.join(OUT, "reports", "PER_SCENE.csv"), "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["id", "scene", "split"] + list(OM) + ["d_" + m for m in OM])
        w.writerows(perscene)
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
        col = {"V3-T1 + D1 (anchor)": "#222222", "MAGICIAN Beam": "#999999", "xa_s2d": "#56B4E9", "xa_sr": "#0072B2", "xa_si": "#009E73", "xa_m2d": "#E69F00", "xa_mr": "#D55E00", "xa_mi": "#CC79A7"}
        fig, axs = plt.subplots(1, 4, figsize=(17, 4.2))
        for ax, gn in zip(axs, ("All-14", "Unseen-8", "Test-3", "Train-6")):
            for nm, c in col.items():
                y = (curves.get(nm) or {}).get(gn)
                if y is not None:
                    ax.plot(np.arange(41), y[:41], color=c, lw=2.0 if nm.startswith("V3") else 1.4, ls="--" if nm.startswith("MAG") else "-", label=nm)
            ax.set_title(f"{gn}: official coverage, first 40 steps", fontsize=9, loc="left")
            ax.set_xlabel("step", fontsize=8)
            ax.grid(alpha=0.3)
            for s_ in ("top", "right"):
                ax.spines[s_].set_visible(False)
        axs[0].legend(fontsize=7, frameon=False)
        fig.tight_layout()
        fig.savefig(os.path.join(OUT, "reports", "EARLY_CURVES.png"), dpi=140)
        plt.close(fig)
    except Exception as e:                                    # noqa: BLE001  (figures never block the tables)
        print("figure skipped:", repr(e))
    print("aggregated", [(r[0], r[2], r[3], r[5]) for r in master if r[3] == "All-14" or (r[0] in IDS and r[3] in ("Dev-5",))])
    print(hashlib.sha256(open(os.path.join(OUT, "reports", "MASTER_TABLE.csv"), "rb").read()).hexdigest()[:16], "MASTER_TABLE.csv")


if __name__ == "__main__":
    main()
