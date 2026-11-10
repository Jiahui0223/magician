"""CVPR overnight -- episode manifest + record validation (CPU). python build_manifest.py
Writes experiments/cvpr_results/manifest/{episodes.csv, protocol.json, provenance.json} from the canonical records.
One row per expected (method, scene, start) cell of every method in cvpr_lib.METHODS:
  COMPLETE    valid record (101 finite values in [0, 1], 101 executed poses, episode file == canonical record, official replay == stored curve)
  INCOMPLETE  no record yet (not run / still running); no curve columns
  FAILED      the queue gave up after 3 tries, or a record exists but failed validation (problems listed)
  BLOCKED     the method has no runnable implementation (reason in provenance.json)
Invalid / missing cells never enter an aggregate; nothing is imputed."""
import glob
import json
import os
import subprocess
import sys
import time

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import cvpr_lib as L  # noqa: E402


def planner_facts(method, ep):
    """K / depth / noise offset actually used, read from the episode's planner handshake when it exists."""
    d = L.METHODS[method]
    K, depth, noise = d["K"], d["depth"], d.get("noise", 0 if d["K"] != "" else "")
    seen = {}
    try:
        h = ep["planner_hello"][0] if isinstance(ep.get("planner_hello"), list) else (ep.get("planner_hello") or {})
        pl = h.get("planner") or {}
        if "K" in pl and pl["K"] is not None:
            seen["K"] = pl["K"]
        if "kd" in pl:
            seen["depth"], seen["noise"] = pl["kd"]["depth"], pl["kd"].get("noise_offset", 0)
        elif "d_policy" in pl:
            seen["depth"] = pl["d_policy"]["policy_cfg"]["depth"]
        seen["ckpt"] = h.get("ckpt_sha256")
    except Exception:
        pass
    return K, depth, noise, seen


def main():
    gave_up = set()
    gp = os.path.join(L.T1W, "logs", "gave_up.log")
    if os.path.exists(gp):
        for line in open(gp):
            if line.startswith("GAVE_UP"):
                gave_up.add(line.split()[1])
    tries = {}
    for d in glob.glob(os.path.join(L.T1W, "failed", "*_try*")):
        b = os.path.basename(d)
        tries[b.rsplit("_try", 1)[0]] = tries.get(b.rsplit("_try", 1)[0], 0) + 1
    header = ["method_id", "method_display", "scene_id", "start_id", "split", "protocol_id", "checkpoint_sha", "seed", "K", "planning_depth", "score_kind",
              "noise_key", "run_status", "problems", "n_failed_tries", "official_replay_equal", "official_replay_hash", "curve_hash", "trajectory_path",
              "source_path", "metrics_path", "record_source", "runtime_s", "host"] + [f"c{t}" for t in range(L.MOVES + 1)]
    rows, summary = [], {}
    for m, d in L.METHODS.items():
        E = L.load(m)
        st = {"COMPLETE": 0, "INCOMPLETE": 0, "FAILED": 0, "BLOCKED": 0, "problems": []}
        extra = [k for k in E if k[0] not in L.SPLIT or not 0 <= k[1] < L.N_START]
        if extra:
            st["problems"].append(f"records outside the frozen grid: {extra}")
        for s in L.SCENE_ORDER:
            for k in range(L.N_START):
                g = "random" if m == "RANDOM" else "macarons" if m == "MACARONS" else (m if m in ("MAG-B", "MAG-G") else "V2M")
                nt = tries.get(f"{m}_{g}_{s}_s{k}_r0", 0)
                base = [m, d["display"], s, k, L.SPLIT[s], L.PROTOCOL_ID]
                if d.get("blocked"):
                    st["BLOCKED"] += 1
                    rows.append(base + ["", "", "", "", "", "", "BLOCKED", d["blocked"], 0, "", "", "", "", "", "", "", "", ""] + [""] * (L.MOVES + 1))
                    continue
                r = E.get((s, k))
                if r is None:
                    status = "FAILED" if f"{m}:{s}:{k}" in gave_up else "INCOMPLETE"
                    st[status] += 1
                    rows.append(base + [d["ckpt"], 0, d["K"], d["depth"], d["score"], "", status, "queue gave up after 3 tries" if status == "FAILED" else "no record", nt,
                                        "", "", "", "", "", "", "", "", ""] + [""] * (L.MOVES + 1))
                    continue
                ok, bad, info = L.check_record(r)
                ep = info.get("ep") or {}
                K, depth, noise, seen = planner_facts(m, ep)
                for key, want in (("K", K), ("depth", depth), ("noise", noise)):
                    if key in seen and want != "" and str(seen[key]) != str(want) and not (key == "depth" and isinstance(want, str)):
                        bad.append(f"planner {key} {seen[key]} != expected {want}")
                if seen.get("ckpt") and d["ckpt"] and seen["ckpt"] != d["ckpt"]:
                    bad.append("planner checkpoint sha != expected")
                if r.get("ckpt_sha256") and d["ckpt"] and r["ckpt_sha256"] != d["ckpt"]:
                    bad.append("record checkpoint sha != expected")
                status = "COMPLETE" if not bad else "FAILED"
                st[status] += 1
                if bad:
                    st["problems"].append(f"{s}_s{k}: {bad}")
                nk = (L.NOISE_KEY.replace("{k}", "{k}" if K == 4 else str(noise)) + (f" ; K = {K}" + ("" if K == 4 else f", single map k = {noise}"))) if K != "" else ""
                rows.append(base + [r.get("ckpt_sha256") or d["ckpt"], 0, K, depth, d["score"], nk, status, "; ".join(bad), nt, info["replay_equal"], info["replay_hash"],
                                    L.sha_curve(r["coverage_official"]), r["episode_file"], r["_path"], r["metrics_file"], r.get("source"),
                                    r.get("runtime_s") or ep.get("seconds_this_process") or ep.get("seconds_total"), r.get("host") or ep.get("host")]
                            + [repr(float(x)) for x in r["coverage_official"]])
        summary[m] = st
    L.write_csv(os.path.join(L.MAN, "episodes.csv"), header, rows)
    head = subprocess.run(["git", "rev-parse", "HEAD"], capture_output=True, text=True, cwd=L.REPO).stdout.strip()
    proto = {"protocol_id": L.PROTOCOL_ID, "protocol": L.PROTOCOL_DOC, "protocol_documents": ["experiments/cvpr_results/table1/PROTOCOL.md", "experiments/cvpr_results/table1/FROZEN_CONFIG.yaml",
                                                                                              "experiments/cvpr_results/protocol_audit/"],
             "main_method": {"id": "V3-T0-D1", "checkpoint": "work_dirs/exp6_v3/task0_vae/checkpoints/final.pt", "sha256": L.T0_SHA, "xattn_enabled": False, "K": 4, "euler_steps": 10,
                             "planner": "D1 = frozen D-v1 policy d1 (depth 2, exhaustive, lambda_risk 0)", "encoder_input": "D100 map", "novelty_reference": "S10 observed points",
                             "latent_refresh": "R10 (steps 0, 10, ..., 90)"},
             "metrics": {"curve": "C[0..100] official MAGICIAN coverage (may decrease; never cummax-ed or smoothed)", "AUC@T": "trapezoid(C[0..T], dx = 1) / T (numpy.trapz; numpy.trapezoid is the same function in numpy >= 2)",
                         "Cov@T": "C[T]", "Final": "C[100]", "aggregation": "mean over the 5 starts of a scene, then unweighted mean over scenes",
                         "uncertainty": "paired scene-level bootstrap, 10,000 draws, seed 0, percentile 95 % interval (resampling unit = scene; 14 scenes, small sample)"},
             "paper_columns_preregistered": ["AUC@20", "AUC@50", "AUC@100", "Cov@20", "Final"], "paper_columns_source": "cvpr_overnight_experiment_plan.md section 3 (fixed before the new runs of 2026-11-10)",
             "T_full": list(L.T_FULL), "scene_order": L.SCENE_ORDER, "splits": {g: v for g, v in L.GROUPS.items()}, "starts": "settings.json start_positions[0..4] (official)",
             "noise_key_template": L.NOISE_KEY, "table3": {"config": "experiments/cvpr_results/scripts/table3/kd_config.yaml (KD-v1)", "runner": "experiments/cvpr_results/scripts/table3/{sim_out_kd,kd_out_planner,kd_policy}.py",
                                                            "hashes": {f: L.sha_file(os.path.join(L.CV, "scripts", "table3", f)) for f in ("kd_config.yaml", "kd_policy.py", "kd_out_planner.py", "sim_out_kd.py")},
                                                            "frozen_d_config_sha256": L.sha_file(os.path.join(L.REPO, "experiments/exp6_v3/ablation_study/experiment_D_h3_lookahead/configs/d_config.yaml")),
                                                            "frozen_d_policy_sha256": L.sha_file(os.path.join(L.REPO, "experiments/exp6_v3/ablation_study/experiment_D_h3_lookahead/scripts/d_policy.py")),
                                                            "frozen_h_policy_sha256": L.sha_file(os.path.join(L.REPO, "experiments/exp6_v3/H/scripts/h_policy.py"))},
             "git_head": head, "generated": time.strftime("%F %T")}
    json.dump(proto, open(os.path.join(L.MAN, "protocol.json"), "w"), indent=1, ensure_ascii=False)
    prov = {}
    for m, d in L.METHODS.items():
        st = summary[m]
        prov[m] = {"display": d["display"], "kind": d["kind"], "source": d["source"], "status": "BLOCKED" if d.get("blocked") else ("COMPLETE" if st["COMPLETE"] == 70 else "PARTIAL"),
                   "n_complete": st["COMPLETE"], "n_expected": 70, "n_incomplete": st["INCOMPLETE"], "n_failed": st["FAILED"], "problems": st["problems"], "blocked_reason": d.get("blocked"),
                   "checkpoint_sha256": d["ckpt"] or None, "planner": d["score"], "K": d["K"], "planning_depth": d["depth"]}
    json.dump({"generated": time.strftime("%F %T"), "git_head": head, "methods": prov}, open(os.path.join(L.MAN, "provenance.json"), "w"), indent=1, ensure_ascii=False)
    print(f"{'method':22s} complete incomplete failed blocked")
    for m, st in summary.items():
        print(f"{m:22s} {st['COMPLETE']:8d} {st['INCOMPLETE']:10d} {st['FAILED']:6d} {st['BLOCKED']:7d}  {'; '.join(st['problems'])[:200]}")
    return summary


if __name__ == "__main__":
    main()
