"""CVPR Table 1 -- canonical per-episode records (CPU). Writes work_dirs/cvpr_results/table1/episodes/<model>/<scene>_s<k>.json:
  model, scene, start, split, coverage_official (episode file curve = MAGICIAN official, 101 values), coverage_mono (metrics
  file), executed_keys, ckpt_sha256 (registry), source (historical / table1_run / table1_replay), paths, runtime, checks.
Modes: --historic   import the 390 audited historical episodes (audit/historical_outdoor_episodes.json; MAIN + optional models)
       --new M S K  import one new Table-1 run from work_dirs/cvpr_results/table1/g3
Checks (raise): 101 executed keys; 101-value official curve; metrics cov_official == episode curve; metrics cov_mono 101 values.
Never overwrites an existing record with different curves (exit 2)."""
import argparse
import json
import os
import time

import yaml

HERE = os.path.dirname(os.path.abspath(__file__))
T1 = os.path.dirname(HERE)
REPO = os.path.dirname(os.path.dirname(os.path.dirname(T1)))
EPI = os.path.join(REPO, "work_dirs", "cvpr_results", "table1", "episodes")
T1W = os.path.join(REPO, "work_dirs", "cvpr_results", "table1", "g3")
GROUP = {"MAG-B": "MAG-B", "MAG-G": "MAG-G", "V2O": "V2O"}


def registries():
    sr = yaml.safe_load(open(os.path.join(T1, "SCENE_REGISTRY.yaml")))
    split = {s["local_id"]: s["split"] for s in sr["scenes"]}
    mr = yaml.safe_load(open(os.path.join(T1, "MODEL_REGISTRY.yaml")))
    sha = {m["id"]: m.get("sha256") for m in mr["models"]}
    return split, sha


def write(rec):
    p = os.path.join(EPI, rec["model"], f"{rec['scene']}_s{rec['start']}.json")
    if os.path.exists(p):
        old = json.load(open(p))
        if old["coverage_official"] != rec["coverage_official"]:
            raise SystemExit(f"STOP: existing record differs {p}")
        return False
    os.makedirs(os.path.dirname(p), exist_ok=True)
    json.dump(rec, open(p + ".tmp", "w"))
    os.replace(p + ".tmp", p)
    return True


def build(model, scene, start, ep_path, mt_path, source, split, sha, extra=None):
    ep = json.load(open(os.path.join(REPO, ep_path)))
    mt = json.load(open(os.path.join(REPO, mt_path)))
    assert len(ep["executed_keys"]) == 101 and len(ep["coverage"]) == 101 and ep["moves"] == 100, ep_path
    assert mt["cov_official"] == ep["coverage"], mt_path
    assert len(mt["cov_mono"]) == 101
    rec = {"model": model, "scene": scene, "start": int(start), "split": split[scene], "coverage_official": ep["coverage"],
           "coverage_mono": mt["cov_mono"], "executed_keys": ep["executed_keys"], "ckpt_sha256": sha.get(model),
           "source": source, "episode_file": ep_path, "metrics_file": mt_path, "n_hover": ep.get("n_hover", 0),
           "eval_protocol": "M-protocol frozen simulator (exp3b m_t6_sim) + t25_out_cov official/mono", "imported": time.strftime("%F %T")}
    rec.update(extra or {})
    return rec


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--historic", action="store_true")
    ap.add_argument("--new", nargs=3, metavar=("MODEL", "SCENE", "START"))
    a = ap.parse_args()
    split, sha = registries()
    if a.historic:
        d = json.load(open(os.path.join(T1, "audit", "historical_outdoor_episodes.json")))
        n = 0
        for e in d["episodes"]:
            assert e["ep_complete"] and e["official_equal_stored"] and e["mt_official_eq_episode"], e["episode"]
            n += write(build(e["model"], e["scene"], e["start"], e["episode"], e["metrics"], "historical", split, sha,
                             {"runtime_s": e.get("seconds_this_process"), "host": e.get("host")}))
        print("historic records written", n, "of", len(d["episodes"]))
    if a.new:
        m, s, k = a.new
        g = GROUP.get(m, "V2M")
        ep = os.path.relpath(os.path.join(T1W, "closed_loop", "outdoor", m, g, "episodes", f"ep_{g}_{s}_s{k}_r0.json"), REPO)
        mt = os.path.relpath(os.path.join(T1W, "metrics", "outdoor", m, g, f"{g}_{s}_s{k}_r0.json"), REPO)
        print("written" if write(build(m, s, k, ep, mt, "table1_run", split, sha)) else "exists", m, s, k)


if __name__ == "__main__":
    main()
