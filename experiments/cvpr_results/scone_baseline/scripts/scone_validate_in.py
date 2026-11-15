"""EXP9-SCONE Phase D -- validation of one MP3D-150 SCONE episode (CPU; evaluation side). Reuses the EXP7 validator
collect_mp3d150.validate() unchanged (V1 151 finite non-decreasing values / 150 moves, V2 C_obs replay from the stored 10 % samples
|d| <= 1e-12, V3 every executed move in the simulator legal list) and adds:
  V4 same start pose as the MAG-B episode of the same level / start (EXP7 manifest);
  V6 provenance: planner_info.json names the official SCONE weights with the expected sha256 and trained_macarons_loaded = False;
  V7 planner guard log empty in every step, no fallback pick.
Writes <ep-dir>/validation.json {"valid": all checks, ...}.
Usage: python scone_validate_in.py --ep-dir D --scene-id LEVEL --start-id K --moves 150"""
import argparse
import csv
import json
import os
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
REPO = os.path.dirname(os.path.dirname(os.path.dirname(os.path.dirname(HERE))))
sys.path.insert(0, os.path.join(REPO, "experiments", "cvpr_results", "exp7", "scripts"))
sys.path.insert(0, HERE)
import collect_mp3d150 as CM  # noqa: E402

SHA = {"pretrained_macarons.pth": "c7d66c9f46ac07e676a0a956390460c067aa5ed9bc5ebbaa298300a85d9e54ad",
       "pretrained_scone_occ.pth": "257674e17f7d693bfdd555eed50e4a2ce9e77d5f3c31e45183d68d0faf404bd9",
       "pretrained_scone_vis.pth": "72c9bfdccde7e50578a7c830f3c09e7653074e9df3f23b8241546210b1a315f0"}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--ep-dir", required=True)
    ap.add_argument("--scene-id", required=True)
    ap.add_argument("--start-id", required=True)
    ap.add_argument("--moves", type=int, default=150)
    a = ap.parse_args()
    assert a.moves == 150, "collect_mp3d150.validate checks the 150-move protocol"
    d = os.path.abspath(a.ep_dir)
    row = {"output_path": os.path.relpath(os.path.join(d, "episode.json"), CM.R), "scene_id": a.scene_id,
           "start_id": a.start_id, "method": "SCONE"}
    out = CM.validate(row)
    ex0 = out["start"]
    mag = [r for r in csv.DictReader(open(os.path.join(CM.E7, "manifests", "mp3d150_manifest.csv")))
           if r["scene_id"] == a.scene_id and r["start_id"] == str(a.start_id) and r["method"] == "MAG-B"]
    e_mag = json.load(open(os.path.join(CM.R, mag[0]["output_path"])))
    out["checks"]["V4_same_start_as_MAGB"] = [int(x) for x in e_mag["executed"][0]] == ex0
    info = json.load(open(os.path.join(d, "planner_info.json")))
    m = info.get("model", {})
    out["checks"]["V6_provenance"] = m.get("sha256") == SHA and m.get("trained_macarons_loaded") is False and info.get("method") == "SCONE"
    ans = [json.loads(x) for x in open(os.path.join(d, "planner.jsonl"))]
    out["checks"]["V7_guard_no_fallback"] = len(ans) == a.moves and all(not x["guard_denied"] for x in ans) and not any(x["fallback"] for x in ans)
    out["valid"] = all(out["checks"].values())
    out["tag"] = f"{a.scene_id}_S{a.start_id}"
    res = {k: v for k, v in out.items() if k != "row"}
    json.dump(res, open(os.path.join(d, "validation.json"), "w"), indent=1)
    print(json.dumps({"tag": out["tag"], "valid": out["valid"], "checks": out["checks"], "replay_max_abs_diff": out["replay_max_abs_diff"],
                      "metrics": out["metrics"]}))
    sys.exit(0 if out["valid"] else 3)


if __name__ == "__main__":
    main()
