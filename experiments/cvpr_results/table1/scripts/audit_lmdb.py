"""Table-1 dataset audit: read-only summary of MAGICIAN's own results LMDB (magician_lmdb).

Run with the magician env (the seen2scene env has no lmdb):
  env -u LD_PRELOAD -u PYTHONPATH /bsuscratch/jiahuizhang/envs/magician/bin/python -I audit_lmdb.py OUT.json
Per key "<scene>/<k>": shapes of X_cam_history / V_cam_history, coverage length, X[0], V[0], final coverage
(coverage[100] and coverage[-1]) and two AUC conventions (exp1: trapz(cov[0..100]) / 100;
evaluation_lmdb.py: unnormalised trapz over cov[:100]).
"""
import json
import pickle
import sys

import lmdb
import numpy as np

LMDB = "/bsuscratch/jiahuizhang/projects/occupancy/MAGICIAN/results/scene_exploration/magician_lmdb"


def main():
    out = sys.argv[1]
    env = lmdb.open(LMDB, readonly=True, lock=False)
    rows = {}
    with env.begin() as txn:
        keys = [k.decode() for k, _ in txn.cursor()]
        for key in keys:
            d = pickle.loads(txn.get(key.encode()))
            X = np.asarray(d["X_cam_history"], dtype=np.float64)
            V = np.asarray(d["V_cam_history"], dtype=np.float64)
            cov = np.asarray([float(c) for c in d["coverage"]], dtype=np.float64)
            rows[key] = {"fields": sorted(d.keys()), "X_shape": list(X.shape), "V_shape": list(V.shape),
                         "n_coverage": int(len(cov)), "X0": X[0].tolist(), "V0": V[0].tolist(),
                         "cov0": float(cov[0]), "cov_final_last": float(cov[-1]),
                         "cov_100": float(cov[100]) if len(cov) > 100 else None,
                         "auc_exp1": float(np.trapz(cov[:101], dx=1.0) / 100.0) if len(cov) >= 101 else None,
                         "auc_evallmdb_unnorm": float(np.trapz(cov[:100], dx=1.0))}
            del d
    env.close()
    scenes = sorted({k.split("/")[0] for k in rows})
    per_scene = {s: sorted(int(k.split("/")[1]) for k in rows if k.split("/")[0] == s) for s in scenes}
    res = {"lmdb": LMDB, "n_keys": len(rows), "scenes": scenes, "starts_per_scene": per_scene, "rows": rows}
    json.dump(res, open(out, "w"), indent=1)
    print("keys", len(rows), "scenes", len(scenes))
    for s in scenes:
        ks = [f"{s}/{k}" for k in per_scene[s]]
        print(s, per_scene[s], [rows[k]["X_shape"] for k in ks][0], [rows[k]["n_coverage"] for k in ks],
              "final(mean)=%.4f auc(mean)=%.4f" % (np.mean([rows[k]["cov_final_last"] for k in ks]),
                                                  np.mean([rows[k]["auc_exp1"] or np.nan for k in ks])))


if __name__ == "__main__":
    main()
