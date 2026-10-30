"""EXP5 Ablation A2 -- additional descriptive metric (DEVIATIONS #10; CPU): phantom occluders.
On labelled M_unknown rays whose GT first hit is the KNOWN surface behind the unknown segment (label category KNOWN), a
predicted UNKNOWN hit is a phantom surface in front of an observed surface. The protocol's false-surface rate covers
only label NOHIT rays; this rate covers label KNOWN rays. Per candidate of the candidate set, per sample; mean over
candidates -> state -> episode -> scene. Output results/a2_phantom.json
"""
import json
import os
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.join(os.path.dirname(HERE), "common"))
import ab_common as A  # noqa: E402
import ab_load as AL  # noqa: E402

import numpy as np  # noqa: E402

CAT_NOHIT, CAT_KNOWN, CAT_HIT = 1, 2, 3


def main():
    out = {}
    for d in ("indoor", "outdoor"):
        rows = []
        for st in A.load_manifest(d)["states"]:
            z = np.load(os.path.join(A.state_dir(d, st["sid"]), "v2.npz"))
            r = {"sid": st["sid"], "scene": st["scene"], "start": st["start"], "src": st["sources"][0]["source"]}
            for m in ("V2O", "V2M"):
                ph, fs, share = [], [], []
                for j in st["cand"]:
                    lm = z["lm"][j].astype(bool)
                    cat = z["lab_cat"][j]
                    hu = z[f"{m}_hit_u"][j].astype(bool)                     # [K, R]
                    kn, nh = lm & (cat == CAT_KNOWN), lm & (cat == CAT_NOHIT)
                    if kn.any():
                        ph.append(float(hu[:, kn].mean()))
                    if nh.any():
                        fs.append(float(hu[:, nh].mean()))
                    wrong = hu[:, kn].sum() + hu[:, nh].sum()
                    allh = hu[:, lm].sum()
                    if allh:
                        share.append(float(hu[:, kn].sum() / allh))
                r[f"{m}_phantom_known"] = float(np.mean(ph)) if ph else None
                r[f"{m}_false_nohit"] = float(np.mean(fs)) if fs else None
                r[f"{m}_share_hits_on_known_rays"] = float(np.mean(share)) if share else None
            r["label_known_share"] = float(np.mean([(z["lm"][j] & (z["lab_cat"][j] == CAT_KNOWN)).sum() / max(1, z["lm"][j].sum())
                                                    for j in st["cand"]]))
            rows.append(r)
        S = {k: AL.boot(AL.hier_mean(rows, k)) for k in rows[0] if k not in ("sid", "scene", "start", "src")}
        ma, mb = AL.hier_mean(rows, "V2M_phantom_known"), AL.hier_mean(rows, "V2O_phantom_known")
        S["paired_phantom_known_V2M_minus_V2O"] = AL.boot({s: ma[s] - mb[s] for s in ma})
        out[d] = S
        print(d, {k: round(v["mean"], 3) for k, v in S.items() if v["mean"] is not None})
    json.dump(out, open(os.path.join(HERE, "results", "a2_phantom.json"), "w"), indent=1)


if __name__ == "__main__":
    main()
