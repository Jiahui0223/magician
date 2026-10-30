"""EXP5 Task 1.1 D-REPORT: data manifest (sha256 of every shard), aggregated checks (T-L0/T-L3/T-L4/T-L5/T-L6, T-R1,
march range), class / label-category statistics per scene and split, and example visualisations of projection classes
and labels. Output: W/data/data_manifest.json, W/data/summary.json, W/data/figures/*.png
"""
import glob
import json
import os
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
import v2_frozen as VF  # noqa: E402

import numpy as np  # noqa: E402

from seen2scene.futureview import geometry as G  # noqa: E402

D = os.path.join(VF.WORK, "data")


def main():
    sp = VF.frozen_values()["split"]["scenes"]
    man = {"files": {}, "scenes": {}}
    summ = {"per_scene": {}, "checks": {}}
    all_ok_l3, all_tr1, all_march = True, [], []
    for s in sp:
        rows = []
        for p in sorted(glob.glob(os.path.join(D, "views", s, "*.json"))):
            r = json.load(open(p))
            rows.append(r)
            man["files"][f"views/{s}/{r['traj']}.npz"] = r["sha256"]
            all_ok_l3 &= all(r["l3_bitwise"].values())
            if "TR1" in r["checks"]:
                all_tr1.append(r["checks"]["TR1"])
            all_march.append(r["checks"].get("march_equal"))
        for p in sorted(glob.glob(os.path.join(D, "tstar", s, "*.npz"))):
            man["files"][f"tstar/{s}/{os.path.basename(p)}"] = VF.sha256_file(p)
        assert len(rows) == 30, (s, len(rows))
        agg = {}
        for split, sel in (("train", [r for r in rows if not r["heldout"]]), ("heldout", [r for r in rows if r["heldout"]])):
            views = sum(r["views"] for r in sel)
            rays = sum(r["rays"] for r in sel)
            cls = {c: sum(r["cls_frac"][c] * r["rays"] for r in sel) / rays for c in ("unknown", "known", "empty")}
            cats = {k: sum(r["m_unknown_cat"][k] for r in sel) for k in G.CAT}
            n_mu = sum(cats.values())
            agg[split] = {"trajs": len(sel), "views": views, "rays": rays, "cls_frac": cls,
                          "m_unknown_rays": n_mu, "m_unknown_cat_frac": {k: v / n_mu for k, v in cats.items()},
                          "infeasible_frac": sum(cats[k] for k in ("INF_EMPTY", "INF_BAND", "INF_OOD", "INF_OTHER")) / n_mu,
                          "invalid_frac": sum(cats[k] for k in ("BLOCKED", "NEAR", "OUT", "INF_EMPTY", "INF_BAND", "INF_OOD", "INF_OTHER")) / n_mu,
                          "hit_frac_of_targets": cats["HIT"] / max(1, cats["HIT"] + cats["NOHIT"] + cats["KNOWN"]),
                          "build_s": sum(r["seconds"] for r in sel)}
        man["scenes"][s] = {"trajs": len(rows), "views": sum(r["views"] for r in rows),
                            "coarse_shape": rows[0]["coarse_shape"]}
        summ["per_scene"][s] = agg
    checks = {"TL5_l3_bitwise_all": all_ok_l3, "TR1_all": all(all_tr1) and len(all_tr1) > 0, "TR1_n": len(all_tr1),
              "march_equal_all": all(x is True for x in all_march)}
    for s in sp:
        c = json.load(open(os.path.join(D, "checks", f"TL34_{s}.json")))["summary"]
        checks[f"TL3_TL4_{s}"] = c
    checks["TL0_TL6_pisa"] = json.load(open(os.path.join(D, "checks", "labels_test_pisa.json")))
    checks["TL3_pass_all"] = all(checks[f"TL3_TL4_{s}"]["TL3_pass"] for s in sp)
    checks["TL4_pass_all"] = all(checks[f"TL3_TL4_{s}"]["TL4_pass"] for s in sp)
    summ["checks"] = checks
    tot_views = sum(v["views"] for v in man["scenes"].values())
    summ["totals"] = {"views": tot_views, "rays": tot_views * 7296,
                      "disk_gb_views": sum(os.path.getsize(p) for p in glob.glob(os.path.join(D, "views", "*", "*.npz"))) / 1e9,
                      "disk_gb_tstar": sum(os.path.getsize(p) for p in glob.glob(os.path.join(D, "tstar", "*", "*.npz"))) / 1e9}
    man["split"] = "frozen_values.json:split"
    json.dump(man, open(os.path.join(D, "data_manifest.json"), "w"), indent=1)
    summ["data_manifest_sha256"] = VF.sha256_file(os.path.join(D, "data_manifest.json"))
    json.dump(summ, open(os.path.join(D, "summary.json"), "w"), indent=1)
    figures()
    print(json.dumps(summ["totals"]), json.dumps({k: v for k, v in checks.items() if not isinstance(v, dict)}))


def figures():
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    out = os.path.join(D, "figures")
    os.makedirs(out, exist_ok=True)
    sp = VF.frozen_values()["split"]["scenes"]
    rng = np.random.default_rng(0)
    for s in sp:
        tj = sp[s]["heldout_trajs"][0]
        z = np.load(os.path.join(D, "views", s, f"{tj}.npz"))
        fig, ax = plt.subplots(3, 4, figsize=(16, 7))
        for col, t in enumerate((0, 5, 20, 60)):
            r = int(np.flatnonzero(z["t"] == t)[rng.integers((z["t"] == t).sum())])
            cls = z["cls"][r].reshape(64, 114)
            cat = z["cat"][r].reshape(64, 114).astype(float)
            y = np.where(z["b"][r] == 1, (z["y"][r].astype(float) + 1) / 2, np.nan).reshape(64, 114)
            ax[0, col].imshow(cls, cmap="viridis", vmin=0, vmax=2)
            ax[0, col].set_title(f"{s}/{tj} t={t} a={int(z['a'][r])}: class (0 unk, 1 known, 2 empty)", fontsize=7)
            ax[1, col].imshow(cat, cmap="tab10", vmin=0, vmax=9)
            ax[1, col].set_title("label category (1 NOHIT, 2 KNOWN, 3 HIT, 6-9 infeasible)", fontsize=7)
            ax[2, col].imshow(y, cmap="magma", vmin=0, vmax=1)
            ax[2, col].set_title("HIT: s along unknown arclength", fontsize=7)
            for a in ax[:, col]:
                a.axis("off")
        fig.tight_layout()
        fig.savefig(os.path.join(out, f"views_{s}.png"), dpi=110)
        plt.close(fig)


if __name__ == "__main__":
    main()
