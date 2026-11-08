"""Experiment XB / Task B3 -- error stratification per camera (seen2scene env, 1 GPU). Same evaluator as Task A (xb_fixed -> ct_fixed) with the per-camera metric
function extended: on labelled UNKNOWN rays the HIT / NOHIT confusion (summed over the K = 4 samples) and the first-hit absolute depth error on true positives are
accumulated in strata of
  depth     truth z (runtime units) bins [0,2) [2,5) [5,10) [10,25) [25,45) [45,70)
  nseg      UNKNOWN segment count 1 / 2 / >= 3
  boundary  relative distance of the first UNKNOWN sample d_lo / tcap bins [0, .1), [.1, .3), [.3, 1]
  lu        UNKNOWN length share L_u bins [0, .1), [.1, .3), [.3, 1]
Rows go to states/<group>/<tag>.jsonl with cams[...]["strata"]. Usage: python xb_b3.py --set ... --group G --ckpts T1@none,XA_SR@none,XA_MI@none --only TAG"""
import os
import sys

R = "/bsuscratch/jiahuizhang/projects/occupancy/seen2scene"
sys.path.insert(0, os.path.join(R, "experiments/exp6_v3/ablation_study/experiment_XB_bottleneck_audit/scripts"))
import xb_fixed as XBF  # noqa: E402

CT, torch, G = XBF.CT, XBF.torch, XBF.CT.G
_cm = CT.camera_metrics
ZB = (0.0, 2.0, 5.0, 10.0, 25.0, 45.0, 70.0)          # runtime units; indoor states fall in the first bins, outdoor in the last
NB = ((1, 1), (2, 2), (3, 10 ** 6))
RB = (0.0, 0.1, 0.3, 1.0001)


def _bins(x, edges):
    return [(x >= edges[i]) & (x < edges[i + 1]) for i in range(len(edges) - 1)]


def camera_metrics(tk, hu, ts, v, geo, kM, new_pred, new_true, eps_cov):
    o = _cm(tk, hu, ts, v, geo, kM, new_pred, new_true, eps_cov)
    pr = v["pr"]
    dz = v["dz"].to(tk.dtype)
    lab = G.encode_labels(v["o"], v["d"], pr, ts, dz, kM, geo.struct, geo.domain)
    cat, b = lab["cat"], lab["b"]
    unk = pr["n_u"] > 0
    lm = unk & (b != 0)
    lh = lm & (cat == G.CAT["HIT"])
    ln = lm & ~lh
    p = hu
    zt = (ts.double() * kM * dz.double())
    zh = (tk.double() * kM * dz.double())
    e = (zh - zt[None]).abs()
    tcap = v["tcap"].to(tk.dtype)
    dlo = torch.where(torch.isfinite(pr["d_lo"]), (pr["d_lo"] / tcap).clamp(0, 1), torch.ones_like(tcap))
    lu = (pr["n_u"].to(tk.dtype) * G.H_STEP / tcap).clamp(0, 1)
    groups = {"depth": _bins(torch.where(torch.isfinite(zt), zt, torch.full_like(zt, 1e9)), ZB), "nseg": [(pr["n_seg"] >= a) & (pr["n_seg"] <= bb) for a, bb in NB],
              "boundary": _bins(dlo, RB), "lu": _bins(lu, RB)}
    st = {}
    for gname, bins in groups.items():
        rows = []
        for m in bins:
            mm = m & lm
            tp = (p & (lh & m)[None]).sum(); fp = (p & (ln & m)[None]).sum(); fn = (~p & (lh & m)[None]).sum(); tn = (~p & (ln & m)[None]).sum()
            tpm = p & (lh & m)[None]
            rows.append({"n_lm": int(mm.sum()), "n_lh": int((lh & m).sum()), "tp": int(tp), "fp": int(fp), "fn": int(fn), "tn": int(tn),
                         "sae_tp": float(e[tpm].sum()) if tpm.any() else 0.0, "n_tp": int(tpm.sum())})
        st[gname] = rows
    o["strata"] = st
    return o


CT.camera_metrics = camera_metrics

if __name__ == "__main__":
    CT.main()
