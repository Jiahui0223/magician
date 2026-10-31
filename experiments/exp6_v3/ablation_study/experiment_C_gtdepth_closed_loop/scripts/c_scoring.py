"""EXP6 V3 Ablation Experiment C -- planner-side scoring helpers (seen2scene env, torch).

gtd_scores : GT-depth DistP1 of every legal candidate on ONE geometry (the frozen sparse-P1 SceneCaster of the current
             state). Line by line the Experiment A rule (ablation_study/shared/ab_score.py, evaluation side), which in
             turn is the EXP5 S2 / T1.3 truedepth_distp1 rule: t* (float32, model units, inf = no hit) capped by the
             ray's tmax (region exit / sensor range) -> scoring.new_set_from_depth on the sparse band_dil with the
             distance weight min(1, (z / z0)^2) -> sum of the unique new-voxel weights. K = 1 (the true observation of a
             deterministic sensor is unique).
p1_hash    : sha256 of the P1 fusion state (fixed-point BAND sums / counts / EMPTY) for the no-leakage check.
"""
import hashlib
import json

import numpy as np
import torch

from seen2scene.futureview import scoring as SCO


@torch.no_grad()
def gtd_scores(geo, legal_R, legal_T, tstar, z0, return_views=False):
    k_M = float(geo.fr.k_M)
    scores, extra = [], []
    for j, (R, T) in enumerate(zip(legal_R, legal_T)):
        v = geo.view(R, T)
        ts = torch.as_tensor(np.asarray(tstar[j]), device=v["d"].device, dtype=v["d"].dtype)
        tt = torch.where(ts <= v["pr"]["tmax"], ts, torch.full_like(ts, float("inf")))
        ids, w = SCO.new_set_from_depth(tt, v["o"], v["d"], v["dz"], k_M, geo.dims, geo.domain, geo.caster.band_dil, z0)
        scores.append(float(w.sum()))
        if return_views:
            new, _, hit = SCO.new_mask_from_depth(tt, v["o"], v["d"], geo.dims, geo.domain, geo.caster.band_dil)
            extra.append({"n_new_vox": int(len(ids)), "n_new_rays": int(new.sum()), "n_hit_rays": int(hit.sum()),
                          "n_unknown_rays": int((v["pr"]["cls"] == 0).sum()), "n_valid": int(v["pr"]["valid"].sum())})
    return (scores, extra) if return_views else scores


def p1_hash(fu):
    """Cheap GPU-side checksum of the P1 fusion state (totals + strided sub-sums of the fixed-point BAND sums, counts and
    EMPTY flags, frame count): the GT scoring never writes to the fusion, so any change shows up here."""
    with torch.no_grad():
        v = [fu.sum.sum(), fu.sum[::9973].sum(), fu.cnt.sum(), fu.cnt[::7919].sum(), fu.empty.sum(), fu.empty[::6007].sum()]
        v = [int(x.item()) for x in v] + [int(fu.n_frames)]
    return hashlib.sha256(json.dumps(v).encode()).hexdigest()
