"""EXP9-SCONE -- shared SCONE pieces for the outdoor runner (scone_sim_out.py) and the indoor planner child
(scone_planner_grid.py). Run with the magician env. Nothing here is a re-implementation of SCONE: the occupancy field and the
coverage-gain prediction are the official MACARONS-repo functions (compute_scene_occupancy_probability_field,
predict_coverage_gain_for_single_camera; identical in MACARONS b23c180, MAGICIAN e697009 and the clean 9521d69 worktree, see
AUDIT_AND_PLAN.md section 2), called with the official ShapeNet-pretrained SCONE weights.

ORIGINAL SCONE (kept): SconeOcc + SconeVis networks and weights (pretrained_scone_occ.pth / pretrained_scone_vis.pth, MACARONS
  Google Drive), proxy-point volume, per-cell occupancy prediction from the partial depth point cloud + view-state harmonics,
  Monte-Carlo coverage-gain integral over occupancy-sampled proxy points in the candidate frustum, greedy NBV = argmax of the
  predicted coverage gain over the direct neighbours of the current pose (SCONE paper section 3.2).
ADAPTED (shared with the MACARONS row, see PROTOCOL_ADAPTATION.md): candidate set = the simulator legal list of the frozen protocol,
  GT depth via the frozen sensor, shared 10 % point sample, pipeline parameters of macarons_default_training_config.json.
"""
import hashlib
import os

import torch

OFFICIAL = "/bsuscratch/jiahuizhang/projects/occupancy/SCONE_official"
W_OCC = os.path.join(OFFICIAL, "weights_gdrive", "occupancy", "pretrained_scone_occ.pth")
W_VIS = os.path.join(OFFICIAL, "weights_gdrive", "coverage_gain", "pretrained_scone_vis.pth")
SHA = {"pretrained_scone_occ.pth": "257674e17f7d693bfdd555eed50e4a2ce9e77d5f3c31e45183d68d0faf404bd9",
       "pretrained_scone_vis.pth": "72c9bfdccde7e50578a7c830f3c09e7653074e9df3f23b8241546210b1a315f0",
       "pretrained_macarons.pth": "c7d66c9f46ac07e676a0a956390460c067aa5ed9bc5ebbaa298300a85d9e54ad"}
GDRIVE = {"pretrained_scone_occ.pth": "https://drive.google.com/uc?id=1FmpKSYkLuuy40JCMuFmh2j6sSYxO_Pim",
          "pretrained_scone_vis.pth": "https://drive.google.com/uc?id=1az0Fc6qEiZ5gj1mV66hbaV-96DVvm1iz"}


def sha256_file(p):
    h = hashlib.sha256()
    with open(p, "rb") as f:
        for b in iter(lambda: f.read(1 << 20), b""):
            h.update(b)
    return h.hexdigest()


def _strip(sd):
    return {k[len("module."):] if k.startswith("module.") else k: v for k, v in sd.items()}


def load_scone_model(load_pretrained_macarons, weights_dir, dev):
    """Macarons wrapper holding the official SCONE modules.

    1. load_pretrained_macarons('pretrained_macarons.pth') = the MACARONS-repo container built by
       weights/generate_macarons_model.py -p from pretrained_scone_occ.pth + pretrained_scone_vis.pth (its depth module is
       never called: depth comes from the sensor);
    2. the two official SCONE files are loaded into scone.occupancy / scone.visibility (strict) and every tensor is checked to be
       bitwise equal to step 1, so the weights are traceable to the downloaded files themselves.
    trained_macarons.pth (MACARONS self-supervised fine-tuning) is NOT loaded."""
    pm = os.path.join(weights_dir, "pretrained_macarons.pth")
    shas = {"pretrained_macarons.pth": sha256_file(pm), "pretrained_scone_occ.pth": sha256_file(W_OCC),
            "pretrained_scone_vis.pth": sha256_file(W_VIS)}
    for k, v in shas.items():
        assert v == SHA[k], f"sha256 mismatch {k}: {v}"
    macarons = load_pretrained_macarons(pretrained_model_path="pretrained_macarons.pth", device=dev, learn_pose=False)
    before = {k: v.detach().clone() for k, v in macarons.scone.state_dict().items()}
    occ = torch.load(W_OCC, map_location=dev, weights_only=False)
    vis = torch.load(W_VIS, map_location=dev, weights_only=False)
    macarons.scone.occupancy.load_state_dict(_strip(occ["model_state_dict"]), strict=True)
    macarons.scone.visibility.load_state_dict(_strip(vis["model_state_dict"]), strict=True)
    after = macarons.scone.state_dict()
    n_diff = sum(int(not torch.equal(before[k], after[k])) for k in before)
    assert n_diff == 0, f"{n_diff} SCONE tensors differ between pretrained_macarons.pth and the official SCONE files"
    macarons.eval()
    info = {"weights": {"occupancy": W_OCC, "coverage_gain": W_VIS, "container": os.path.realpath(pm)},
            "sha256": shas, "gdrive": GDRIVE, "occ_epoch": int(occ["epoch"]), "vis_epoch": int(vis["epoch"]),
            "n_params_occ": int(sum(p.numel() for p in macarons.scone.occupancy.parameters())),
            "n_params_vis": int(sum(p.numel() for p in macarons.scone.visibility.parameters())),
            "container_equals_official": True, "trained_macarons_loaded": False}
    return macarons, info


def occ_stats(X_world, occ_probs, proxy_scene):
    """Diagnostics of the occupancy field (never fed back into the decision)."""
    o = occ_probs.reshape(-1).float()
    n = int(o.numel())
    if n == 0:
        return {"n": 0}
    q = torch.quantile(o[:min(n, 2_000_000)], torch.tensor([0.05, 0.5, 0.95], device=o.device)).tolist()
    oof = int((proxy_scene.out_of_field > 0.).sum().item())
    return {"n": n, "n_out_of_field": oof, "n_predicted": n - oof, "mean": float(o.mean()), "q05": q[0], "q50": q[1],
            "q95": q[2], "frac_gt_0.5": float((o > 0.5).float().mean()), "finite": bool(torch.isfinite(o).all())}


def scone_greedy(params, scone, camera, surface_scene, proxy_scene, candidates, dev, occupancy_field, predict_gain):
    """SCONE NBV for one step. candidates: list of (X (1, 3), fov_camera) in the caller's order.
    Official order (MACARONS testers/scene.py compute_trajectory): occupancy field once, then for every candidate the
    predicted coverage gain, argmax with strict '>' starting from -1 (ties -> earliest candidate); a candidate whose frustum
    holds no proxy point with occ > min_occ gets the official dummy gain 0. Default = first candidate (official
    next_idx = valid_neighbors[0]; EXP3b R3: first LEGAL candidate); None only when there is no candidate (hover).
    Returns (pick or None, gains [float | None], X_world, occ_probs)."""
    with torch.no_grad():
        X_world, view_harmonics, occ_probs = occupancy_field(params, scone, camera, surface_scene, proxy_scene, dev)
    gains, pick, best = [], (0 if candidates else None), -1.
    for X_c, fov_c in candidates:
        with torch.no_grad():
            _, _, _, cg = predict_gain(params=params, macarons=scone, proxy_scene=proxy_scene, surface_scene=surface_scene,
                                       X_world=X_world, proxy_view_harmonics=view_harmonics, occ_probs=occ_probs,
                                       camera=camera, X_cam_world=X_c, fov_camera=fov_c)
        g = float(cg.reshape(-1)[0].item()) if cg.shape[0] > 0 else None
        gains.append(g)
        if cg.shape[0] > 0 and cg > best:
            best = cg
            pick = len(gains) - 1
    return pick, gains, X_world, occ_probs
