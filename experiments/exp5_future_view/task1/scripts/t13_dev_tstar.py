"""EXP5 Task 1.3 (magician env, 1 GPU; EVALUATION SIDE, Dev scenes, purpose "dev_eval"): first GT intersection t* of
the 114 x 64 block-centre rays of every legal candidate at the S-a states (73 T5-D A1 states) and the S-b states (B0
closed-loop states), for the true-depth DistP1 upper bound only (protocol §4.6). Same rasteriser as t11_labels.py.
Output W/fixed_state/dev_tstar/<set>_<scene>_s<k>.npz  (t, tstar [n_views, 7296], view index of each candidate)
"""
import argparse
import glob
import os
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
import v2_frozen as VF  # noqa: E402
import v2_common as LC  # noqa: E402
import t11_labels as TL  # noqa: E402

np, torch, MC, C0 = TL.np, TL.torch, TL.MC, TL.C0


def load_mesh_dev(scene, dev):
    import exp3b_paths as P
    from pytorch3d.io import load_obj
    from pytorch3d.structures import Meshes
    C0.assert_geometry_allowed(scene, "dev_eval")
    objs = sorted(glob.glob(os.path.join(P.scene_dir(scene), "*.obj")))
    assert len(objs) == 1, objs
    verts, faces, _ = load_obj(objs[0], load_textures=False, device=dev)
    return Meshes(verts=[verts * 10.0], faces=[faces.verts_idx])


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--set", choices=["sa", "sb"], required=True)
    ap.add_argument("--scene", required=True)
    a = ap.parse_args()
    import t5_diagnostic as TD
    states_a = [x for x in MC.T5_STATES if x not in {("eiffel", 2, 0), ("colosseum", 2, 0)}]   # = g1_common.STATES_A_FULL
    dev = torch.device("cuda")
    mesh = load_mesh_dev(a.scene, dev)
    fr = TL.frame_of(a.scene)
    ren = TL.Renderer(mesh, dev, 0, None, 8)
    od = os.path.join(VF.WORK, "fixed_state", "dev_tstar")
    os.makedirs(od, exist_ok=True)
    if a.set == "sa":
        groups = {}
        for s, k, t in states_a:
            if s == a.scene:
                groups.setdefault(k, []).append((t, TD.svc_path(s, k, t)))
    else:
        groups = {}
        for k in LC.STARTS:
            d = os.path.join(LC.B0_ROOT, "svc", f"A2_{a.scene}_s{k}_r0")
            groups[k] = [(t, os.path.join(d, f"svc_{a.scene}_s{k}_t{t}.npz")) for t in range(LC.MOVES)]
    for k, items in groups.items():
        p = os.path.join(od, f"{a.set}_{a.scene}_s{k}.npz")
        if os.path.exists(p):
            continue
        T_, A_, Rs, Ts = [], [], [], []
        for t, f in items:
            z = np.load(f)
            for ai in range(len(z["legal_R"])):
                T_.append(t); A_.append(ai); Rs.append(z["legal_R"][ai]); Ts.append(z["legal_T"][ai])
        ts = ren.tstar(Rs, Ts, [(0.0, 0.0)] * len(Rs), fr.k_M).astype(np.float32) if Rs else np.zeros((0, 7296), np.float32)
        np.savez(p + ".tmp.npz", t=np.asarray(T_, np.int16), a=np.asarray(A_, np.int8), tstar=ts)
        os.replace(p + ".tmp.npz", p)
        print(f"{a.set} {a.scene} s{k}: {len(Rs)} views", flush=True)


if __name__ == "__main__":
    main()
