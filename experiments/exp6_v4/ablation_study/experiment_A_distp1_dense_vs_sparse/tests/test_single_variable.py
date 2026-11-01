"""G0 single-variable proof (GPU): from the SAME executed records (the A0 smoke episode's service records), build the B2
fusion (P1FusionB2 = dense + sparse twin) and the full fusion (P1FusionFull), build the planner geometries under the b2
patch and without it, and assert that everything the model / projection / decoder reads is identical (input volume,
structure, domain, cast volume) while only caster.band_dil differs (sparse twin vs dense), the dense band_dil being a
superset-sized map. Also: the fusions hold exactly the executed records (no unexecuted candidate observation)."""
import glob
import os
import sys

import pytest
import torch

R = "/bsuscratch/jiahuizhang/projects/occupancy/seen2scene"
EXP5 = os.path.join(R, "experiments", "exp5_future_view")
SMOKE = os.path.join(R, "work_dirs", "exp6_v4", "ablation_study", "experiment_A", "smoke", "indoor_b2", "V2M", "Z6MFQCViBuw_L0_S6", "svc")
pytestmark = pytest.mark.skipif(not torch.cuda.is_available() or not os.path.isdir(SMOKE), reason="needs a GPU and the A0 smoke records")


def test_b2_and_full_differ_only_in_band_dil():
    sys.path.insert(0, os.path.join(EXP5, "task2", "scripts"))
    import t23_v2_in_planner  # noqa: F401  (sets the planner import paths)
    sys.path.insert(0, os.path.join(EXP5, "task3_full_depth_p1", "scripts"))
    sys.path.insert(0, os.path.join(EXP5, "task1", "scripts"))
    import p1_full as PF
    import v2_plan_core as PC
    import m_common as MC
    import t2_common as C
    import t21_build as B21
    dev = torch.device("cuda")
    recs = [MC.load_service_record(p) for p in sorted(glob.glob(os.path.join(SMOKE, "svc_t*.npz")))[:3]]
    assert len(recs) >= 2
    card = C.load_card("Z6MFQCViBuw_L0")
    region = B21.region(card, recs[0]["cam_center"])[:3]
    fb2, ffull = PF.P1FusionB2(*region, device=dev), PF.P1FusionFull(*region, device=dev)
    for r in recs:
        fb2.integrate_record(r); ffull.integrate_record(r)
    assert fb2.n_frames == ffull.n_frames == len(recs) == fb2.sparse.n_frames        # only the executed records
    assert torch.equal(fb2.input_volume(), ffull.input_volume()) and torch.equal(fb2.structure(), ffull.structure())
    orig = PC.Geometry.from_fusion
    PF._patch_geometry_b2()                                                          # as the b2 planner child
    g_b2 = PC.Geometry.from_fusion(*region, fb2, dev)
    PC.Geometry.from_fusion = orig                                                   # as the full planner child (no patch)
    g_full = PC.Geometry.from_fusion(*region, ffull, dev)
    assert torch.equal(g_b2.caster.domain, g_full.caster.domain) and torch.equal(g_b2.vol_obs, g_full.vol_obs) and torch.equal(g_b2.struct, g_full.struct)
    assert torch.equal(g_b2.coarse(), g_full.coarse())                                # model input identical
    assert not torch.equal(g_b2.caster.band_dil, g_full.caster.band_dil)              # the single variable
    n_sp, n_de = int(g_b2.caster.band_dil.sum()), int(g_full.caster.band_dil.sum())
    assert n_de > n_sp, (n_sp, n_de)
    assert torch.equal(g_b2.caster.band_dil, torch.nn.functional.max_pool3d(((fb2.sparse.input_volume().to(dev).abs() < MC.TRUNC * 0.999) & g_b2.caster.domain)[None, None].half(), 3, 1, 1)[0, 0] > 0)
    print("band_dil voxels sparse", n_sp, "dense", n_de, "ratio %.2f" % (n_de / max(n_sp, 1)))
