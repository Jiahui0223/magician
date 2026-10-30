"""Seen2Scene future-view prediction (EXP5 Task 1, "V2"): for every legal candidate camera, predict K samples of the
first visible surface (hit / depth) on the frozen 114 x 64 scoring rays, conditioned on the observed partial TSDF.

    geometry.py   ray marching on the observed state (same sample positions as the frozen SceneCaster), unknown
                  samples of a ray, projection statistics / input channels, label encoding, decoding to hits
    coarse.py     8^3 block statistics of the observed state (input of the shared geometry encoder)
    rays.py       camera rays of the MAGICIAN back-projection convention with a sub-pixel shift, and the custom
                  PyTorch3D intrinsics whose pixel centres are those rays (label rendering)

Everything here works on tensors / arrays; the frozen EXP3b / EXP4 modules (SceneCaster, P1Fusion, DistP1) are glued in
by the experiment scripts (experiments/exp5_future_view/task1/scripts). Frozen rules: task1/protocol.md v1.0.
"""
