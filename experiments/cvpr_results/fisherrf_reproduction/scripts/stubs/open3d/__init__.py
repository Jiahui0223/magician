"""open3d STUB for the FisherRF adapter process only (decision D8, user-approved 2026-11-11).

open3d cannot be installed in the fisherrf env (glibc 2.17; ENVIRONMENT.md B1). The official FisherRF-active-mapping
code imports it at module top level (models/SLAM/gaussian.py:7, models/SLAM/utils/gs_helpers.py:2) but never calls it on
the code path we use: gaussian.py contains no `o3d.` attribute access at all, and the only use in the repository,
gs_helpers.o3d_knn (gs_helpers.py:176-181), is defined but never called anywhere (grep 2026-11-11).

This directory is put on sys.path ONLY by fisherrf_planner.py. Any attribute access on the module raises, so a real use
of open3d would fail loudly instead of silently computing something else. (The planner identifies the stub by
`open3d.__file__`, a regular module attribute set by the import system.)
"""


def __getattr__(name):
    raise RuntimeError(f"open3d stub: called, not allowed (attribute {name!r})")
