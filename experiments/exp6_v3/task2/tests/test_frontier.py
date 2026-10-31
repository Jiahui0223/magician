"""Task 2 -- frontier map unit tests on a synthetic grid (seen2scene env, CPU)."""
import os
import sys

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.join(os.path.dirname(HERE), "common"))
import t2_frontier as TF  # noqa: E402


def test_bfs_and_frontier_progress():
    passable = np.zeros((5, 5), bool)
    passable[2, :] = True                       # a corridor along h at l = 2
    d = TF.FrontierMap.bfs(passable, [(2, 0)])
    assert d[2, 4] == 4 and d[0, 0] == -1 and d[2, 2] == 2
    fm = TF.FrontierMap.__new__(TF.FrontierMap)
    fm.n_l, fm.n_w, fm.n_h = 5, 1, 5
    fm.free = np.zeros((5, 5), np.int32); fm.occ = np.zeros((5, 5), np.int32); fm.visited = np.zeros((5, 5), bool)
    fm.free[2, :3] = 5                          # cells (2,0..2) free; (2,3) unknown -> frontier
    free, occ, unk = fm.states()
    fr = fm.frontier(free, unk)
    assert fr[2, 3] and fr[1, 0] and not fr[2, 0] and fr.sum() == 3 + 3 + 1   # unknown neighbours of the free run
    # progress: current cell (2,0); candidates at (2,1) [towards], (2,0) rotation, (1,0) frontier itself (unknown, legal)
    fm.x_min = np.array([0.0, 0.0, 0.0]); fm.x_max = np.array([5.0, 1.0, 5.0]); fm.cell = np.array([1.0, 1.0, 1.0]); fm.n_records = 1
    rec = {"cam_center": np.array([2.5, 0.5, 0.5]), "legal_center": np.array([[2.5, 0.5, 1.5], [2.5, 0.5, 0.5], [1.5, 0.5, 0.5]])}
    fp, info = fm.progress(rec, None, None)
    assert info["dist_to_frontier"] == 1 and fp[1] == 0.0 and fp[2] == 1.0 and fp[0] == 0.0
    fm.visited[1, 0] = True
    fp, _ = fm.progress(rec, None, None)
    assert fp[2] == 0.5
