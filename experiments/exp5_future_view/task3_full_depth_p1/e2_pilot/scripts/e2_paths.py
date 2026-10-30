"""EXP5 Task 3 / E2 pilot -- paths. The dense (D100) training data live in their own roots that mirror the Task 2 / Task 1
layouts, so the frozen builders and loaders run unchanged with a redirected root:
  W2D   fake Task 2 work root: mp3d/<everything> = symlinks to task2/mp3d/* (trajectories, t* labels, manifests, cards,
        meshes), except mp3d/views (D100 shards, new) and mp3d/views_claims (new)
  OUTD  Macarons++ T1.1-layout view shards <scene>/<traj>.npz built from the D100 P1 (labels: the frozen T1.1 t* files)
Nothing under the Task 1 / Task 2 work roots is written."""
import os

HERE = os.path.dirname(os.path.abspath(__file__))
E2 = os.path.dirname(HERE)
EXP5 = os.path.dirname(os.path.dirname(E2))
REPO = os.path.dirname(os.path.dirname(EXP5))
T2W = os.path.join(REPO, "work_dirs", "exp5_future_view", "task2")
W = os.path.join(REPO, "work_dirs", "exp5_future_view", "task3_full_depth_p1", "e2")
W2D = os.path.join(W, "w2_d100")
OUTD = os.path.join(W, "out_views_d100")
CHECK = os.path.join(W, "smoke_check")


def setup_roots():
    os.makedirs(os.path.join(W2D, "mp3d", "views"), exist_ok=True)
    os.makedirs(os.path.join(W2D, "mp3d", "views_claims"), exist_ok=True)
    os.makedirs(OUTD, exist_ok=True)
    for x in os.listdir(os.path.join(T2W, "mp3d")):
        if x in ("views", "views_claims"):
            continue
        dst = os.path.join(W2D, "mp3d", x)
        if not os.path.lexists(dst):
            os.symlink(os.path.join(T2W, "mp3d", x), dst)


if __name__ == "__main__":
    setup_roots()
    print(sorted(os.listdir(os.path.join(W2D, "mp3d"))))
