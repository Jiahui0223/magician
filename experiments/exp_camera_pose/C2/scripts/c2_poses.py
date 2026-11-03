"""Camera Pose C2 -- the C2 pose lattice and candidate families (numpy only; shared by the Stage A oracle, the Stage B
fixed-state model scoring and the C2 closed-loop simulator). Frozen in C2/protocol.md before any C2 result.

Fine lattice on top of the original MP3D grid g (grid_env.grid_def: ~1 m cells, 5 pitch bins of 30 deg, 8 yaw bins of
45 deg): position index step = 1/4 cell (PS = 4), pitch step 5 deg (ES = 6 per original bin), yaw step 5 deg (AS = 9).
Every original pose maps exactly onto the lattice: fine = (4 l, 4 w, 4 h, 6 e, 9 a). Index axes as grid_env: 0 = x,
1 = height (up), 2 = the second horizontal axis; runtime frame rt = 7 * (x, z_up, -y).
  position(g, f) = x_min + (0.5 + f[:3] / 4) * (x_max - x_min) / n        pitch = -60 + 5 je   yaw = 5 ja (mod 360)
Bounds: f[:3] in [0, 4 (n - 1)] (the original cell-centre hull), je in [0, 24] (pitch -60 .. +60, the original range).

Candidate families (offsets in fine units; translations never exceed one original cell):
  C0   the original neighbours: +-4 on one position axis, +-6 pitch, +-9 yaw (= grid_env.neighbours on the lattice)
  C2a  orientation only:  yaw +-2, +-4 (+-10, +-20 deg); pitch +-1, +-2 (+-5, +-10 deg)                  8 new
  C2b  position only:     x, h, y axis +-1, +-2 (0.25, 0.5 cell); horizontal diagonals (+-2, 0, +-2)      16 new
  C2c  joint:             8 horizontal directions at 0.5 cell (axis +-2; diagonal (+-2, 0, +-2)) x yaw +-4 (+-20 deg)
                          + height +-2 (0.5 cell) x pitch +-2 (+-10 deg)                                    20 new
Legality (simulator side, GT only for the motion constraint): endpoint inside the bounds; the straight segment from the
current to the target position does not intersect the GT mesh (the same Blocker as the original legal_actions; for a
straight-line kinematic path this checks the whole path); rotation-only moves are always legal; the current pose and
duplicates are dropped. The executed kinematic path: linear position, shortest-arc yaw, linear pitch; one observation at
the end point only."""
import numpy as np

PS, ES, AS = 4, 6, 9
N_E, N_A = 25, 72


def to_fine(idx):
    i = [int(x) for x in idx]
    return (PS * i[0], PS * i[1], PS * i[2], ES * i[3], AS * i[4])


def bounds(g):
    n = [int(x) for x in g["n"]]
    return [PS * (k - 1) for k in n]


def position(g, f):
    n = np.asarray(g["n"], np.float64)
    return np.asarray(g["x_min"], np.float64) + (0.5 + np.asarray(f[:3], np.float64) / PS) * (np.asarray(g["x_max"], np.float64) - np.asarray(g["x_min"], np.float64)) / n


def angles(g, f):
    return -60.0 + 5.0 * f[3], 5.0 * (f[4] % N_A)


def _c0():
    out = []
    for ax, step in ((0, PS), (1, PS), (2, PS), (3, ES), (4, AS)):
        for s in (-1, 1):
            d = [0] * 5
            d[ax] = s * step
            out.append(tuple(d))
    return out


def _c2a():
    return [(0, 0, 0, 0, s * k) for k in (2, 4) for s in (-1, 1)] + [(0, 0, 0, s * k, 0) for k in (1, 2) for s in (-1, 1)]


def _c2b():
    out = []
    for ax in (0, 1, 2):
        for k in (1, 2):
            for s in (-1, 1):
                d = [0] * 5
                d[ax] = s * k
                out.append(tuple(d))
    out += [(sx * 2, 0, sz * 2, 0, 0) for sx in (-1, 1) for sz in (-1, 1)]
    return out


def _c2c():
    dirs = [(2, 0, 0), (-2, 0, 0), (0, 0, 2), (0, 0, -2)] + [(sx * 2, 0, sz * 2) for sx in (-1, 1) for sz in (-1, 1)]
    out = [(dx, dy, dz, 0, s * 4) for dx, dy, dz in dirs for s in (-1, 1)]
    out += [(0, sh * 2, 0, sp * 2, 0) for sh in (-1, 1) for sp in (-1, 1)]
    return out


FAMILIES = {"C0": _c0(), "C2a": _c2a(), "C2b": _c2b(), "C2c": _c2c()}
VARIANTS = {"C0": ("C0",), "C2a": ("C0", "C2a"), "C2b": ("C0", "C2b"), "C2c": ("C0", "C2c"), "ALL": ("C0", "C2a", "C2b", "C2c")}


def apply(f, d):
    return (f[0] + d[0], f[1] + d[1], f[2] + d[2], f[3] + d[3], (f[4] + d[4]) % N_A)


def in_bounds(g, f):
    b = bounds(g)
    return all(0 <= f[i] <= b[i] for i in range(3)) and 0 <= f[3] <= N_E - 1


def zup_m(g, f, rt_to_zup_m):
    return np.asarray(rt_to_zup_m(position(g, f)), np.float64)


def move_cost(g, f0, f1, rt_to_zup_m):
    """Translation (m, z-up metric), |d yaw| (shortest arc, deg), |d pitch| (deg) of the kinematic path f0 -> f1."""
    tr = float(np.linalg.norm(zup_m(g, f1, rt_to_zup_m) - zup_m(g, f0, rt_to_zup_m)))
    dy = abs(((5.0 * (f1[4] - f0[4]) + 180.0) % 360.0) - 180.0)
    dp = abs(5.0 * (f1[3] - f0[3]))
    return tr, dy, dp


def candidates(g, f0, variant, blocked, rt_to_zup_m):
    """Legal candidates of `variant` from fine pose f0, sorted lexicographically (the original tie-break order).
    blocked(p_zup_m, q_zup_m) -> bool is the simulator's GT segment test. Returns (legal list of dicts, n_generated,
    n_out_of_bounds, n_blocked)."""
    fam_of = {}
    for fam in VARIANTS[variant]:
        for d in FAMILIES[fam]:
            q = apply(f0, d)
            if q == tuple(f0) or q in fam_of:
                continue
            fam_of[q] = fam
    p0 = zup_m(g, f0, rt_to_zup_m)
    legal, n_oob, n_blk = [], 0, 0
    for q in sorted(fam_of):
        if not in_bounds(g, q):
            n_oob += 1
            continue
        if tuple(q[:3]) != tuple(f0[:3]) and blocked(p0, zup_m(g, q, rt_to_zup_m)):
            n_blk += 1
            continue
        tr, dy, dp = move_cost(g, f0, q, rt_to_zup_m)
        legal.append({"f": q, "family": fam_of[q], "trans_m": tr, "dyaw": dy, "dpitch": dp,
                      "on_original_lattice": q[0] % PS == 0 and q[1] % PS == 0 and q[2] % PS == 0 and q[3] % ES == 0 and q[4] % AS == 0})
    return legal, len(fam_of), n_oob, n_blk
