"""CVPR-style architecture figure of V3-T0 + D1 (no cross-attention; matplotlib, vector PDF + PNG). python v3_t0_d1_architecture.py"""
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import FancyBboxPatch, FancyArrowPatch

C = {"geo": "#ECECEC", "net": "#D3E4F4", "gen": "#FBE2C6", "plan": "#D7EBD3", "mem": "#EADFF2", "edge": "#3C3C3C", "text": "#111111", "sub": "#4F4F4F", "loop": "#8A8A8A"}
fig, ax = plt.subplots(figsize=(18, 8.6))
ax.set_xlim(0, 182); ax.set_ylim(0, 82); ax.axis("off")


def box(x, y, w, h, title, sub=None, fc=C["geo"], ls="-", fs=9.0, subfs=7.3):
    ax.add_patch(FancyBboxPatch((x, y), w, h, boxstyle="round,pad=0.25,rounding_size=1.1", fc=fc, ec=C["edge"], lw=1.0, ls=ls))
    if sub:
        ax.text(x + w / 2, y + h * 0.64, title, ha="center", va="center", fontsize=fs, fontweight="bold", color=C["text"])
        ax.text(x + w / 2, y + h * 0.28, sub, ha="center", va="center", fontsize=subfs, color=C["sub"], linespacing=1.25)
    else:
        ax.text(x + w / 2, y + h / 2, title, ha="center", va="center", fontsize=fs, fontweight="bold", color=C["text"])
    return (x, y, w, h)


def arrow(p, q, ls="-", lw=1.1, color=None):
    ax.add_patch(FancyArrowPatch(p, q, arrowstyle="-|>", mutation_scale=11, lw=lw, ls=ls, color=color or C["edge"], shrinkA=0, shrinkB=0))


def poly(pts, ls="-", lw=1.1, color=None):
    """orthogonal polyline with an arrow head on the last segment"""
    c = color or C["edge"]
    for a_, b_ in zip(pts[:-2], pts[1:-1]):
        ax.plot([a_[0], b_[0]], [a_[1], b_[1]], ls=ls, lw=lw, color=c, solid_capstyle="round")
    arrow(pts[-2], pts[-1], ls=ls, lw=lw, color=c)


def R(b): return (b[0] + b[2], b[1] + b[3] / 2)
def L(b): return (b[0], b[1] + b[3] / 2)
def T(b): return (b[0] + b[2] / 2, b[1] + b[3])
def B(b): return (b[0] + b[2] / 2, b[1])


# ------------------------------------------------------------------ lane titles
ax.text(2, 77.5, "(a) Scene side: one latent grid per refresh (R10, every 10 steps)", fontsize=10.5, fontweight="bold")
ax.text(2, 54.5, "(b) Candidate side: one conditional map per candidate pose, every step", fontsize=10.5, fontweight="bold")
ax.text(2, 27.5, "(c) Rectified-flow prediction of future first hits and D1 planning", fontsize=10.5, fontweight="bold")

# ------------------------------------------------------------------ (a)
ya, ha = 64, 11
obs = box(2, ya, 18, ha, "Observations 0..t", "depth + pose per step\n(simulator)", C["geo"])
p1 = box(24, ya, 20, ha, "P1 partial TSDF", "fine voxels (0.011)\nBAND / EMPTY / UNKNOWN", C["geo"])
agg = box(48, ya, 20, ha, "4x aggregation", "[sdf, band, free, valid]\n4 x X x Y x Z", C["geo"])
vae = box(72, ya, 24, ha, "3D VAE encoder", "Conv3d stem 32, Res3D, stride-2 64,\n2 x Res3D, 1x1 -> (mu, logvar)", C["net"])
mu = box(100, ya, 18, ha, "Latent grid  mu_t", "16 ch @ 8x voxels\n(logvar: training only)", C["net"])
for a_, b_ in ((obs, p1), (p1, agg), (agg, vae), (vae, mu)):
    arrow(R(a_), L(b_))

# ------------------------------------------------------------------ (b)
yb, hb = 41, 11
cand = box(2, yb, 18, hb, "Candidate pose c_i", "legal set of the simulator\n(6 moves + 4 rotations)", C["geo"])
rays = box(24, yb, 20, hb, "7,296 scoring rays", "64 x 114 grid, (o, d, t_cap)", C["geo"])
proj = box(48, yb + 5.6, 44, 7.0, "Ray marching on P1  ->  11 projection channels", "class (3), t_known, d_lo, d_hi, L_u, n_seg, d_z, cam_z, log t_cap", C["geo"], fs=8.2, subfs=6.8)
samp = box(98, yb - 0.4, 22, 6.6, "Sample mu_t along ray", "32 points, trilinear + valid flag", C["net"], fs=8.2, subfs=6.8)
renc = box(124, yb - 0.4, 22, 6.6, "Latent ray encoder", "Conv1d x3 + fc -> 64-d per ray", C["net"], fs=8.2, subfs=6.8)
cond = box(152, yb, 22, hb, "Conditional map", "[11 proj | 64 ray | 1 valid]\n= 76 x 64 x 128", C["net"])
arrow(R(cand), L(rays))
poly([R(rays), (46, R(rays)[1]), (46, proj[1] + 3.5), (48, proj[1] + 3.5)])
poly([R(rays), (46, R(rays)[1]), (46, samp[1] + 3.3), (98, samp[1] + 3.3)])
arrow(R(samp), L(renc))
poly([R(proj), (149, proj[1] + 3.5), (149, yb + 8.2), (152, yb + 8.2)])
poly([R(renc), (149, renc[1] + 3.3), (149, yb + 3.0), (152, yb + 3.0)])
arrow(B(mu), T(samp))
ax.text(110.5, 54.0, "mu_t", fontsize=7.3, color=C["sub"], ha="left")

# ------------------------------------------------------------------ (c)
yc = 6
un = box(2, yc, 100, 18, "", None, "#FFFFFF")
ax.text(52, yc + 16.3, "Rectified-flow U-Net   input [x_tau (2) | cond (76)] = 78 ch @ 64 x 128   ->   velocity (2 ch)", ha="center", fontsize=9.0, fontweight="bold")
ax.text(52, yc + 13.6, "4 levels down, mirrored up path with skip connections; Res2D with FiLM(tau); self-attention at the 8 x 16 level", ha="center", fontsize=7.0, color=C["sub"])
lv = [("64", "64 x 128", 5), ("128", "32 x 64", 16), ("192", "16 x 32", 27), ("256", "8 x 16", 38)]
for name, res, x in lv:
    box(x, yc + 3.0, 9.5, 6.5, name, res, C["gen"], fs=8.4, subfs=6.8)
    if x < 38:
        arrow((x + 9.5, yc + 6.25), (x + 11, yc + 6.25), lw=0.9)
bot = box(52, yc + 2.2, 47, 8.4, "Bottleneck 8 x 16", "Res2D - Self-Attn (128 tokens, 4 heads) - Res2D\nintegrates information across all rays of the candidate view", C["gen"], fs=8.4, subfs=6.8)
arrow((47.5, yc + 6.25), (52, yc + 6.25), lw=0.9)
# connectors into the U-Net (orthogonal buses between lanes (b) and (c))
poly([B(cond), (163, 30.5), (24, 30.5), (24, yc + 18)])
ax.text(92, 31.3, "cond", fontsize=7.3, color=C["sub"])
# generative head and planner
eul = box(106, yc + 10.2, 20, 7.2, "Euler x10, K = 4", "noise eps_k shared by all candidates", C["gen"], fs=8.4, subfs=6.8)
by = box(106, yc + 1.2, 20, 7.2, "(b, y) per ray", "b: hit in UNKNOWN?\ny: position on the UNKNOWN arc", C["gen"], fs=8.4, subfs=6.6)
dec = box(130, yc + 1.2, 20, 7.2, "Geometric decode", "y -> UNKNOWN sample\n-> first-hit distance t_k", C["geo"], fs=8.4, subfs=6.6)
val = box(130, yc + 10.2, 20, 7.2, "Novel-coverage value", "hits > 1 m from observed points,\n1 m cells  ->  V_k(c)", C["plan"], fs=8.4, subfs=6.6)
d1 = box(154, yc + 1.2, 26, 16.2, "D1 two-step lookahead", "S(c_i) = max_j mean_k |V_k(c_i) U V_k(c_j)|\nc_j = grid neighbours of c_i\n(same model, same noise)\nargmax S -> execute -> observe", C["plan"], fs=8.6, subfs=6.6)
arrow((102, yc + 13.8), L(eul)); arrow(B(eul), T(by)); arrow(R(by), L(dec)); arrow(T(dec), B(val)); arrow(R(val), (154, val[1] + 3.6))
# closed loop back to the observations (orthogonal, dotted)
poly([(180, yc + 9.3), (181.2, yc + 9.3), (181.2, 1.5), (1, 1.5), (1, ya + 5.5), (2, ya + 5.5)], ls=":", lw=1.1, color=C["loop"])
ax.text(91, 2.6, "next observation (closed loop, 100 steps)", fontsize=7.3, color=C["loop"], ha="center")

# ------------------------------------------------------------------ legend
lx, ly = 128, 70
for i, (k, lab) in enumerate((("geo", "geometric / frozen"), ("net", "learned (3D VAE encoder + ray encoder)"), ("gen", "generative head (rectified-flow U-Net)"), ("plan", "planner D1 (frozen, no learning)"))):
    ax.add_patch(FancyBboxPatch((lx, ly + 5.2 - i * 2.3), 3.0, 1.6, boxstyle="round,pad=0.1", fc=C[k], ec=C["edge"], lw=0.7, ls="--" if k == "mem" else "-"))
    ax.text(lx + 4.0, ly + 6.0 - i * 2.3, lab, fontsize=7.2, va="center")
ax.text(2, 36.0, "V3-T0: 23.5 M parameters, no cross-attention.  Inference: D100 map as encoder input, S10 map for novelty, R10 refresh, K = 4, Euler 10.  Only UNKNOWN rays are generated; EMPTY / KNOWN rays keep the geometric outcome.", fontsize=7.0, color=C["sub"])
fig.savefig("experiments/exp6_v3/figures/V3_T0_D1_architecture.pdf", bbox_inches="tight")
fig.savefig("experiments/exp6_v3/figures/V3_T0_D1_architecture.png", dpi=170, bbox_inches="tight")
print("saved")
