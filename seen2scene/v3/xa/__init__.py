"""EXP6 V3 / Experiment XA: multi-scale geometry-aware ray-scene attention (experiments/exp6_v3/XA_Experiment_Protocol.md).

Only the network changes with respect to V3-T1: the VAE feature pyramid, the scene keys / values, the queries and the
cross-attention. P1, labels, losses, flow, decoding and the planners are the frozen V3 code.

    vae_pyramid.py  V3 encoder with the stride-4 feature kept (F4), mu (F8), valid-aware 2x pooling of mu (F16)
    memory.py       observation-supported scene tokens of one scale + the T1 frustum selection; global key / value tensor
    queries.py      the three query builders on the 16 x 32 anchor grid: spatial 2D, representative ray, UNKNOWN interval;
                    on-demand local samples of F4 / F8 along a query
    attention.py    XABlock: one cross-attention over global (shared per view) and local (per query) keys, zero-init output
    xa_model.py     XAModel (V3Model subclass): the block at the 16 x 32 level of the flow U-Net; build() / load_core()

Six variants = memory {s: mu @8x, <= 512 tokens (the T1 memory) | m: F16 256 + F8 256 global tokens + 16 local F4 / F8
samples per query} x query {2d | ray | interval}. All new parameters live in unet.xattn, created after every module shared
with T0 / T1, so the shared modules get the T1 initial values under the same seed (checked by the common-state hash).
"""
