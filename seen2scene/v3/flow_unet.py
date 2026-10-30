"""The V2 2D rectified-flow U-Net (seen2scene.futureview.model.UNet2D, unchanged layers) with one optional cross-
attention block at the bottleneck, applied after the bottleneck self-attention: H' = H + CrossAttn(LN(H), M).
With xattn=None the forward is the V2 forward; the cross-attention is created last, so the parameters of every
common module are created in the same order (identical initial values under the same seed) for T0 and T1.
"""
import torch

from seen2scene.futureview.model import Res2D, UNet2D, timestep_embedding
from seen2scene.v3.cross_attention import CrossAttnBlock


class FlowUNet(UNet2D):
    def __init__(self, in_ch, out_ch=2, ch=(64, 128, 192, 256), blocks=2, attn_levels=(3,), heads=4, t_emb=256, groups=16,
                 xattn=None):
        super().__init__(in_ch, out_ch, ch, blocks, attn_levels, heads, t_emb, groups)
        self.xattn = CrossAttnBlock(ch[-1], xattn["mem_dim"], xattn["q_pe_dim"], xattn["heads"]) if xattn else None

    def forward(self, x, tau, mem=None, return_features=False):
        """mem: None or (tokens [B, N, F], mask [B, N], query pe [B, L, P]) for the cross-attention block."""
        te = self.temb(timestep_embedding(tau))
        h = self.inp(x)
        hs = [h]
        for m in self.down:
            if isinstance(m[0], Res2D):
                h = m[0](h, te)
                for a in m[1:]:
                    h = a(h)
            else:
                h = m[0](h)
            hs.append(h)
        h = self.mid[1](self.mid[0](h, te))
        if self.xattn is not None:
            assert mem is not None, "T1 needs the scene memory"
            h = self.xattn(h, *mem)
        h = self.mid[2](h, te)
        for m in self.up:
            if isinstance(m[0], Res2D):
                h = m[0](torch.cat([h, hs.pop()], 1), te)
                for a in m[1:]:
                    h = a(h)
            else:
                h = m[1](m[0](h))
        out = self.out(h)
        return (out, h) if return_features else out
