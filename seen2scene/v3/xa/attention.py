"""XABlock (Experiment XA): one cross-attention from the queries of the 16 x 32 anchor grid onto the scene keys / values.

  q   = W_q LN(h_cell + E_q(query features))            h_cell = the flow U-Net feature of the query's anchor cell
  key = W_k LN(e),  value = W_v e                       e = E_scale(global token) or E_local(local sample)
  keys of one query = the view's global tokens (shared by its queries) + the query's own local samples; one softmax over
  both (masked); 4 heads. Queries of the same cell (intervals of one ray) are averaged; cells without a query get 0.
  h' = h + W_o(.) with W_o initialised to zero, so the model starts as the network without cross-attention (T0).
"""
import math

import torch
import torch.nn as nn
import torch.nn.functional as F


class XABlock(nn.Module):
    def __init__(self, dim, q_dim, g_dim, n_scales, l_dim=0, heads=4, per_cell=True):
        super().__init__()
        assert dim % heads == 0
        self.h, self.per_cell = heads, per_cell
        self.ln_q, self.ln_k = nn.LayerNorm(dim), nn.LayerNorm(dim)
        self.e_q = nn.Linear(q_dim, dim)
        self.e_g = nn.ModuleList([nn.Linear(g_dim, dim) for _ in range(n_scales)])
        self.e_l = nn.Linear(l_dim, dim) if l_dim else None
        self.wq, self.wk, self.wv, self.wo = nn.Linear(dim, dim), nn.Linear(dim, dim), nn.Linear(dim, dim), nn.Linear(dim, dim)
        nn.init.zeros_(self.wo.weight)
        nn.init.zeros_(self.wo.bias)

    def forward(self, hmap, G, Q, L):
        """hmap [B, C, 16, 32]; G [B, Ng, g_dim + 2]; Q [B, Nq, 2 + q_dim]; L [B, Nq, Nl, l_dim + 1] (ignored without e_l)."""
        B, C, gh, gw = hmap.shape
        dt = hmap.dtype
        hd = C // self.h
        htok = hmap.flatten(2).transpose(1, 2)
        cell = Q[..., 0].round().long()
        qmask = Q[..., 1] > 0.5
        hq = htok if self.per_cell else htok.gather(1, cell[..., None].expand(-1, -1, C))
        q = self.wq(self.ln_q(hq + self.e_q(Q[..., 2:].to(dt))))
        gf, gs, gmask = G[..., :-2].to(dt), G[..., -2], G[..., -1] > 0.5
        e = self.e_g[0](gf)
        for s in range(1, len(self.e_g)):
            e = torch.where((gs == s)[..., None], self.e_g[s](gf), e)
        kg, vg = self.wk(self.ln_k(e)), self.wv(e)
        heads = lambda t: t.reshape(B, -1, self.h, hd).transpose(1, 2)  # noqa: E731
        qh, kgh, vgh = heads(q), heads(kg), heads(vg)
        Nq, Ng = qh.shape[2], kgh.shape[2]
        logit = (qh.float() @ kgh.float().transpose(-1, -2)) / math.sqrt(hd)
        am = gmask[:, None, None, :].expand(-1, -1, Nq, -1)
        if self.e_l is not None:
            el = self.e_l(L[..., :-1].to(dt))
            Nl = el.shape[2]
            kl = self.wk(self.ln_k(el)).reshape(B, Nq, Nl, self.h, hd).permute(0, 3, 1, 2, 4)
            vl = self.wv(el).reshape(B, Nq, Nl, self.h, hd).permute(0, 3, 1, 2, 4)
            logit = torch.cat([logit, torch.einsum("bhqd,bhqld->bhql", qh.float(), kl.float()) / math.sqrt(hd)], -1)
            am = torch.cat([am, (L[..., -1] > 0.5)[:, None]], -1)
        none = ~am.any(-1, keepdim=True)
        w = torch.softmax(logit.masked_fill(~(am | none), float("-inf")), -1)
        out = w[..., :Ng].to(vgh.dtype) @ vgh
        if self.e_l is not None:
            out = out + torch.einsum("bhql,bhqld->bhqd", w[..., Ng:].to(vl.dtype), vl)
        out = out.masked_fill(none, 0.0).transpose(1, 2).reshape(B, Nq, C)
        qm = qmask[..., None].to(out.dtype)
        if self.per_cell:
            agg = out * qm
        else:
            A = F.one_hot(cell, gh * gw).to(out.dtype) * qm
            agg = (A.transpose(1, 2) @ out) / A.sum(1).clamp(min=1.0)[..., None]
        return hmap + self.wo(agg).transpose(1, 2).reshape(B, C, gh, gw)
