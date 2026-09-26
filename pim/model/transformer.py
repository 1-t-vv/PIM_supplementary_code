import math
import warnings
from typing import List, Optional, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn.attention import SDPBackend, sdpa_kernel


_FLASH_BATCH_CHUNK = 32768


def _flash_attention(
    query: torch.Tensor,
    key: torch.Tensor,
    value: torch.Tensor,
) -> torch.Tensor:
    """
    Run SDPA with a guaranteed FlashAttention backend on CUDA.

    Keeping FLASH_ATTENTION as the only allowed CUDA backend prevents PyTorch
    from silently falling back to the memory-efficient or math kernels. CPU is
    intentionally left on the math implementation for local testing/evaluation.
    """
    if query.ndim != 4 or key.ndim != 4 or value.ndim != 4:
        raise ValueError(
            "FlashAttention expects 4D (batch, heads, sequence, head_dim) "
            f"tensors, got query={query.shape}, key={key.shape}, value={value.shape}"
        )

    if query.device.type == "cuda":
        with sdpa_kernel(backends=[SDPBackend.FLASH_ATTENTION]):
            if query.size(0) <= _FLASH_BATCH_CHUNK:
                return F.scaled_dot_product_attention(
                    query,
                    key,
                    value,
                    dropout_p=0.0,
                    is_causal=False,
                )

            # Fine evaluation can flatten B*h*V into more than 65k independent
            # neighborhoods. Chunk that dimension to stay within fused-kernel
            # launch limits without changing any attention neighborhoods.
            outputs = []
            for start in range(0, query.size(0), _FLASH_BATCH_CHUNK):
                end = min(start + _FLASH_BATCH_CHUNK, query.size(0))
                outputs.append(
                    F.scaled_dot_product_attention(
                        query[start:end],
                        key[start:end],
                        value[start:end],
                        dropout_p=0.0,
                        is_causal=False,
                    )
                )
            return torch.cat(outputs, dim=0)

    return F.scaled_dot_product_attention(
        query,
        key,
        value,
        dropout_p=0.0,
        is_causal=False,
    )


@torch.no_grad()
def verify_flash_attention_support(
    device: torch.device,
    head_dim: int,
    key_length: int,
) -> None:
    """Fail early when the active CUDA/PyTorch stack cannot run FlashAttention."""
    device = torch.device(device)
    if device.type != "cuda":
        return

    q = torch.zeros((1, 1, 1, head_dim), device=device, dtype=torch.float16)
    k = torch.zeros((1, 1, key_length, head_dim), device=device, dtype=torch.float16)
    v = torch.zeros_like(k)
    try:
        _flash_attention(q, k, v)
        torch.cuda.synchronize(device)
    except Exception as exc:
        raise RuntimeError(
            "FlashAttention is required, but the current GPU/PyTorch/CUDA "
            f"stack rejected head_dim={head_dim}, key_length={key_length}."
        ) from exc


class DropPath(nn.Module):
    def __init__(self, p: float = 0.0):
        super().__init__()
        self.p = float(p)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if self.p == 0.0 or (not self.training):
            return x
        keep = 1.0 - self.p
        shape = (x.shape[0],) + (1,) * (x.ndim - 1)
        mask = x.new_empty(shape).bernoulli_(keep)
        return x * mask / keep


@torch.no_grad()
def knn_idx_chunked(x: torch.Tensor, k: int = 12, chunk: int = 1024) -> torch.Tensor:
    """
    Brute-force KNN using torch.cdist, computed in chunks to cap peak memory.
    x: (B, V, 3)
    return: (B, V, min(k, V - 1))
    """
    if x.ndim != 3 or x.size(-1) != 3:
        raise ValueError(f"x must be (B,V,3), got {tuple(x.shape)}")
    if k < 0:
        raise ValueError(f"k must be non-negative, got {k}")
    if chunk <= 0:
        raise ValueError(f"chunk must be positive, got {chunk}")

    B, V, _ = x.shape
    device = x.device
    effective_k = min(int(k), max(V - 1, 0))
    idx_out = torch.empty(B, V, effective_k, dtype=torch.long, device=device)
    if effective_k == 0:
        return idx_out

    x32 = x.to(torch.float32)

    for s in range(0, V, chunk):
        e = min(V, s + chunk)
        dist = torch.cdist(x32[:, s:e], x32)  # (B, e-s, V)
        local_query = torch.arange(e - s, device=device)
        global_query = torch.arange(s, e, device=device)
        dist[:, local_query, global_query] = torch.inf
        topk = dist.topk(k=effective_k, dim=-1, largest=False).indices
        idx_out[:, s:e] = topk
        del dist, topk

    return idx_out


@torch.no_grad()
def uniform_anchor_indices(V: int, k: int, device: torch.device) -> torch.Tensor:
    """
    Return (k,) long tensor of "uniform anchors" in [0, V-1].
    If V < k, repeat indices.
    """
    if V <= 0:
        raise ValueError(f"V must be positive, got {V}")
    if k <= 0:
        raise ValueError(f"k must be positive, got {k}")
    if V >= k:
        # Match ``np.linspace(0, V - 1, num=k, dtype=np.int64)`` used by
        # existing cache files.  Using round() here changes almost half of the
        # anchors for the current V=161/642 datasets.
        if k == 1:
            anchors = torch.zeros(1, device=device, dtype=torch.long)
        else:
            anchors = torch.arange(k, device=device, dtype=torch.long)
            anchors = anchors.mul(V - 1).div(k - 1, rounding_mode="floor")
    else:
        base = torch.arange(V, device=device, dtype=torch.long)
        reps = (k + V - 1) // V
        anchors = base.repeat(reps)[:k]
    return anchors


@torch.no_grad()
def farthest_point_anchor_indices(points: torch.Tensor, k: int) -> torch.Tensor:
    """Return deterministic per-sample FPS indices with shape (B, k)."""
    if points.ndim != 3 or points.size(-1) != 3:
        raise ValueError(f"points must be (B,V,3), got {tuple(points.shape)}")
    if points.size(1) <= 0:
        raise ValueError("points must contain at least one vertex")
    if k <= 0:
        raise ValueError(f"k must be positive, got {k}")

    B, V, _ = points.shape
    if V <= k:
        shared = uniform_anchor_indices(V, k, points.device)
        return shared.unsqueeze(0).expand(B, -1)

    xyz = points.to(dtype=torch.float32)
    anchors = torch.empty(B, k, device=points.device, dtype=torch.long)
    batch = torch.arange(B, device=points.device)

    # A fixed initial vertex keeps the sampling strategy independent of RNG.
    anchors[:, 0] = 0
    selected = xyz[:, 0]
    min_dist_sq = (xyz - selected[:, None, :]).square().sum(dim=-1)

    for i in range(1, k):
        farthest = min_dist_sq.argmax(dim=1)
        anchors[:, i] = farthest
        selected = xyz[batch, farthest]
        dist_sq = (xyz - selected[:, None, :]).square().sum(dim=-1)
        min_dist_sq = torch.minimum(min_dist_sq, dist_sq)

    return anchors


class LocalSA_Block(nn.Module):
    """
    Local KNN self-attention inside one object.
    Uses cosine attention (L2-normalized Q/K) + forced FlashAttention on CUDA.
    """

    def __init__(
        self,
        d_model: int,
        nhead: int,
        dropout: float = 0.1,
        droppath: float = 0.1,
        gamma_init: float = 1e-3,
        query_chunk: int = 256,
    ):
        super().__init__()
        assert d_model % nhead == 0
        if query_chunk <= 0:
            raise ValueError(f"query_chunk must be positive, got {query_chunk}")
        self.nhead = nhead
        self.hd = d_model // nhead
        self.query_chunk = int(query_chunk)

        self.ln = nn.LayerNorm(d_model)
        self.q = nn.Linear(d_model, d_model, bias=False)
        self.k = nn.Linear(d_model, d_model, bias=False)
        self.v = nn.Linear(d_model, d_model, bias=False)
        self.o = nn.Linear(d_model, d_model, bias=False)
        self.drop = nn.Dropout(dropout)

        self.gamma = nn.Parameter(torch.ones(d_model) * gamma_init)
        self.drop_res = DropPath(droppath)

    @staticmethod
    def _l2n(x: torch.Tensor, eps: float = 1e-6) -> torch.Tensor:
        return x / (x.norm(dim=-1, keepdim=True) + eps)

    def forward(self, x: torch.Tensor, idx: torch.Tensor) -> torch.Tensor:
        """
        x:   (B, V, D)
        idx: (B, V, k)
        """
        B, V, D = x.shape
        h, hd = self.nhead, self.hd
        k = idx.size(-1)

        # A single-vertex object has no valid neighbor. Keep the local
        # attention residual as an identity instead of invoking SDPA with an
        # empty key/value sequence.
        if k == 0:
            return x

        x_ln = self.ln(x)

        # (B,h,V,hd)
        Q = self.q(x_ln).view(B, V, h, hd).permute(0, 2, 1, 3).contiguous()
        K = self.k(x_ln).view(B, V, h, hd).permute(0, 2, 1, 3).contiguous()
        Vv = self.v(x_ln).view(B, V, h, hd).permute(0, 2, 1, 3).contiguous()

        Bh = B * h
        Q = Q.view(Bh, V, hd)
        K = K.view(Bh, V, hd)
        Vv = Vv.view(Bh, V, hd)

        bh = torch.arange(Bh, device=x.device)[:, None, None]  # (Bh,1,1)

        # Only materialize neighbor K/V for one query-vertex chunk at a time.
        # Q/K/V projections remain shared across all chunks, so this changes
        # neither parameters nor attention neighborhoods.
        out_chunks: List[torch.Tensor] = []
        for start in range(0, V, self.query_chunk):
            end = min(start + self.query_chunk, V)
            query_vertices = end - start

            # (B,q,k) -> (Bh,q,k), without expanding indices for all V.
            idx_h = (
                idx[:, start:end]
                .unsqueeze(1)
                .expand(B, h, query_vertices, k)
                .reshape(Bh, query_vertices, k)
            )
            Knb = K[bh, idx_h]   # (Bh,q,k,hd)
            Vnb = Vv[bh, idx_h]  # (Bh,q,k,hd)

            # Cosine attention. The original heads are folded into batch, so
            # the explicit FlashAttention head dimension is one.
            Qn = self._l2n(Q[:, start:end])
            Kn = self._l2n(Knb)
            q_sdpa = (Qn * math.sqrt(hd)).reshape(Bh * query_vertices, 1, 1, hd)
            k_sdpa = Kn.reshape(Bh * query_vertices, 1, k, hd)
            v_sdpa = Vnb.reshape(Bh * query_vertices, 1, k, hd)

            out_chunk = _flash_attention(q_sdpa, k_sdpa, v_sdpa)
            out_chunks.append(out_chunk.reshape(Bh, query_vertices, hd))

        out = torch.cat(out_chunks, dim=1)
        out = out.view(B, h, V, hd).permute(0, 2, 1, 3).reshape(B, V, D)
        out = self.o(self.drop(out))
        return x + self.drop_res(self.gamma * out)


class LocalCrossMHA(nn.Module):
    """
    Cross-attention: each query point attends to k selected points in another object.
    Uses cosine attention + forced FlashAttention on CUDA.
    """

    def __init__(self, d_model: int, nhead: int, dropout: float = 0.1):
        super().__init__()
        assert d_model % nhead == 0
        self.nhead = nhead
        self.hd = d_model // nhead

        self.ln_q = nn.LayerNorm(d_model)
        self.ln_kv = nn.LayerNorm(d_model)
        self.q = nn.Linear(d_model, d_model, bias=False)
        self.k = nn.Linear(d_model, d_model, bias=False)
        self.v = nn.Linear(d_model, d_model, bias=False)
        # Bias-free output projection makes source aggregation exactly linear:
        # mean(o(x_j)) == o(mean(x_j)). Existing checkpoints use this layout.
        self.o = nn.Linear(d_model, d_model, bias=False)
        self.drop = nn.Dropout(dropout)

    @staticmethod
    def _l2n(x: torch.Tensor, eps: float = 1e-6) -> torch.Tensor:
        return x / (x.norm(dim=-1, keepdim=True) + eps)

    def project_q(self, q_in: torch.Tensor) -> torch.Tensor:
        """Project one object's queries once for reuse across source objects."""
        B, Nq, D = q_in.shape
        h, hd = self.nhead, self.hd
        q_ln = self.ln_q(q_in)
        Q = self.q(q_ln).view(B, Nq, h, hd).permute(0, 2, 1, 3).contiguous()
        return self._l2n(Q)  # (B,h,Nq,hd)

    def project_kv(self, kv_in: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """Project one object's keys/values once for reuse across target objects."""
        B, Nk, D = kv_in.shape
        h, hd = self.nhead, self.hd
        kv_ln = self.ln_kv(kv_in)
        K = self.k(kv_ln).view(B, Nk, h, hd).permute(0, 2, 1, 3).contiguous()
        Vv = self.v(kv_ln).view(B, Nk, h, hd).permute(0, 2, 1, 3).contiguous()
        return K, Vv  # each (B,h,Nk,hd)

    def attend_projected_pairs(
        self,
        Qn: torch.Tensor,
        K_sources: torch.Tensor,
        V_sources: torch.Tensor,
        idx_sources: torch.Tensor,
        project_output: bool = True,
    ) -> torch.Tensor:
        """
        Attend from one target object to P source objects in one batched call.

        Qn:          (B,h,Nq,hd), already L2-normalized
        K_sources:   (B,P,h,Nk,hd)
        V_sources:   (B,P,h,Nk,hd)
        idx_sources: (B,P,Nq,k)
        return:      (B,P,Nq,D)
        """
        B, h, Nq, hd = Qn.shape
        Bk, P, hk, Nk, hdk = K_sources.shape
        if (Bk, hk, hdk) != (B, h, hd):
            raise ValueError(
                "Projected cross-attention shape mismatch: "
                f"Q={Qn.shape}, K={K_sources.shape}"
            )
        if V_sources.shape != K_sources.shape:
            raise ValueError(f"K/V shape mismatch: K={K_sources.shape}, V={V_sources.shape}")
        if idx_sources.shape[:3] != (B, P, Nq):
            raise ValueError(
                "Cross-index shape mismatch: "
                f"expected prefix {(B, P, Nq)}, got {idx_sources.shape}"
            )

        k = idx_sources.size(-1)
        BPh = B * P * h

        # Pair is folded into batch so all P source-object attentions share one
        # FlashAttention launch while retaining independent softmax operations.
        Q_pair = Qn[:, None].expand(B, P, h, Nq, hd).reshape(BPh, Nq, hd)
        K_pair = K_sources.reshape(BPh, Nk, hd)
        V_pair = V_sources.reshape(BPh, Nk, hd)
        idx_h = idx_sources[:, :, None].expand(B, P, h, Nq, k).reshape(BPh, Nq, k)
        bph = torch.arange(BPh, device=Qn.device)[:, None, None]

        Knb = K_pair[bph, idx_h]   # (BPh,Nq,k,hd)
        Vnb = V_pair[bph, idx_h]   # (BPh,Nq,k,hd)
        Kn = self._l2n(Knb)

        q_sdpa = (Q_pair * math.sqrt(hd)).reshape(BPh * Nq, 1, 1, hd)
        k_sdpa = Kn.reshape(BPh * Nq, 1, k, hd)
        v_sdpa = Vnb.reshape(BPh * Nq, 1, k, hd)
        out = _flash_attention(q_sdpa, k_sdpa, v_sdpa)

        out = out.reshape(B, P, h, Nq, hd)
        out = out.permute(0, 1, 3, 2, 4).reshape(B, P, Nq, h * hd)
        out = self.drop(out)
        return self.o(out) if project_output else out

    def attend_projected_anchors(
        self,
        Qn: torch.Tensor,
        K_sources: torch.Tensor,
        V_sources: torch.Tensor,
        anchor_indices: torch.Tensor,
        project_output: bool = True,
    ) -> torch.Tensor:
        """
        Attend to one shared anchor set per source object without expanding
        anchors over every query point.

        Qn:             (B,h,Nq,hd), already L2-normalized
        K/V_sources:    (B,P,h,Nk,hd)
        anchor_indices: (k,) for a shared set, or (B,P,k)
        return:         (B,P,Nq,D)

        The old implementation materialized K/V as (B*P*h,Nq,k,hd) and ran
        Nq independent length-1 queries.  Source anchors do not depend on the
        query, so this is exactly the same attention expressed as standard
        Q=(Nq,hd), K/V=(k,hd) SDPA sequences.
        """
        B, h, Nq, hd = Qn.shape
        Bk, P, hk, Nk, hdk = K_sources.shape
        if (Bk, hk, hdk) != (B, h, hd):
            raise ValueError(
                "Projected anchor-attention shape mismatch: "
                f"Q={Qn.shape}, K={K_sources.shape}"
            )
        if V_sources.shape != K_sources.shape:
            raise ValueError(f"K/V shape mismatch: K={K_sources.shape}, V={V_sources.shape}")

        anchors = anchor_indices.to(device=Qn.device, dtype=torch.long)
        if anchors.ndim == 1:
            if anchors.numel() == 0:
                raise ValueError("anchor_indices must not be empty")
            K_anchor = K_sources.index_select(3, anchors)
            V_anchor = V_sources.index_select(3, anchors)
        elif anchors.ndim == 3:
            if anchors.shape[:2] != (B, P) or anchors.size(-1) == 0:
                raise ValueError(
                    f"Expected batched anchors (B,P,k) with prefix {(B, P)}, "
                    f"got {anchors.shape}"
                )
            gather_idx = anchors[:, :, None, :, None].expand(B, P, h, -1, hd)
            K_anchor = K_sources.gather(3, gather_idx)
            V_anchor = V_sources.gather(3, gather_idx)
        else:
            raise ValueError(
                f"anchor_indices must be (k,) or (B,P,k), got {anchors.shape}"
            )

        Kn = self._l2n(K_anchor)
        return self.attend_projected_anchor_values(
            Qn, Kn, V_anchor, project_output=project_output
        )

    def attend_projected_anchor_values(
        self,
        Qn: torch.Tensor,
        Kn_anchor: torch.Tensor,
        V_anchor: torch.Tensor,
        project_output: bool = True,
    ) -> torch.Tensor:
        """Attend to already-selected, normalized anchor keys.

        Qn:        (B,h,Nq,hd), normalized queries
        Kn_anchor: (B,P,h,k,hd), normalized anchor keys
        V_anchor:  (B,P,h,k,hd), anchor values
        """
        B, h, Nq, hd = Qn.shape
        Bk, P, hk, k, hdk = Kn_anchor.shape
        if (Bk, hk, hdk) != (B, h, hd):
            raise ValueError(
                "Projected anchor-value shape mismatch: "
                f"Q={Qn.shape}, K={Kn_anchor.shape}"
            )
        if V_anchor.shape != Kn_anchor.shape:
            raise ValueError(
                f"Anchor K/V shape mismatch: K={Kn_anchor.shape}, V={V_anchor.shape}"
            )

        # Preserve cosine attention: SDPA applies 1/sqrt(hd), so scaling Q by
        # sqrt(hd) yields the same logits as the previous implementation.
        Q_pair = Qn[:, None].expand(B, P, h, Nq, hd).reshape(B * P, h, Nq, hd)
        K_pair = Kn_anchor.reshape(B * P, h, k, hd)
        V_pair = V_anchor.reshape(B * P, h, k, hd)

        out = _flash_attention(Q_pair * math.sqrt(hd), K_pair, V_pair)
        out = out.reshape(B, P, h, Nq, hd)
        out = out.permute(0, 1, 3, 2, 4).reshape(B, P, Nq, h * hd)
        out = self.drop(out)
        return self.o(out) if project_output else out

    def forward(self, q_in: torch.Tensor, kv_in: torch.Tensor, idx: torch.Tensor) -> torch.Tensor:
        """
        q_in:  (B, Nq, D)
        kv_in: (B, Nk, D)
        idx:   (B, Nq, k) indices into kv_in's Nk
        """
        Qn = self.project_q(q_in)
        K, Vv = self.project_kv(kv_in)
        out = self.attend_projected_pairs(
            Qn,
            K[:, None],
            Vv[:, None],
            idx[:, None],
        )
        return out[:, 0]


class SwiGLU(nn.Module):
    def __init__(self, d_model: int, mult: int = 4, dropout: float = 0.0, bias: bool = False):
        super().__init__()
        h = int(mult * d_model)
        self.w1 = nn.Linear(d_model, h, bias=bias)
        self.w2 = nn.Linear(d_model, h, bias=bias)
        self.w3 = nn.Linear(h, d_model, bias=bias)
        self.drop = nn.Dropout(dropout)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = F.silu(self.w1(x)) * self.w2(x)
        return self.w3(self.drop(x))


class FFN_Block(nn.Module):
    def __init__(self, d_model: int, mult: int = 4, dropout: float = 0.1, droppath: float = 0.1, gamma_init: float = 1e-3):
        super().__init__()
        self.ln = nn.LayerNorm(d_model)
        self.ffn = SwiGLU(d_model, mult, dropout=dropout)
        self.drop = nn.Dropout(dropout)

        self.gamma = nn.Parameter(torch.ones(d_model) * gamma_init)
        self.drop_res = DropPath(droppath)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        out = self.ffn(self.ln(x))
        return x + self.drop_res(self.gamma * out)

class SplitConditionEncoder(nn.Module):
    """
    Per-point encoder (no tokens):

      D split:
        - D/2  : rel_dir  = [x - center]              (3 dims input)
        - D/16 : rad      = [||x-center||]            (1 dim input)
        - D/8  : center   = center                    (3 dims input, broadcast to V)
        - D/4  : obj_cond = vel(3) + force(3) + rho(1) + mu(3) => 10 dims (broadcast to V)
        - D/16 : global   = [ft,fspin,froll,gz]       (4 dims, broadcast to V)

    Each part: Linear + GELU + LN, concat -> D, then Linear + LN.
    """

    def __init__(self, d_model: int, knn_k: int, knn_chunk: int):
        super().__init__()
        assert d_model % 16 == 0, f"d_model must be divisible by 16, got {d_model}"

        self.knn_k = int(knn_k)
        self.knn_chunk = int(knn_chunk)

        d_rel = d_model // 2
        d_rad = d_model // 16
        d_cen = d_model // 8
        d_obj = d_model // 4
        d_glb = d_model // 16

        d_rel_hidden = d_rel  

        self.rel_proj = nn.Sequential(
            nn.Linear(3, d_rel_hidden, bias=True),
            nn.GELU(),
            nn.Linear(d_rel_hidden, d_rel, bias=True),
            nn.GELU(),
            nn.LayerNorm(d_rel),
        )
        self.rad_proj = nn.Sequential(nn.Linear(1, d_rad), nn.GELU(), nn.LayerNorm(d_rad))
        self.cen_proj = nn.Sequential(nn.Linear(3, d_cen), nn.GELU(), nn.LayerNorm(d_cen))
        self.obj_proj = nn.Sequential(nn.Linear(10, d_obj), nn.GELU(), nn.LayerNorm(d_obj))
        self.glb_proj = nn.Sequential(nn.Linear(4, d_glb), nn.GELU(), nn.LayerNorm(d_glb))

        self.fuse = nn.Linear(d_model, d_model)
        self.ln = nn.LayerNorm(d_model)

    def forward(
        self,
        X0: torch.Tensor,                 # (B,V,3)
        vel: torch.Tensor,                # (B,3)
        force: torch.Tensor,              # (B,3)
        rho: torch.Tensor,                # (B,1) or (B,)
        friction: torch.Tensor,           # (B,3) or (B,1,3)
        global_cond: torch.Tensor,        # (B,4) == [ft,fspin,froll,gz]
        idx_local: Optional[torch.Tensor] = None,  # (B,V,k)
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        B, V, _ = X0.shape
        device = X0.device
        dtype = X0.dtype

        center = X0.mean(dim=1, keepdim=True)        # (B,1,3)
        rel = X0 - center                            # (B,V,3)

        # radius per point
        r = torch.linalg.norm(rel, dim=-1, keepdim=True)  # (B,V,1)

        if idx_local is None:
            idx_local = knn_idx_chunked(X0, k=self.knn_k, chunk=self.knn_chunk)

        rho = rho.to(device=device, dtype=dtype).view(B, 1)

        fr = friction.to(device=device, dtype=dtype)
        if fr.ndim == 3 and fr.size(1) == 1 and fr.size(2) == 3:
            fr = fr[:, 0, :]
        if not (fr.ndim == 2 and fr.size(1) == 3):
            raise ValueError(f"[Encoder] friction must be (B,3) or (B,1,3), got {friction.shape}")

        vel = vel.to(device=device, dtype=dtype)
        force = force.to(device=device, dtype=dtype)
        global_cond = global_cond.to(device=device, dtype=dtype).view(B, 4)

        cen_full = center.expand(B, V, 3)  # (B,V,3)
        obj_cond = torch.cat([vel, force, rho, fr], dim=-1)  # (B,10)
        obj_full = obj_cond[:, None, :].expand(-1, V, -1)    # (B,V,10)
        glb_full = global_cond[:, None, :].expand(-1, V, -1) # (B,V,4)

        h_rel = self.rel_proj(rel)     # (B,V,D/2)
        h_rad = self.rad_proj(r)       # (B,V,D/16)
        h_cen = self.cen_proj(cen_full)# (B,V,D/8)
        h_obj = self.obj_proj(obj_full)# (B,V,D/4)
        h_glb = self.glb_proj(glb_full)# (B,V,D/16)

        H = torch.cat([h_rel, h_rad, h_cen, h_obj, h_glb], dim=-1)  # (B,V,D)
        H = self.ln(self.fuse(H))
        return H, center, rel, idx_local



class CrossLayerMulti(nn.Module):
    """
    One post-layer: cross-attn (i <- j for all j!=i) + local SA + FFN.
    """

    def __init__(
        self,
        d_model: int,
        nhead: int,
        ff_mult: int,
        dropout: float,
        droppath: float,
        gamma_init: float,
        num_anchors: int,
        pair_chunk: int = 8,
        local_query_chunk: int = 256,
    ):
        super().__init__()
        if pair_chunk <= 0:
            raise ValueError(f"pair_chunk must be positive, got {pair_chunk}")
        if num_anchors <= 0:
            raise ValueError(f"num_anchors must be positive, got {num_anchors}")
        self.pair_chunk = int(pair_chunk)
        self.num_anchors = int(num_anchors)
        self.cross = LocalCrossMHA(d_model, nhead, dropout=dropout)
        self.sa = LocalSA_Block(
            d_model,
            nhead,
            dropout=dropout,
            droppath=droppath,
            gamma_init=gamma_init,
            query_chunk=local_query_chunk,
        )
        self.ff = FFN_Block(d_model, mult=ff_mult, dropout=dropout, droppath=droppath, gamma_init=gamma_init)

        self.gamma = nn.Parameter(torch.ones(d_model) * gamma_init)
        self.drop_res = DropPath(droppath)

    def forward(
        self,
        H_list: List[torch.Tensor],
        idx_local_list: List[torch.Tensor],
        anchor_indices_list: List[torch.Tensor],
    ) -> List[torch.Tensor]:
        M = len(H_list)
        out_list: List[torch.Tensor] = []

        # Every object's projections are shared by all ordered object pairs in
        # this post layer. Previously each projection was recomputed M-1 times.
        Q_list = [self.cross.project_q(H) for H in H_list]
        KV_list = [self.cross.project_kv(H) for H in H_list]
        # Select and normalize each source object's anchors once per layer.
        # They are reused for all M-1 target objects in this layer.
        anchor_KV_list: List[Tuple[torch.Tensor, torch.Tensor]] = []
        for j, (K, Vv) in enumerate(KV_list):
            anchors = anchor_indices_list[j].to(device=K.device, dtype=torch.long)
            if anchors.ndim == 1:
                K_anchor = K.index_select(2, anchors)
                V_anchor = Vv.index_select(2, anchors)
            elif anchors.ndim == 2:
                if anchors.size(0) != K.size(0) or anchors.size(1) == 0:
                    raise ValueError(
                        f"Expected batched anchors (B,k) with B={K.size(0)}, "
                        f"got {anchors.shape}"
                    )
                gather_idx = anchors[:, None, :, None].expand(
                    -1, K.size(1), -1, K.size(3)
                )
                K_anchor = K.gather(2, gather_idx)
                V_anchor = Vv.gather(2, gather_idx)
            else:
                raise ValueError(
                    f"anchor indices must be (k,) or (B,k), got {anchors.shape}"
                )
            anchor_KV_list.append((self.cross._l2n(K_anchor), V_anchor))

        for i in range(M):
            Hi = H_list[i]
            cross_sum = None
            cnt = 0
            source_ids = [j for j in range(M) if j != i]

            for start in range(0, len(source_ids), self.pair_chunk):
                group_ids = source_ids[start:start + self.pair_chunk]
                K_group = torch.stack(
                    [anchor_KV_list[j][0] for j in group_ids], dim=1
                )
                V_group = torch.stack(
                    [anchor_KV_list[j][1] for j in group_ids], dim=1
                )
                pair_out = self.cross.attend_projected_anchor_values(
                    Q_list[i], K_group, V_group, project_output=False
                )
                chunk_sum = pair_out.sum(dim=1)
                cross_sum = chunk_sum if cross_sum is None else cross_sum + chunk_sum
                cnt += len(group_ids)

            if cnt > 0:
                # Output projection is linear, so project once after averaging
                # all sources instead of once for each of the M-1 pairs.
                cross_out = self.cross.o(cross_sum / float(cnt))
                Hi = Hi + self.drop_res(self.gamma * cross_out)

            Hi = self.sa(Hi, idx_local_list[i])
            Hi = self.ff(Hi)
            out_list.append(Hi)

        return out_list


class CenterDecoder(nn.Module):
    """
    Mean-pooling over points -> (B,D), then MLP -> delta_c (B,3)
    """
    def __init__(self, d_model: int):
        super().__init__()
        self.mlp = nn.Sequential(
            nn.LayerNorm(d_model),
            nn.Linear(d_model, d_model),
            nn.GELU(),
            nn.Linear(d_model, 3),
        )

    def forward(self, H: torch.Tensor) -> torch.Tensor:
        # H: (B,V,D)
        pooled = H.mean(dim=1)  # (B,D)
        return self.mlp(pooled)  # (B,3)


class RelDecoder(nn.Module):
    """
    Per-point -> delta_r (B,V,3)
    """
    def __init__(self, d_model: int):
        super().__init__()
        self.mlp = nn.Sequential(
            nn.LayerNorm(d_model),
            nn.Linear(d_model, d_model),
            nn.GELU(),
            nn.Linear(d_model, 3),
        )

    def forward(self, H: torch.Tensor) -> torch.Tensor:
        return self.mlp(H)


class Predictor(nn.Module):
    """
    No center_broadcast / no inject / no self_token / no pair_token.

    Pipeline:
      - Encode each object into per-point features (rel+center+obj_cond+global_cond).
      - Pre layers: local SA + FFN (pure intra-object).
      - Post layers: cross-attn among objects + local SA + FFN.
      - Decode:
          * CenterDecoder: mean-pool(H) -> delta_c
          * RelDecoder:    H -> delta_r, then enforce zero-mean over V
        Final:
          X_hat = X0 + delta_c[:,None,:] + (delta_r - mean(delta_r))
    """

    def __init__(
        self,
        d_model: int = 512,
        nhead: int = 8,
        pre_layers: int = 4,
        post_layers: int = 4,
        ff_mult: int = 4,
        dropout: float = 0.0,
        knn_k: int = 32,
        num_anchors: Optional[int] = None,
        knn_chunk: int = 1024,
        cross_pair_chunk: int = 8,
        droppath_rate: float = 0.10,
        gamma_init: float = 1e-3,
        zero_init_residual: bool = False,
        zero_init_decoder: bool = True,
        local_query_chunk: int = 256,
    ):
        super().__init__()
        self.knn_k = int(knn_k)
        self.num_anchors = self.knn_k if num_anchors is None else int(num_anchors)
        if self.knn_k <= 0:
            raise ValueError(f"knn_k must be positive, got {self.knn_k}")
        if self.num_anchors <= 0:
            raise ValueError(
                f"num_anchors must be positive, got {self.num_anchors}"
            )
        self.knn_chunk = int(knn_chunk)
        self.cross_pair_chunk = int(cross_pair_chunk)
        if local_query_chunk <= 0:
            raise ValueError(
                f"local_query_chunk must be positive, got {local_query_chunk}"
            )
        self.local_query_chunk = int(local_query_chunk)

        self.enc = SplitConditionEncoder(d_model=d_model, knn_k=self.knn_k, knn_chunk=self.knn_chunk)

        self.pre = nn.ModuleList(
            [
                nn.ModuleDict(
                    {
                        "sa": LocalSA_Block(
                            d_model,
                            nhead,
                            dropout=dropout,
                            droppath=droppath_rate,
                            gamma_init=gamma_init,
                            query_chunk=self.local_query_chunk,
                        ),
                        "ff": FFN_Block(d_model, mult=ff_mult, dropout=dropout, droppath=droppath_rate, gamma_init=gamma_init),
                    }
                )
                for _ in range(int(pre_layers))
            ]
        )

        self.post = nn.ModuleList(
            [
                CrossLayerMulti(
                    d_model=d_model,
                    nhead=nhead,
                    ff_mult=ff_mult,
                    dropout=dropout,
                    droppath=droppath_rate,
                    gamma_init=gamma_init,
                    num_anchors=self.num_anchors,
                    pair_chunk=self.cross_pair_chunk,
                    local_query_chunk=self.local_query_chunk,
                )
                for _ in range(int(post_layers))
            ]
        )

        self.center_dec = CenterDecoder(d_model=d_model)
        self.rel_dec = RelDecoder(d_model=d_model)

        self._warn_local_once = False

        if zero_init_residual:
            self._zero_init_transformer_residuals()
        if zero_init_decoder:
            self._zero_init_decoders()

    def _zero_init_transformer_residuals(self) -> None:
        for m in self.modules():
            if isinstance(m, LocalSA_Block):
                nn.init.zeros_(m.o.weight)
            elif isinstance(m, LocalCrossMHA):
                nn.init.zeros_(m.o.weight)
                if m.o.bias is not None:
                    nn.init.zeros_(m.o.bias)
            elif isinstance(m, FFN_Block):
                nn.init.zeros_(m.ffn.w3.weight)
                if m.ffn.w3.bias is not None:
                    nn.init.zeros_(m.ffn.w3.bias)

    def _zero_init_decoders(self) -> None:
        for m in (self.center_dec, self.rel_dec):
            if isinstance(m, CenterDecoder):
                last = m.mlp[-1]
                if isinstance(last, nn.Linear):
                    nn.init.zeros_(last.weight)
                    if last.bias is not None:
                        nn.init.zeros_(last.bias)
            elif isinstance(m, RelDecoder):
                last = m.mlp[-1]
                if isinstance(last, nn.Linear):
                    nn.init.zeros_(last.weight)
                    if last.bias is not None:
                        nn.init.zeros_(last.bias)

    def _build_anchor_indices(self, X0: torch.Tensor) -> torch.Tensor:
        return farthest_point_anchor_indices(X0, self.num_anchors)

    def forward(
        self,
        X0_list: List[torch.Tensor],
        v0_list: List[torch.Tensor],
        f0_list: List[torch.Tensor],
        rho0_list: List[torch.Tensor],
        fric_list: List[torch.Tensor],
        ft: torch.Tensor,
        fspin: torch.Tensor,
        froll: torch.Tensor,
        gz: torch.Tensor,
        idx_local_list: Optional[List[torch.Tensor]] = None,
    ):
        """
        Notes:
          - idx_local_list[i] expected (B, Vi, k) if provided
          - Cross-attention anchors are always generated model-side; there is
            no cross-index input, cache, or host-to-device transfer.
        """
        M = len(X0_list)
        if M < 2:
            raise ValueError("Need at least two objects")

        B = X0_list[0].size(0)
        device = X0_list[0].device
        dtype = X0_list[0].dtype

        # global cond: (B,4)
        ft = ft.to(device=device, dtype=dtype).view(B, 1)
        fspin = fspin.to(device=device, dtype=dtype).view(B, 1)
        froll = froll.to(device=device, dtype=dtype).view(B, 1)
        gz = gz.to(device=device, dtype=dtype).view(B, 1)
        global_cond = torch.cat([ft, fspin, froll, gz], dim=1)

        # local idx
        if idx_local_list is None:
            idx_local_list = [None] * M
        elif len(idx_local_list) != M:
            raise ValueError(
                f"Expected {M} local-index tensors, got {len(idx_local_list)}"
            )
        for i in range(M):
            idx_local = idx_local_list[i]
            missing_local_idx = (
                not isinstance(idx_local, torch.Tensor)
                or idx_local.ndim == 0
                or idx_local.size(-1) == 0
            )
            if missing_local_idx:
                if not self._warn_local_once:
                    warnings.warn(
                        "[Predictor] local KNN indices are missing or empty; "
                        "computing them on-the-fly."
                    )
                    self._warn_local_once = True
                idx_local_list[i] = knn_idx_chunked(X0_list[i], k=self.knn_k, chunk=self.knn_chunk)
            else:
                idx_local = idx_local.to(device, non_blocking=True)
                idx_local_list[i] = idx_local[..., :self.knn_k]

        # Compute anchor indices once per source object and reuse them across
        # every target object and every post-attention layer.
        anchor_indices_list = [
            self._build_anchor_indices(X0)
            for X0 in X0_list
        ]

        # encode per object
        H_list: List[torch.Tensor] = []
        centers0: List[torch.Tensor] = []
        for i in range(M):
            rho = rho0_list[i].to(device=device, dtype=dtype)
            fr = fric_list[i].to(device=device, dtype=dtype)

            Hi, c0, _r0, idxi = self.enc(
                X0=X0_list[i],
                vel=v0_list[i],
                force=f0_list[i],
                rho=rho,
                friction=fr,
                global_cond=global_cond,
                idx_local=idx_local_list[i],
            )
            H_list.append(Hi)
            centers0.append(c0)
            idx_local_list[i] = idxi

        # pre: intra-object only
        for layer in self.pre:
            new_list: List[torch.Tensor] = []
            for i in range(M):
                Hi = layer["sa"](H_list[i], idx_local_list[i])
                Hi = layer["ff"](Hi)
                new_list.append(Hi)
            H_list = new_list

        # post: cross + refine
        for layer in self.post:
            H_list = layer(
                H_list,
                idx_local_list,
                anchor_indices_list,
            )

        # decode: delta_c + delta_r
        X_hat_list: List[torch.Tensor] = []
        dC_list: List[torch.Tensor] = []
        dR_list: List[torch.Tensor] = []

        for i in range(M):
            dC = self.center_dec(H_list[i])            # (B,3)
            dR = self.rel_dec(H_list[i])               # (B,Vi,3)
            dR = dR - dR.mean(dim=1, keepdim=True)     # enforce zero-mean over points

            X_hat = X0_list[i] + dC[:, None, :] + dR   # (B,Vi,3)

            dC_list.append(dC)
            dR_list.append(dR)
            X_hat_list.append(X_hat)

        aux = {
            "centers0": centers0,
            "idx_local_list": idx_local_list,
            "global_cond": global_cond,
            "dC_list": dC_list,
            "dR_list": dR_list,
        }
        return X_hat_list, aux
