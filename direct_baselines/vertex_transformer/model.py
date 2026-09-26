"""Vertex Transformer baseline and shared endpoint metrics."""

from __future__ import annotations

import math
from typing import Any

import torch
import torch.nn as nn
import torch.nn.functional as F


def object_numeric_features(
    x0_list: list[torch.Tensor], v0_list: list[torch.Tensor],
    f0_list: list[torch.Tensor], rho_list: list[torch.Tensor],
    fric_list: list[torch.Tensor], ft: torch.Tensor, fspin: torch.Tensor,
    froll: torch.Tensor, gz: torch.Tensor,
) -> torch.Tensor:
    """Build the same 30-D object condition used by Rigid-Pose MLP."""
    centers = [x.mean(dim=1) for x in x0_list]
    center_stack = torch.stack(centers, dim=1)
    scene_center = center_stack.mean(dim=1, keepdim=True)
    global_cond = torch.cat(
        [ft.view(ft.size(0), 1), fspin.view(ft.size(0), 1),
         froll.view(ft.size(0), 1), gz.view(gz.size(0), 1)], dim=1
    )
    features = []
    for center, x0, velocity, force, density, friction in zip(
        centers, x0_list, v0_list, f0_list, rho_list, fric_list
    ):
        centered = x0 - center[:, None, :]
        extent = x0.amax(dim=1) - x0.amin(dim=1)
        rms_radius = centered.square().mean(dim=(1, 2)).sqrt().unsqueeze(1)
        covariance = torch.einsum("bvi,bvj->bij", centered, centered) / float(x0.size(1))
        covariance_upper = torch.stack(
            [covariance[:, 0, 0], covariance[:, 0, 1], covariance[:, 0, 2],
             covariance[:, 1, 1], covariance[:, 1, 2], covariance[:, 2, 2]], dim=1
        )
        features.append(torch.cat(
            [center, center - scene_center[:, 0, :], extent, rms_radius,
             covariance_upper, velocity, force,
             density.reshape(x0.size(0), -1), friction, global_cond], dim=1
        ))
    return torch.stack(features, dim=1)


def point_geometry_features(x0: torch.Tensor, fourier_bands: int) -> torch.Tensor:
    """Encode centered point coordinates without using target geometry."""
    centered = x0 - x0.mean(dim=1, keepdim=True)
    scale = centered.square().sum(dim=-1).mean(dim=1, keepdim=True).sqrt().clamp_min(1e-6)
    normalized = centered / scale[:, :, None]
    radius = normalized.square().sum(dim=-1, keepdim=True).sqrt()
    features = [centered, normalized, radius]
    for band in range(fourier_bands):
        frequency = math.pi * float(2 ** band)
        features.extend([
            torch.sin(frequency * normalized),
            torch.cos(frequency * normalized),
        ])
    return torch.cat(features, dim=-1)


def gather_neighbors(values: torch.Tensor, indices: torch.Tensor) -> torch.Tensor:
    """Gather ``(B,Q,K,D)`` neighbor values from ``values=(B,V,D)``."""
    batch_size, num_vertices, width = values.shape
    if indices.ndim != 3 or indices.size(0) != batch_size:
        raise ValueError(
            f"indices must be (B,Q,K), got {tuple(indices.shape)} for B={batch_size}"
        )
    offsets = (
        torch.arange(batch_size, device=values.device, dtype=torch.long)
        * num_vertices
    )[:, None, None]
    flat_indices = (indices.long() + offsets).reshape(-1)
    return values.reshape(batch_size * num_vertices, width)[flat_indices].reshape(
        batch_size, indices.size(1), indices.size(2), width
    )


class VertexAttentionBlock(nn.Module):
    """KNN point self-attention with learned relative-position features."""

    def __init__(
        self, d_model: int, nhead: int, ff_mult: int,
        dropout: float, query_chunk: int,
    ) -> None:
        super().__init__()
        if d_model % nhead != 0:
            raise ValueError(f"d_model={d_model} must be divisible by nhead={nhead}")
        self.nhead = int(nhead)
        self.head_dim = d_model // nhead
        self.query_chunk = int(query_chunk)
        self.norm1 = nn.LayerNorm(d_model)
        self.q = nn.Linear(d_model, d_model, bias=False)
        self.k = nn.Linear(d_model, d_model, bias=False)
        self.v = nn.Linear(d_model, d_model, bias=False)
        self.position = nn.Sequential(
            nn.Linear(3, d_model // 2), nn.GELU(),
            nn.Linear(d_model // 2, d_model),
        )
        self.attention_bias = nn.Linear(d_model, nhead, bias=False)
        self.out = nn.Linear(d_model, d_model, bias=False)
        self.dropout = nn.Dropout(dropout)
        self.norm2 = nn.LayerNorm(d_model)
        hidden = d_model * ff_mult
        self.ffn = nn.Sequential(
            nn.Linear(d_model, hidden), nn.GELU(), nn.Dropout(dropout),
            nn.Linear(hidden, d_model), nn.Dropout(dropout),
        )

    def forward(
        self, tokens: torch.Tensor, coordinates: torch.Tensor,
        indices: torch.Tensor,
    ) -> torch.Tensor:
        if indices.ndim != 3 or indices.size(1) != tokens.size(1):
            raise ValueError(
                f"KNN indices must be (B,V,K), got {tuple(indices.shape)} "
                f"for tokens {tuple(tokens.shape)}"
            )
        normalized = self.norm1(tokens)
        q_all = self.q(normalized)
        k_all = self.k(normalized)
        v_all = self.v(normalized)
        outputs = []
        for start in range(0, tokens.size(1), self.query_chunk):
            end = min(tokens.size(1), start + self.query_chunk)
            local_indices = indices[:, start:end]
            q = q_all[:, start:end].reshape(
                tokens.size(0), end - start, self.nhead, self.head_dim
            )
            k = gather_neighbors(k_all, local_indices).reshape(
                tokens.size(0), end - start, local_indices.size(2),
                self.nhead, self.head_dim,
            )
            v = gather_neighbors(v_all, local_indices).reshape_as(k)
            neighbor_coordinates = gather_neighbors(coordinates, local_indices)
            relative = neighbor_coordinates - coordinates[:, start:end, None, :]
            position = self.position(relative).reshape_as(k)
            logits = (
                q[:, :, None] * (k + position)
            ).sum(dim=-1) / math.sqrt(float(self.head_dim))
            logits = logits + self.attention_bias(
                position.reshape(
                    tokens.size(0), end - start, local_indices.size(2), -1
                )
            )
            attention = F.softmax(logits.float(), dim=2).to(v.dtype)
            output = (attention[..., None] * (v + position)).sum(dim=2)
            outputs.append(output.reshape(tokens.size(0), end - start, -1))
        tokens = tokens + self.dropout(self.out(torch.cat(outputs, dim=1)))
        return tokens + self.ffn(self.norm2(tokens))


class VertexTransformer(nn.Module):
    """Hierarchical Vertex Transformer with object-token scene interaction.

    Point attention is restricted to cached intra-object KNN neighborhoods.
    Inter-object communication occurs only through pooled object tokens.  The
    decoder directly predicts per-point displacement and does not use the main
    model's point-level cross-object attention or dual-branch decoder.
    """

    def __init__(
        self, num_objects: int, d_model: int = 384, nhead: int = 6,
        pre_layers: int = 1, post_layers: int = 0, object_layers: int = 1,
        ff_mult: int = 4, object_embedding_dim: int = 16,
        fourier_bands: int = 3, dropout: float = 0.05,
        query_chunk: int = 512,
    ) -> None:
        super().__init__()
        if min(pre_layers, post_layers, object_layers) < 0:
            raise ValueError("Transformer layer counts must be non-negative")
        if query_chunk <= 0:
            raise ValueError("query_chunk must be positive")
        self.num_objects = int(num_objects)
        self.d_model = int(d_model)
        self.fourier_bands = int(fourier_bands)
        point_dim = 7 + 6 * fourier_bands
        self.object_embedding = nn.Embedding(num_objects, object_embedding_dim)
        self.point_encoder = nn.Sequential(
            nn.LayerNorm(point_dim), nn.Linear(point_dim, d_model), nn.GELU(),
        )
        self.condition_encoder = nn.Sequential(
            nn.LayerNorm(30 + object_embedding_dim),
            nn.Linear(30 + object_embedding_dim, d_model), nn.GELU(),
        )
        self.input_norm = nn.LayerNorm(d_model)
        block_args = (d_model, nhead, ff_mult, dropout, query_chunk)
        self.pre = nn.ModuleList([
            VertexAttentionBlock(*block_args) for _ in range(pre_layers)
        ])
        self.pool_projection = nn.Sequential(
            nn.LayerNorm(d_model * 2), nn.Linear(d_model * 2, d_model), nn.GELU(),
        )
        if object_layers > 0:
            object_layer = nn.TransformerEncoderLayer(
                d_model=d_model, nhead=nhead,
                dim_feedforward=d_model * ff_mult, dropout=dropout,
                activation="gelu", batch_first=True, norm_first=True,
            )
            self.object_transformer: nn.Module = nn.TransformerEncoder(
                object_layer, num_layers=object_layers,
                norm=nn.LayerNorm(d_model), enable_nested_tensor=False,
            )
        else:
            self.object_transformer = nn.Identity()
        self.context_projection = nn.Sequential(
            nn.LayerNorm(d_model), nn.Linear(d_model, d_model),
        )
        self.post = nn.ModuleList([
            VertexAttentionBlock(*block_args) for _ in range(post_layers)
        ])
        self.decoder = nn.Sequential(
            nn.LayerNorm(d_model), nn.Linear(d_model, d_model), nn.GELU(),
            nn.Linear(d_model, 3),
        )
        nn.init.zeros_(self.decoder[-1].weight)
        nn.init.zeros_(self.decoder[-1].bias)

    def forward(self, batch: dict[str, Any]) -> list[torch.Tensor]:
        x0_list = batch["x0"]
        idx_list = batch["idx"]
        num_objects = len(x0_list)
        if num_objects != self.num_objects:
            raise ValueError(
                f"Expected {self.num_objects} objects, received {num_objects}"
            )
        object_features = object_numeric_features(
            x0_list, batch["v0"], batch["f0"], batch["rho"], batch["fric"],
            batch["ft"], batch["fspin"], batch["froll"], batch["gz"],
        )
        batch_size = x0_list[0].size(0)
        object_ids = torch.arange(
            num_objects, device=x0_list[0].device
        ).unsqueeze(0).expand(batch_size, -1)
        conditions = self.condition_encoder(torch.cat(
            [object_features, self.object_embedding(object_ids)], dim=-1
        ))

        point_tokens = []
        for object_index, (x0, indices) in enumerate(zip(x0_list, idx_list)):
            geometry = point_geometry_features(x0, self.fourier_bands)
            tokens = self.input_norm(
                self.point_encoder(geometry) + conditions[:, object_index, None, :]
            )
            for block in self.pre:
                tokens = block(tokens, x0, indices)
            point_tokens.append(tokens)

        pooled = torch.stack([
            self.pool_projection(torch.cat(
                [tokens.mean(dim=1), tokens.amax(dim=1)], dim=-1
            ))
            for tokens in point_tokens
        ], dim=1)
        object_context = self.object_transformer(pooled)
        object_context = self.context_projection(object_context)

        predictions = []
        for object_index, (x0, indices, tokens) in enumerate(
            zip(x0_list, idx_list, point_tokens)
        ):
            tokens = tokens + object_context[:, object_index, None, :]
            for block in self.post:
                tokens = block(tokens, x0, indices)
            predictions.append(x0 + self.decoder(tokens))
        return predictions
