"""Rigid-Pose MLP and its rigid-transform utilities."""

from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F


def object_numeric_features(
    x0_list: list[torch.Tensor], v0_list: list[torch.Tensor],
    f0_list: list[torch.Tensor], rho_list: list[torch.Tensor],
    fric_list: list[torch.Tensor], ft: torch.Tensor, fspin: torch.Tensor,
    froll: torch.Tensor, gz: torch.Tensor,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Build (B,M,30) geometric/physical features and centers (B,M,3)."""
    centers = [x.mean(dim=1) for x in x0_list]
    center_stack = torch.stack(centers, dim=1)
    scene_center = center_stack.mean(dim=1, keepdim=True)
    global_cond = torch.cat(
        [ft.view(ft.size(0), 1), fspin.view(ft.size(0), 1),
         froll.view(ft.size(0), 1), gz.view(gz.size(0), 1)], dim=1
    )
    feature_list = []
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
        relative_center = center - scene_center[:, 0, :]
        feature_list.append(torch.cat(
            [center, relative_center, extent, rms_radius, covariance_upper,
             velocity, force, density.reshape(x0.size(0), -1), friction, global_cond], dim=1
        ))
    return torch.stack(feature_list, dim=1), center_stack


def rotation6d_to_matrix(rotation6d: torch.Tensor) -> torch.Tensor:
    first = F.normalize(rotation6d[..., 0:3], dim=-1)
    second_raw = rotation6d[..., 3:6]
    second = F.normalize(
        second_raw - (first * second_raw).sum(dim=-1, keepdim=True) * first, dim=-1
    )
    third = torch.cross(first, second, dim=-1)
    return torch.stack([first, second, third], dim=-1)


def transform_points(
    x0_list: list[torch.Tensor], center_delta: torch.Tensor,
    rotations: torch.Tensor,
) -> list[torch.Tensor]:
    predictions = []
    for index, x0 in enumerate(x0_list):
        center = x0.mean(dim=1, keepdim=True)
        predictions.append(
            (x0 - center) @ rotations[:, index] + center
            + center_delta[:, index, None, :]
        )
    return predictions


class MLPBlock(nn.Module):
    def __init__(self, width: int, dropout: float) -> None:
        super().__init__()
        self.norm = nn.LayerNorm(width)
        self.fc1 = nn.Linear(width, width * 2)
        self.fc2 = nn.Linear(width * 2, width)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        update = self.fc2(F.gelu(self.fc1(self.norm(x))))
        return x + self.dropout(update)


class RigidPoseMLP(nn.Module):
    """Rigid-Pose MLP with pooled scene context and no attention."""

    def __init__(
        self, num_objects: int, input_dim: int = 30, width: int = 256,
        depth: int = 1, object_embedding_dim: int = 16, dropout: float = 0.0,
    ) -> None:
        super().__init__()
        self.num_objects = int(num_objects)
        self.object_embedding = nn.Embedding(num_objects, object_embedding_dim)
        self.object_encoder = nn.Sequential(
            nn.LayerNorm(input_dim + object_embedding_dim), nn.Linear(input_dim + object_embedding_dim, width),
            nn.GELU(), nn.Dropout(dropout),
        )
        self.context_encoder = nn.Sequential(
            nn.LayerNorm(width * 2), nn.Linear(width * 2, width), nn.GELU(), nn.Dropout(dropout)
        )
        self.input_projection = nn.Linear(width * 2, width)
        self.trunk = nn.ModuleList([MLPBlock(width, dropout) for _ in range(depth)])
        self.center_head = nn.Sequential(
            nn.LayerNorm(width), nn.Linear(width, width), nn.GELU(), nn.Linear(width, 3)
        )
        self.rotation_head = nn.Sequential(
            nn.LayerNorm(width), nn.Linear(width, width), nn.GELU(), nn.Linear(width, 6)
        )
        nn.init.zeros_(self.center_head[-1].weight)
        nn.init.zeros_(self.center_head[-1].bias)
        nn.init.zeros_(self.rotation_head[-1].weight)
        with torch.no_grad():
            self.rotation_head[-1].bias.copy_(torch.tensor([1.0, 0.0, 0.0, 0.0, 1.0, 0.0]))

    def forward(self, features: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        batch_size, num_objects, _ = features.shape
        ids = torch.arange(num_objects, device=features.device).unsqueeze(0).expand(batch_size, -1)
        encoded = self.object_encoder(torch.cat([features, self.object_embedding(ids)], dim=-1))
        context = self.context_encoder(torch.cat([encoded.mean(dim=1), encoded.amax(dim=1)], dim=-1))
        hidden = self.input_projection(torch.cat(
            [encoded, context[:, None, :].expand(-1, num_objects, -1)], dim=-1
        ))
        for block in self.trunk:
            hidden = block(hidden)
        return self.center_head(hidden), rotation6d_to_matrix(self.rotation_head(hidden))
