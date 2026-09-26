from __future__ import annotations

from pathlib import Path
import sys
from typing import Any

import torch
from torch import nn


FIGNET_ROOT = Path(__file__).resolve().parents[1]
if str(FIGNET_ROOT) not in sys.path:
    sys.path.insert(0, str(FIGNET_ROOT))

from fignet.simulator import LearnedSimulator


NORMALIZER_NAMES = (
    "_node_normalizer",
    "_regular_edge_normalizer",
    "_face_edge_normalizer",
    "_output_normalizer",
)


def build_model(config: dict[str, Any], device: torch.device) -> LearnedSimulator:
    return LearnedSimulator(
        mesh_dimensions=3,
        latent_dim=int(config["latent_dim"]),
        nmessage_passing_steps=int(config["message_passing_steps"]),
        nmlp_layers=int(config["mlp_layers"]),
        input_seq_length=int(config["input_sequence_length"]),
        property_dim=int(config["property_dim"]),
        mlp_hidden_dim=int(config["latent_dim"]),
        device=device,
        leave_out_mm=bool(config["leave_out_mm"]),
    ).to(device)


def set_normalizer_accumulation(model: LearnedSimulator, enabled: bool) -> None:
    maximum = 10**6 if enabled else 0
    for name in NORMALIZER_NAMES:
        getattr(model, name)._max_accumulations = maximum


def capture_normalizers(model: LearnedSimulator) -> dict[str, dict[str, Any]]:
    result: dict[str, dict[str, Any]] = {}
    for name in NORMALIZER_NAMES:
        state: dict[str, Any] = {}
        for key, value in getattr(model, name).get_variable().items():
            state[key] = value.detach().cpu() if isinstance(value, torch.Tensor) else value
        result[name] = state
    return result


def restore_normalizers(
    model: LearnedSimulator,
    state: dict[str, dict[str, Any]],
    device: torch.device,
) -> None:
    if set(state) != set(NORMALIZER_NAMES):
        raise ValueError("Incomplete FiGNet normalizer state")
    for name in NORMALIZER_NAMES:
        normalizer = getattr(model, name)
        for key, value in state[name].items():
            setattr(normalizer, key, value.to(device) if isinstance(value, torch.Tensor) else value)


def trainable_parameter_count(model: nn.Module) -> int:
    return sum(parameter.numel() for parameter in model.parameters() if parameter.requires_grad)
