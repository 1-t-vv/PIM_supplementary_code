from __future__ import annotations

from pathlib import Path
import sys

import torch
from torch import nn
from torch_geometric.data import Data


MESHGRAPHNET_ROOT = Path(__file__).resolve().parents[1]
if str(MESHGRAPHNET_ROOT) not in sys.path:
    sys.path.insert(0, str(MESHGRAPHNET_ROOT))

from meshGraphNets_pytorch.model.model import EncoderProcesserDecoder


class RunningNormalizer(nn.Module):
    def __init__(self, size: int, epsilon: float = 1e-8) -> None:
        super().__init__()
        self.epsilon = float(epsilon)
        self.register_buffer("count", torch.tensor(0.0, dtype=torch.float64))
        self.register_buffer("sum", torch.zeros(size, dtype=torch.float64))
        self.register_buffer("sum_squared", torch.zeros(size, dtype=torch.float64))

    @torch.no_grad()
    def update(self, value: torch.Tensor) -> None:
        flattened = value.detach().reshape(-1, value.shape[-1]).to(torch.float64)
        self.count.add_(flattened.shape[0])
        self.sum.add_(flattened.sum(dim=0))
        self.sum_squared.add_((flattened * flattened).sum(dim=0))

    def statistics(self) -> tuple[torch.Tensor, torch.Tensor]:
        count = self.count.clamp_min(1.0)
        mean = self.sum / count
        variance = (self.sum_squared / count - mean * mean).clamp_min(0.0)
        std = variance.sqrt().clamp_min(self.epsilon)
        return mean.to(torch.float32), std.to(torch.float32)

    def normalize(self, value: torch.Tensor, accumulate: bool = False) -> torch.Tensor:
        if accumulate:
            self.update(value)
        mean, std = self.statistics()
        return (value - mean) / std

    def inverse(self, value: torch.Tensor) -> torch.Tensor:
        mean, std = self.statistics()
        return value * std + mean


class MeshGraphNetFlat(nn.Module):
    def __init__(
        self,
        node_input_size: int = 14,
        edge_input_size: int = 6,
        hidden_size: int = 128,
        message_passing_steps: int = 15,
        output_size: int = 3,
    ) -> None:
        super().__init__()
        if output_size != 3:
            raise ValueError("The Flat MGN adapter requires a three-dimensional output")
        self.node_input_size = int(node_input_size)
        self.edge_input_size = int(edge_input_size)
        self.output_size = int(output_size)
        self.node_normalizer = RunningNormalizer(self.node_input_size)
        self.edge_normalizer = RunningNormalizer(self.edge_input_size)
        self.output_normalizer = RunningNormalizer(self.output_size)
        self.network = EncoderProcesserDecoder(
            message_passing_num=int(message_passing_steps),
            node_input_size=self.node_input_size,
            edge_input_size=self.edge_input_size,
            hidden_size=int(hidden_size),
            output_size=self.output_size,
        )
        self.network.apply(self._init_weights)

    @staticmethod
    def _init_weights(module: nn.Module) -> None:
        if isinstance(module, nn.Linear):
            nn.init.xavier_uniform_(module.weight)
            if module.bias is not None:
                nn.init.zeros_(module.bias)

    def normalized_prediction(
        self,
        graph: Data,
        accumulate: bool,
    ) -> torch.Tensor:
        normalized = Data(
            x=self.node_normalizer.normalize(graph.x, accumulate=accumulate),
            edge_attr=self.edge_normalizer.normalize(
                graph.edge_attr, accumulate=accumulate
            ),
            edge_index=graph.edge_index,
        )
        return self.network(normalized)

    def training_pair(
        self,
        graph: Data,
        target_velocity_delta: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        prediction = self.normalized_prediction(graph, accumulate=True)
        target = self.output_normalizer.normalize(
            target_velocity_delta, accumulate=True
        )
        return prediction, target

    def predict_velocity_delta(self, graph: Data) -> torch.Tensor:
        prediction = self.normalized_prediction(graph, accumulate=False)
        return self.output_normalizer.inverse(prediction)


def build_model(config: dict[str, object]) -> MeshGraphNetFlat:
    return MeshGraphNetFlat(
        node_input_size=int(config["node_input_size"]),
        edge_input_size=int(config["edge_input_size"]),
        hidden_size=int(config["hidden_size"]),
        message_passing_steps=int(config["message_passing_steps"]),
        output_size=int(config["output_size"]),
    )


def trainable_parameter_count(model: nn.Module) -> int:
    return sum(parameter.numel() for parameter in model.parameters() if parameter.requires_grad)
