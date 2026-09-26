from typing import List, Tuple

import torch


def _rotate_z_c4(values: torch.Tensor, quarter_turns: torch.Tensor) -> torch.Tensor:
    """Rotate batched 3D points or vectors around the world z axis."""
    shape = (values.size(0),) + (1,) * (values.ndim - 2)
    cos_values = values.new_tensor((1.0, 0.0, -1.0, 0.0))[quarter_turns].view(shape)
    sin_values = values.new_tensor((0.0, 1.0, 0.0, -1.0))[quarter_turns].view(shape)
    x = values[..., 0]
    y = values[..., 1]
    return torch.stack(
        (cos_values * x - sin_values * y,
         sin_values * x + cos_values * y,
         values[..., 2]),
        dim=-1,
    )


@torch.no_grad()
def apply_random_c4_rotation(
    X0_list: List[torch.Tensor],
    v0_list: List[torch.Tensor],
    f0_list: List[torch.Tensor],
    Y_list: List[torch.Tensor],
) -> Tuple[List[torch.Tensor], List[torch.Tensor], List[torch.Tensor], List[torch.Tensor]]:
    """Apply one shared random C4 rotation per scene in a training batch."""
    batch_size = X0_list[0].size(0)
    quarter_turns = torch.randint(
        0, 4, (batch_size,), device=X0_list[0].device, dtype=torch.long
    )
    return (
        [_rotate_z_c4(values, quarter_turns) for values in X0_list],
        [_rotate_z_c4(values, quarter_turns) for values in v0_list],
        [_rotate_z_c4(values, quarter_turns) for values in f0_list],
        [_rotate_z_c4(values, quarter_turns) for values in Y_list],
    )
