from __future__ import annotations
from typing import Any, Sequence
import torch
from torch import nn

class SquaredReLU(nn.Module):

    def forward(self, value: torch.Tensor) -> torch.Tensor:
        return torch.nn.functional.relu(value).square()

class ProjectorRMSNorm(nn.Module):

    def __init__(self, hidden_size: int, eps: float=1e-05) -> None:
        super().__init__()
        self.weight = nn.Parameter(torch.ones(hidden_size))
        self.eps = eps

    def forward(self, hidden_states: torch.Tensor) -> torch.Tensor:
        input_dtype = hidden_states.dtype
        fp32_states = hidden_states.float()
        variance = fp32_states.square().mean(-1, keepdim=True)
        normalized = fp32_states * torch.rsqrt(variance + self.eps)
        return (self.weight.float() * normalized).to(input_dtype)

def radio_h_pixel_shuffle_indices(image_grid_thw: torch.Tensor | Sequence[Sequence[int]], *, device: torch.device | str | None=None) -> torch.Tensor:
    grid = torch.as_tensor(image_grid_thw)
    if grid.ndim == 1:
        grid = grid.unsqueeze(0)
    if grid.ndim != 2 or grid.shape[-1] != 3:
        raise ValueError('image_grid_thw must have shape (num_images, 3)')
    if torch.any(grid[:, 0] != 1):
        raise ValueError('RADIO-H image evaluation requires T=1')
    if torch.any(grid[:, 1:] <= 0) or torch.any(grid[:, 1:] % 2):
        raise ValueError('RADIO-H patch grids must be positive and even')
    target_device = torch.device(device) if device is not None else grid.device
    indices: list[torch.Tensor] = []
    offset = 0
    for (_, height_value, width_value) in grid.tolist():
        (height, width) = (int(height_value), int(width_value))
        (half_height, half_width) = (height // 2, width // 2)
        flat_output = torch.arange(half_height * half_width, device=target_device)
        output_row = flat_output // half_width
        output_col = flat_output % half_width
        top_left = offset + 2 * output_row * width + 2 * output_col
        indices.append(torch.stack([top_left, top_left + 1, top_left + width, top_left + width + 1], dim=1).reshape(-1))
        offset += height * width
    return torch.cat(indices) if indices else torch.empty(0, dtype=torch.long)

class RadioHProjector(nn.Module):

    def __init__(self, config: Any) -> None:
        super().__init__()
        vision_width = 1280
        hidden_width = int(config.hidden_size)
        projector_width = getattr(config, 'projector_hidden_size', hidden_width)
        if projector_width is None:
            projector_width = hidden_width
        input_width = vision_width * 4
        self.mlp = nn.Sequential(ProjectorRMSNorm(input_width), nn.Linear(input_width, int(projector_width), bias=False), SquaredReLU(), nn.Linear(int(projector_width), hidden_width, bias=False))

    def forward(self, image_features: torch.Tensor, image_grid_thw: torch.Tensor) -> torch.Tensor:
        if image_features.ndim != 2:
            raise ValueError('image_features must be two-dimensional')
        gather = radio_h_pixel_shuffle_indices(image_grid_thw, device=image_features.device)
        if gather.numel() != image_features.shape[0]:
            raise ValueError(f'RADIO-H grid/feature mismatch: grid has {gather.numel()} patches, features have {image_features.shape[0]}')
        merged = image_features[gather].view(-1, 4 * image_features.shape[-1])
        return 0.01 * self.mlp(merged)
__all__ = ['RadioHProjector', 'radio_h_pixel_shuffle_indices']
