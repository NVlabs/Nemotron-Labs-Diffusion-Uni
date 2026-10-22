from __future__ import annotations
from collections.abc import Sequence
from typing import Any
import torch
_IMAGE_PAD_TOKEN_INDEX = -500

def _contiguous_offsets(positions: Sequence[int]) -> list[tuple[int, int]]:
    if not positions:
        return []
    offsets: list[tuple[int, int]] = []
    start = previous = int(positions[0])
    for position_value in positions[1:]:
        position = int(position_value)
        if position != previous + 1:
            offsets.append((start, previous))
            start = position
        previous = position
    offsets.append((start, previous))
    return offsets

def build_sglang_radio_h_request(prompt_ids: torch.Tensor, images: Sequence[tuple[torch.Tensor, torch.Tensor]]) -> tuple[list[int], list[dict[str, Any]] | list[None]]:
    if prompt_ids.ndim != 2 or prompt_ids.shape[0] != 1:
        raise ValueError('prompt_ids must have shape (1, sequence)')
    input_ids = [int(value) for value in prompt_ids[0].detach().cpu().tolist()]
    slot_positions = [index for (index, token_id) in enumerate(input_ids) if token_id == _IMAGE_PAD_TOKEN_INDEX]
    if not images:
        if slot_positions:
            raise ValueError('image slots were provided without RADIO-H images')
        return (input_ids, [None])
    merge_size = 2
    counts: list[int] = []
    normalized: list[tuple[torch.Tensor, torch.Tensor]] = []
    for (image_index, image) in enumerate(images):
        if not isinstance(image, (tuple, list)) or len(image) != 2:
            raise TypeError(f'images[{image_index}] must be (pixel_values, image_grid_thw)')
        (pixels, grid) = image
        grid = torch.as_tensor(grid, dtype=torch.long)
        if grid.ndim == 1:
            grid = grid.unsqueeze(0)
        if grid.shape != (1, 3) or int(grid[0, 0]) != 1:
            raise ValueError('each RADIO-H grid must have shape (1, 3) with T=1')
        (height, width) = (int(grid[0, 1]), int(grid[0, 2]))
        if height % merge_size or width % merge_size:
            raise ValueError('RADIO-H grid is not divisible by its merge size')
        counts.append(height // merge_size * (width // merge_size))
        normalized.append((torch.as_tensor(pixels), grid))
    expected_slots = sum(counts)
    if len(slot_positions) != expected_slots:
        raise ValueError(f'RADIO-H prompt/media mismatch: prompt has {len(slot_positions)} slots but grids produce {expected_slots}')
    payloads: list[dict[str, Any]] = []
    cursor = 0
    for ((pixels, grid), count) in zip(normalized, counts, strict=True):
        image_positions = slot_positions[cursor:cursor + count]
        cursor += count
        for position in image_positions:
            input_ids[position] = 0
        payloads.append({'format': 'processor_output', 'modality': 'image', 'pixel_values': pixels.detach().cpu().contiguous(), 'image_grid_thw': grid.detach().cpu().contiguous(), 'offsets': _contiguous_offsets(image_positions)})
    return (input_ids, payloads)
__all__ = ['build_sglang_radio_h_request']
