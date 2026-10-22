"""Continuous-time semantic-tree corruption (paper Eqs. 15–18).

Here K counts levels including clean and fully noisy endpoints. The paper's K
counts transitions, so its K equals this module's num_tree_levels - 1.
"""
from dataclasses import dataclass
import torch


@dataclass
class TreeState:
    representatives: torch.Tensor
    ages: torch.Tensor
    depths: torch.Tensor
    noise: torch.Tensor
    supervised: torch.Tensor
    num_tree_levels: int
    epsilon: float
    time: float

    @property
    def decoding_steps(self):
        return self.num_tree_levels - 1


def ages_at_time(time, starts, epsilon, num_tree_levels):
    """Token age: 0 is clean; K-1 is the root (uniform-code) state."""
    if not 2 <= num_tree_levels <= 18:
        raise ValueError('num_tree_levels must be in [2, 18]')
    if not 0 < epsilon < 1:
        raise ValueError('epsilon must be in (0, 1)')
    if not 0 <= time <= 1:
        raise ValueError('time must be in [0, 1]')
    jumps = num_tree_levels - 1
    return torch.floor(jumps * (time - starts) / epsilon).long().clamp(0, jumps)


def corrupt(clean, tree, *, generator=None, num_tree_levels=None, epsilon=None,
            time=None, structural_mask=None):
    """Sample K, epsilon, t, then independent per-token transition intervals.

    There are no precomputed stage ratios or flip perturbations. Structural
    markers stay clean and carry no image-token loss. Root representatives are
    uniform over the codebook and independent of the clean endpoint.
    """
    device = clean.device
    if num_tree_levels is None:
        num_tree_levels = int(torch.randint(2, tree.depth + 2, (), device=device,
                                            generator=generator))
    def uniform_open():
        # Avoid the zero endpoint of torch.rand; the upper endpoint is excluded.
        return max(float(torch.rand((), device=device, generator=generator)), 1e-7)
    epsilon = uniform_open() if epsilon is None else float(epsilon)
    time = uniform_open() if time is None else float(time)
    starts = torch.rand(clean.shape, device=device, generator=generator) * (1 - epsilon)
    ages = ages_at_time(time, starts, epsilon, num_tree_levels)
    jumps = num_tree_levels - 1
    depths = tree.depth - torch.div(ages * tree.depth, jumps, rounding_mode='floor')
    structural_mask = torch.zeros_like(clean, dtype=torch.bool) if structural_mask is None else structural_mask
    ages = ages.masked_fill(structural_mask, 0)
    depths = depths.masked_fill(structural_mask, tree.depth)
    safe_clean = clean.masked_fill(structural_mask, 0)
    representatives = tree.sample_representatives(safe_clean, depths, generator=generator)
    representatives = torch.where(structural_mask, clean, representatives)
    noise = ages.float() / jumps
    return TreeState(representatives, ages, depths, noise, ages.gt(0) & ~structural_mask,
                     num_tree_levels, epsilon, time)


def embeddings(state, embedding, mask_token_id):
    return embedding(state.representatives) + state.noise.unsqueeze(-1).to(embedding.weight.dtype) * embedding.weight[mask_token_id]
