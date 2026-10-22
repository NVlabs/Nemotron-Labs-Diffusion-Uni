import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import torch
from attention import block_diffusion_mask, image_mask, preserve_rotary_precision
from teachers import matched_sigmas


def test_noisy_stream_cannot_read_clean_answer_in_its_block():
    mask = block_diffusion_mask(8, 4)
    assert mask[2, :4].all() and not mask[2, 4:].any()
    assert mask[5, 4:8].all() and mask[5, 8:12].all()
    assert not mask[5, 12:].any() and not mask[5, :4].any()
    assert not mask[10, :8].any()
    assert mask[10, 8:11].all() and not mask[10, 11:].any()


def test_image_block_and_causal_prefix():
    mask = image_mask(torch.tensor([False, False, True, True]))
    assert mask[2, 3] and not mask[0, 1] and not mask[1, 2]


def test_rotary_phases_survive_bfloat16_autocast():
    class Rotary(torch.nn.Module):
        def __init__(self):
            super().__init__()
            self.register_buffer('inv_freq', 1 / (1000000 ** (torch.arange(0, 128, 2).float() / 128)))
        def forward(self, x, position_ids):
            phase = self.inv_freq[None, :, None] @ position_ids[:, None].float()
            return phase.cos().to(x.dtype)
    module = Rotary()
    x, positions = torch.zeros(1, dtype=torch.bfloat16), torch.arange(4096)[None]
    expected = module(x, positions)
    with torch.autocast('cpu', dtype=torch.bfloat16):
        assert not torch.equal(module(x, positions), expected)
    preserve_rotary_precision(module)
    with torch.autocast('cpu', dtype=torch.bfloat16):
        assert torch.equal(module(x, positions), expected)


def test_qwen_initial_sigma_is_point_nine():
    for length in (256, 1024, 4096):
        raw, effective = matched_sigmas(.9, length)
        assert abs(effective[0] - .9) < 1e-10
        assert abs(effective[-1] - .02) < 1e-10
        assert all(a > b for a,b in zip(effective, effective[1:]))
