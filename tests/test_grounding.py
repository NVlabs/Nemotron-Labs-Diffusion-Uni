from types import SimpleNamespace
from unittest.mock import patch

import pytest
import torch
from PIL import Image

import grounding


def test_tags():
    assert grounding.parse_tags(' dog, red collar\n the person on the left ') == ['dog', 'red collar', 'the person on the left']
    for text in (' ', '<SPECIAL_100>', ','.join(['dog'] * 17)):
        with pytest.raises(ValueError):
            grounding.parse_tags(text)


def test_coordinate_mapping():
    assert grounding.pixel_box([0, 256, 1024, 768], (800, 400)) == [0, 0, 800, 400]
    assert grounding.pixel_box([256, 0, 768, 1024], (400, 800)) == [0, 0, 400, 800]
    assert grounding.pixel_box([0, 0, 1024, 1024], (800, 400)) == [0, 0, 800, 400]


class TinyDecoder:
    def __init__(self, fail=False):
        self.mask_token_id = 100
        self.attention = SimpleNamespace(diffusion_lm=False, mode='initial')
        self.encoder = self
        self.layers = [SimpleNamespace(self_attn=self.attention)]
        self.cache = object()
        self.calls = []
        self.fail = fail

    def __call__(self, inputs_embeds, use_cache, past_key_values=None, **kwargs):
        self.calls.append((inputs_embeds.clone(), use_cache))
        if use_cache:
            assert not self.attention.diffusion_lm
            assert kwargs['use_causal_mask']
            return SimpleNamespace(past_key_values=self.cache)
        assert self.attention.diffusion_lm and past_key_values is self.cache
        if self.fail:
            raise RuntimeError('test failure')
        length = inputs_embeds.shape[1]
        return SimpleNamespace(last_hidden_state=torch.arange(length).view(1, length, 1).float())

    def diffusion_head(self, hidden):
        values = torch.full((len(hidden), 50), -100.)
        for row, position in enumerate(hidden[:, 0].long()):
            values[row, 10 + position] = 1. + position
        return values

    def get_input_embeddings(self):
        return lambda ids: ids.float().unsqueeze(-1)


@pytest.mark.parametrize('steps', [1, 2, 4])
def test_shift_cache_and_restoration(steps):
    model = TinyDecoder()
    ids = torch.tensor([[1, 2, 3, 100, 100, 100, 100, 4, 100, 100, 100, 100, 5]])
    embeddings = ids.float().unsqueeze(-1)
    result = grounding._fill_coordinates(model, ids, embeddings, 2, steps)
    assert result == [[10, 11, 12, 13], [15, 16, 17, 18]]
    assert len(model.calls) == steps + 1
    assert model.calls[0][0].shape[1] == 2
    assert model.attention.diffusion_lm is False and model.attention.mode == 'initial'
    assert torch.equal(embeddings[..., 0], ids)


def test_failure_restores_attention():
    model = TinyDecoder(fail=True)
    ids = torch.tensor([[1, 2, 3, 100, 100, 100, 100]])
    with pytest.raises(RuntimeError, match='test failure'):
        grounding._fill_coordinates(model, ids, ids.float().unsqueeze(-1), 1, 1)
    assert model.attention.diffusion_lm is False and model.attention.mode == 'initial'


def test_overlay_and_invalid_output():
    model = SimpleNamespace(tokenizer=SimpleNamespace(convert_ids_to_tokens=lambda ids: [f'<LOC_{i}>' for i in ids]))
    image = Image.new('RGB', (800, 400), 'white')
    with patch.object(grounding, '_image_prompt', return_value=(None, None)), patch.object(grounding, '_fill_coordinates', return_value=[[100, 300, 900, 700], [600, 600, 400, 400]]):
        result, boxes = grounding.ground_objects(model, image, 'dog, cat')
        assert result.size == image.size and result.tobytes() != image.tobytes()
        assert [x['tag'] for x in boxes] == ['dog', 'cat']
        assert boxes[0]['valid'] and not boxes[1]['valid']
    model.tokenizer.convert_ids_to_tokens = lambda ids: ['not-a-coordinate'] * 4
    with patch.object(grounding, '_image_prompt', return_value=(None, None)), patch.object(grounding, '_fill_coordinates', return_value=[[1, 2, 3, 4]]):
        with pytest.raises(RuntimeError, match='No valid coordinate'):
            grounding.ground_objects(model, image, 'dog')


def test_gradio_build_and_shared_queue():
    import gradio_demo
    demo = gradio_demo.build_demo(object())
    assert len(demo.fns) == 4
    assert all(fn.concurrency_id == 'model' and fn.concurrency_limit == 1 for fn in demo.fns.values())
