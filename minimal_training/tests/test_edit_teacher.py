"""Exercise the edit service response without downloading models or using GPUs."""
import json
import sys
from types import SimpleNamespace
from unittest.mock import Mock

import torch
from PIL import Image

from teachers import EditTeacher
from transport import decode_image, encode_image


def test_refined_image_is_returned_directly(monkeypatch):
    refined = Image.new('RGB', (16, 16), 'green')
    pipe = Mock(return_value=SimpleNamespace(images=[refined]))
    pipe.hf_device_map = {'transformer': 0}
    pipe._execution_device = 'cpu'
    pipe.latent_channels = pipe.vae_scale_factor = 1
    pipe.vae = torch.nn.Linear(1, 1)
    pipe.image_processor.preprocess.return_value = torch.zeros(1, 1, 16, 16)
    pipe._encode_vae_image.return_value = torch.ones(1, 1, 1, 16, 16)
    pipe._pack_latents.side_effect = lambda tensor, *args: tensor.reshape(1, 256, 1)
    loader = Mock(return_value=pipe)
    monkeypatch.setitem(sys.modules, 'diffusers', SimpleNamespace(
        DiffusionPipeline=SimpleNamespace(from_pretrained=loader)))
    monkeypatch.setitem(sys.modules, 'diffusers.utils.torch_utils', SimpleNamespace(
        randn_tensor=lambda shape, **kwargs: torch.randn(shape, **kwargs)))

    teacher = EditTeacher('qwen-image-edit')
    image = encode_image(Image.new('RGB', (16, 16), 'white'))
    result = json.loads(teacher({'source': image, 'proposal': image,
                                 'prompt': 'Make the image green.', 'seed': 42}))

    loader.assert_called_once()
    assert loader.call_args.args == ('qwen-image-edit',)
    assert set(result) == {'image', 'strength', 'initial_sigma', 'teacher_steps'}
    assert decode_image(result['image']).tobytes() == refined.tobytes()
    assert result['strength'] == .9
    assert abs(result['initial_sigma'] - .9) < 1e-10
    assert result['teacher_steps'] == pipe.call_args.kwargs['num_inference_steps']
    assert pipe.call_args.kwargs['prompt'] == 'Make the image green.'
