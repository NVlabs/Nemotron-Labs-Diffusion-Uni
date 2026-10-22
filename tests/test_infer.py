import sys
from types import SimpleNamespace
from unittest.mock import Mock

import pytest
from PIL import Image

import infer


@pytest.mark.parametrize('task', ['text-to-image', 'image-edit', 'understand', 'grounding'])
def test_cli_dispatch_and_outputs(task, tmp_path, monkeypatch, capsys):
    image_path = tmp_path / 'input.png'
    output_path = tmp_path / 'output.png'
    image = Image.new('RGB', (8, 8), 'blue')
    image.save(image_path)
    model = Mock()
    model.to.return_value = model
    model.eval.return_value = model
    model.text_to_image.return_value = image
    model.image_edit.return_value = image
    model.generate.return_value = 'A blue image.'
    loader = Mock(return_value=model)
    monkeypatch.setitem(sys.modules, 'transformers', SimpleNamespace(
        AutoModelForCausalLM=SimpleNamespace(from_pretrained=loader)))
    ground = Mock(return_value=(image, []))
    monkeypatch.setitem(sys.modules, 'grounding', SimpleNamespace(ground_objects=ground))
    monkeypatch.setattr(sys, 'argv', ['infer.py', '--task', task, '--prompt', 'blue',
                                    '--image', str(image_path), '--output', str(output_path)])
    infer.main()
    assert loader.call_args.args == ('nvidia/NLD-Uni',)
    if task == 'understand':
        assert not output_path.exists()
        assert 'A blue image.' in capsys.readouterr().out
        assert model.generate.call_args.kwargs['max_new_tokens'] == 512
    else:
        assert Image.open(output_path).tobytes() == image.tobytes()
        if task == 'grounding':
            assert ground.call_args.kwargs['steps'] == 1
        else:
            method = model.image_edit if task == 'image-edit' else model.text_to_image
            assert method.call_args.kwargs == {'seed': 42}


def test_edit_requires_source_image_before_loading(monkeypatch):
    monkeypatch.setattr(sys, 'argv', ['infer.py', '--task', 'image-edit', '--prompt', 'blue'])
    with pytest.raises(SystemExit) as exc:
        infer.main()
    assert exc.value.code == 2
