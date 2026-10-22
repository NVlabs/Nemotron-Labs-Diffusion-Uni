"""Native SGLang visual understanding with bundled CPU preprocessing."""
import argparse
import json
import os
import tempfile
from pathlib import Path
import torch
from PIL import Image
from transformers import AutoTokenizer
from transformers.dynamic_module_utils import get_class_from_dynamic_module


def prepare_request(model_path, prompt, image=None):
    tokenizer = AutoTokenizer.from_pretrained(model_path)
    Conversation = get_class_from_dynamic_module('conversation.Conversation', model_path)
    RadioImageProcessor = get_class_from_dynamic_module('vision.RadioImageProcessor', model_path)
    conv = Conversation()
    images = []
    if image is not None:
        if not isinstance(image, Image.Image):
            image = Image.open(image)
        pixels = RadioImageProcessor()(images=[image.convert('RGB')]).pixel_values
        h, w = pixels.shape[-2:]
        images = [(pixels, torch.tensor([[1, h // 16, w // 16]]))]
        slots = '\n'.join(['<SPECIAL_23>' * (w // 32)] * (h // 32))
        prompt = slots + '\n' + prompt
    conv.append_message(conv.roles[0], prompt)
    conv.append_message(conv.roles[1], None)
    ids = tokenizer(conv.get_prompt(), return_tensors='pt').input_ids
    ids[ids == 23] = -500
    from sglang_nlduni.request import build_sglang_radio_h_request
    return build_sglang_radio_h_request(ids, images)


def create_engine(model_path, config_dir, *, threshold=.9, tensor_parallel_size=1):
    root = str(Path(__file__).resolve().parent)
    os.environ['PYTHONPATH'] = root + os.pathsep + os.environ.get('PYTHONPATH', '')
    os.environ['SGLANG_EXTERNAL_MODEL_PACKAGE'] = 'sglang_nlduni.models'
    os.environ['SGLANG_EXTERNAL_MM_PROCESSOR_PACKAGE'] = 'sglang_nlduni.processors'
    os.environ['SGLANG_EXTERNAL_MM_MODEL_ARCH'] = 'NLDUniForCausalLM'
    # This environment uses CUDA 12.9 kernels, while the fork's generic
    # minimum-version check targets its newer CUDA 13 default environment.
    os.environ['SGLANG_SKIP_SGL_KERNEL_VERSION_CHECK'] = '1'
    cfg = Path(config_dir) / 'diffusion.json'
    cfg.write_text(json.dumps(dict(causal_context=True, temperature=0., block_size=32,
                                   max_steps=32, threshold=threshold, shift_logits=True)))
    import sglang
    return sglang.Engine(
        model_path=str(model_path), tokenizer_path=str(model_path), trust_remote_code=True,
        dtype='bfloat16', tp_size=tensor_parallel_size, context_length=16384,
        mem_fraction_static=.8, max_running_requests=1, chunked_prefill_size=16384,
        attention_backend='flashinfer', disable_cuda_graph=True,
        dllm_algorithm='NLDUniFastDiffuser', dllm_algorithm_config=str(cfg),
        json_model_override_args=json.dumps({'architectures': ['NLDUniForCausalLM']}),
    )


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('model')
    parser.add_argument('--prompt', default='Describe this image.')
    parser.add_argument('--image')
    parser.add_argument('--max-new-tokens', type=int, default=512)
    parser.add_argument('--threshold', type=float, default=.9)
    parser.add_argument('--tensor-parallel-size', type=int, default=1)
    args = parser.parse_args()
    with tempfile.TemporaryDirectory(prefix='nlduni-sglang-') as tmp:
        engine = create_engine(args.model, tmp, threshold=args.threshold,
                               tensor_parallel_size=args.tensor_parallel_size)
        try:
            ids, media = prepare_request(args.model, args.prompt, args.image)
            result = engine.generate(input_ids=ids, image_data=media,
                sampling_params={'max_new_tokens': args.max_new_tokens, 'temperature': 0.})
            print(result['text'])
        finally:
            engine.shutdown()


if __name__ == '__main__':
    main()
