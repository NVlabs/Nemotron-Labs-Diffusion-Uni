"""Plain Transformers/Diffusers teacher services. Each sees four GPUs."""
import argparse
import importlib
import io
import json
import math
from http.server import BaseHTTPRequestHandler, HTTPServer
import traceback
import numpy as np
import torch
from transport import decode_image, encode_image
from prompts import SYSTEM
from loss_math import teacher_topk


def matched_sigmas(strength, sequence_length, steps=40):
    """Invert Qwen's dynamic shift and terminal stretch to start exactly at r."""
    if not .02 < strength <= 1: raise ValueError('strength must be in (.02, 1]')
    scale = math.exp(.5 + (sequence_length - 256) * (.9 - .5) / (8192 - 256))
    raw = np.linspace(1., 1. / steps, steps)
    shift = lambda x: scale * x / (1 + (scale - 1) * x)
    stretch = (1 - shift(raw[-1])) / .98
    shifted_start = 1 - stretch * (1 - strength)
    start = shifted_start / (scale - (scale - 1) * shifted_start)
    selected = np.concatenate(([start], raw[raw < start - 1e-12]))
    effective = 1 - (1 - shift(selected)) / stretch
    if not np.isclose(effective[0], strength) or not np.isclose(effective[-1], .02):
        raise RuntimeError('Qwen sigma inversion failed')
    return selected.tolist(), effective.tolist()


class VLMTeacher:
    def __init__(self, path):
        from transformers import AutoConfig, AutoModelForCausalLM, AutoProcessor
        self.processor = AutoProcessor.from_pretrained(path, trust_remote_code=True)
        config = AutoConfig.from_pretrained(path, trust_remote_code=True, attn_implementation='eager')
        config.llm_config._attn_implementation = 'eager'
        config.vision_config.use_flash_attn = False
        self.model = AutoModelForCausalLM.from_pretrained(path, trust_remote_code=True,
            config=config,
            dtype=torch.bfloat16, device_map='balanced', max_memory={i: '70GiB' for i in range(4)},
            attn_implementation='eager').eval()
        implementation = importlib.import_module(type(self.model.language_model).__module__)
        if not implementation.is_fast_path_available:
            raise RuntimeError('Fused Mamba kernels are required for teacher fidelity; install kernels and cache its version-1 causal-conv1d/mamba-ssm builds')
        print('VLM teacher device map:', self.model.hf_device_map, flush=True)

    @torch.inference_mode()
    def __call__(self, payload):
        images = [decode_image(x) for x in payload['images']]
        content = [{'type': 'image', 'image': im} for im in images]
        content.append({'type': 'text', 'text': payload['prompt'].replace('<image>', '')})
        text = self.processor.apply_chat_template([
            {'role': 'system', 'content': SYSTEM},
            {'role': 'user', 'content': content}],
            tokenize=False, add_generation_prompt=True, enable_thinking=True)
        inputs = self.processor(text=[text], images=images or None, return_tensors='pt')
        m = self.model
        device = m.language_model.get_input_embeddings().weight.device
        prefix = inputs.input_ids.to(device)
        response = torch.tensor(payload['response_ids'], device=device, dtype=torch.long).reshape(1, -1)
        # Feed preceding response tokens; the last R positions predict all R targets.
        ids = torch.cat([prefix, response[:, :-1]], 1)
        embeds = m.language_model.get_input_embeddings()(ids)
        if images:
            pixels = inputs.pixel_values
            vision_device = next(m.vision_model.parameters()).device
            if isinstance(pixels, list):
                features = torch.cat([m.extract_feature(x.to(vision_device, torch.bfloat16)).reshape(-1, embeds.shape[-1])
                                      for x in pixels], dim=0).to(embeds)
            else:
                features = m.extract_feature(pixels.to(vision_device, torch.bfloat16)).reshape(-1, embeds.shape[-1]).to(embeds)
            mask = ids.eq(m.img_context_token_id)
            if int(mask.sum()) != len(features): raise ValueError('Teacher image token mismatch')
            embeds[mask] = features
        output = m.language_model(inputs_embeds=embeds, attention_mask=torch.ones_like(ids),
            logits_to_keep=response.numel(), use_cache=False)
        logits = output.logits[0].float()
        if logits.shape != (response.numel(), 131072): raise ValueError('Teacher vocabulary/position mismatch')
        if not torch.isfinite(logits).all(): raise ValueError('Nonfinite teacher logits')
        buffer = io.BytesIO()
        logp, indices = teacher_topk(logits)
        np.savez(buffer, logprobs=logp.cpu().numpy(), indices=indices.cpu().numpy())
        return buffer.getvalue()


class EditTeacher:
    def __init__(self, path, strength=.9, steps=40):
        from diffusers import DiffusionPipeline
        self.pipe = DiffusionPipeline.from_pretrained(path, torch_dtype=torch.bfloat16,
            device_map='balanced', max_memory={0:'44GiB', 1:'60GiB', 2:'60GiB', 3:'60GiB'})
        self.strength, self.steps = strength, steps
        print('Qwen teacher device map:', self.pipe.hf_device_map, flush=True)

    @torch.inference_mode()
    def __call__(self, payload):
        from diffusers.utils.torch_utils import randn_tensor
        source, proposal = decode_image(payload['source']), decode_image(payload['proposal'])
        pipe = self.pipe
        width, height = proposal.size
        device = pipe._execution_device
        generator = torch.Generator(device=device).manual_seed(payload['seed'])
        pixels = pipe.image_processor.preprocess(proposal, height=height, width=width).unsqueeze(2)
        vae_device = next(pipe.vae.parameters()).device
        proposal_latent = pipe._encode_vae_image(pixels.to(vae_device, torch.bfloat16), generator=generator)
        channels, lh, lw = pipe.latent_channels, height // pipe.vae_scale_factor, width // pipe.vae_scale_factor
        pack = lambda x: pipe._pack_latents(x, 1, channels, lh, lw)
        latent = pack(proposal_latent).to(device)
        noise = pack(randn_tensor((1, 1, channels, lh, lw), generator=generator, device=device, dtype=torch.bfloat16))
        initial = (1 - self.strength) * latent + self.strength * noise
        sigmas, effective = matched_sigmas(self.strength, initial.shape[1], self.steps)
        result = pipe(image=[source], prompt=payload['prompt'], negative_prompt=' ',
            true_cfg_scale=4., guidance_scale=1., height=height, width=width,
            num_inference_steps=len(sigmas), sigmas=sigmas, latents=initial,
            generator=generator).images[0]
        return json.dumps(dict(image=encode_image(result), strength=self.strength,
                               initial_sigma=effective[0], teacher_steps=len(sigmas))).encode()


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('kind', choices=['vlm','edit']); parser.add_argument('--model', required=True)
    parser.add_argument('--host', default='0.0.0.0')
    parser.add_argument('--port', type=int, required=True)
    args = parser.parse_args()
    if torch.cuda.device_count() != 4: raise RuntimeError('Each teacher requires exactly four visible GPUs')
    teacher = VLMTeacher(args.model) if args.kind == 'vlm' else EditTeacher(args.model)
    class Handler(BaseHTTPRequestHandler):
        def do_GET(self):
            self.send_response(200); self.end_headers(); self.wfile.write(b'ready')
        def do_POST(self):
            try:
                request = json.loads(self.rfile.read(int(self.headers['Content-Length'])))
                result = teacher(request)
                self.send_response(200); self.end_headers(); self.wfile.write(result)
            except Exception as error:
                traceback.print_exc()
                self.send_response(500); self.end_headers(); self.wfile.write(str(error).encode())
    print(f'{args.kind} ready on {args.port}', flush=True)
    HTTPServer((args.host, args.port), Handler).serve_forever()


if __name__ == '__main__': main()
