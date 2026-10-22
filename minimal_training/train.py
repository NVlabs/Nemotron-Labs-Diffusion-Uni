"""Single-student-node, batch-one FSDP trainer. Launch with torchrun."""
import argparse
from datetime import timedelta
import functools
import io
import json
import os
from pathlib import Path
import random
import time
import numpy as np
import torch
import torch.distributed as dist
import yaml
from torch.distributed.fsdp import FullyShardedDataParallel as FSDP, MixedPrecision
from torch.distributed.fsdp.wrap import transformer_auto_wrap_policy
from torch.distributed.algorithms._checkpoint.checkpoint_wrapper import (
    apply_activation_checkpointing, checkpoint_wrapper, CheckpointImpl)
from transformers import AutoModelForCausalLM
from data import Mixture
from objectives import Student
from transport import request, encode_image, decode_image


def broadcast(value):
    values = [value]
    dist.broadcast_object_list(values, src=0)
    return values[0]


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--config', default=str(Path(__file__).parent / 'configs/default.yaml'))
    parser.add_argument('--model'); parser.add_argument('--output'); parser.add_argument('--teacher-host')
    args = parser.parse_args()
    config_path = Path(args.config).resolve()
    cfg = yaml.safe_load(config_path.read_text())
    for name in ('model', 'output'):
        if getattr(args, name): cfg[name] = getattr(args, name)
    if args.teacher_host:
        cfg['vlm_teacher'] = f'http://{args.teacher_host}:8123'
        cfg['edit_teacher'] = f'http://{args.teacher_host}:8124'
    local_rank = int(os.environ['LOCAL_RANK']); torch.cuda.set_device(local_rank)
    dist.init_process_group('nccl', timeout=timedelta(minutes=45))
    rank, world = dist.get_rank(), dist.get_world_size()
    if world != int(os.environ['LOCAL_WORLD_SIZE']):
        raise ValueError('Only one student node is supported; teachers run separately')
    torch.manual_seed(cfg['seed']); random.seed(cfg['seed']); np.random.seed(cfg['seed'])
    output = Path(cfg['output']); output.mkdir(parents=True, exist_ok=True)
    if rank == 0:
        (output / 'config.yaml').write_text(yaml.safe_dump(cfg, sort_keys=False))
        print(json.dumps({'event':'loading', 'world_size':world, 'global_batch_size':1,
                         'gpu':torch.cuda.get_device_name(), 'model':cfg['model']}), flush=True)
    model = AutoModelForCausalLM.from_pretrained(cfg['model'], trust_remote_code=True,
        torch_dtype=torch.bfloat16, attn_implementation='flash_attention_2')
    frozen = [model.encoder.vision_tower, model.encoder.vqvae]
    for module in frozen:
        module.requires_grad_(False); module.to(local_rank)
    # FP32 master parameters/Adam states; BF16 computation and FP32 gradient reduction.
    for parameter in model.parameters():
        if parameter.requires_grad: parameter.data = parameter.data.float()
    student = Student(model, cfg)
    layer_class = type(model.encoder.layers[0])
    apply_activation_checkpointing(student,
        checkpoint_wrapper_fn=functools.partial(checkpoint_wrapper,
            checkpoint_impl=CheckpointImpl.NO_REENTRANT),
        check_fn=lambda module: isinstance(module, layer_class))
    wrapped = FSDP(student, device_id=local_rank, use_orig_params=True, ignored_modules=frozen,
        auto_wrap_policy=functools.partial(transformer_auto_wrap_policy,
            transformer_layer_cls={layer_class}),
        mixed_precision=MixedPrecision(param_dtype=torch.bfloat16, reduce_dtype=torch.float32,
                                       buffer_dtype=torch.float32), limit_all_gathers=True)
    optimizer = torch.optim.AdamW((p for p in wrapped.parameters() if p.requires_grad),
        lr=cfg['learning_rate'], betas=(.9,.95), weight_decay=cfg['weight_decay'], foreach=False)
    mixtures = {task: Mixture(config_path.parent / path, task, cfg['seed'])
                for task, path in cfg['datasets'].items()} if rank == 0 else {}
    counts = {task: 0 for task in cfg['datasets']}
    attempts = {task: 0 for task in counts}
    log = (output / 'steps.jsonl').open('a') if rank == 0 else None
    global_step = 0
    while min(counts.values()) < cfg['steps_per_task']:
        for task in counts:
            if counts[task] >= cfg['steps_per_task']: continue
            attempts[task] += 1
            if attempts[task] > cfg['max_attempts_per_task']: raise RuntimeError('Exceeded task attempt limit: '+task)
            sample = broadcast(mixtures[task].sample() if rank == 0 else None)
            seed = cfg['seed'] + 1009 * global_step + attempts[task]
            torch.manual_seed(seed); torch.cuda.manual_seed_all(seed)
            target = response = teacher = None
            metadata = {}
            started = time.monotonic()
            optimizer.zero_grad(set_to_none=True)
            if task.endswith('_opd'):
                wrapped.eval()
                with torch.no_grad(), torch.autocast('cuda', dtype=torch.bfloat16):
                    proposal = wrapped('rollout', sample, task)
                if task == 'vlm_opd':
                    response = proposal
                    if rank == 0:
                        raw = request(cfg['vlm_teacher'], {'prompt': sample.prompt,
                            'images':[encode_image(im) for im in sample.images],
                            'response_ids':response[0].tolist()})
                        payload = np.load(io.BytesIO(raw))
                        teacher = (torch.from_numpy(payload['logprobs']), torch.from_numpy(payload['indices']))
                        metadata['response'] = model.tokenizer.decode(response[0], skip_special_tokens=True)
                    teacher = broadcast(teacher)
                else:
                    if rank == 0:
                        source = student.parts['image_utils'].pad_to_square_and_resize(sample.images[0], cfg['resolution'])
                        result = json.loads(request(cfg['edit_teacher'], {'prompt':sample.prompt,
                            'source':encode_image(source), 'proposal':encode_image(proposal), 'seed':seed}))
                        target = decode_image(result.pop('image'))
                        metadata = result
                        folder = output / f'edit-attempt-{attempts[task]:03d}'; folder.mkdir(exist_ok=True)
                        source.save(folder/'source.png'); proposal.save(folder/'student.png'); target.save(folder/'teacher.png')
                        (folder/'refinement.json').write_text(json.dumps({'prompt':sample.prompt, **metadata}, indent=2))
                    metadata = broadcast(metadata)
                    target = broadcast(target)
            wrapped.train()
            for module in frozen: module.eval()
            with torch.autocast('cuda', dtype=torch.bfloat16):
                loss, components = wrapped('loss', sample, task, response=response, teacher=teacher, target=target)
            if not torch.isfinite(loss): raise FloatingPointError(f'Nonfinite {task} loss')
            loss.backward()
            norm = wrapped.clip_grad_norm_(cfg['max_grad_norm'])
            if not torch.isfinite(norm): raise FloatingPointError(f'Nonfinite {task} gradients')
            if norm == 0:
                if rank == 0: print('Zero-supervision draw; resampling without optimizer update', flush=True)
                continue
            optimizer.step()
            counts[task] += 1; global_step += 1
            row = {'event':'optimizer_step','global_step':global_step,'task':task,
                'task_step':counts[task],'attempt':attempts[task], 'loss':float(loss.detach()),
                'grad_norm':float(norm), 'seconds':time.monotonic()-started, 'seed':seed,
                'peak_gpu_gb':torch.cuda.max_memory_allocated()/2**30, **components, **metadata}
            if rank == 0:
                log.write(json.dumps(row)+'\n'); log.flush(); print(json.dumps(row), flush=True)
    if cfg['save_model']:
        from torch.distributed.fsdp import StateDictType, FullStateDictConfig
        with FSDP.state_dict_type(wrapped, StateDictType.FULL_STATE_DICT,
                FullStateDictConfig(offload_to_cpu=True, rank0_only=True)):
            state = wrapped.state_dict()
        if rank == 0:
            state = {key.removeprefix('model.'):value.to(torch.bfloat16) for key,value in state.items()}
            model.save_pretrained(output/'model', state_dict=state, max_shard_size='5GB')
            # HF observes FP32 master parameters; describe the BF16 saved tensors.
            model.config.dtype = 'bfloat16'
            model.config.save_pretrained(output/'model')
    if rank == 0:
        (output/'complete.json').write_text(json.dumps({'steps':counts,'attempts':attempts,'global_batch_size':1},indent=2))
        log.close()
    dist.barrier(); dist.destroy_process_group()


if __name__ == '__main__': main()
