"""SFT and on-policy objectives using only the standalone model and PyTorch."""
import importlib
import random
from pathlib import Path
import torch
from torch import nn
import torch.nn.functional as F
from torch.utils.checkpoint import checkpoint
from attention import block_diffusion_mask, image_mask, install
from corruption import corrupt, embeddings
from loss_math import (NOISE_EPS, opd_token_losses, text_denominators, image_mean,
                       image_group_table, image_token_losses, image_condition_dropout)
from prompts import SYSTEM


class Student(nn.Module):
    def __init__(self, model, config):
        super().__init__()
        self.model, self.options = model, config
        root = type(model).__module__.rsplit('.', 1)[0]
        self.parts = {name: importlib.import_module(root + '.' + name)
                      for name in ('image_utils', 'conversation', 'markers', 'tree', 'padding')}
        tree_api = self.parts['tree']
        self.tree = tree_api.BalancedSemanticTree.from_artifact(
            tree_api.production_semantic_tree_contract(model.config.image_token_tree_path))
        groups = torch.load(Path(__file__).with_name('assets')/'image_groups.pt', weights_only=True)
        members, valid = image_group_table(groups)
        self.register_buffer('image_groups', groups, persistent=False)
        self.register_buffer('group_members', members, persistent=False)
        self.register_buffer('group_valid', valid, persistent=False)
        install(model)

    def tokenize(self, prompt, images, assistant=None, *, thinking=False):
        m, util = self.model, self.parts['image_utils']
        conv = self.parts['conversation'].Conversation()
        if images and '<image>' not in prompt:
            prompt = '<image>\n' * len(images) + prompt
        if prompt.count('<image>') != len(images):
            raise ValueError('Image placeholder count must match images')
        processed = util.process_images(images, m.encoder.vision_tower.image_processor, m.config) if images else []
        conv.append_message(conv.roles[0], prompt)
        conv.append_message(conv.roles[1], assistant)
        rendered = conv.get_prompt()
        if thinking: rendered += '<think>'
        if images:
            grids = [(int(grid[0, 1]) // 2, int(grid[0, 2]) // 2) for _, grid in processed]
            rendered = util.fix_one_prompt(rendered, m.config, grids)
        ids = util.tokenizer_image_token(rendered, m.tokenizer, return_tensors='pt', constant_version='ministral')[None].to(m.device)
        processed = [(pixels.to(m.device, torch.bfloat16), grid.to(m.device)) for pixels, grid in processed]
        embeds = m.prepare_inputs_labels_for_multimodal(ids, None, None, None, None, processed)[4]
        return ids, embeds

    def sft_tokens(self, example):
        """Keep chat turns, assistant labels, EOS augmentation and block padding."""
        m, util = self.model, self.parts['image_utils']
        turns = example.turns or [{'role':'user', 'content':example.prompt},
                                  {'role':'assistant', 'content':example.answer}]
        rendered = '<|im_start|>system\n' + SYSTEM + '<|im_end|>\n'
        canonical = []
        for turn in turns:
            role = turn.get('role', turn.get('from'))
            role = {'human':'user', 'gpt':'assistant'}.get(role, role)
            content = turn.get('content', turn.get('value', ''))
            if isinstance(content, list):
                content = '\n'.join(x.get('text', '') for x in content if isinstance(x, dict))
            if role == 'system': continue
            if role not in ('user', 'assistant'): raise ValueError('Unsupported conversation role')
            canonical.append((role, str(content)))
        if not canonical or canonical[0][0] != 'user': raise ValueError('SFT needs a user turn')
        if example.images and not any('<image>' in text for _, text in canonical):
            canonical[0] = ('user', '<image>\n' * len(example.images) + canonical[0][1])
        for role, content in canonical:
            rendered += '<|im_start|>' + role + '\n' + content + '<|im_end|>\n'
        processed = util.process_images(example.images, m.encoder.vision_tower.image_processor, m.config) if example.images else []
        if processed:
            grids = [(int(grid[0, 1]) // 2, int(grid[0, 2]) // 2) for _, grid in processed]
            rendered = util.fix_one_prompt(rendered, m.config, grids)
        values, labels = [], []
        for part in rendered.split('<|im_start|>')[1:]:
            segment = '<|im_start|>' + part
            row = util.tokenizer_image_token(segment, m.tokenizer, constant_version='ministral')
            assistant = part.startswith('assistant\n')
            if assistant:
                header = '<|im_start|>assistant\n' + ('<think>' if part.startswith('assistant\n<think>') else '')
                prefix_length = len(m.tokenizer(header, add_special_tokens=False).input_ids)
                eos_at = len(row) - 1 - row[::-1].index(11)
                # Truncate content, retaining the true terminal EOS/newline.
                row = row[:prefix_length] + row[prefix_length:eos_at][:self.options['max_answer_tokens']] + row[eos_at:]
                eos_at = len(row) - 1 - row[::-1].index(11)
                row[eos_at+1:eos_at+1] = [11] * random.randint(0, self.options['block_size'])
                label = [-100] * prefix_length + row[prefix_length:]
            else:
                label = [-100] * len(row)
            values.extend(row); labels.extend(label)
        padding = (-len(values)) % self.options['block_size']
        values.extend([11] * padding); labels.extend([11] * padding)
        ids = torch.tensor([values], device=m.device)
        labels = torch.tensor([labels], device=m.device).masked_fill(ids.eq(11), 11)
        processed = [(pixels.to(m.device, torch.bfloat16), grid.to(m.device)) for pixels, grid in processed]
        clean = m.prepare_inputs_labels_for_multimodal(ids, None, None, None, None, processed)[4]
        return ids, clean, labels

    @torch.no_grad()
    def rollout(self, example, task):
        m = self.model
        if task == 'image_edit_opd':
            return m.image_edit(example.images[0], example.prompt,
                resolution=self.options['resolution'], steps=self.options['rollout_image_steps'],
                seed=self.options['seed'], crop_to_source=False)
        ids, embeds = self.tokenize(example.prompt, example.images, thinking=True)
        cache = None; result = []
        attention = [layer.self_attn for layer in m.encoder.layers]
        previous = [a.diffusion_lm for a in attention]
        try:
            for a in attention: a.diffusion_lm = False
            for _ in range(self.options['max_new_tokens']):
                output = m.encoder(inputs_embeds=embeds, past_key_values=cache, use_cache=True,
                    overwrite_attn_impl='flash_attn', use_causal_mask=True)
                cache = output.past_key_values
                logits = m.diffusion_head(output.last_hidden_state[:, -1]).float()[:, :131072]
                logits[:, 18:30] = -torch.inf; logits[:, 100] = -torch.inf
                temperature = self.options['rollout_temperature']
                token = logits.argmax(-1, keepdim=True) if temperature == 0 else torch.multinomial((logits / temperature).softmax(-1), 1)
                result.append(token)
                if token.item() == 11: break
                embeds = m.encoder.embed_tokens(token)
        finally:
            for a, old in zip(attention, previous): a.diffusion_lm = old
        return torch.cat(result, dim=1).cpu()

    def language_loss(self, example, response=None, teacher=None):
        m = self.model
        if teacher is None:
            ids, clean, labels = self.sft_tokens(example)
        else:
            prompt_ids, prompt = self.tokenize(example.prompt, example.images, thinking=True)
            response = response.to(m.device)
            ids = torch.cat([prompt_ids, response], 1)
            clean = torch.cat([prompt, m.encoder.embed_tokens(response)], 1)
            labels = torch.cat([torch.full_like(prompt_ids, -100), response], 1)
        length = ids.shape[1]
        time = NOISE_EPS + (1 - NOISE_EPS) * torch.rand((), device=m.device)
        selected = (torch.rand(ids.shape, device=m.device) < time) & labels.ne(-100)
        prompt_dropped = teacher is None and bool(torch.rand((), device=m.device) < .1)
        if prompt_dropped:
            clean = torch.where(labels.eq(-100)[..., None], m.encoder.embed_tokens(torch.tensor(100, device=m.device)), clean)
        noisy = torch.where(selected[..., None], m.encoder.embed_tokens(torch.tensor(100, device=m.device)), clean)
        states = m.encoder(inputs_embeds=torch.cat([noisy, clean], 1), use_cache=False,
            position_ids=torch.arange(length, device=m.device)[None].repeat(1, 2),
            training_mask=block_diffusion_mask(length, self.options['block_size'], m.device),
            overwrite_attn_impl='flash_attn').last_hidden_state
        target_positions = labels[0, 1:].ne(-100).nonzero().flatten() + 1
        diff = states[0, target_positions - 1]
        ar = states[0, length + target_positions - 1]
        targets = labels[0, target_positions]
        selected = selected[0, target_positions]
        ar_total, diff_total = [], []
        for offset in range(0, targets.numel(), 32):
            end = min(offset + 32, targets.numel())
            labels_chunk = targets[offset:end]
            mask = selected[offset:end]
            teacher_chunk = None if teacher is None else tuple(x[offset:end].to(m.device) for x in teacher)
            def loss_chunk(a, d, labels=labels_chunk, mask=mask, target=teacher_chunk):
                al = m.diffusion_head(a).float()
                dl = m.diffusion_head(d).float()
                if target is None:
                    la = F.cross_entropy(al, labels, reduction='sum')
                    ld = (F.cross_entropy(dl, labels, reduction='none') * mask).sum() / time
                else:
                    ar_kl, diff_kl = opd_token_losses(al, dl, *target)
                    la, ld = ar_kl.sum(), (diff_kl * mask).sum()
                return torch.stack([la, ld])
            pair = checkpoint(loss_chunk, ar[offset:end], diff[offset:end], use_reentrant=False)
            ar_total.append(pair[0]); diff_total.append(pair[1])
        ar_count, diff_count = text_denominators(targets.numel(), int(selected.sum()), opd=teacher is not None)
        ar_loss, diff_loss = sum(ar_total) / ar_count, sum(diff_total) / diff_count
        weight = self.options['opd_ar_weight'] if teacher is not None else 1.
        loss = weight * ar_loss + diff_loss
        return loss, {'ar': float(ar_loss.detach()), 'diffusion': float(diff_loss.detach()),
                      'text_time': float(time), 'response_tokens': targets.numel(),
                      'masked_tokens': int(selected.sum()), 'diffusion_denominator': diff_count,
                      'prompt_dropped': prompt_dropped}

    def image_loss(self, example, target):
        m, marks, util = self.model, self.parts['markers'], self.parts['image_utils']
        resolution = self.options['resolution']
        with torch.no_grad():
            image = util.pad_to_square_and_resize(target, resolution)
            codes, shape = m.encode_image_gen(m.encoder.image_processor_gen.preprocess(image).to(m.device, torch.bfloat16))
        codes, shape, structural = marks.append_image_row_markers(codes, shape,
            row_marker_id=131072, image_end_id=131074, markers_per_row=2)
        parents = shape[0] * shape[1] // 4
        source = None
        if example.images:
            with torch.no_grad():
                pixels = util.pad_to_square_and_resize(example.images[0], resolution)
                source_codes, source_shape = m.encode_image_gen(m.encoder.image_processor_gen.preprocess(pixels).to(m.device, torch.bfloat16))
            source = m.encoder.call_gen_embedding(source_codes, source_shape, enc=True)
            question = '<image> ' + '<SPECIAL_19>' * source.shape[1] + '\n ' + example.prompt + ' '
        else:
            question = f'Generate an image with the caption:\n {example.prompt} ORIGINAL WIDTH : {resolution}; ORIGINAL HEIGHT : {resolution}; TOP : 0; LEFT : 0; SCORE : 7.110;'
        ids, inputs = self.tokenize(question, example.images, 'Sure <SPECIAL_21>' + '<SPECIAL_18>' * (parents + 128) + '<SPECIAL_22>')
        fixed, generated, fixed_ids = marks.split_generation_fixed_prefix(ids.eq(18), codebook_size=131072, count=128)
        inputs[fixed] = m.encoder.call_gen_fixed_token_embedding(fixed_ids).to(inputs.dtype)
        if source is not None: inputs[ids.eq(19)] = source.flatten(0, 1).to(inputs.dtype)
        tree = self.tree.to(m.device)
        protected = structural.clone()
        if source is not None:
            padding = self.parts['padding']._source_padding_vq_mask(example.images[0].size, source_shape, device=m.device)
            protected[~structural] = padding.flatten()
        state = corrupt(codes, tree, structural_mask=protected)
        child = embeddings(state, m.encoder.gen_embedding, 131073)
        inputs[generated] = m.encoder.call_gen_embedding(gen_shape=shape, child_embeddings=child).flatten(0, 1).to(inputs.dtype)
        route = fixed | generated | ids.eq(19)
        drop, drop_route, cfg_mode = image_condition_dropout(ids, fixed | generated, ids.eq(19), float(torch.rand((), device=m.device)))
        inputs = torch.where(drop[..., None], m.encoder.embed_tokens(torch.tensor(100, device=m.device)), inputs)
        route = route & ~drop_route
        hidden = m.encoder(inputs_embeds=inputs, routing=route, use_cache=False,
            training_mask=image_mask((fixed | generated)[0]), overwrite_attn_impl='flash_attn').last_hidden_state
        expanded = m.encoder.packed_image_codec.expand_fixed_grid(hidden[generated][None], shape)[0]
        mask = state.supervised[0]
        selected, labels = expanded[mask], codes[0, mask]
        values = []
        for offset in range(0, len(labels), 32):
            target_chunk = labels[offset:offset+32]
            def ce(x, y=target_chunk):
                exact, grouped = image_token_losses(m.encoder.gen_predictor(x).float(), y,
                    self.image_groups, self.group_members, self.group_valid)
                return torch.stack([exact.sum(), grouped.sum()])
            values.append(checkpoint(ce, selected[offset:offset+32], use_reentrant=False))
        terms = image_mean(sum(values) if values else expanded.sum().repeat(2) * 0., int(mask.sum()))
        loss = terms[0] + .2 * terms[1]
        return loss, {'image_ce': float(terms[0].detach()), 'grouped_ce': float(terms[1].detach()),
            'num_tree_levels': state.num_tree_levels, 'cfg_mode': cfg_mode,
            'epsilon': state.epsilon, 'time': state.time, 'supervised_tokens': int(mask.sum())}

    def forward(self, action, example, task, response=None, teacher=None, target=None):
        if action == 'rollout': return self.rollout(example, task)
        if task in ('t2i_sft', 'image_edit_opd'):
            return self.image_loss(example, example.target if target is None else target)
        return self.language_loss(example, response, teacher)
