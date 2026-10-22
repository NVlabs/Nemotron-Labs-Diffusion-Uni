"""Explicit training masks; the exported model keeps its inference attention."""
import importlib
import types
import torch
import torch.nn.functional as F


def block_diffusion_mask(length, block_size, device='cpu'):
    # Order is [noisy, clean]. A noisy block sees itself and earlier clean blocks.
    index = torch.arange(2 * length, device=device)
    q, k = index[:, None], index[None, :]
    qp, kp = q % length, k % length
    return ((q < length) & (((k < length) & (qp // block_size == kp // block_size)) |
            ((k >= length) & (kp // block_size < qp // block_size)))) | (
            (q >= length) & (k >= length) & (kp <= qp))


def image_mask(generation):
    length = generation.numel()
    causal = torch.ones(length, length, dtype=torch.bool, device=generation.device).tril()
    return causal | (generation[:, None] & generation[None, :])


def preserve_rotary_precision(rotary):
    """Keep position phases in FP32 even inside the trainer's BF16 autocast."""
    original = rotary.forward
    def forward(self, x, position_ids):
        with torch.autocast(device_type=x.device.type, enabled=False):
            return original(x, position_ids)
    rotary.forward = types.MethodType(forward, rotary)


def install(model):
    core = importlib.import_module(type(model).__module__.rsplit('.', 1)[0] + '.core')
    preserve_rotary_precision(model.encoder.rotary_emb)
    for layer in model.encoder.layers:
        attention = layer.self_attn
        original = attention.forward
        def forward(self, hidden_states, position_embeddings, attention_mask=None,
                    routing=None, training_mask=None, position_ids=None,
                    cache_position=None, _original=original, **kwargs):
            if training_mask is None:
                return _original(hidden_states=hidden_states, position_embeddings=position_embeddings,
                    attention_mask=attention_mask, routing=routing, position_ids=position_ids,
                    cache_position=cache_position, **kwargs)
            shape = (*hidden_states.shape[:-1], -1, self.head_dim)
            q, k, v = [x.view(shape).transpose(1, 2) for x in self._project_qkv(hidden_states, routing)]
            q, k = core.apply_rotary_pos_emb(q, k, *position_embeddings)
            positions = position_ids[0] if position_ids is not None else cache_position
            q = q * core._get_llama_4_attn_scale(positions,
                self.config.rope_parameters['llama_4_scaling_beta'],
                self.config.rope_parameters['original_max_position_embeddings']).to(q.dtype)
            mask = training_mask
            if self.split_tower and routing is not None:
                route = core._routing_mask(routing, hidden_states)[0]
                mask = mask & (route[:, None] == route[None, :])
            out = F.scaled_dot_product_attention(q, core.repeat_kv(k, self.num_key_value_groups),
                core.repeat_kv(v, self.num_key_value_groups), attn_mask=mask[None, None],
                dropout_p=0., scale=self.scaling)
            out = out.transpose(1, 2).reshape(*hidden_states.shape[:-1], -1)
            return self._project_output(out, routing), None
        attention.forward = types.MethodType(forward, attention)
