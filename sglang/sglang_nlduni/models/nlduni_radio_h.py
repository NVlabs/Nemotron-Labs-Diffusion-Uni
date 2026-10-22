from __future__ import annotations
import json
from collections.abc import Iterable
from dataclasses import dataclass, field
from itertools import chain
from pathlib import Path
from typing import Any
import torch
from sglang_nlduni.projector import RadioHProjector
from sglang.srt.configs.radio import RadioConfig
from sglang.srt.layers.quantization.base_config import QuantizationConfig
from sglang.srt.managers.mm_utils import general_mm_embed_routine
from sglang.srt.managers.schedule_batch import Modality, MultimodalDataItem
from sglang.srt.model_executor.forward_batch_info import ForwardBatch, PPProxyTensors
from sglang.srt.model_loader.weight_utils import default_weight_loader
from sglang.srt.models.nemotron_labs_dllm import NemotronLabsDiffusionModel as _TextDiffusionModel
from sglang.srt.models.radio import RadioModel
_ShardId = str | int | None
_STACKED_TEXT_MAPPING: tuple[tuple[str, str, _ShardId], ...] = (('.self_attn.qkv_proj', '.self_attn.q_proj', 'q'), ('.self_attn.qkv_proj', '.self_attn.k_proj', 'k'), ('.self_attn.qkv_proj', '.self_attn.v_proj', 'v'), ('.mlp.gate_up_proj', '.mlp.gate_proj', 0), ('.mlp.gate_up_proj', '.mlp.up_proj', 1))
_IGNORED_VISION_SOURCE_NAMES = frozenset({'radio_model.summary_idxs'})

@dataclass
class _RadioHWeightLoadState:
    projector_params: dict[str, torch.nn.Parameter]
    vision_param_map: dict[str, torch.nn.Parameter]
    text_params: set[str]
    loaded_projector: set[str] = field(default_factory=set)
    loaded_vision: set[str] = field(default_factory=set)
    loaded_text_parts: dict[str, set[_ShardId]] = field(default_factory=dict)
    unmatched_projector_sources: list[str] = field(default_factory=list)
    unmatched_vision_sources: list[str] = field(default_factory=list)
    unmatched_text_sources: list[str] = field(default_factory=list)

def _load_radio_config(config):
    return (RadioConfig(model_name='vit_huge_patch16_224', image_size=768, patch_size=16, max_img_size=2048, reg_tokens=10, min_num_patches=1024, max_num_patches=13312, video_temporal_patch_size=1, separate_video_embedder=False), 13312)

class NLDUniForCausalLM(_TextDiffusionModel):

    def __init__(self, config, quant_config: QuantizationConfig | None=None, prefix: str='') -> None:
        super().__init__(config, quant_config=quant_config, prefix=prefix)
        (radio_config, max_patches) = _load_radio_config(config)
        self.vision_model = RadioModel(radio_config, quant_config=None)
        self.mm_projector = RadioHProjector(config)
        self._max_vision_patches_per_forward = max(1, max_patches)
        dtype = self.model.embed_tokens.weight.dtype
        self.vision_model.to(dtype=dtype)
        self.mm_projector.to(dtype=dtype)

    def get_input_embeddings(self):
        return self.model.embed_tokens

    @staticmethod
    def pad_input_ids(input_ids: list[int], mm_inputs) -> list[int]:
        padded = list(input_ids)
        for item in mm_inputs.mm_items:
            if not item.is_image() or item.offsets is None:
                continue
            if item.pad_value is None:
                raise ValueError('RADIO-H multimodal item is missing pad_value')
            for (start, end) in item.offsets:
                padded[int(start):int(end) + 1] = [int(item.pad_value)] * (int(end) - int(start) + 1)
        return padded

    def _encode_image_chunk(self, items: list[MultimodalDataItem]) -> torch.Tensor:
        reference = next(self.vision_model.parameters())
        pixels = [item.feature.to(device=reference.device, dtype=reference.dtype) for item in items]
        (features, _) = self.vision_model(pixels)
        packed_features = features.squeeze(0)
        grids = torch.cat([torch.as_tensor(item.image_grid_thw) for item in items], dim=0).to(device=packed_features.device)
        return self.mm_projector(packed_features, grids)

    def get_image_feature(self, items: list[MultimodalDataItem]) -> torch.Tensor:
        outputs: list[torch.Tensor] = []
        active: list[MultimodalDataItem] = []
        active_patches = 0
        for item in items:
            grid = torch.as_tensor(item.image_grid_thw)
            patch_count = int(grid.prod(dim=-1).sum().item())
            if active and active_patches + patch_count > self._max_vision_patches_per_forward:
                outputs.append(self._encode_image_chunk(active))
                active = []
                active_patches = 0
            active.append(item)
            active_patches += patch_count
        if active:
            outputs.append(self._encode_image_chunk(active))
        if not outputs:
            return self.model.embed_tokens.weight.new_empty((0, self.config.hidden_size))
        return torch.cat(outputs, dim=0)

    @torch.no_grad()
    def forward(self, input_ids: torch.Tensor | None, positions: torch.Tensor, forward_batch: ForwardBatch, input_embeds: torch.Tensor | None=None, pp_proxy_tensors: PPProxyTensors | None=None):
        if input_embeds is not None or not self.pp_group.is_first_rank:
            return super().forward(input_ids=input_ids, positions=positions, forward_batch=forward_batch, input_embeds=input_embeds, pp_proxy_tensors=pp_proxy_tensors)
        if input_ids is None:
            raise ValueError('input_ids are required before multimodal embedding')
        if forward_batch.forward_mode.is_dllm_extend():
            return super().forward(input_ids=input_ids, positions=positions, forward_batch=forward_batch, input_embeds=None, pp_proxy_tensors=pp_proxy_tensors)
        return general_mm_embed_routine(input_ids=input_ids, positions=positions, forward_batch=forward_batch, language_model=self, multimodal_model=self, data_embedding_funcs={Modality.IMAGE: self.get_image_feature}, pp_proxy_tensors=pp_proxy_tensors)

    def _new_weight_load_state(self) -> _RadioHWeightLoadState:
        return _RadioHWeightLoadState(projector_params=dict(self.mm_projector.named_parameters()), vision_param_map=dict(self.vision_model.named_parameters()), text_params={name for (name, _) in self.named_parameters() if not name.startswith(('vision_model.', 'mm_projector.'))})

    @staticmethod
    def _text_target(name: str) -> tuple[str, _ShardId]:
        target_name = name
        if target_name.startswith('encoder.'):
            target_name = 'model.' + target_name[len('encoder.'):]
        shard: _ShardId = None
        for (param_name, shard_name, shard_id) in _STACKED_TEXT_MAPPING:
            if shard_name in target_name:
                target_name = target_name.replace(shard_name, param_name)
                shard = shard_id
                break
        return (target_name, shard)

    def _dispatch_weight_stream(self, weights: Iterable[tuple[str, torch.Tensor]], state: _RadioHWeightLoadState) -> Any:

        def text_weights():
            for (name, loaded_weight) in weights:
                if name.startswith('encoder.vision_tower.vision_tower.'):
                    radio_name = name[len('encoder.vision_tower.vision_tower.'):]
                    loaded = self.vision_model.load_weights([(radio_name, loaded_weight)])
                    if not loaded:
                        if radio_name not in _IGNORED_VISION_SOURCE_NAMES:
                            state.unmatched_vision_sources.append(name)
                        continue
                    duplicates = loaded & state.loaded_vision
                    if duplicates:
                        raise ValueError(f'duplicate RADIO-H vision targets in one load: {sorted(duplicates)}')
                    state.loaded_vision.update(loaded)
                    continue
                if name.startswith('encoder.mm_projector.'):
                    projector_name = name[len('encoder.mm_projector.'):]
                    parameter = state.projector_params.get(projector_name)
                    if parameter is None:
                        state.unmatched_projector_sources.append(name)
                        continue
                    if projector_name in state.loaded_projector:
                        raise ValueError(f'duplicate RADIO-H projector target in one load: {projector_name}')
                    loader = getattr(parameter, 'weight_loader', default_weight_loader)
                    loader(parameter, loaded_weight)
                    state.loaded_projector.add(projector_name)
                    continue
                if is_ignored_radio_h_source_name(name):
                    continue
                (target_name, shard) = self._text_target(name)
                if target_name not in state.text_params:
                    state.unmatched_text_sources.append(name)
                else:
                    loaded_parts = state.loaded_text_parts.setdefault(target_name, set())
                    if shard in loaded_parts:
                        raise ValueError(f'duplicate RADIO-H text target part in one load: source={name!r}, target={target_name!r}, shard={shard!r}')
                    loaded_parts.add(shard)
                yield (name, loaded_weight)
        return super().load_weights(text_weights())

    def _validate_complete_weight_load(self, state: _RadioHWeightLoadState, *, operation: str) -> None:
        missing_projector = set(state.projector_params) - state.loaded_projector
        missing_vision = set(state.vision_param_map) - state.loaded_vision
        missing_text_parts: dict[str, list[str]] = {}
        for name in sorted(state.text_params):
            if '.self_attn.qkv_proj' in name:
                expected_parts: set[_ShardId] = {'q', 'k', 'v'}
            elif '.mlp.gate_up_proj' in name:
                expected_parts = {0, 1}
            else:
                expected_parts = {None}
            missing_parts = expected_parts - state.loaded_text_parts.get(name, set())
            if missing_parts:
                missing_text_parts[name] = sorted(map(str, missing_parts))
        identity_layer_scales = {f'model.encoder.layers.{layer}.ls{branch}' for layer in range(self.vision_model.config.num_hidden_layers) for branch in (1, 2)}
        missing_identity_scales = missing_vision & identity_layer_scales
        partially_missing_scales = bool(missing_identity_scales) and missing_identity_scales != identity_layer_scales
        if missing_identity_scales == identity_layer_scales:
            with torch.no_grad():
                for name in missing_identity_scales:
                    state.vision_param_map[name].fill_(1)
        nonidentity_scales = [name for name in sorted(missing_identity_scales) if not torch.equal(state.vision_param_map[name].detach(), torch.ones_like(state.vision_param_map[name]))]
        unexpected_missing_vision = missing_vision - identity_layer_scales
        if missing_projector or missing_text_parts or state.unmatched_projector_sources or state.unmatched_vision_sources or state.unmatched_text_sources or unexpected_missing_vision or partially_missing_scales or nonidentity_scales:
            raise RuntimeError(f'incomplete RADIO-H {operation}: missing text parts={missing_text_parts}, unmatched text sources={sorted(state.unmatched_text_sources)}, missing projector={sorted(missing_projector)}, unmatched projector sources={sorted(state.unmatched_projector_sources)}, missing vision={sorted(unexpected_missing_vision)}, unmatched vision sources={sorted(state.unmatched_vision_sources)}, partially missing layer scales={partially_missing_scales}, non-identity target-only layer scales={nonidentity_scales}')

    def load_weights(self, weights: Iterable[tuple[str, torch.Tensor]]):
        state = self._new_weight_load_state()
        result = self._dispatch_weight_stream(weights, state)
        self._validate_complete_weight_load(state, operation='checkpoint load')
        return result
EntryClass = NLDUniForCausalLM
__all__ = ['EntryClass', 'NLDUniForCausalLM']

def is_ignored_radio_h_source_name(name):
    return name.startswith(('encoder.vqvae.', 'encoder.gen_embedding.', 'encoder.gen_predictor.', 'encoder.packed_image_codec.', 'encoder.norm_gen.')) or name == 'encoder.image_newline' or any((part in name for part in ('.q_proj_gen.', '.k_proj_gen.', '.v_proj_gen.', '.o_proj_gen.', '.gate_proj_gen.', '.up_proj_gen.', '.down_proj_gen.', '.input_layernorm_gen.', '.post_attention_layernorm_gen.')))
