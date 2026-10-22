from __future__ import annotations
import copy
import functools
import json
import time
from collections import OrderedDict
from dataclasses import dataclass
from typing import Any
import torch

class _RequestNfeModelRunner:

    def __init__(self, model_runner: Any) -> None:
        self._model_runner = model_runner
        self.counts: dict[str, int] = {}

    def __getattr__(self, name: str) -> Any:
        return getattr(self._model_runner, name)

    def forward(self, forward_batch: Any, *args: Any, **kwargs: Any) -> Any:
        rids = list(getattr(forward_batch, 'rids', ()) or ())
        for rid in rids[:int(forward_batch.batch_size)]:
            request_id = str(rid)
            self.counts[request_id] = self.counts.get(request_id, 0) + 1
        return self._model_runner.forward(forward_batch, *args, **kwargs)

@dataclass
class _ThinkingBudgetState:
    reasoning_tokens: int = 0
    closed: bool = False

@dataclass(frozen=True)
class _ThinkingBudgetDecision:
    output_tokens: torch.Tensor
    force_close_seed: bool = False

class _ExactThinkingBudget:

    def __init__(self, *, max_thinking_tokens: int, end_think_token_id: int, max_states: int) -> None:
        if isinstance(max_thinking_tokens, bool) or max_thinking_tokens < 0:
            raise ValueError('max_thinking_tokens must be a non-negative integer')
        if isinstance(end_think_token_id, bool) or end_think_token_id < 0:
            raise ValueError('end_think_token_id must be a non-negative integer')
        if max_states < 1:
            raise ValueError('max_states must be positive')
        self.max_thinking_tokens = int(max_thinking_tokens)
        self.end_think_token_id = int(end_think_token_id)
        self._max_states = int(max_states)
        self._states: OrderedDict[Any, _ThinkingBudgetState] = OrderedDict()

    @classmethod
    def from_config(cls, config: Any) -> _ExactThinkingBudget | None:
        algorithm_config = config.algorithm_config
        raw_budget = algorithm_config.get('max_thinking_tokens')
        if raw_budget is None:
            return None
        if isinstance(raw_budget, bool):
            raise TypeError('max_thinking_tokens must be an integer or null')
        try:
            budget = int(raw_budget)
        except (TypeError, ValueError) as error:
            raise TypeError('max_thinking_tokens must be an integer or null') from error
        if budget < 0:
            raise ValueError('max_thinking_tokens must be non-negative')
        raw_end_token = algorithm_config.get('end_think_token_id')
        if raw_end_token is None:
            raise ValueError('end_think_token_id is required when max_thinking_tokens is set')
        if isinstance(raw_end_token, bool):
            raise TypeError('end_think_token_id must be an integer')
        try:
            end_token = int(raw_end_token)
        except (TypeError, ValueError) as error:
            raise TypeError('end_think_token_id must be an integer') from error
        if end_token < 0:
            raise ValueError('end_think_token_id must be non-negative')
        return cls(max_thinking_tokens=budget, end_think_token_id=end_token, max_states=max(64, int(config.max_running_requests) * 8))

    def _state(self, rid: Any) -> _ThinkingBudgetState:
        state = self._states.pop(rid, None)
        if state is None:
            state = _ThinkingBudgetState()
        self._states[rid] = state
        while len(self._states) > self._max_states:
            self._states.popitem(last=False)
        return state

    def forget(self, rid: Any) -> None:
        self._states.pop(rid, None)

    def prepare_seed(self, rid: Any, seed: int) -> int:
        state = self._state(rid)
        if not state.closed and state.reasoning_tokens == self.max_thinking_tokens:
            return self.end_think_token_id
        return int(seed)

    @staticmethod
    def _first_offset(tokens: torch.Tensor, token_id: int | None) -> int | None:
        if token_id is None or tokens.numel() == 0:
            return None
        offsets = torch.nonzero(tokens.eq(token_id), as_tuple=False).flatten()
        return int(offsets[0].item()) if offsets.numel() else None

    def apply(self, rid: Any, output_tokens: torch.Tensor, *, eos_token_id: int | None) -> _ThinkingBudgetDecision:
        if output_tokens.ndim != 1:
            raise ValueError('SGLang DLLM output tokens must be one-dimensional')
        state = self._state(rid)
        eos_offset = self._first_offset(output_tokens, eos_token_id)
        if state.closed:
            if eos_offset is not None:
                self.forget(rid)
            return _ThinkingBudgetDecision(output_tokens)
        remaining = self.max_thinking_tokens - state.reasoning_tokens
        if remaining < 0:
            raise RuntimeError('thinking-budget state exceeded its configured boundary')
        close_offset = self._first_offset(output_tokens, self.end_think_token_id)
        if eos_offset is not None and eos_offset < remaining and (close_offset is None or eos_offset < close_offset):
            self.forget(rid)
            return _ThinkingBudgetDecision(output_tokens)
        if close_offset is not None and close_offset <= remaining and (eos_offset is None or close_offset <= eos_offset):
            state.reasoning_tokens += close_offset
            state.closed = True
            if eos_offset is not None:
                self.forget(rid)
            return _ThinkingBudgetDecision(output_tokens)
        row_length = int(output_tokens.numel())
        if row_length < remaining:
            state.reasoning_tokens += row_length
            return _ThinkingBudgetDecision(output_tokens)
        if remaining == 0:
            raise RuntimeError('thinking-budget boundary seed was not prepared before generation')
        state.reasoning_tokens = self.max_thinking_tokens
        return _ThinkingBudgetDecision(output_tokens[:remaining], force_close_seed=True)

def install_nlduni_scheduler_metrics() -> None:
    from sglang.srt.dllm.mixin.scheduler import SchedulerDllmMixin
    process_result = SchedulerDllmMixin.process_batch_result_dllm
    if getattr(process_result, '_nlduni_records_prefill_finished_time', False):
        return
    SchedulerDllmMixin.process_batch_result_dllm = _record_dllm_prefill_finished(process_result)

def _align_shifted_block_logits(full_logits: torch.Tensor, *, block_size: int) -> torch.Tensor:
    if full_logits.ndim != 2:
        raise ValueError('SGLang DLLM full_logits must have shape [tokens, vocab]')
    if block_size < 1:
        raise ValueError('block_size must be positive')
    if full_logits.shape[0] % block_size:
        raise ValueError('SGLang DLLM full_logits token count must be divisible by block_size')
    blocks = full_logits.reshape(-1, block_size, full_logits.shape[-1])
    aligned = torch.cat([blocks[:, :1], blocks[:, :-1]], dim=1)
    return aligned.reshape_as(full_logits)

class _ShiftedLogitsModelRunner:

    def __init__(self, model_runner: Any, *, block_size: int) -> None:
        self._model_runner = model_runner
        self._block_size = block_size
        self.latest_raw_logits_output: Any | None = None

    def __getattr__(self, name: str) -> Any:
        return getattr(self._model_runner, name)

    def forward(self, *args: Any, **kwargs: Any) -> Any:
        raw_output = self._model_runner.forward(*args, **kwargs)
        raw_logits_output = raw_output.logits_output
        self.latest_raw_logits_output = raw_logits_output
        full_logits = getattr(raw_logits_output, 'full_logits', None)
        if full_logits is None:
            return raw_output
        aligned_logits_output = copy.copy(raw_logits_output)
        aligned_logits_output.full_logits = _align_shifted_block_logits(full_logits, block_size=self._block_size)
        aligned_output = copy.copy(raw_output)
        aligned_output.logits_output = aligned_logits_output
        return aligned_output

def install_nlduni_fastdiffuser() -> None:
    from sglang.srt.dllm.algorithm import algo_name_to_cls
    from sglang.srt.dllm.algorithm.fastdiffuser import FastDiffuser
    from sglang.srt.layers.logits_processor import LogitsProcessorOutput
    if 'NLDUniFastDiffuser' in algo_name_to_cls:
        return

    class NLDUniFastDiffuser(FastDiffuser):

        def __init__(self, config: Any) -> None:
            super().__init__(config)
            shift_logits = config.algorithm_config.get('shift_logits', False)
            if not isinstance(shift_logits, bool):
                raise TypeError('NLDUniFastDiffuser shift_logits must be boolean')
            self.shift_logits = shift_logits
            self._nlduni_thinking_budget = _ExactThinkingBudget.from_config(config)
            self._nlduni_stats_file = self._stats_file
            self._stats_file = None
            self._nlduni_seed_tokens: OrderedDict[str, int] = OrderedDict()
            self._nlduni_max_seed_states = max(64, int(config.max_running_requests) * 8)

        def _set_seed(self, rid: str, seed: int) -> None:
            self._nlduni_seed_tokens[rid] = seed
            self._nlduni_seed_tokens.move_to_end(rid)
            while len(self._nlduni_seed_tokens) > self._nlduni_max_seed_states:
                (oldest, _) = self._nlduni_seed_tokens.popitem(last=False)
                if self._nlduni_thinking_budget is not None:
                    self._nlduni_thinking_budget.forget(oldest)

        def _apply_thinking_budget(self, rids: list[str], output_rows: list[torch.Tensor], *, eos_id: int | None) -> list[torch.Tensor]:
            budget = self._nlduni_thinking_budget
            if budget is None:
                return output_rows
            bounded_rows: list[torch.Tensor] = []
            for (rid, output_tokens) in zip(rids, output_rows, strict=True):
                decision = budget.apply(rid, output_tokens, eos_token_id=eos_id)
                if decision.force_close_seed:
                    self._set_seed(rid, budget.end_think_token_id)
                bounded_rows.append(decision.output_tokens)
            return bounded_rows

        def _sample_seed(self, logits: torch.Tensor) -> int:
            logits = logits.clone()
            logits[self.mask_id] = -torch.inf
            if self.temperature and self.temperature > 0:
                probabilities = torch.softmax(logits.float() / float(self.temperature), dim=-1)
                return int(torch.multinomial(probabilities, 1).item())
            return int(torch.argmax(logits).item())

        def _capture_prefill_seeds(self, forward_batch, logits_output) -> None:
            rids = list(getattr(forward_batch, 'rids', ()) or ())
            next_logits = getattr(logits_output, 'next_token_logits', None)
            if not rids or next_logits is None:
                return
            for (row, rid) in enumerate(rids[:next_logits.shape[0]]):
                request_id = str(rid)
                seed = self._sample_seed(next_logits[row])
                if self._nlduni_thinking_budget is not None:
                    seed = self._nlduni_thinking_budget.prepare_seed(request_id, seed)
                self._set_seed(request_id, seed)

        def _compact_forward_batch(self, forward_batch, active_indices):
            compacted = super()._compact_forward_batch(forward_batch, active_indices)
            if forward_batch.rids is not None:
                compacted.rids = [forward_batch.rids[index] for index in active_indices]
            return compacted

        def _write_nfe_stats(self, rids: list[str], counts: dict[str, int]) -> None:
            if not self._nlduni_stats_file:
                return
            with open(self._nlduni_stats_file, 'a', encoding='utf-8') as stream:
                for rid in rids:
                    stream.write(json.dumps({'rid': rid, 'nfe': counts.get(rid, 0)}) + '\n')

        def run(self, model_runner, forward_batch):
            has_masks = bool((forward_batch.input_ids == self.mask_id).any())
            if not has_masks:
                result = super().run(model_runner, forward_batch)
                self._capture_prefill_seeds(forward_batch, result[0])
                return result
            batch_size = int(forward_batch.batch_size)
            rids = [str(rid) for rid in forward_batch.rids[:batch_size]]
            eos_id = self._get_eos_id(model_runner)
            injected: list[int | None] = []
            terminal_seed: list[bool] = []
            for (row, rid) in enumerate(rids):
                seed = self._nlduni_seed_tokens.get(rid)
                if seed is not None:
                    if self._nlduni_thinking_budget is not None:
                        seed = self._nlduni_thinking_budget.prepare_seed(rid, seed)
                        self._nlduni_seed_tokens[rid] = seed
                    self._nlduni_seed_tokens.move_to_end(rid)
                block_start = row * self.block_size
                block_end = block_start + self.block_size
                block = forward_batch.input_ids[block_start:block_end]
                mask_positions = torch.nonzero(block.eq(self.mask_id), as_tuple=False).flatten()
                if seed is None or mask_positions.numel() == 0:
                    injected.append(None)
                    terminal_seed.append(False)
                    continue
                block[int(mask_positions[0].item())] = seed
                injected.append(seed)
                is_terminal = eos_id is not None and seed == eos_id
                terminal_seed.append(is_terminal)
                if is_terminal:
                    block[block.eq(self.mask_id)] = eos_id
            if self.causal_context and any((seed is None for seed in injected)):
                missing = [rid for (rid, seed) in zip(rids, injected) if seed is None]
                raise RuntimeError(f'NLDUni FastDiffuser has no causal block seed for requests {missing}')
            active_indices = [index for (index, is_terminal) in enumerate(terminal_seed) if not is_terminal]
            counting_runner = _RequestNfeModelRunner(model_runner)
            output_rows = [forward_batch.input_ids.new_empty(0) for _ in range(batch_size)]
            if active_indices:
                active_batch = forward_batch if len(active_indices) == batch_size else self._compact_forward_batch(forward_batch, active_indices)
                shifted_runner = None
                fastdiffuser_runner = counting_runner
                if self.shift_logits:
                    shifted_runner = _ShiftedLogitsModelRunner(counting_runner, block_size=self.block_size)
                    fastdiffuser_runner = shifted_runner
                (logits_output, active_output_rows, can_run_graph) = super().run(fastdiffuser_runner, active_batch)
                for (original_row, row_tokens) in zip(active_indices, active_output_rows, strict=True):
                    output_rows[original_row] = row_tokens
                if shifted_runner is not None:
                    raw_logits_output = shifted_runner.latest_raw_logits_output
                    if raw_logits_output is None:
                        raise RuntimeError('shifted NLDUni FastDiffuser completed without raw logits')
                    logits_output = raw_logits_output
            else:
                logits_output = LogitsProcessorOutput(next_token_logits=None, full_logits=None)
                can_run_graph = False
            self._write_nfe_stats(rids, counting_runner.counts)
            full_logits = getattr(logits_output, 'full_logits', None)
            if full_logits is not None:
                for (active_row, original_row) in enumerate(active_indices):
                    last_position = (active_row + 1) * self.block_size - 1
                    self._set_seed(rids[original_row], self._sample_seed(full_logits[last_position]))
            seeded_rows: list[torch.Tensor] = []
            for (rid, seed, row_tokens) in zip(rids, injected, output_rows):
                if seed is None:
                    seeded = row_tokens
                else:
                    seeded = torch.cat([row_tokens.new_tensor([seed]), row_tokens])
                if eos_id is not None:
                    eos_positions = torch.nonzero(seeded.eq(eos_id), as_tuple=False).flatten()
                    if eos_positions.numel():
                        seeded = seeded[:int(eos_positions[0].item()) + 1]
                        self._nlduni_seed_tokens.pop(rid, None)
                seeded_rows.append(seeded)
            if self._nlduni_thinking_budget is not None:
                seeded_rows = self._apply_thinking_budget(rids, seeded_rows, eos_id=eos_id)
            return (logits_output, seeded_rows, can_run_graph)
    NLDUniFastDiffuser.__name__ = 'NLDUniFastDiffuser'
    algo_name_to_cls['NLDUniFastDiffuser'] = NLDUniFastDiffuser
__all__ = ['install_nlduni_fastdiffuser']
