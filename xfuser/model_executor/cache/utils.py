"""
adapted from https://github.com/ali-vilab/TeaCache.git
adapted from https://github.com/chengzeyi/ParaAttention.git
"""
import contextlib
import dataclasses
from collections import defaultdict
from typing import Dict, Optional, List, DefaultDict, Union, Any
from xfuser.core.distributed import (
    get_sp_group,
    get_sequence_parallel_world_size,
)
import torch
from torch.nn import Module
from abc import ABC, abstractmethod
from xfuser.logger import init_logger

logger = init_logger(__name__)


# --------- CacheContext --------- #
class CacheContext(Module):
    def __init__(self):
        super().__init__()
        self.register_buffer("default_coef", torch.tensor([1.0, 0.0]).cuda())
        self.register_buffer("flux_coef",
                             torch.tensor([498.651651, -283.781631, 55.8554382, -3.82021401, 0.264230861]).cuda())

        self.register_buffer("original_hidden_states", None, persistent=False)
        self.register_buffer("original_encoder_hidden_states", None, persistent=False)
        self.register_buffer("hidden_states_residual", None, persistent=False)
        self.register_buffer("encoder_hidden_states_residual", None, persistent=False)
        self.register_buffer("modulated_inputs", None, persistent=False)

    def get_coef(self, name: str) -> torch.Tensor:
        return getattr(self, f"{name}_coef")


#---------  CacheCallback  ---------#
@dataclasses.dataclass
class CacheState:
    transformer: Optional[torch.nn.Module] = None
    transformer_blocks: Optional[List[torch.nn.Module]] = None
    single_transformer_blocks: Optional[List[torch.nn.Module]] = None
    cache_context: Optional[CacheContext] = None
    rel_l1_thresh: float = 0.6
    return_hidden_states_first: bool = True
    use_cache: torch.Tensor = torch.tensor(False, dtype=torch.bool)
    num_steps: int = 8
    name: str = "default"


class CacheCallback:
    def on_init_end(self, state: CacheState, **kwargs): pass

    def on_forward_begin(self, state: CacheState, **kwargs): pass

    def on_forward_remaining_begin(self, state: CacheState, **kwargs): pass

    def on_forward_end(self, state: CacheState, **kwargs): pass


class CallbackHandler(CacheCallback):
    def __init__(self, callbacks: Optional[List[CacheCallback]] = None):
        self.callbacks = list(callbacks) if callbacks else []

    def trigger_event(self, event: str, state: CacheState):
        for cb in self.callbacks:
            getattr(cb, event)(state)


# --------- Vectorized Poly1D --------- #
class VectorizedPoly1D(Module):
    def __init__(self, coefficients: torch.Tensor):
        super().__init__()
        self.register_buffer("coefficients", coefficients)
        self.degree = len(coefficients) - 1

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        result = torch.zeros_like(x)
        for i, coef in enumerate(self.coefficients):
            result += coef * (x ** (self.degree - i))
        return result


class CachedTransformerBlocks(torch.nn.Module, ABC):
    def __init__(
        self,
        transformer_blocks: List[Module],
        single_transformer_blocks: Optional[List[Module]] = None,
        *,
        transformer: Optional[Module] = None,
        rel_l1_thresh: float = 0.6,
        return_hidden_states_first: bool = True,
        num_steps: int = -1,
        name: str = "default",
        callbacks: Optional[List[CacheCallback]] = None,
    ):
        super().__init__()
        self.transformer_blocks = torch.nn.ModuleList(transformer_blocks)
        self.single_transformer_blocks = torch.nn.ModuleList(
            single_transformer_blocks) if single_transformer_blocks else None
        self.transformer = transformer
        self.register_buffer("cnt", torch.tensor(0).cuda())
        self.register_buffer("accumulated_rel_l1_distance", torch.tensor([0.0]).cuda())
        self.register_buffer("use_cache", torch.tensor(False, dtype=torch.bool).cuda())

        self.cache_context = CacheContext()
        self.callback_handler = CallbackHandler(callbacks)

        self.rel_l1_thresh = torch.tensor(rel_l1_thresh).cuda()
        self.return_hidden_states_first = return_hidden_states_first
        self.num_steps = num_steps
        self.name = name
        self.callback_handler.trigger_event("on_init_begin", self)

    @property
    def is_parallelized(self) -> bool:
        return get_sequence_parallel_world_size() > 1

    def all_reduce(self, input_: torch.Tensor, op=torch.distributed.ReduceOp.AVG) -> torch.Tensor:
        return get_sp_group().all_reduce(input_, op=op) if self.is_parallelized else input_

    def l1_distance(self, t1: torch.Tensor, t2: torch.Tensor) -> torch.Tensor:
        diff = (t1 - t2).abs().mean()
        norm = t1.abs().mean()
        diff, norm = self.all_reduce(diff.unsqueeze(0)), self.all_reduce(norm.unsqueeze(0))
        return (diff / norm).squeeze()

    @abstractmethod
    def are_two_tensor_similar(self, t1: torch.Tensor, t2: torch.Tensor, threshold: float) -> torch.Tensor:
        pass

    @abstractmethod
    def get_start_idx(self) -> int:
        pass

    @abstractmethod
    def get_modulated_inputs(self, hidden_states: torch.Tensor, encoder_hidden_states: torch.Tensor, *args, **kwargs):
        pass

    def process_blocks(self, start_idx: int, hidden: torch.Tensor, encoder: torch.Tensor, *args, **kwargs):
        for block in self.transformer_blocks[start_idx:]:
            hidden, encoder = block(hidden, encoder, *args, **kwargs)
            hidden, encoder = (hidden, encoder) if self.return_hidden_states_first else (encoder, hidden)

        if self.single_transformer_blocks:
            for block in self.single_transformer_blocks:
                hidden, encoder = block(hidden, encoder, *args, **kwargs)
                hidden, encoder = (hidden, encoder) if self.return_hidden_states_first else (encoder, hidden)

        self.cache_context.hidden_states_residual = hidden - self.cache_context.original_hidden_states
        self.cache_context.encoder_hidden_states_residual = encoder - self.cache_context.original_encoder_hidden_states
        return hidden, encoder

    def forward(self, hidden_states, encoder_hidden_states, *args, **kwargs):
        self.callback_handler.trigger_event("on_forward_begin", self)

        modulated, prev_modulated, orig_hidden, orig_encoder = \
            self.get_modulated_inputs(hidden_states, encoder_hidden_states, *args, **kwargs)

        self.cache_context.original_hidden_states = orig_hidden
        self.cache_context.original_encoder_hidden_states = orig_encoder

        self.use_cache = self.are_two_tensor_similar(prev_modulated, modulated, self.rel_l1_thresh) \
            if prev_modulated is not None else torch.tensor(False, dtype=torch.bool)

        self.callback_handler.trigger_event("on_forward_remaining_begin", self)
        if self.use_cache:
            hidden = hidden_states + self.cache_context.hidden_states_residual
            encoder = encoder_hidden_states + self.cache_context.encoder_hidden_states_residual
        else:
            hidden, encoder = self.process_blocks(self.get_start_idx(), orig_hidden, orig_encoder, *args, **kwargs)
        self.callback_handler.trigger_event("on_forward_end", self)
        return ((hidden, encoder) if self.return_hidden_states_first else (encoder, hidden))


class FBCachedTransformerBlocks(CachedTransformerBlocks):
    def __init__(
        self,
        transformer_blocks,
        single_transformer_blocks=None,
        *,
        transformer=None,
        rel_l1_thresh=0.6,
        return_hidden_states_first=True,
        num_steps=-1,
        name="default",
        callbacks: Optional[List[CacheCallback]] = None,
    ):
        super().__init__(transformer_blocks,
                         single_transformer_blocks=single_transformer_blocks,
                         transformer=transformer,
                         rel_l1_thresh=rel_l1_thresh,
                         num_steps=num_steps,
                         return_hidden_states_first=return_hidden_states_first,
                         name=name,
                         callbacks=callbacks)

    def get_start_idx(self) -> int:
        return 1

    def are_two_tensor_similar(self, t1: torch.Tensor, t2: torch.Tensor, threshold: torch.Tensor) -> torch.Tensor:
        return self.l1_distance(t1, t2) < threshold

    def get_modulated_inputs(self, hidden_states, encoder_hidden_states, *args, **kwargs):
        original_hidden_states = hidden_states
        first_transformer_block = self.transformer_blocks[0]
        hidden_states, encoder_hidden_states = first_transformer_block(hidden_states, encoder_hidden_states, *args,
                                                                       **kwargs)
        hidden_states, encoder_hidden_states = (
        hidden_states, encoder_hidden_states) if self.return_hidden_states_first else (
        encoder_hidden_states, hidden_states)
        first_hidden_states_residual = hidden_states - original_hidden_states
        prev_first_hidden_states_residual = self.cache_context.modulated_inputs
        if not self.use_cache:
            self.cache_context.modulated_inputs = first_hidden_states_residual
        return first_hidden_states_residual, prev_first_hidden_states_residual, hidden_states, encoder_hidden_states


class TeaCachedTransformerBlocks(CachedTransformerBlocks):
    def __init__(
        self,
        transformer_blocks,
        single_transformer_blocks=None,
        *,
        transformer=None,
        rel_l1_thresh=0.6,
        return_hidden_states_first=True,
        num_steps=-1,
        name="default",
        callbacks: Optional[List[CacheCallback]] = None,
    ):
        super().__init__(transformer_blocks,
                         single_transformer_blocks=single_transformer_blocks,
                         transformer=transformer,
                         rel_l1_thresh=rel_l1_thresh,
                         num_steps=num_steps,
                         return_hidden_states_first=return_hidden_states_first,
                         name=name,
                         callbacks=callbacks)
        self.rescale_func = VectorizedPoly1D(self.cache_context.get_coef(self.name))

    def get_start_idx(self) -> int:
        return 0

    def are_two_tensor_similar(self, t1: torch.Tensor, t2: torch.Tensor, threshold: float) -> torch.Tensor:
        diff = self.l1_distance(t1, t2)
        new_accum = self.accumulated_rel_l1_distance + self.rescale_func(diff)
        reset_mask = (self.cnt == 0) or (self.cnt == self.num_steps - 1)
        self.use_cache = torch.logical_and(new_accum < threshold, torch.logical_not(reset_mask))
        self.accumulated_rel_l1_distance[0] = torch.where(self.use_cache, new_accum[0], 0.0)
        self.cnt = torch.where(self.cnt + 1 < self.num_steps, self.cnt + 1, 0)

        return self.use_cache

    def get_modulated_inputs(self, hidden_states, encoder_hidden_states, *args, **kwargs):
        inp = hidden_states.clone()
        temb_ = kwargs.get("temb", None).clone()
        modulated, gate_msa, shift_mlp, scale_mlp, gate_mlp = self.transformer_blocks[0].norm1(inp, emb=temb_)
        prev_modulated = self.cache_context.modulated_inputs
        self.cache_context.modulated_inputs = modulated
        return modulated, prev_modulated, hidden_states, encoder_hidden_states


@dataclasses.dataclass
class CacheContextWan:
    residual_diff_threshold: Union[torch.Tensor, float] = 0.0
    downsample_factor: int = 1
    num_inference_steps: int = -1
    warmup_steps: int = 0
    buffers: Dict[str, Any] = dataclasses.field(default_factory=dict)
    executed_steps: int = 0

    def get_residual_diff_threshold(self):
        residual_diff_threshold = self.residual_diff_threshold
        if isinstance(residual_diff_threshold, torch.Tensor):
            residual_diff_threshold = residual_diff_threshold.item()
        return residual_diff_threshold

    def get_wan_coef(self):
        wan_coef = self.wan_coef
        return wan_coef

    def get_num_inference_steps(self):
        num_inference_steps = self.num_inference_steps
        return num_inference_steps

    def get_buffer(self, name):
        return self.buffers.get(name)

    def set_buffer(self, name, buffer):
        self.buffers[name] = buffer

    def remove_buffer(self, name):
        if name in self.buffers:
            del self.buffers[name]

    def clear_buffers(self):
        self.buffers.clear()

    def mark_step_begin(self):
        self.executed_steps += 1

    def get_current_step(self):
        return self.executed_steps

    def set_current_step(self):
        self.executed_steps = 0

    def is_in_warmup(self):
        return self.get_current_step() < self.warmup_steps


@torch.compiler.disable
def get_residual_diff_threshold():
    cache_context = get_current_cache_context()
    assert cache_context is not None, "cache_context must be set before"
    return cache_context.get_residual_diff_threshold()


@torch.compiler.disable
def get_buffer(name):
    cache_context = get_current_cache_context()
    assert cache_context is not None, "cache_context must be set before"
    return cache_context.get_buffer(name)


@torch.compiler.disable
def set_buffer(name, buffer):
    cache_context = get_current_cache_context()
    assert cache_context is not None, "cache_context must be set before"
    cache_context.set_buffer(name, buffer)


@torch.compiler.disable
def remove_buffer(name):
    cache_context = get_current_cache_context()
    assert cache_context is not None, "cache_context must be set before"
    cache_context.remove_buffer(name)


@torch.compiler.disable
def mark_step_begin():
    cache_context = get_current_cache_context()
    assert cache_context is not None, "cache_context must be set before"
    cache_context.mark_step_begin()


@torch.compiler.disable
def get_current_step():
    cache_context = get_current_cache_context()
    assert cache_context is not None, "cache_context must be set before"
    return cache_context.get_current_step()


@torch.compiler.disable
def set_current_step():
    cache_context = get_current_cache_context()
    assert cache_context is not None, "cache_context must be set before"
    cache_context.set_current_step()


@torch.compiler.disable
def get_num_inference_steps():
    cache_context = get_current_cache_context()
    assert cache_context is not None, "cache_context must be set before"
    return cache_context.get_num_inference_steps()


@torch.compiler.disable
def is_in_warmup():
    cache_context = get_current_cache_context()
    assert cache_context is not None, "cache_context must be set before"
    return cache_context.is_in_warmup()


_current_cache_context = None


def create_cache_context(*args, **kwargs):
    return CacheContextWan(*args, **kwargs)


def get_current_cache_context():
    return _current_cache_context


def set_current_cache_context(cache_context=None):
    global _current_cache_context
    _current_cache_context = cache_context


@contextlib.contextmanager
def cache_context(cache_context):
    global _current_cache_context
    old_cache_context = _current_cache_context
    _current_cache_context = cache_context
    try:
        yield
    finally:
        _current_cache_context = old_cache_context


@torch.compiler.disable
def apply_prev_hidden_states_residual(hidden_states, encoder_hidden_states):
    hidden_states_residual = get_hidden_states_residual()
    assert hidden_states_residual is not None, "hidden_states_residual must be set before"
    hidden_states = hidden_states_residual + hidden_states

    encoder_hidden_states_residual = get_encoder_hidden_states_residual()
    assert encoder_hidden_states_residual is not None, "encoder_hidden_states_residual must be set before"
    encoder_hidden_states = encoder_hidden_states_residual + encoder_hidden_states

    hidden_states = hidden_states.contiguous()
    encoder_hidden_states = encoder_hidden_states.contiguous()

    return hidden_states, encoder_hidden_states


@torch.compiler.disable
def get_downsample_factor():
    cache_context = get_current_cache_context()
    assert cache_context is not None, "cache_context must be set before"
    return cache_context.downsample_factor


@torch.compiler.disable
def set_first_hidden_states_residual(first_hidden_states_residual):
    downsample_factor = get_downsample_factor()
    if downsample_factor > 1:
        first_hidden_states_residual = first_hidden_states_residual[..., ::downsample_factor]
        first_hidden_states_residual = first_hidden_states_residual.contiguous()
    set_buffer("first_hidden_states_residual", first_hidden_states_residual)


@torch.compiler.disable
def get_first_hidden_states_residual():
    return get_buffer("first_hidden_states_residual")


@torch.compiler.disable
def set_hidden_states_residual(hidden_states_residual):
    set_buffer("hidden_states_residual", hidden_states_residual)


@torch.compiler.disable
def get_hidden_states_residual():
    return get_buffer("hidden_states_residual")


@torch.compiler.disable
def set_encoder_hidden_states_residual(encoder_hidden_states_residual):
    set_buffer("encoder_hidden_states_residual", encoder_hidden_states_residual)


@torch.compiler.disable
def get_encoder_hidden_states_residual():
    return get_buffer("encoder_hidden_states_residual")


class TeaCachedWanTransformerBlocks(torch.nn.Module):
    def __init__(
        self,
        transformer_blocks,
        single_transformer_blocks=None,
        *,
        transformer=None,
        return_hidden_states_first=True,
        return_hidden_states_only=False,
    ):
        super().__init__()
        self.transformer = transformer
        self.transformer_blocks = transformer_blocks
        self.single_transformer_blocks = single_transformer_blocks
        self.__class__.accumulated_rel_l1_distance = 0
        self.__class__.wan_coef = [-3.03318725e+05, 4.90537029e+04, -2.65530556e+03, 5.87365115e+01, -3.15583525e-01]
        self.__class__.previous_residual = None
        self.__class__.previous_residual_encoder = None
        self.__class__.previous_modulated_input = None
        logger.info("use TeaCache")

    def forward(self, hidden_states, encoder_hidden_states, timestep_proj, *args, **kwargs):
        if get_current_step() < 5 or get_current_step() >= (get_num_inference_steps() - 1):
            should_calc = True
            self.accumulated_rel_l1_distance = 0
        else:
            diff = self.l1_distance(self.previous_modulated_input, timestep_proj)
            result = torch.zeros_like(diff)
            for i, coef in enumerate(self.wan_coef):
                result += coef * (diff ** (len(self.wan_coef) - 1 - i))
            self.accumulated_rel_l1_distance += result
            if self.accumulated_rel_l1_distance < get_residual_diff_threshold():
                should_calc = False
            else:
                should_calc = True
                self.accumulated_rel_l1_distance = 0
        self.previous_modulated_input = timestep_proj
        mark_step_begin()

        if not should_calc:
            hidden_states += self.previous_residual
            encoder_hidden_states += self.previous_residual_encoder
        else:
            ori_hidden_states = hidden_states.clone()
            ori_encoder_hidden_states = encoder_hidden_states.clone()
            # actual Transformer blocks
            if torch.is_grad_enabled() and self.gradient_checkpointing:
                for i, encoder_block in enumerate(self.transformer_blocks[0:]):
                    hidden_states = self._gradient_checkpointing_func(
                        encoder_block, hidden_states, encoder_hidden_states, timestep_proj, *args, **kwargs)
            else:
                for i, encoder_block in enumerate(self.transformer_blocks[0:]):
                    hidden_states = encoder_block(hidden_states, encoder_hidden_states, timestep_proj, *args, **kwargs)

            self.previous_residual = hidden_states - ori_hidden_states
            self.previous_residual_encoder = encoder_hidden_states - ori_encoder_hidden_states

        return hidden_states

    @property
    def is_parallelized(self) -> bool:
        return get_sequence_parallel_world_size() > 1

    def all_reduce(self, input_: torch.Tensor, op=torch.distributed.ReduceOp.AVG) -> torch.Tensor:
        return get_sp_group().all_reduce(input_, op=op) if self.is_parallelized else input_

    def l1_distance(self, t1: torch.Tensor, t2: torch.Tensor):
        diff = (t1 - t2).abs().mean()
        norm = t1.abs().mean()
        diff, norm = self.all_reduce(diff.unsqueeze(0)), self.all_reduce(norm.unsqueeze(0))
        return (diff / norm).squeeze()


class FBCachedWanTransformerBlocks(torch.nn.Module):
    def __init__(
        self,
        transformer_blocks,
        single_transformer_blocks=None,
        *,
        transformer=None,
        return_hidden_states_first=True,
        return_hidden_states_only=False,
    ):
        super().__init__()

        self.transformer = transformer
        self.transformer_blocks = transformer_blocks
        self.single_transformer_blocks = single_transformer_blocks
        self.return_hidden_states_first = return_hidden_states_first
        self.return_hidden_states_only = return_hidden_states_only
        logger.info("use FBCache")

    def forward(self, hidden_states, encoder_hidden_states, *args, **kwargs):
        original_hidden_states = hidden_states
        first_transformer_block = self.transformer_blocks[0]
        hidden_states = first_transformer_block(hidden_states, encoder_hidden_states, *args, **kwargs)
        if not isinstance(hidden_states, torch.Tensor):
            hidden_states, encoder_hidden_states = hidden_states
            if not self.return_hidden_states_first:
                hidden_states, encoder_hidden_states = encoder_hidden_states, hidden_states
        first_hidden_states_residual = hidden_states - original_hidden_states
        del original_hidden_states

        mark_step_begin()
        #使能位置可调整
        if get_current_step() < 6:
            can_use_cache = False
        else:
            can_use_cache = self.get_can_use_cache(
                first_hidden_states_residual,
            )

        if can_use_cache:
            del first_hidden_states_residual
            hidden_states, encoder_hidden_states = apply_prev_hidden_states_residual(
                hidden_states, encoder_hidden_states
            )
        else:
            set_first_hidden_states_residual(first_hidden_states_residual)
            del first_hidden_states_residual
            (
                hidden_states,
                encoder_hidden_states,
                hidden_states_residual,
                encoder_hidden_states_residual,
            ) = self.call_remaining_transformer_blocks(hidden_states, encoder_hidden_states, *args, **kwargs)
            set_hidden_states_residual(hidden_states_residual)
            set_encoder_hidden_states_residual(encoder_hidden_states_residual)
        torch._dynamo.graph_break()

        return (
            hidden_states
            if self.return_hidden_states_only
            else (
                (hidden_states, encoder_hidden_states)
                if self.return_hidden_states_first
                else (encoder_hidden_states, hidden_states)
            )
        )

    def call_remaining_transformer_blocks(self, hidden_states, encoder_hidden_states, *args, **kwargs):
        original_hidden_states = hidden_states
        original_encoder_hidden_states = encoder_hidden_states
        for i, encoder_block in enumerate(self.transformer_blocks[1:]):
            hidden_states = encoder_block(hidden_states, encoder_hidden_states, *args, **kwargs)
            if not isinstance(hidden_states, torch.Tensor):
                hidden_states, encoder_hidden_states = hidden_states
                if not self.return_hidden_states_first:
                    hidden_states, encoder_hidden_states = encoder_hidden_states, hidden_states
        if self.single_transformer_blocks is not None:
            hidden_states = encoder_block(hidden_states, encoder_hidden_states, *args, **kwargs)
            if not isinstance(hidden_states, torch.Tensor):
                hidden_states, encoder_hidden_states = hidden_states
                if not self.return_hidden_states_first:
                    hidden_states, encoder_hidden_states = encoder_hidden_states, hidden_states

        hidden_states = hidden_states.reshape(-1).contiguous().reshape(original_hidden_states.shape)
        encoder_hidden_states = (
            encoder_hidden_states.reshape(-1).contiguous().reshape(original_encoder_hidden_states.shape)
        )

        hidden_states_residual = hidden_states - original_hidden_states
        encoder_hidden_states_residual = encoder_hidden_states - original_encoder_hidden_states

        hidden_states_residual = hidden_states_residual.reshape(-1).contiguous().reshape(original_hidden_states.shape)
        encoder_hidden_states_residual = (
            encoder_hidden_states_residual.reshape(-1).contiguous().reshape(original_encoder_hidden_states.shape)
        )

        return hidden_states, encoder_hidden_states, hidden_states_residual, encoder_hidden_states_residual

    def get_can_use_cache(self, first_hidden_states_residual):
        if is_in_warmup():
            return False
        threshold = get_residual_diff_threshold()
        if threshold <= 0.0:
            return False
        downsample_factor = get_downsample_factor()
        if downsample_factor > 1:
            first_hidden_states_residual = first_hidden_states_residual[..., ::downsample_factor]
        prev_first_hidden_states_residual = get_first_hidden_states_residual()
        can_use_cache = prev_first_hidden_states_residual is not None and self.are_two_tensors_similar(
            prev_first_hidden_states_residual,
            first_hidden_states_residual,
            threshold=threshold,
        )
        return can_use_cache

    @property
    def is_parallelized(self) -> bool:
        return get_sequence_parallel_world_size() > 1

    def all_reduce(self, input_: torch.Tensor, op=torch.distributed.ReduceOp.AVG) -> torch.Tensor:
        return get_sp_group().all_reduce(input_, op=op) if self.is_parallelized else input_

    def are_two_tensors_similar(self, t1, t2, *, threshold):
        if threshold <= 0.0:
            return False
        if t1.shape != t2.shape:
            return False

        mean_diff = (t1 - t2).abs().mean()
        mean_t1 = t1.abs().mean()
        mean_diff, mean_t1 = self.all_reduce(mean_diff.unsqueeze(0)), self.all_reduce(mean_t1.unsqueeze(0))
        diff = (mean_diff / mean_t1).squeeze()
        return diff < threshold
