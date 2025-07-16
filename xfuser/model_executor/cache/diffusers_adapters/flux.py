"""
adapted from https://github.com/ali-vilab/TeaCache.git
adapted from https://github.com/chengzeyi/ParaAttention.git
"""
import functools
import unittest
from types import MethodType

import torch
from torch import nn
from diffusers import DiffusionPipeline, FluxTransformer2DModel
from xfuser.model_executor.cache.diffusers_adapters.registry import TRANSFORMER_ADAPTER_REGISTRY

from xfuser.model_executor.cache import utils

def create_cached_transformer_blocks(use_cache, transformer, rel_l1_thresh, return_hidden_states_first, num_steps):
    cached_transformer_class = {
        "Fb": utils.FBCachedTransformerBlocks,
        "Tea": utils.TeaCachedTransformerBlocks,
    }.get(use_cache)

    if not cached_transformer_class:
        raise ValueError(f"Unsupported use_cache value: {use_cache}")

    return cached_transformer_class(
        transformer.transformer_blocks,
        transformer.single_transformer_blocks,
        transformer=transformer,
        rel_l1_thresh=rel_l1_thresh,
        return_hidden_states_first=return_hidden_states_first,
        num_steps=num_steps,
        name=TRANSFORMER_ADAPTER_REGISTRY.get(type(transformer)),
    )


def apply_cache_on_transformer(
    transformer: FluxTransformer2DModel,
    *,
    rel_l1_thresh=0.12,
    return_hidden_states_first=False,
    num_steps=8,
    use_cache="Fb",
):
    cached_transformer_blocks = nn.ModuleList([
        create_cached_transformer_blocks(use_cache, transformer, rel_l1_thresh, return_hidden_states_first, num_steps)
    ])

    dummy_single_transformer_blocks = torch.nn.ModuleList()

    original_forward = transformer.forward

    object.__setattr__(transformer, '_custom_cached_transformer_blocks', cached_transformer_blocks)

    @functools.wraps(original_forward)
    def new_forward(
        self,
        *args,
        **kwargs,
    ):
        with unittest.mock.patch.object(
            self,
            "transformer_blocks",
            self._custom_cached_transformer_blocks,
        ), unittest.mock.patch.object(
            self,
            "single_transformer_blocks",
            dummy_single_transformer_blocks,
        ):
            return original_forward(
                *args,
                **kwargs,
            )

    def clear_cache_modulated_inputs(self):
        for block in self._custom_cached_transformer_blocks:
            block.cache_context.modulated_inputs = None

    transformer.clear_cache_modulated_inputs = MethodType(clear_cache_modulated_inputs, transformer)

    transformer.forward = new_forward.__get__(transformer)

    return transformer
