"""
adapted from https://github.com/ali-vilab/TeaCache.git
adapted from https://github.com/chengzeyi/ParaAttention.git
"""
import functools
import unittest

import torch
from torch import nn
from diffusers import DiffusionPipeline, WanTransformer3DModel
from xfuser.model_executor.cache.diffusers_adapters.registry import TRANSFORMER_ADAPTER_REGISTRY

from xfuser.model_executor.cache import utils

def create_cached_transformer_blocks(use_cache, transformer):
    cached_transformer_class = {
        "Fb": utils.FBCachedWanTransformerBlocks,
        "Tea": utils.TeaCachedWanTransformerBlocks,
    }.get(use_cache)

    if not cached_transformer_class:
        raise ValueError(f"Unsupported use_cache value: {use_cache}")
    
    return cached_transformer_class(
        transformer.blocks,
        transformer=transformer,
        return_hidden_states_only=True,
    )


def apply_cache_on_transformer(
    transformer: WanTransformer3DModel,
    *,
    rel_l1_thresh=0.12,
    return_hidden_states_first=False,
    num_steps=8,
    use_cache="Fb",
    shallow_patch: bool = False,
    residual_diff_threshold=0.12,
    downsample_factor=1,
):
    if getattr(transformer, "_is_cached", False):
        return transformer
    
    blocks = nn.ModuleList([
        create_cached_transformer_blocks(
            use_cache=use_cache, 
            transformer=transformer,
        )
    ])

    original_forward = transformer.forward

    @functools.wraps(transformer.__class__.forward)
    def new_forward(
        self,
        *args,
        **kwargs,
    ):
        with unittest.mock.patch.object(
            self,
            "blocks",
            blocks,
        ):
            return original_forward(
                *args,
                **kwargs,
            )

    transformer.forward = new_forward.__get__(transformer)
    transformer._is_cached = True
    return transformer

def apply_cache_on_pipe(
    pipe: DiffusionPipeline,
    *,
    shallow_patch: bool = False,
    residual_diff_threshold=0.12,
    downsample_factor=1,
    use_cache="Fb",
    **kwargs,
):
    if not getattr(pipe, "_is_cached", False):
        original_call = pipe.__class__.__call__

        @functools.wraps(original_call)
        def new_call(self, *args, **kwargs):
            num_inference_steps = kwargs.get("num_inference_steps", 50)
            with utils.cache_context(
                utils.create_cache_context(
                    residual_diff_threshold=residual_diff_threshold,
                    downsample_factor=downsample_factor,
                    num_inference_steps=num_inference_steps,
                )
            ):
                return original_call(self, *args, **kwargs)
        pipe.__class__.__call__ = new_call
        pipe.__class__._is_cached = True

    if not shallow_patch:
        apply_cache_on_transformer(use_cache=use_cache, transformer=pipe.transformer, **kwargs)

    return pipe
