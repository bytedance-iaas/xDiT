from typing import Optional, Dict, Any, Union, List, Optional, Tuple, Type
import torch
import torch.distributed
import torch.nn as nn

from diffusers.models.embeddings import PatchEmbed
from diffusers.models.transformers import WanTransformer3DModel
from diffusers.models.modeling_outputs import Transformer2DModelOutput
from diffusers.utils import USE_PEFT_BACKEND, is_torch_version, scale_lora_layers, USE_PEFT_BACKEND, unscale_lora_layers

from xfuser.model_executor.models import xFuserModelBaseWrapper
from xfuser.logger import init_logger
from xfuser.model_executor.base_wrapper import xFuserBaseWrapper
from xfuser.core.distributed import (
    get_sequence_parallel_world_size,
    get_classifier_free_guidance_world_size,
    get_classifier_free_guidance_rank,
    get_sequence_parallel_rank,
    get_sp_group,
    get_cfg_group
)

from xfuser.model_executor.models.transformers.register import xFuserTransformerWrappersRegister
from xfuser.model_executor.models.transformers.base_transformer import xFuserTransformerBaseWrapper
from xfuser.model_executor.layers.attention_processor import xFuserWanAttnProcessor2_0


from xfuser.core.distributed import is_pipeline_first_stage,is_pipeline_last_stage

from xfuser.logger import init_logger

logger = init_logger(__name__)

@xFuserTransformerWrappersRegister.register(WanTransformer3DModel)
class xFuserWanTransformer3DWrapper(xFuserTransformerBaseWrapper):
    def __init__(
        self,
        transformer: WanTransformer3DModel,
    ):
        super().__init__(
            transformer=transformer,
            transformer_blocks_name=["blocks"],
            # submodule_classes_to_wrap=[nn.Conv2d, PatchEmbed],
            # submodule_name_to_wrap=["attn1"]
        )
        transformer: WanTransformer3DModel
        for block in transformer.blocks:
            block.attn1.processor = xFuserWanAttnProcessor2_0()
            block.attn2.processor = xFuserWanAttnProcessor2_0()

    @xFuserBaseWrapper.forward_check_condition
    def forward(
        self,
        hidden_states: torch.Tensor,
        timestep: torch.LongTensor,
        encoder_hidden_states: torch.Tensor,
        encoder_hidden_states_image: Optional[torch.Tensor] = None,
        return_dict: bool = True,
        attention_kwargs: Optional[Dict[str, Any]] = None,
    ) -> Union[torch.Tensor, Dict[str, torch.Tensor]]:
        if attention_kwargs is not None:
            attention_kwargs = attention_kwargs.copy()
            lora_scale = attention_kwargs.pop("scale", 1.0)
        else:
            lora_scale = 1.0

        if USE_PEFT_BACKEND:
            # weight the lora layers by setting `lora_scale` for each PEFT layer
            scale_lora_layers(self, lora_scale)
        else:
            if attention_kwargs is not None and attention_kwargs.get("scale", None) is not None:
                logger.warning(
                    "Passing `scale` via `attention_kwargs` when not using the PEFT backend is ineffective."
                )

        batch_size, num_channels, num_frames, height, width = hidden_states.shape
        p_t, p_h, p_w = self.config.patch_size
        post_patch_num_frames = num_frames // p_t
        post_patch_height = height // p_h
        post_patch_width = width // p_w

        rotary_emb = self.rope(hidden_states)

        hidden_states = self.patch_embedding(hidden_states)
        hidden_states = hidden_states.flatten(2).transpose(1, 2)

        #split timestep hidden_states
        timestep = torch.chunk(timestep, get_classifier_free_guidance_world_size(),dim=0)[get_classifier_free_guidance_rank()]
        hidden_states = torch.chunk(hidden_states,
                                    get_classifier_free_guidance_world_size(),
                                    dim=0)[get_classifier_free_guidance_rank()]
        hidden_states = torch.chunk(hidden_states, get_sequence_parallel_world_size(), dim=-2)[get_sequence_parallel_rank()]

        temb, timestep_proj, encoder_hidden_states, encoder_hidden_states_image = self.condition_embedder(
            timestep, encoder_hidden_states, encoder_hidden_states_image
        )
        timestep_proj = timestep_proj.unflatten(1, (6, -1))

        if encoder_hidden_states_image is not None:
            encoder_hidden_states = torch.concat([encoder_hidden_states_image, encoder_hidden_states], dim=1)

        if encoder_hidden_states.shape[-2] % get_sequence_parallel_world_size() != 0:
            split_text_embed_in_sp = False
        else:
            split_text_embed_in_sp = True
        encoder_hidden_states = torch.chunk(encoder_hidden_states,get_classifier_free_guidance_world_size(),dim=0)[get_classifier_free_guidance_rank()]
        if split_text_embed_in_sp:
            encoder_hidden_states = torch.chunk(encoder_hidden_states, get_sequence_parallel_world_size(), dim=-2)[get_sequence_parallel_rank()]

        freqs_cos, freqs_sin = rotary_emb
        def get_rotary_emb_chunk(freqs):
            freqs = torch.chunk(freqs, get_sequence_parallel_world_size(), dim=2)[get_sequence_parallel_rank()]
            return freqs
        freqs_cos = get_rotary_emb_chunk(freqs_cos)
        freqs_sin = get_rotary_emb_chunk(freqs_sin)
        rotary_emb = (freqs_cos, freqs_sin)
        # CFG Done

        # 4. Transformer blocks
        if torch.is_grad_enabled() and self.gradient_checkpointing:
            for block in self.blocks:
                hidden_states = self._gradient_checkpointing_func(
                    block, hidden_states, encoder_hidden_states, timestep_proj, rotary_emb
                )
        else:
            for block in self.blocks:
                hidden_states = block(hidden_states, encoder_hidden_states, timestep_proj, rotary_emb)

        # 5. Output norm, projection & unpatchify
        if is_pipeline_last_stage():
            shift, scale = (self.scale_shift_table + temb.unsqueeze(1)).chunk(2, dim=1)

            # Move the shift and scale tensors to the same device as hidden_states.
            # When using multi-GPU inference via accelerate these will be on the
            # first device rather than the last device, which hidden_states ends up
            # on.
            shift = shift.to(hidden_states.device)
            scale = scale.to(hidden_states.device)

            hidden_states = (self.norm_out(hidden_states.float()) * (1 + scale) + shift).type_as(hidden_states)
            hidden_states = self.proj_out(hidden_states)

            hidden_states = get_sp_group().all_gather(hidden_states, dim=-2)
            hidden_states = get_cfg_group().all_gather(hidden_states, dim=0)

            hidden_states = hidden_states.reshape(
                batch_size, post_patch_num_frames, post_patch_height, post_patch_width, p_t, p_h, p_w, -1
            )
            hidden_states = hidden_states.permute(0, 7, 1, 4, 2, 5, 3, 6)
            output = hidden_states.flatten(6, 7).flatten(4, 5).flatten(2, 3)
        else:
            output = hidden_states

        if USE_PEFT_BACKEND:
            # remove `lora_scale` from each PEFT layer
            unscale_lora_layers(self, lora_scale)

        if not return_dict:
            return (output,)

        return Transformer2DModelOutput(sample=output)
