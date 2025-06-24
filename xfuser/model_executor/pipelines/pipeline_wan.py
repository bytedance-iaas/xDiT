from typing import Optional, Union, List, Dict, Callable, Any
import inspect
import os
from typing import Any, Callable, Dict, List, Optional, Tuple, Union

import torch
import torch.distributed
from diffusers import WanPipeline
from diffusers.callbacks import MultiPipelineCallbacks, PipelineCallback
from diffusers.pipelines.wan.pipeline_wan import WanPipelineOutput
from diffusers.schedulers import FlowMatchEulerDiscreteScheduler
from xfuser.model_executor.pipelines import xFuserPipelineBaseWrapper
from xfuser.model_executor.pipelines.register import xFuserPipelineWrapperRegister

from diffusers.utils.import_utils import is_torch_xla_available

from xfuser.config import EngineConfig
from xfuser.core.distributed import (
    get_cfg_group,
    get_classifier_free_guidance_world_size,
    get_pipeline_parallel_world_size,
    get_runtime_state,
    get_sequence_parallel_rank,
    get_sequence_parallel_world_size,
    get_pp_group,
    get_sp_group,
    is_dp_last_group,
    get_pipeline_parallel_rank,
    get_pp_group,
    is_pipeline_first_stage,
    is_pipeline_last_stage,
)

from xfuser.logger import init_logger

logger = init_logger(__name__)

if is_torch_xla_available():
    import torch_xla.core.xla_model as xm

    XLA_AVAILABLE = True
else:
    XLA_AVAILABLE = False

@xFuserPipelineWrapperRegister.register(WanPipeline)
class xFuserWanPipeline(xFuserPipelineBaseWrapper):
    @classmethod
    def from_pretrained(
        cls,
        pretrained_model_name_or_path: Optional[Union[str, os.PathLike]],
        engine_config: EngineConfig,
        **kwargs,
    ):
        pipeline = WanPipeline.from_pretrained(
            pretrained_model_name_or_path, **kwargs
        )
        return cls(pipeline, engine_config)

    @property
    def guidance_scale(self):
        return self._guidance_scale

    @property
    def do_classifier_free_guidance(self):
        return self._guidance_scale > 1.0

    @property
    def num_timesteps(self):
        return self._num_timesteps

    @property
    def current_timestep(self):
        return self._current_timestep

    @property
    def interrupt(self):
        return self._interrupt

    @property
    def attention_kwargs(self):
        return self._attention_kwargs

    @torch.no_grad()
    @xFuserPipelineBaseWrapper.enable_data_parallel
    @xFuserPipelineBaseWrapper.check_to_use_naive_forward
    def __call__(
        self,
        prompt: Union[str, List[str]] = None,
        negative_prompt: Union[str, List[str]] = None,
        height: int = 480,
        width: int = 832,
        num_frames: int = 81,
        num_inference_steps: int = 50,
        guidance_scale: float = 5.0,
        num_videos_per_prompt: Optional[int] = 1,
        generator: Optional[Union[torch.Generator, List[torch.Generator]]] = None,
        latents: Optional[torch.Tensor] = None,
        prompt_embeds: Optional[torch.Tensor] = None,
        negative_prompt_embeds: Optional[torch.Tensor] = None,
        output_type: Optional[str] = "np",
        return_dict: bool = True,
        attention_kwargs: Optional[Dict[str, Any]] = None,
        callback_on_step_end: Optional[
            Union[Callable[[int, int, Dict], None], PipelineCallback, MultiPipelineCallbacks]
        ] = None,
        callback_on_step_end_tensor_inputs: List[str] = ["latents"],
        max_sequence_length: int = 512,
    ):
        r"""
        The call function to the pipeline for generation.

        Args:
            prompt (`str` or `List[str]`, *optional*):
                The prompt or prompts to guide the image generation. If not defined, one has to pass `prompt_embeds`.
                instead.
            height (`int`, defaults to `480`):
                The height in pixels of the generated image.
            width (`int`, defaults to `832`):
                The width in pixels of the generated image.
            num_frames (`int`, defaults to `81`):
                The number of frames in the generated video.
            num_inference_steps (`int`, defaults to `50`):
                The number of denoising steps. More denoising steps usually lead to a higher quality image at the
                expense of slower inference.
            guidance_scale (`float`, defaults to `5.0`):
                Guidance scale as defined in [Classifier-Free Diffusion
                Guidance](https://huggingface.co/papers/2207.12598). `guidance_scale` is defined as `w` of equation 2.
                of [Imagen Paper](https://huggingface.co/papers/2205.11487). Guidance scale is enabled by setting
                `guidance_scale > 1`. Higher guidance scale encourages to generate images that are closely linked to
                the text `prompt`, usually at the expense of lower image quality.
            num_videos_per_prompt (`int`, *optional*, defaults to 1):
                The number of images to generate per prompt.
            generator (`torch.Generator` or `List[torch.Generator]`, *optional*):
                A [`torch.Generator`](https://pytorch.org/docs/stable/generated/torch.Generator.html) to make
                generation deterministic.
            latents (`torch.Tensor`, *optional*):
                Pre-generated noisy latents sampled from a Gaussian distribution, to be used as inputs for image
                generation. Can be used to tweak the same generation with different prompts. If not provided, a latents
                tensor is generated by sampling using the supplied random `generator`.
            prompt_embeds (`torch.Tensor`, *optional*):
                Pre-generated text embeddings. Can be used to easily tweak text inputs (prompt weighting). If not
                provided, text embeddings are generated from the `prompt` input argument.
            output_type (`str`, *optional*, defaults to `"np"`):
                The output format of the generated image. Choose between `PIL.Image` or `np.array`.
            return_dict (`bool`, *optional*, defaults to `True`):
                Whether or not to return a [`WanPipelineOutput`] instead of a plain tuple.
            attention_kwargs (`dict`, *optional*):
                A kwargs dictionary that if specified is passed along to the `AttentionProcessor` as defined under
                `self.processor` in
                [diffusers.models.attention_processor](https://github.com/huggingface/diffusers/blob/main/src/diffusers/models/attention_processor.py).
            callback_on_step_end (`Callable`, `PipelineCallback`, `MultiPipelineCallbacks`, *optional*):
                A function or a subclass of `PipelineCallback` or `MultiPipelineCallbacks` that is called at the end of
                each denoising step during the inference. with the following arguments: `callback_on_step_end(self:
                DiffusionPipeline, step: int, timestep: int, callback_kwargs: Dict)`. `callback_kwargs` will include a
                list of all tensors as specified by `callback_on_step_end_tensor_inputs`.
            callback_on_step_end_tensor_inputs (`List`, *optional*):
                The list of tensor inputs for the `callback_on_step_end` function. The tensors specified in the list
                will be passed as `callback_kwargs` argument. You will only be able to include variables listed in the
                `._callback_tensor_inputs` attribute of your pipeline class.
            autocast_dtype (`torch.dtype`, *optional*, defaults to `torch.bfloat16`):
                The dtype to use for the torch.amp.autocast.

        Examples:

        Returns:
            [`~WanPipelineOutput`] or `tuple`:
                If `return_dict` is `True`, [`WanPipelineOutput`] is returned, otherwise a `tuple` is returned where
                the first element is a list with the generated images and the second element is a list of `bool`s
                indicating whether the corresponding generated image contains "not-safe-for-work" (nsfw) content.
        """
        if isinstance(callback_on_step_end, (PipelineCallback, MultiPipelineCallbacks)):
            callback_on_step_end_tensor_inputs = callback_on_step_end.tensor_inputs

        # 1. Check inputs. Raise error if not correct
        self.check_inputs(
            prompt,
            negative_prompt,
            height,
            width,
            prompt_embeds,
            negative_prompt_embeds,
            callback_on_step_end_tensor_inputs,
        )

        if num_frames % self.vae_scale_factor_temporal != 1:
            logger.warning(
                f"`num_frames - 1` has to be divisible by {self.vae_scale_factor_temporal}. Rounding to the nearest number."
            )
            num_frames = num_frames // self.vae_scale_factor_temporal * self.vae_scale_factor_temporal + 1
        num_frames = max(num_frames, 1)

        self._guidance_scale = guidance_scale
        self._attention_kwargs = attention_kwargs
        self._current_timestep = None
        self._interrupt = False

        device = self._execution_device

        # 2. Define call parameters
        if prompt is not None and isinstance(prompt, str):
            batch_size = 1
        elif prompt is not None and isinstance(prompt, list):
            batch_size = len(prompt)
        else:
            batch_size = prompt_embeds.shape[0]
    
        do_classifier_free_guidance = guidance_scale > 1.0

        get_runtime_state().set_video_input_parameters(
            height=height,
            width=width,
            num_frames=num_frames,
            batch_size=batch_size,
            num_inference_steps=num_inference_steps,
            split_text_embed_in_sp=get_pipeline_parallel_world_size() == 1,
        )

        # 3. Encode input prompt
        prompt_embeds, negative_prompt_embeds = self.encode_prompt(
            prompt=prompt,
            negative_prompt=negative_prompt,
            do_classifier_free_guidance=do_classifier_free_guidance,
            num_videos_per_prompt=num_videos_per_prompt,
            prompt_embeds=prompt_embeds,
            negative_prompt_embeds=negative_prompt_embeds,
            max_sequence_length=max_sequence_length,
            device=device,
        )
        
        transformer_dtype = self.transformer.dtype
        prompt_embeds = prompt_embeds.to(transformer_dtype)
        if negative_prompt_embeds is not None:
            negative_prompt_embeds = negative_prompt_embeds.to(transformer_dtype)
        
        if self.do_classifier_free_guidance:
            prompt_embeds = self._process_cfg_split_batch(negative_prompt_embeds, prompt_embeds)

        # 4. Prepare timesteps
        self.scheduler.set_timesteps(num_inference_steps, device=device)
        timesteps = self.scheduler.timesteps

        # 5. Prepare latent variables
        num_channels_latents = self.transformer.config.in_channels
        latents = self.prepare_latents(
            batch_size * num_videos_per_prompt,
            num_channels_latents,
            height,
            width,
            num_frames,
            torch.float32,
            device,
            generator,
            latents,
        )

        # image_rotary_emb = (
        #     self._prepare_rotary_positional_embeddings(height, width, latents.size(1), device)
        #     if self.transformer.config.use_rotary_positional_embeddings
        #     else None
        # )

        # 6. Denoising loop
        num_warmup_steps = len(timesteps) - num_inference_steps * self.scheduler.order
        self._num_timesteps = len(timesteps)

        num_pipeline_warmup_steps = get_runtime_state().runtime_config.warmup_steps

        p_t = self.transformer.config.patch_size[0] or 1
        latents, prompt_embeds = self._init_sync_pipeline(
            latents, prompt_embeds, (latents.size(1) + p_t - 1) // p_t
        )

        with self.progress_bar(total=num_inference_steps) as progress_bar:
            if (
                get_pipeline_parallel_world_size() > 1
                and len(timesteps) > num_pipeline_warmup_steps
            ):
                # * warmup stage
                latents = self._sync_pipeline(
                    latents=latents,
                    guidance_scale=guidance_scale,
                    prompt_embeds=prompt_embeds,
                    negative_prompt_embeds=negative_prompt_embeds,
                    transformer_dtype=transformer_dtype,
                    attention_kwargs=attention_kwargs,
                    timesteps=timesteps[:num_pipeline_warmup_steps],
                    num_warmup_steps=num_warmup_steps,
                    progress_bar=progress_bar,
                    callback_on_step_end=callback_on_step_end,
                    callback_on_step_end_tensor_inputs=callback_on_step_end_tensor_inputs
                )

                # * pipefusion stage
                latents = self._async_pipeline(
                    latents=latents,
                    guidance_scale=guidance_scale,
                    prompt_embeds=prompt_embeds,
                    negative_prompt_embeds=negative_prompt_embeds,
                    transformer_dtype=transformer_dtype,
                    attention_kwargs=attention_kwargs,
                    timesteps=timesteps[num_pipeline_warmup_steps:],
                    num_warmup_steps=num_warmup_steps,
                    progress_bar=progress_bar,
                    callback_on_step_end=callback_on_step_end,
                    callback_on_step_end_tensor_inputs=callback_on_step_end_tensor_inputs
                )
            else:
                latents = self._sync_pipeline(
                    latents=latents,
                    guidance_scale=guidance_scale,
                    prompt_embeds=prompt_embeds,
                    negative_prompt_embeds=negative_prompt_embeds,
                    transformer_dtype=transformer_dtype,
                    attention_kwargs=attention_kwargs,
                    timesteps=timesteps,
                    num_warmup_steps=num_warmup_steps,
                    progress_bar=progress_bar,
                    callback_on_step_end=callback_on_step_end,
                    callback_on_step_end_tensor_inputs=callback_on_step_end_tensor_inputs
                )

        self._current_timestep = None
        if not output_type == "latent":
            latents = latents.to(self.vae.dtype)
            latents_mean = (
                torch.tensor(self.vae.config.latents_mean)
                .view(1, self.vae.config.z_dim, 1, 1, 1)
                .to(latents.device, latents.dtype)
            )
            latents_std = 1.0 / torch.tensor(self.vae.config.latents_std).view(1, self.vae.config.z_dim, 1, 1, 1).to(
                latents.device, latents.dtype
            )
            latents = latents / latents_std + latents_mean
            video = self.vae.decode(latents, return_dict=False)[0]
            video = self.video_processor.postprocess_video(video, output_type=output_type)
        else:
            video = latents

        # Offload all models
        self.maybe_free_model_hooks()

        if not return_dict:
            return (video,)

        return WanPipelineOutput(frames=video)

    def _init_sync_pipeline(
        self,
        latents: torch.Tensor,
        prompt_embeds: torch.Tensor,
        latents_frames: Optional[int] = None,
    ):
        latents = super()._init_video_sync_pipeline(latents)

        if get_runtime_state().split_text_embed_in_sp:
            if prompt_embeds.shape[-2] % get_sequence_parallel_world_size() == 0:
                prompt_embeds = torch.chunk(prompt_embeds, get_sequence_parallel_world_size(), dim=-2)[
                    get_sequence_parallel_rank()
                ]
            else:
                get_runtime_state().split_text_embed_in_sp = False

        return latents, prompt_embeds

    def _sync_pipeline(
            self,
            latents: torch.Tensor,
            guidance_scale: float,
            prompt_embeds: torch.Tensor,
            negative_prompt_embeds: torch.Tensor,
            transformer_dtype: torch.dtype,
            attention_kwargs: Dict[str, Any],
            timesteps: List[int],
            num_warmup_steps: int,
            progress_bar,
            callback_on_step_end: Union[Callable[[int, int , Dict], None], PipelineCallback, MultiPipelineCallbacks],
            callback_on_step_end_tensor_inputs: List[str],
            sync_only: bool = False
    ):
        skips = None
        for i, t in enumerate(timesteps):
            if self.interrupt:
                continue

            if is_pipeline_last_stage():
                last_timestep_latents = latents

            # when there is only one pp stage, no need to recv
            if get_pipeline_parallel_world_size() == 1:
                pass
            # all ranks should recv the latent from the previous rank except
            #   the first rank in the first pipeline forward which should use
            #   the input latent
            elif is_pipeline_first_stage() and i == 0:
                pass
            else:
                latents = get_pp_group().pipeline_recv()
                if (
                    get_pipeline_parallel_rank() >= get_pipeline_parallel_world_size() // 2
                ):
                    skips = get_pp_group().pipeline_recv_skip()

            noise_pred = self._backbone_forward(
                latents=latents,
                guidance_scale=guidance_scale,
                prompt_embeds=prompt_embeds,
                negative_prompt_embeds=negative_prompt_embeds,
                attention_kwargs=attention_kwargs,
                transformer_dtype=transformer_dtype,
                t=t
            )

            if is_pipeline_last_stage():
                latents = self.scheduler.step(noise_pred, t, last_timestep_latents, return_dict=False)[0]
            elif (
                get_pipeline_parallel_rank() >= get_pipeline_parallel_world_size()
            ):
                pass
            else:
                latents, skips = latents

            if sync_only and is_pipeline_last_stage() and i == len(timesteps) - 1:
                pass
            elif get_pipeline_parallel_world_size() > 1:
                get_pp_group().pipeline_send(latents)
                if ( get_pipeline_parallel_rank() < get_pipeline_parallel_world_size() // 2):
                    get_pp_group().pipeline_send_skip(skips)

            # call the callback, if provided
            if callback_on_step_end is not None:
                callback_kwargs = {}
                for k in callback_on_step_end_tensor_inputs:
                    callback_kwargs[k] = locals()[k]
                callback_outputs = callback_on_step_end(self, i, t, callback_kwargs)

                latents = callback_outputs.pop("latents", latents)
                prompt_embeds = callback_outputs.pop("prompt_embeds", prompt_embeds)
                negative_prompt_embeds = callback_outputs.pop("negative_prompt_embeds", negative_prompt_embeds)

            if i == len(timesteps) - 1 or ((i + 1) > num_warmup_steps and (i + 1) % self.scheduler.order == 0):
                progress_bar.update()

            if XLA_AVAILABLE:
                xm.mark_step()

            if get_sequence_parallel_world_size() > 1:
                latents = get_sp_group().all_gather(latents, dim=-2)

        if (
            sync_only
            and get_sequence_parallel_world_size() > 1
            and is_pipeline_last_stage()
        ):
            sp_degree = get_sequence_parallel_world_size()
            sp_latents_list = get_sp_group().all_gather(latents, separate_tensors=True)
            latents_list = []
            for pp_patch_idx in range(get_runtime_state().num_pipeline_patch):
                latents_list += [
                    sp_latents_list[sp_patch_idx][
                        :,
                        :,
                        get_runtime_state()
                        .pp_patches_start_idx_local[pp_patch_idx] : get_runtime_state()
                        .pp_patches_start_idx_local[pp_patch_idx + 1],
                        :,
                    ]
                    for sp_patch_idx in range(sp_degree)
                ]
            latents = torch.cat(latents_list, dim=-2)

        return latents

    def _async_pipeline(
        self,
        latents: torch.Tensor,
        guidance_scale: float,
        prompt_embeds: torch.Tensor,
        negative_prompt_embeds: torch.Tensor,
        transformer_dtype: torch.dtype,
        attention_kwargs: Dict[str, Any],
        timesteps: List[int],
        num_warmup_steps: int,
        progress_bar,
        callback_on_step_end: Union[Callable[[int, int , Dict], None], PipelineCallback, MultiPipelineCallbacks],
        callback_on_step_end_tensor_inputs: List[str],
    ):
        if len(timesteps) == 0:
            return latents
        num_pipeline_patch = get_runtime_state().num_pipeline_patch
        num_pipeline_warmup_steps = get_runtime_state().runtime_config.warmup_steps
        patch_latents = self._init_async_pipeline(
            num_timesteps=len(timesteps),
            latents=latents,
            num_pipeline_warmup_steps=num_pipeline_warmup_steps
        )
        last_patch_latents = (
            [None for _ in range(num_pipeline_patch)]
            if (is_pipeline_last_stage())
            else None
        )

        first_async_recv = True
        for i, t in enumerate(timesteps):
            for patch_idx in range(num_pipeline_patch):
                if is_pipeline_last_stage():
                    last_patch_latents[patch_idx] = patch_latents[patch_idx]

                if is_pipeline_first_stage() and i == 0:
                    pass
                else:
                    if first_async_recv:
                        get_pp_group().recv_next()
                        first_async_recv = False
                    patch_latents[patch_idx] = get_pp_group().get_pipeline_recv_data(idx=patch_idx)

                patch_latents[patch_idx] = self._backbone_forward(
                    latents=patch_latents[patch_idx],
                    guidance_scale=guidance_scale,
                    prompt_embeds=prompt_embeds,
                    negative_prompt_embeds=negative_prompt_embeds,
                    attention_kwargs=attention_kwargs,
                    t=t,
                    transformer_dtype=transformer_dtype
                )

                if is_pipeline_last_stage():
                    patch_latents[patch_idx] = self.scheduler.step(
                        patch_latents[patch_idx],
                        t,
                        last_patch_latents[patch_idx],
                        return_dict=False,
                    )[0]
                    if i != len(timesteps) - 1:
                        get_pp_group().pipeline_isend(
                            patch_latents[patch_idx], segment_idx=patch_idx
                        )
                elif (
                    get_pipeline_parallel_rank()
                    >= get_pipeline_parallel_world_size() // 2
                ):
                    get_pp_group().pipeline_isend(
                        patch_latents[patch_idx], segment_idx=patch_idx
                    )
                else:
                    patch_latents[patch_idx], skips = patch_latents[patch_idx]
                    get_pp_group().pipeline_isend(
                        patch_latents[patch_idx], segment_idx=patch_idx
                    )
                    get_pp_group().pipeline_isend_skip(skips)

                if is_pipeline_first_stage() and i == 0:
                    pass
                else:
                    if i == len(timesteps) - 1 and patch_idx == num_pipeline_patch - 1:
                        pass
                    else:
                        get_pp_group().recv_next()
                        if (
                            get_pipeline_parallel_rank()
                            >= get_pipeline_parallel_world_size() // 2
                        ):
                            get_pp_group().recv_skip_next()

                get_runtime_state().next_patch()

            if i == len(timesteps) - 1 or (
                (i + num_pipeline_warmup_steps + 1) > num_warmup_steps
                and (i + num_pipeline_warmup_steps + 1) % self.scheduler.order == 0
            ):
                progress_bar.update()
                if callback_on_step_end is not None:
                    callback_kwargs = {}
                    for k in callback_on_step_end_tensor_inputs:
                        callback_kwargs[k] = locals()[k]
                    callback_outputs = callback_on_step_end(self, i, t, callback_kwargs)

                    latents = callback_outputs.pop("latents", latents)
                    prompt_embeds = callback_outputs.pop("prompt_embeds", prompt_embeds)
                    negative_prompt_embeds = callback_outputs.pop(
                        "negative_prompt_embeds", negative_prompt_embeds
                    )
                    prompt_embeds_2 = callback_outputs.pop(
                        "prompt_embeds_2", prompt_embeds_2
                    )
                    negative_prompt_embeds_2 = callback_outputs.pop(
                        "negative_prompt_embeds_2", negative_prompt_embeds_2
                    )

        latents = None
        if is_pipeline_last_stage():
            latents = torch.cat(patch_latents, dim=2)
            if get_sequence_parallel_world_size() > 1:
                sp_degree = get_sequence_parallel_world_size()
                sp_latents_list = get_sp_group().all_gather(
                    latents, separate_tensors=True
                )
                latents_list = []
                for pp_patch_idx in range(get_runtime_state().num_pipeline_patch):
                    latents_list += [
                        sp_latents_list[sp_patch_idx][
                            ...,
                            get_runtime_state()
                            .pp_patches_start_idx_local[
                                pp_patch_idx
                            ] : get_runtime_state()
                            .pp_patches_start_idx_local[pp_patch_idx + 1],
                            :,
                        ]
                        for sp_patch_idx in range(sp_degree)
                    ]
                latents = torch.cat(latents_list, dim=-2)

        return latents

    def _backbone_forward(
        self,
        latents: torch.Tensor,
        guidance_scale: int,
        prompt_embeds: torch.Tensor,
        negative_prompt_embeds: torch.Tensor,
        attention_kwargs: Dict[str, Any],
        transformer_dtype: str,
        t: Union[float, torch.Tensor]
    ):
        self._current_timestep = t
        latent_model_input = latents.to(transformer_dtype)
        latent_model_input = torch.cat([latent_model_input] * 2) if self.do_classifier_free_guidance else latents
        timestep = t.expand(latent_model_input.shape[0])

        noise_pred = self.transformer(
            hidden_states=latent_model_input,
            timestep=timestep,
            encoder_hidden_states=prompt_embeds,
            attention_kwargs=attention_kwargs,
            return_dict=False,
        )[0]

        if is_pipeline_last_stage():
            if get_classifier_free_guidance_world_size() == 1:
                noise_pred_uncond, noise_pred_text = noise_pred.chunk(2)
            elif get_classifier_free_guidance_world_size() == 2:
                noise_pred_uncond, noise_pred_text = get_cfg_group().all_gather(
                    noise_pred, separate_tensors=True
                )
            latents = noise_pred_uncond + guidance_scale * (noise_pred_text - noise_pred_uncond)

        else:
            latents = noise_pred

        return latents
