import os
os.environ['NCCL_DEBUG'] = 'ERROR'
import sys
import functools
from typing import List, Optional, Tuple, Union, Dict, Any
import torch.distributed as dist

import logging
import time
import torch
import torch.distributed
from diffusers import AutoModel, DiffusionPipeline, AutoencoderKLWan
from diffusers.schedulers.scheduling_unipc_multistep import UniPCMultistepScheduler
from xfuser import xFuserArgs, xFuserWanPipeline
from xfuser.config import FlexibleArgumentParser
from xfuser.core.distributed import (
    get_world_group,
    get_data_parallel_rank,
    get_data_parallel_world_size,
    get_sequence_parallel_world_size,
    get_sequence_parallel_rank,
    get_runtime_state,
    is_dp_last_group,
    init_distributed_environment,
    initialize_model_parallel,
    get_world_group,
    get_sp_group,
    get_classifier_free_guidance_world_size,
    get_classifier_free_guidance_rank,
    get_cfg_group,
    initialize_runtime_state,
    get_pipeline_parallel_world_size,
)
from diffusers.utils import export_to_video
from diffusers.utils import scale_lora_layers, unscale_lora_layers, USE_PEFT_BACKEND
from diffusers.models.attention import Attention
from diffusers.models.transformers.transformer_wan import WanAttnProcessor2_0
from xfuser.core.long_ctx_attention import xFuserLongContextAttention
from transformers import UMT5EncoderModel

from xfuser.model_executor.layers.attention_processor import xFuserWanAttnProcessor2_0

def main():
    dist.init_process_group("nccl")
    init_distributed_environment(
        rank=dist.get_rank(),
        world_size=dist.get_world_size()
    )

    parser = FlexibleArgumentParser(description="xFuser Arguments")
    args = xFuserArgs.add_cli_args(parser).parse_args()
    engine_args = xFuserArgs.from_cli_args(args)
    engine_config, input_config = engine_args.create_config()
    
    if args.enable_fa3:
        assert torch.cuda.get_device_capability()[0] >= 9, (
            "FlashAttention v3 requires SM >= 90. "
        )
        import yunchang
        from yunchang.kernels import AttnType
        try:
            import flash_attn_interface
            FLASH_ATTN_3_AVAILABLE = True
        except ModuleNotFoundError:
            FLASH_ATTN_3_AVAILABLE = False
        assert FLASH_ATTN_3_AVAILABLE == True, ("FlashAttention v3 is not installed")

        setattr(xFuserWanAttnProcessor2_0, "enable_fa3", True)
    else:
        setattr(xFuserWanAttnProcessor2_0, "enable_fa3", False)
    
    assert engine_args.pipefusion_parallel_degree == 1, "This script does not support PipeFusion."
    assert engine_args.use_parallel_vae is False, "parallel VAE not implemented for Wan2.1"

    text_encoder = UMT5EncoderModel.from_pretrained(engine_config.model_config.model, subfolder="text_encoder", torch_dtype=torch.bfloat16)
    vae = AutoencoderKLWan.from_pretrained(engine_config.model_config.model, subfolder="vae", torch_dtype=torch.float32)
    transformer = AutoModel.from_pretrained(engine_config.model_config.model, subfolder="transformer", torch_dtype=torch.bfloat16)

    pipe = xFuserWanPipeline.from_pretrained(
        pretrained_model_name_or_path=engine_config.model_config.model,
        transformer=transformer,
        vae=vae,
        text_encoder=text_encoder,
        engine_config=engine_config,
        torch_dtype=torch.bfloat16,
    )
    flow_shift = 5.0
    scheduler = UniPCMultistepScheduler(prediction_type='flow_prediction', use_flow_sigmas=True, num_train_timesteps=1000, flow_shift=flow_shift)
    pipe.scheduler = scheduler
    local_rank = get_world_group().local_rank
    device = torch.device(f"cuda:{local_rank}")
    pipe.to(device)

    initialize_runtime_state(pipe, engine_config)

    if args.enable_sequential_cpu_offload:
        pipe.enable_sequential_cpu_offload(gpu_id=local_rank)
        logging.info(f"rank {local_rank} sequential CPU offload enabled")
    elif args.enable_model_cpu_offload:
        pipe.enable_model_cpu_offload(gpu_id=local_rank)
        logging.info(f"rank {local_rank} model CPU offload enabled")
    else:
        device = torch.device(f"cuda:{local_rank}")
        pipe = pipe.to(device)

    if args.enable_tiling:
        pipe.vae.enable_tiling()

    if args.enable_slicing:
        pipe.vae.enable_slicing()

    parameter_peak_memory = torch.cuda.max_memory_allocated(device=f"cuda:{local_rank}")

    if engine_config.runtime_config.use_torch_compile:
        torch._inductor.config.reorder_for_compute_comm_overlap = True
        pipe.transformer = torch.compile(pipe.transformer,
            mode="max-autotune-no-cudagraphs")
        output = pipe(
            height=input_config.height,
            width=input_config.width,
            num_frames=input_config.num_frames,
            prompt=input_config.prompt,
            negative_prompt=input_config.negative_prompt,
            num_inference_steps=1,
            guidance_scale=5.0,
            generator=torch.Generator(device="cuda").manual_seed(input_config.seed),
        ).frames[0]

    torch.cuda.reset_peak_memory_stats()
    start_time = time.time()
    output = pipe(
        height=input_config.height,
        width=input_config.width,
        num_frames=input_config.num_frames,
        prompt=input_config.prompt,
        negative_prompt=input_config.negative_prompt,
        num_inference_steps=input_config.num_inference_steps,
        guidance_scale=5.0,
        generator=torch.Generator(device="cuda").manual_seed(input_config.seed),
    ).frames[0]

    end_time = time.time()
    elapsed_time = end_time - start_time
    peak_memory = torch.cuda.max_memory_allocated(device=f"cuda:{local_rank}")
    parallel_info = (
        f"dp{engine_args.data_parallel_degree}_cfg{engine_config.parallel_config.cfg_degree}_"
        f"ulysses{engine_args.ulysses_degree}_ring{engine_args.ring_degree}_"
        f"tp{engine_args.tensor_parallel_degree}_"
        f"pp{engine_args.pipefusion_parallel_degree}_patch{engine_args.num_pipeline_patch}"
    )

    if is_dp_last_group():
        resolution = f"{input_config.width}x{input_config.height}"
        output_filename = f"results/wan_{parallel_info}_{resolution}.mp4"
        export_to_video(output, output_filename, fps=16, quality=8)
        print(f"output saved to {output_filename}")

    if get_world_group().rank == get_world_group().world_size - 1:
        print(f"epoch time: {elapsed_time:.2f} sec, parameter memory: {parameter_peak_memory/1e9:.2f} GB, memory: {peak_memory/1e9} GB")

    dist.destroy_process_group()

if __name__ == "__main__":
    main()
