import os
os.environ['NCCL_DEBUG'] = 'ERROR'
import torch.distributed as dist

import logging
import time
import torch
import torch.distributed
from diffusers import AutoModel, DiffusionPipeline, AutoencoderKLWan
from diffusers.schedulers.scheduling_unipc_multistep import UniPCMultistepScheduler
from xfuser import xFuserArgs, xFuserWanPipeline
from xfuser.model_executor.pipelines.pipeline_wan import parallelize_transformer
from xfuser.config import FlexibleArgumentParser
from xfuser.core.distributed import (
    is_dp_last_group,
    init_distributed_environment,
    get_world_group,
    initialize_runtime_state,
)
from diffusers.utils import export_to_video
from transformers import UMT5EncoderModel

from xfuser.model_executor.layers.attention_processor import xFuserWanAttnProcessor2_0
from xfuser.model_executor.cache.diffusers_adapters.wan import apply_cache_on_pipe
from xfuser.logger import init_logger
logger = init_logger(__name__)

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

    if args.enable_sage_attn:
        setattr(xFuserWanAttnProcessor2_0, "enable_sage_attn", True)
    else:
        setattr(xFuserWanAttnProcessor2_0, "enable_sage_attn", False)

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

    start_time = time.time()
    text_encoder = UMT5EncoderModel.from_pretrained(engine_config.model_config.model, subfolder="text_encoder", torch_dtype=torch.bfloat16)
    vae = AutoencoderKLWan.from_pretrained(engine_config.model_config.model, subfolder="vae", torch_dtype=torch.float32)
    transformer = AutoModel.from_pretrained(engine_config.model_config.model, subfolder="transformer", torch_dtype=torch.bfloat16)
    load_elapsed = time.time() - start_time
    logger.info(f"loading checkpoint elapsed: {load_elapsed:.2f}")

    if engine_config.enable_quantize:
        from torchao.quantization import quantize_, Float8WeightOnlyConfig, float8_weight_only
        quantize_(text_encoder, float8_weight_only())
        quantize_(transformer, Float8WeightOnlyConfig())
        if "Wan2.2" in engine_config.model_config.model:
            quantize_(transformer_2, Float8WeightOnlyConfig())

    pipe = xFuserWanPipeline.from_pretrained(
        pretrained_model_name_or_path=engine_config.model_config.model,
        transformer=transformer,
        vae=vae,
        text_encoder=text_encoder,
        engine_config=engine_config,
        torch_dtype=torch.bfloat16,
    )

    if engine_config.enable_quantize:
        from torchao.quantization import quantize_, float8_dynamic_activation_float8_weight, float8_weight_only
        quantize_(pipe.text_encoder, float8_weight_only())
        quantize_(pipe.transformer, float8_dynamic_activation_float8_weight())
        if pipe.transformer_2 is not None:
            quantize_(pipe.transformer_2, float8_dynamic_activation_float8_weight())

    flow_shift = 5.0
    scheduler = UniPCMultistepScheduler(prediction_type='flow_prediction', use_flow_sigmas=True, num_train_timesteps=1000, flow_shift=flow_shift)
    pipe.scheduler = scheduler
    local_rank = get_world_group().local_rank
    device = torch.device(f"cuda:{local_rank}")
    pipe.to(device)

    initialize_runtime_state(pipe, engine_config)

    if args.enable_sequential_cpu_offload:
        pipe.enable_sequential_cpu_offload(gpu_id=local_rank)
        logger.info(f"rank {local_rank} sequential CPU offload enabled")
    elif args.enable_model_cpu_offload:
        pipe.enable_model_cpu_offload(gpu_id=local_rank)
        logger.info(f"rank {local_rank} model CPU offload enabled")
    else:
        device = torch.device(f"cuda:{local_rank}")
        pipe = pipe.to(device)

    if args.enable_tiling:
        pipe.vae.enable_tiling()

    if args.enable_slicing:
        pipe.vae.enable_slicing()

    parameter_peak_memory = torch.cuda.max_memory_allocated(device=f"cuda:{local_rank}")

    parallelize_transformer(pipe)
    if args.use_teacache or args.use_fbcache:
        if args.use_teacache and args.use_fbcache:
            logger.warning(f"apply --use_teacache and --use_fbcache togather. we use FBCache")
            use_cache="Fb"
        elif args.use_teacache:
            use_cache="Tea"
        elif args.use_fbcache:
            use_cache="Fb"
        apply_cache_on_pipe(pipe=pipe, use_cache=use_cache, residual_diff_threshold=args.cache_threshold)

    if engine_config.runtime_config.use_torch_compile:
        torch._inductor.config.reorder_for_compute_comm_overlap = True
        from xfuser.model_executor.torch_compile import apply_torch_compile_dynamic_shape_monkey_patch
        apply_torch_compile_dynamic_shape_monkey_patch()
        components = {
            'vae': pipe.vae,
            'transformer': pipe.transformer,
            'transformer_2': pipe.transformer_2,
        }
        for name, component in components.items():
            if component is not None:
                if hasattr(pipe, name):
                    if hasattr(component, 'forward'):
                        optimized_forward = torch.compile(
                            component.forward,
                            mode="default",
                            dynamic=True,
                        )
                        setattr(component, 'forward', optimized_forward)
                        print(f"Finish compiling the {name.replace('_', ' ').title()} forward function")
                else:
                    print(f"Skip compiling the {name.replace('_', ' ').title()}")
            else:
                print(f"Skip compiling the {name.replace('_', ' ').title()}")

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
        start_time = time.time()
        export_to_video(output, output_filename, fps=16, quality=8)
        save_elapsed = time.time() - start_time
        print(f"output saved to {output_filename} elapsed {save_elapsed:.2f} s")

    if get_world_group().rank == get_world_group().world_size - 1:
        print(f"epoch time: {elapsed_time:.2f} sec, parameter memory: {parameter_peak_memory/1e9:.2f} GB, memory: {peak_memory/1e9} GB")

    dist.destroy_process_group()

if __name__ == "__main__":
    main()
