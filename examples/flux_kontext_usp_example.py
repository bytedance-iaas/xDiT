import functools
from typing import List, Optional

import logging
import time
import torch
from diffusers import DiffusionPipeline
from diffusers.utils import load_image
from xfuser import xFuserArgs, xFuserFluxKontextPipeline
from xfuser.config import FlexibleArgumentParser
from xfuser.core.distributed import (
    get_world_group,
    get_data_parallel_world_size,
    get_data_parallel_rank,
    get_runtime_state,
    get_classifier_free_guidance_world_size,
    get_classifier_free_guidance_rank,
    get_cfg_group,
    get_sequence_parallel_world_size,
    get_sequence_parallel_rank,
    get_sp_group,
    is_dp_last_group,
    initialize_runtime_state,
    get_pipeline_parallel_world_size,
)
from xfuser.model_executor.pipelines.pipeline_flux_kontext import parallelize_transformer


def main():
    parser = FlexibleArgumentParser(description="xFuser Arguments")
    args = xFuserArgs.add_cli_args(parser).parse_args()
    engine_args = xFuserArgs.from_cli_args(args)
    engine_config, input_config = engine_args.create_config()
    engine_config.runtime_config.dtype = torch.bfloat16
    local_rank = get_world_group().local_rank

    assert engine_args.pipefusion_parallel_degree == 1, "This script does not support PipeFusion."

    cache_args = {
        "use_teacache": engine_args.use_teacache,
        "use_fbcache": engine_args.use_fbcache,
        "rel_l1_thresh": 0.12,
        "return_hidden_states_first": False,
        "num_steps": input_config.num_inference_steps,
    }

    pipe = xFuserFluxKontextPipeline.from_pretrained(
        pretrained_model_name_or_path=engine_config.model_config.model,
        cache_args=cache_args,
        torch_dtype=torch.bfloat16,
        engine_config=engine_config,
    )

    if args.enable_sequential_cpu_offload:
        pipe.enable_sequential_cpu_offload(gpu_id=local_rank)
        logging.info(f"rank {local_rank} sequential CPU offload enabled")
    else:
        pipe = pipe.to(f"cuda:{local_rank}")

    parameter_peak_memory = torch.cuda.max_memory_allocated(device=f"cuda:{local_rank}")

    if get_sequence_parallel_world_size() > 1:
        parallelize_transformer(pipe)
    get_runtime_state().set_input_parameters(
        height=input_config.height,
        width=input_config.width,
        batch_size=1,
        num_inference_steps=1,
        max_condition_sequence_length=512,
        split_text_embed_in_sp=get_pipeline_parallel_world_size() == 1,
    )
    pipe.prepare_run(input_config, steps=1)
    image = load_image("/data00/datasets/kontext-bench/test/images/0000.jpg").convert("RGB")

    if engine_config.runtime_config.use_torch_compile:
        torch._inductor.config.reorder_for_compute_comm_overlap = True
        pipe.transformer = torch.compile(pipe.transformer, mode="max-autotune-no-cudagraphs")

        # one step to warmup the torch compiler
        output = pipe(
            image=image,
            height=input_config.height,
            width=input_config.width,
            prompt=input_config.prompt,
            num_inference_steps=1,
            output_type=input_config.output_type,
            guidance_scale=2.5,
            generator=torch.Generator(device="cuda").manual_seed(input_config.seed),
        ).images

    torch.cuda.reset_peak_memory_stats()
    start_time = time.time()

    if engine_args.use_teacache is True or engine_args.use_fbcache is True and engine_config.use_svdq is False:
        pipe.transformer.clear_cache_modulated_inputs()

    output = pipe(
        image=image,
        height=input_config.height,
        width=input_config.width,
        prompt=input_config.prompt,
        num_inference_steps=input_config.num_inference_steps,
        output_type=input_config.output_type,
        guidance_scale=2.5,
        generator=torch.Generator(device="cuda").manual_seed(input_config.seed),
    )
    end_time = time.time()
    elapsed_time = end_time - start_time
    peak_memory = torch.cuda.max_memory_allocated(device=f"cuda:{local_rank}")

    parallel_info = (
        f"dp{engine_args.data_parallel_degree}_cfg{engine_config.parallel_config.cfg_degree}_"
        f"ulysses{engine_args.ulysses_degree}_ring{engine_args.ring_degree}_"
        f"tp{engine_args.tensor_parallel_degree}_"
        f"pp{engine_args.pipefusion_parallel_degree}_patch{engine_args.num_pipeline_patch}"
    )
    if input_config.output_type == "pil":
        dp_group_index = get_data_parallel_rank()
        num_dp_groups = get_data_parallel_world_size()
        dp_batch_size = (input_config.batch_size + num_dp_groups - 1) // num_dp_groups
        if is_dp_last_group():
            for i, image in enumerate(output.images):
                image_rank = dp_group_index * dp_batch_size + i
                image_name = f"flux_result_{parallel_info}_{image_rank}_tc_{engine_args.use_torch_compile}.png"
                image.save(f"./results/{image_name}")
                print(f"image {i} saved to ./results/{image_name}")


    if get_world_group().rank == get_world_group().world_size - 1:
        print(
            f"epoch time: {elapsed_time:.2f} sec, parameter memory: {parameter_peak_memory / 1e9:.2f} GB, memory: {peak_memory / 1e9:.2f} GB"
        )
    get_runtime_state().destroy_distributed_env()


if __name__ == "__main__":
    main()
