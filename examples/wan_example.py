import time
import torch

# from xfuser.model_executor.pipelines.pipeline_wan import xFuserWanPipeline

from xfuser import xFuserWanPipeline, xFuserArgs
from xfuser.config import FlexibleArgumentParser
from xfuser.core.distributed import (
    get_world_group,
    get_runtime_state,
)

from diffusers.utils import export_to_video
from diffusers import AutoencoderKLWan
from diffusers.schedulers.scheduling_unipc_multistep import UniPCMultistepScheduler

from xfuser.logger import init_logger

logger = init_logger(__name__)

def main():
    parser = FlexibleArgumentParser(description="xFuser Arguments")
    args = xFuserArgs.add_cli_args(parser).parse_args()
    engine_args = xFuserArgs.from_cli_args(args)
    engine_config, input_config = engine_args.create_config()
    local_rank = get_world_group().local_rank

    vae = AutoencoderKLWan.from_pretrained(
        engine_config.model_config.model,
        subfolder="vae",
        torch_dtype=torch.float32
    )

    scheduler = UniPCMultistepScheduler(
        prediction_type='flow_prediction',
        use_flow_sigmas=True,
        num_train_timesteps=1000,
        flow_shift=5.0
    )

    pipe = xFuserWanPipeline.from_pretrained(
        pretrained_model_name_or_path=engine_config.model_config.model,
        torch_dtype=torch.float32,
        vae=vae,
        scheduler=scheduler,
        engine_config=engine_config,
    ).to(f"cuda:{local_rank}")
    model_memory = torch.cuda.max_memory_allocated(device=f"cuda:{local_rank}")

    torch.cuda.reset_peak_memory_stats()
    start_time = time.time()
    output = pipe(
        height=input_config.height,
        width=input_config.width,
        prompt=input_config.prompt,
        negative_prompt=input_config.negative_prompt,
        num_frame=input_config.num_frames,
        num_inference_steps=input_config.num_inference_steps,
        output_type=input_config.output_type,
        guidance_scale=input_config.guidance_scale,
        generator=torch.Generator(device="cuda").manual_seed(input_config.seed)
    )
    end_time = time.time()
    elapsed_time = end_time - start_time
    peak_memory = torch.cuda.max_memory_allocated(device=f"cuda:{local_rank}")
    
    parallel_info = (
        f"dp{engine_args.data_parallel_degree}_cfg{engine_config.parallel_config.cfg_degree}_"
        f"ulysses{engine_args.ulysses_degree}_ring{engine_args.ring_degree}_"
        f"pp{engine_args.pipefusion_parallel_degree}_patch{engine_args.num_pipeline_patch}_tc_{engine_args.use_torch_compile}"
    )

    export_to_video(output, "output.mp4", fps=16)

    if get_world_group().rank == get_world_group().world_size - 1:
        print(f"epoch time: {elapsed_time:.2f} sec, model memory: {model_memory/1e9:.2f} GB, overall memory: {peak_memory/1e9:.2f} GB")

    get_runtime_state().destroy_distributed_env()

if __name__ == "__main__":
    main()