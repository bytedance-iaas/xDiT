import os
os.environ['NCCL_DEBUG'] = 'ERROR'
import time
import torch
import ray
import logging
import base64
from io import BytesIO
import imageio
from xfuser.logger import init_logger

logger = init_logger(__name__)
from fastapi import FastAPI, HTTPException
from diffusers.schedulers.scheduling_unipc_multistep import UniPCMultistepScheduler

from pydantic import BaseModel
from typing import Optional
import argparse
from xfuser import (
    xFuserWanPipeline,
    xFuserArgs,
)
from xfuser.model_executor.cache.diffusers_adapters.wan import apply_cache_on_pipe
from xfuser.model_executor.pipelines import pipeline_wan
from xfuser.model_executor.layers.attention_processor import xFuserWanAttnProcessor2_0

from xfuser.core.distributed import (
    get_runtime_state,
    initialize_runtime_state,
    get_pipeline_parallel_world_size,
)

args = None

# Define request model
class GenerateRequest(BaseModel):
    prompt: str
    negative_prompt: Optional[str] = None
    num_inference_steps: Optional[int] = 50
    seed: Optional[int] = 0
    cfg: Optional[float] = 5
    save_server: Optional[str] = "False"
    height: Optional[int] = 720
    width: Optional[int] = 1280
    num_frames: Optional[int] = 81

    # Add input validation
    class Config:
        json_schema_extra = {
            "example": {
                "prompt": "a beautiful landscape",
                "save_server": "False",
                "num_inference_steps": 50,
                "warmup_steps": 1,
                "seed": 0,
                "cfg": 5,
                "height": 720,
                "width": 1280,
                "num_frames": 81
            }
        }


app = FastAPI()


@ray.remote(num_gpus=1)
class VideoGenerator:
    def __init__(self, xfuser_args: xFuserArgs, rank: int, world_size: int):
        # Set PyTorch distributed environment variables
        os.environ["RANK"] = str(rank)
        os.environ["WORLD_SIZE"] = str(world_size)
        os.environ["MASTER_ADDR"] = "127.0.0.1"
        os.environ["MASTER_PORT"] = "29500"

        self.rank = rank
        self.setup_logger()
        self.initialize_model(xfuser_args)

    def setup_logger(self):
        self.logger = logging.getLogger(__name__)
        # Add console handler if not already present
        if not self.logger.handlers:
            console_handler = logging.StreamHandler()
            console_handler.setLevel(logging.INFO)
            formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
            console_handler.setFormatter(formatter)
            self.logger.addHandler(console_handler)
            self.logger.setLevel(logging.INFO)

    def initialize_model(self, xfuser_args: xFuserArgs):

        # init distributed environment in create_config
        self.engine_config, self.input_config = xfuser_args.create_config()
        model_name = self.engine_config.model_config.model.split("/")[-1]
        self.logger.info(f"model_name, {model_name}")
        pipeline_map = {
            "Wan2.1-T2V-14B-Diffusers": xFuserWanPipeline,
        }

        PipelineClass = pipeline_map.get(model_name)
        if PipelineClass is None:
            raise NotImplementedError(f"{model_name} is currently not supported!")

        self.logger.info(f"Initializing model {model_name} from {xfuser_args.model}")

        self.pipe = PipelineClass.from_pretrained(
            pretrained_model_name_or_path=xfuser_args.model,
            engine_config=self.engine_config,
            torch_dtype=torch.bfloat16,
        ).to("cuda")

        initialize_runtime_state(self.pipe, self.engine_config)
        if self.pipe.__class__.__name__.startswith("xFuserWan"):
            scheduler = UniPCMultistepScheduler(prediction_type='flow_prediction', use_flow_sigmas=True,
                                                num_train_timesteps=1000, flow_shift=5.0)
            self.pipe.scheduler = scheduler
            if xfuser_args.enable_sage_attn:
                setattr(xFuserWanAttnProcessor2_0, "enable_sage_attn", True)
            else:
                setattr(xFuserWanAttnProcessor2_0, "enable_sage_attn", False)
            setattr(xFuserWanAttnProcessor2_0, "enable_fa3", False)

            pipeline_wan.parallelize_transformer(self.pipe)

            if xfuser_args.use_teacache or xfuser_args.use_fbcache:
                if xfuser_args.use_teacache and xfuser_args.use_fbcache:
                    logger.warning(f"apply --use_teacache and --use_fbcache togather. we use FBCache")
                    use_cache = "Fb"
                elif xfuser_args.use_teacache:
                    use_cache = "Tea"
                elif xfuser_args.use_fbcache:
                    use_cache = "Fb"
                apply_cache_on_pipe(pipe=self.pipe, use_cache=use_cache,
                                    residual_diff_threshold=xfuser_args.cache_threshold)

        if xfuser_args.use_torch_compile:
            self.pipe.transformer = torch.compile(self.pipe.transformer,
                                                  mode="default")

        get_runtime_state().set_video_input_parameters(
            height=self.input_config.height,
            width=self.input_config.width,
            batch_size=1,
            num_inference_steps=self.input_config.num_inference_steps,
            split_text_embed_in_sp=get_pipeline_parallel_world_size() == 1,
        )

        self.pipe.prepare_run(self.input_config, steps=1)
        self.logger.info("Model initialization completed")

    def generate(self, request: GenerateRequest):
        try:
            start_time = time.time()
            output = self.pipe(
                height=request.height,
                width=request.width,
                prompt=request.prompt,
                negative_prompt=request.negative_prompt,
                num_inference_steps=request.num_inference_steps,
                num_frames=request.num_frames,
                output_type="np",
                generator=torch.Generator(device="cuda").manual_seed(request.seed),
                guidance_scale=request.cfg,
            )
            elapsed_time = time.time() - start_time

            if self.pipe.is_dp_last_group():
                video_frames = output.frames[0]
                buffer = BytesIO()
                with imageio.get_writer(buffer, format="mp4", codec="libx264", fps=16, quality=8) as writer:
                    for frame in video_frames:
                        writer.append_data(frame)
                writer.close()
                video_bytes = buffer.getvalue()
                if str(request.save_server).lower() == "true":
                    global args
                    timestamp = time.strftime("%Y%m%d-%H%M%S")
                    filename = f"generated_video_{timestamp}.mp4"
                    file_path = os.path.join(args.save_path, filename)
                    os.makedirs(args.save_path, exist_ok=True)
                    with open(file_path, "wb") as f:
                        f.write(video_bytes)
                    logger.info(f"Video saved in {file_path}")
                    return {
                        "message": "Video generated successfully",
                        "elapsed_time": f"{elapsed_time:.2f} sec",
                        "output": file_path,
                        "save_to_disk": True
                    }
                else:
                    # Convert to base64
                    video_str = base64.b64encode(video_bytes).decode()
                    return {
                        "message": "Video generated successfully",
                        "elapsed_time": f"{elapsed_time:.2f} sec",
                        "output": video_str,
                        "save_to_disk": False
                    }
            return None

        except Exception as e:
            self.logger.error(f"Error generating video: {str(e)}")
            raise HTTPException(status_code=500, detail=str(e))


class Engine:
    def __init__(self, world_size: int, xfuser_args: xFuserArgs):
        # Ensure Ray is initialized
        if not ray.is_initialized():
            ray.init()

        num_workers = world_size
        self.workers = [
            VideoGenerator.remote(xfuser_args, rank=rank, world_size=world_size)
            for rank in range(num_workers)
        ]

    async def generate(self, request: GenerateRequest):
        results = ray.get([
            worker.generate.remote(request)
            for worker in self.workers
        ])

        return next(path for path in results if path is not None)


@app.post("/generate")
async def generate_video(request: GenerateRequest):
    try:
        # Add input validation
        if not request.prompt:
            raise HTTPException(status_code=400, detail="Prompt cannot be empty")
        if request.height <= 0 or request.width <= 0:
            raise HTTPException(status_code=400, detail="Height and width must be positive")
        if request.num_inference_steps <= 0:
            raise HTTPException(status_code=400, detail="num_inference_steps must be positive")

        result = await engine.generate(request)
        return result
    except Exception as e:
        if isinstance(e, HTTPException):
            raise e
        raise HTTPException(status_code=500, detail=str(e))


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='xDiT HTTP Service')
    parser.add_argument('--model_path', type=str, help='Path to the model', required=True)
    parser.add_argument('--save_path', type=str, default='output', help='Path to save generated videos')
    parser.add_argument('--world_size', type=int, default=1, help='Number of parallel workers')
    parser.add_argument('--pipefusion_parallel_degree', type=int, default=1,
                        help='Degree of pipeline fusion parallelism')
    parser.add_argument('--ulysses_parallel_degree', type=int, default=1,
                        help='Degree of Ulysses parallelism')
    parser.add_argument('--ring_degree', type=int, default=1, help='Degree of ring parallelism')
    parser.add_argument('--use_cfg_parallel', action='store_true', help='Whether to use CFG parallel')
    parser.add_argument('--use_torch_compile', action='store_true', help='Whether to use torch compile')
    parser.add_argument('--enable_sage_attn', action='store_true', help='Whether to enable sage attn')
    parser.add_argument('--use_fbcache', action='store_true', help='Whether to use FBcache')
    parser.add_argument('--use_teacache', action='store_true', help='Whether to use Teacache')
    parser.add_argument('--cache_threshold', type=float, default=0.16, help='Threshold of teacache or fbcache')
    args = parser.parse_args()

    xfuser_args = xFuserArgs(
        model=args.model_path,
        trust_remote_code=True,
        ulysses_degree=args.ulysses_parallel_degree,
        use_cfg_parallel=args.use_cfg_parallel,
        use_torch_compile=args.use_torch_compile,
        enable_sage_attn=args.enable_sage_attn,
        use_fbcache=args.use_fbcache,
        use_teacache=args.use_teacache,
        cache_threshold=args.cache_threshold,
        height=720,
        width=1280,
        num_inference_steps=50,
    )

    engine = Engine(
        world_size=args.world_size,
        xfuser_args=xfuser_args
    )

    # Start the server
    import uvicorn

    uvicorn.run(app, host="0.0.0.0", port=6000)
