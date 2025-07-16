set -x

export PYTHONPATH=$PWD:$PYTHONPATH
mkdir -p ./results
MODEL_ID="/data00/models/FLUX.1-Kontext-dev"
# task args
TASK_ARGS="--height 1024 --width 1024 --no_use_resolution_binning"

# cache args
# CACHE_ARGS="--use_teacache"
CACHE_ARGS="--use_fbcache"

# On 8 gpus, pp=2, ulysses=2, ring=1, cfg_parallel=2 (split batch)
N_GPUS=1
PARALLEL_ARGS="--ulysses_degree 1 --ring_degree 1"
INFERENCE_STEP=28
# CFG_ARGS="--use_cfg_parallel"

# By default, num_pipeline_patch = pipefusion_degree, and you can tune this parameter to achieve optimal performance.
# PIPEFUSION_ARGS="--num_pipeline_patch 8 "

# For high-resolution images, we use the latent output type to avoid runing the vae module. Used for measuring speed.
# OUTPUT_ARGS="--output_type latent"

# PARALLLEL_VAE="--use_parallel_vae"

# Another compile option is `--use_onediff` which will use onediff's compiler.
# COMPILE_FLAG="--use_torch_compile"


# Use this flag to quantize the T5 text encoder, which could reduce the memory usage and have no effect on the result quality.
# QUANTIZE_FLAG="--use_fp8_t5_encoder"

# SVDQuant flag
SVDQ_FLAG="--use_svdq --svdq_quantized_model_path /data00/models/nunchaku-flux.1-kontext-dev/svdq-int4_r32-flux.1-kontext-dev.safetensors"

# export CUDA_VISIBLE_DEVICES=4,5,6,7

torchrun --nproc_per_node=$N_GPUS ./examples/flux_kontext_usp_example.py \
--model $MODEL_ID \
$PARALLEL_ARGS \
$TASK_ARGS \
$PIPEFUSION_ARGS \
$OUTPUT_ARGS \
--num_inference_steps $INFERENCE_STEP \
--warmup_steps 1 \
--prompt "make the cat very fat" \
$CFG_ARGS \
$PARALLLEL_VAE \
$COMPILE_FLAG \
$QUANTIZE_FLAG \
$CACHE_ARGS \
$SVDQ_FLAG
