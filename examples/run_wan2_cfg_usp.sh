#!/bin/bash
set -x

export PYTHONPATH=$PWD:$PYTHONPATH

# Wan2.1 configuration
SCRIPT="wan2_cfg_usp_example.py"
MODEL_ID="/data00/models/Wan2.1-T2V-14B-Diffusers/"
INFERENCE_STEP=50

mkdir -p ./results

# Wan2.1 specific task args
TASK_ARGS="--height 720 --width 1280 --num_frames 81 --seed 0 --enable_sage_attn --use_torch_compile --use_fbcache --cache_threshold 0.16"
N_GPUS=8
PARALLEL_ARGS="--ulysses_degree 4 --ring_degree 1"
CFG_ARGS="--use_cfg_parallel"

# Uncomment and modify these as needed
# PIPEFUSION_ARGS="--num_pipeline_patch 8"
# OUTPUT_ARGS="--output_type latent"
# PARALLLEL_VAE="--use_parallel_vae"
# ENABLE_TILING="--enable_tiling"
# COMPILE_FLAG="--use_torch_compile"

torchrun --nproc_per_node=$N_GPUS ./examples/$SCRIPT \
--model $MODEL_ID \
$PARALLEL_ARGS \
$TASK_ARGS \
$PIPEFUSION_ARGS \
$OUTPUT_ARGS \
--num_inference_steps $INFERENCE_STEP \
--warmup_steps 0 \
--prompt "Two anthropomorphic cats in comfy boxing gear and bright gloves fight intensely on a spotlighted stage." \
$CFG_ARGS \
$PARALLLEL_VAE \
$ENABLE_TILING \
$COMPILE_FLAG  \
--negative_prompt "色调艳丽，过曝，静态，细节模糊不清，字幕，风格，作品，画作，画面，静止，整体发灰，最差质量，低质量，JPEG压缩残留，丑陋的，残缺的，多余的手指，画得不好的手部，画得不好的脸部，畸形的，毁容的，形态畸形的肢体，手指融合，静止不动的画面，杂乱的背景，三条腿，背景人 很多，倒着走"
