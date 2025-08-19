#!/bin/bash
set -x

export PYTHONPATH=$PWD:$PYTHONPATH

# Wan2.1 configuration
SCRIPT="wan21_i2v_example.py"
MODEL_ID="/data00/models/Wan2.1-I2V-14B-720P-Diffusers"
# MODEL_ID="/data00/models/Wan2.2-I2V-A14B-Diffusers"
INFERENCE_STEP=40

mkdir -p ./results

# Wan2.1 specific task args
TASK_ARGS="--height 720 --width 1280 --num_frames 81 --seed 0 --use_torch_compile --enable_sage_attn --use_fbcache --cache_threshold 0.2"
# "--enable_quantize"
N_GPUS=8
PARALLEL_ARGS="--ulysses_degree 4 --ring_degree 1"
CFG_ARGS="--use_cfg_parallel"

# Uncomment and modify these as needed
# PIPEFUSION_ARGS="--num_pipeline_patch 8"
# OUTPUT_ARGS="--output_type latent"
PARALLLEL_VAE="--use_parallel_vae"
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
--prompt "Summer beach vacation style, a white cat wearing sunglasses sits on a surfboard. The fluffy-furred feline gazes directly at the camera with a relaxed expression. Blurred beach scenery forms the background featuring crystal-clear waters, distant green hills, and a blue sky dotted with white clouds. The cat assumes a naturally relaxed posture, as if savoring the sea breeze and warm sunlight. A close-up shot highlights the felines intricate details and the refreshing atmosphere of the seaside." \
$CFG_ARGS \
$PARALLLEL_VAE \
$ENABLE_TILING \
$COMPILE_FLAG  \
--negative_prompt "镜头晃动，色调艳丽，过曝，静态，细节模糊不清，字幕，风格，作品，画作，画面，静止，整体发灰，最差质量，低质量，JPEG压缩残留，丑陋的，残缺的，多余的手指，画得不好的手部，画得不好的脸部，畸形的，毁容的，形态畸形的肢体，手指融合，静止不动的画面，杂乱的背景，三条腿，背景人 很多，倒着走"
