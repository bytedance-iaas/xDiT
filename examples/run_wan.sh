#!/bin/bash
set -x

export PYTHONPATH=$PWD:$PYTHONPATH

# Wan configuration
SCRIPT="wan_example.py"
MODEL_ID="/data00/models/Wan2.1-T2V-14B-Diffusers"
INFERENCE_STEP=50

mkdir -p ./results

TASK_ARGS="--height 720 --width 1280 --num_frames 81 --guidance_scale 5.0"

N_GPUS=8
PARALLEL_ARGS="--ulysses_degree 2 --ring_degree 2 --pipefusion_parallel_degree 2"
PIPEFUSION_ARGS="--num_pipeline_patch 8"

torchrun --nproc_per_node=$N_GPUS ./examples/$SCRIPT \
--model $MODEL_ID \
$PARALLEL_ARGS \
$TASK_ARGS \
$PIPEFUSION_ARGS \
$OUTPUT_ARGS \
--num_inference_steps $INFERENCE_STEP \
--warmup_steps 10 \
--prompt "A cat and a dog baking a cake together in a kitchen. The cat is carefully measuring flour, while the dog is stirring the batter with a wooden spoon. The kitchen is cozy, with sunlight streaming through the window." \
--negative_prompt "Bright tones, overexposed, static, blurred details, subtitles, style, works, paintings, images, static, overall gray, worst quality, low quality, JPEG compression residue, ugly, incomplete, extra fingers, poorly drawn hands, poorly drawn faces, deformed, disfigured, misshapen limbs, fused fingers, still picture, messy background, three legs, many people in the background, walking backwards" \
$CFG_ARGS \
$PARALLLEL_VAE \
$ENABLE_TILING \
$ENABLE_MODEL_CPU_OFFLOAD \
$COMPILE_FLAGS