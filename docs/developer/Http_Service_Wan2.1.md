## Launch a Text-to-Video Http Service

Launch an HTTP-based text-to-video service that generates videos from textual descriptions (prompts) using the DiT model. 
The generated videos can either be returned directly to users or saved to a specified disk location.
For example, the following command launches a HTTP service with 8 GPUs, 4 Ulysses parallel degree, cfg parallel, and the model path is `/data00/models/Wan2.1-T2V-14B-Diffusers`.

```bash
python ./entrypoints/launch_video.py \
--world_size 8 \
--ulysses_parallel_degree 4 \
--ring_degree 1 \
--use_cfg_parallel \
--use_torch_compile \
--enable_sage_attn \
--use_fbcache \
--cache_threshold 0.16 \
--model_path /data00/models/Wan2.1-T2V-14B-Diffusers
```


To an example HTTP request is shown below. The `SAVE_SERVER` parameter is optional - if not set, the video will be returned directly; if set true, the generated video will be saved to the specified directory on disk.

```bash
#!/bin/bash
serverIP=${serverIP:-"127.0.0.1"}
SAVE_SERVER=${SAVE_SERVER:-"False"}
TMP_DIR="./tmp"
mkdir -p $TMP_DIR
PAYLOAD_FILE="$TMP_DIR/payload_$(date +"%Y%m%d_%H%M%S").json"
HEADER_FILE="$TMP_DIR/headers_$(date +"%Y%m%d_%H%M%S").txt"
OUTPUT_FILE="$TMP_DIR/output_$(date +"%Y%m%d_%H%M%S").bin"

{
    echo '{'
    echo '"prompt": "Two anthropomorphic cats in comfy boxing gear and bright gloves fight intensely on a spotlighted stage.",'
    echo '"negative_prompt": "色调艳丽，过曝，静态，细节模糊不清，字幕，风格，作品，画作，画面，静止，整体发灰，最差质量，低质量，JPEG压缩残留，丑陋的，残缺的，多余的手指，画得不好的手部，画得不好的脸部，畸形的，毁容的，形态畸形的肢体，手指融合，静止不动的画面，杂乱的背景，三条腿，背景人 很多，倒着走",'
    echo '"width": 1280,'
    echo '"height": 720,'
    echo '"num_frames": 81',
    echo '"num_inference_steps": 50,'
    echo "\"save_server\": \"$SAVE_SERVER\"",
    echo '"seed": 0,'
    echo '"cfg": 5'
    echo '}'
} > $PAYLOAD_FILE
echo "[INFO] Payload JSON created at $PAYLOAD_FILE"
cat $PAYLOAD_FILE

echo "[INFO] SAVE_SERVER: $SAVE_SERVER"
if [ "$(echo "$SAVE_SERVER" | tr '[:upper:]' '[:lower:]')" = "true" ]; then
    curl -X POST "http://$serverIP:6000/generate" \
        -H "Content-Type: application/json" \
        --data-binary @"$PAYLOAD_FILE" \
        -w '\nResponse Time: %{time_total}s\n'
else
    curl -X POST "http://$serverIP:6000/generate" \
        -H "Content-Type: application/json" \
        --data-binary @"$PAYLOAD_FILE" \
        -w '\nResponse Time: %{time_total}s\n' \
        -D $HEADER_FILE \
        --output $OUTPUT_FILE

    output_data=$(jq -r '.output' $OUTPUT_FILE 2>/dev/null)
    if [[ "$output_data" != "null" ]]; then
        echo $output_data | python3 -c 'import sys,base64; sys.stdout.buffer.write(base64.b64decode(sys.stdin.read()))' > output.mp4
        echo "[INFO] Video saved to output.mp4 (Size: $(du -h "output.mp4" | cut -f1))"
    else
        cat $OUTPUT_FILE
        echo "[Error] An error has occured"
    fi
fi
rm -rf $TMP_DIR
```
