#!/bin/bash
SAVE_DISK=${SAVE_DISK:-"False"}
SAVE_DISK_PATH="/tmp"
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
    if [ $SAVE_DISK != "False" ] ;then
        echo "\"save_disk_path\": \"$SAVE_DISK_PATH\"",
    fi
    echo '"seed": 0,'
    echo '"cfg": 5'
    echo '}'
} > $PAYLOAD_FILE
echo "[INFO] Payload JSON created at $PAYLOAD_FILE"
cat $PAYLOAD_FILE

echo "[INFO] SAVE DISK: $SAVE_DISK"
if [ $SAVE_DISK = "True" ]; then
    curl -X POST "http://localhost:6000/generate" \
        -H "Content-Type: application/json" \
        --data-binary @"$PAYLOAD_FILE" \
        -w '\nResponse Time: %{time_total}s\n'
else
    curl -X POST "http://localhost:6000/generate" \
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
