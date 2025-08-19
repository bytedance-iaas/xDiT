#!/bin/bash
serverIP=${serverIP:-"127.0.0.1"}
SAVE_SERVER=${SAVE_SERVER:-"False"}
TMP_DIR="./tmp"
mkdir -p $TMP_DIR
PAYLOAD_FILE="$TMP_DIR/payload_$(date +"%Y%m%d_%H%M%S").json"
HEADER_FILE="$TMP_DIR/headers_$(date +"%Y%m%d_%H%M%S").txt"
OUTPUT_FILE="$TMP_DIR/output_$(date +"%Y%m%d_%H%M%S").bin"
ORIGIN_IMAGE_FILE="./entrypoints/image_examples/i2v_input.JPG"   #revise input image here

result=$(python3 -c "\
from PIL.JpegImagePlugin import JpegImageFile; \
from PIL import Image; \
from io import BytesIO; \
import base64; \
image: JpegImageFile = Image.open('$ORIGIN_IMAGE_FILE'); \
buffer = BytesIO(); \
image.save(buffer, format='PNG'); \
base64_str = base64.b64encode(buffer.getvalue()).decode('utf-8'); \
width, height = image.size; \
print(f'{base64_str},{width},{height}')\
")

base64_str=$(echo "$result" | cut -d',' -f1)
width=$(echo "$result" | cut -d',' -f2)
height=$(echo "$result" | cut -d',' -f3)

{
    echo '{'
    echo '"prompt": "Summer beach vacation style, a white cat wearing sunglasses sits on a surfboard. The fluffy-furred feline gazes directly at the camera with a relaxed expression. Blurred beach scenery forms the background featuring crystal-clear waters, distant green hills, and a blue sky dotted with white clouds. The cat assumes a naturally relaxed posture, as if savoring the sea breeze and warm sunlight. A close-up shot highlights the felines intricate details and the refreshing atmosphere of the seaside.",'
    echo '"negative_prompt": "镜头抖动，色调艳丽，过曝，静态，细节模糊不清，字幕，风格，作品，画作，画面，静止，整体发灰，最差质量，低质量，JPEG压缩残留，丑陋的，残缺的，多余的手指，画得不好的手部，画得不好的脸部，畸形的，毁容的，形态畸形的肢体，手指融合，静止不动的画面，杂乱的背景，三条腿，背景人 很多，倒着走",'
    echo "\"image\": \"$base64_str\"",
    echo '"height": "1280",'
    echo '"width": "720",'
    echo '"num_frames": 81,'
    echo '"num_inference_steps": 40,'
    echo "\"save_server\": \"$SAVE_SERVER\"",
    echo '"seed": 0,'
    echo '"cfg": 5'
    echo '}'
} > $PAYLOAD_FILE
echo "[INFO] Payload JSON created at $PAYLOAD_FILE"
# cat $PAYLOAD_FILE

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

    start_time=$(date +%s)
    output_data=$(jq -r '.output' $OUTPUT_FILE 2>/dev/null)
    elapsed_time=$(jq -r '.elapsed_time' $OUTPUT_FILE 2>/dev/null)
    if [[ "$output_data" != "null" ]]; then
        echo $output_data | python3 -c 'import sys,base64; sys.stdout.buffer.write(base64.b64decode(sys.stdin.read()))' > output.mp4
        end_time=$(date +%s)
        elapsed=$((end_time - start_time))
        echo "[INFO] Video saved to output.mp4 (size: $(du -h "output.mp4" | cut -f1), cost: ${elapsed} sec, serving elapsed: ${elapsed_time})"
    else
        cat $OUTPUT_FILE
        echo "[Error] An error has occured"
    fi
fi
rm -rf $TMP_DIR
