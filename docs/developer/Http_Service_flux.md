## Launch a Text-to-Image Http Service

Launch an HTTP-based text-to-image service that generates images from textual descriptions (prompts) using the DiT model. 
The generated images can either be returned directly to users or saved to a specified disk location.
For example, the following command launches a HTTP service with 8 GPUs, 8 Ulysses parallel degree, and the model path is `/data00/models/FLUX.1-schnell`.

```bash
python ./entrypoints/launch.py \
--world_size 8 \
--ulysses_parallel_degree 8 \
--model_path /data00/models/FLUX.1-schnell \
--save_path output \
--use_fbcache \
--cache_threshold 0.16
```


To an example HTTP request is shown below. The `SAVE_SERVER` parameter is optional - if not set, the image will be returned directly; if set true, the generated image will be saved to the specified directory on disk.

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
    echo '"prompt": "a cute rabbit",'
    echo '"height": 1024,'
    echo '"width": 1024,'
    echo '"num_inference_steps": 50,'
    echo "\"save_server\": \"$SAVE_SERVER\"",
    echo '"seed": 42,'
    echo '"cfg": 7.5'
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
        echo $output_data | python3 -c 'import sys,base64; sys.stdout.buffer.write(base64.b64decode(sys.stdin.read()))' > output.png
        echo "[INFO] Image saved to output.png (Size: $(du -h "output.png" | cut -f1))"
    else
        cat $OUTPUT_FILE
        echo "[Error] An error has occured"
    fi
fi
rm -rf $TMP_DIR

```
