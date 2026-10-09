#!/bin/bash

if [ -z "$1" ]; then
    echo "provide target directory path!"
    exit 1
fi

TARGET_DIR="${1%/}"
TARGET_DIR="${TARGET_DIR//\\//}"

if [ ! -d "$TARGET_DIR" ]; then
    echo "provide valid target directory path!"
    exit 1
fi

for file in "$TARGET_DIR"/*.mp4; do
    [ -e "$file" ] || break

    base_name=$(basename "$file" .mp4)
    dirname=$(dirname "$file")
    
    output="$dirname/$base_name.wav"

    echo "processing: $file"

    ffmpeg -i "$file" -vn -acodec pcm_s16le -ar 16000 "$output" && rm "$file"

done

echo "done"