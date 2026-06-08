#!/bin/bash

OUTPUT_FILE="filelist.txt"

> "$OUTPUT_FILE"

find . -type f -name "*.npy" | while read -r filepath; do
    filename=$(basename "$filepath" .npy)
    echo "$filename" >> "$OUTPUT_FILE"
done

