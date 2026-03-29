#!/bin/bash

# Root directory of the project.
ROOT_DIR="$(dirname "$(dirname "$(dirname "$(realpath "$0")")")")"

# Configuration of names.
TARGET_DIR="$ROOT_DIR/data/image"
URL="https://s3.amazonaws.com/fast-ai-imageclas/imagenette2-320.tgz"
FILE_NAME="dataset.tgz"

# Create the target directory if it doesn't exist.
mkdir -p "$TARGET_DIR"

# Download the dataset silently only if it doesn't already exist.
if [ ! -f "$TARGET_DIR/$FILE_NAME" ]; then
    curl -L -o "$TARGET_DIR/$FILE_NAME" "$URL" --silent
fi

# Extract the dataset on the target directory.
tar -xzf "$TARGET_DIR/$FILE_NAME" -C "$TARGET_DIR" --strip-components=1
