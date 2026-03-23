#!/bin/bash

# Definir la ruta de destino relativa (sube un nivel desde 'scripts' y entra en 'data/raw')
TARGET_DIR="../data/img"
URL="https://s3.amazonaws.com/fast-ai-imageclas/imagenette2-320.tgz"
FILE_NAME="imagenette2-320.tgz"

mkdir -p "$TARGET_DIR"
echo "Descargando Imagenette desde $URL..."
curl -o "$TARGET_DIR/$FILE_NAME" "$URL"

tar -xzf "$TARGET_DIR/$FILE_NAME" -C "$TARGET_DIR"

rm "$TARGET_DIR/$FILE_NAME"

