#!/bin/bash

BEATMAPS="/media/nick/HDD/Projects/osu/beatmaps/zipped"
TARGET_DIR="/media/nick/HDD/Projects/osu/beatmaps/unzipped"

rm -rf "$TARGET_DIR"
mkdir -p "$TARGET_DIR"

for file in "$BEATMAPS"/*.osz
do
  echo "$file"
  filename=$(basename "$file")
  filename=${filename%.*}
  dirname="$TARGET_DIR/$filename"
  unzip -j -d "$dirname" "$file" "*.osu" "*.mp3"
done
