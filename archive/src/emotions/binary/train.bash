#!/usr/bin/env bash
set -e

BASE="src/emotions/binary/configs/train_binary.yaml"

for emo in anger disgust sadness tenderness; do
  cfg="/tmp/train_binary_${emo}.yaml"
  cp "$BASE" "$cfg"

  # replace the target emotion line
  sed -i "s|target_emotion: .*|target_emotion: \"emotion-${emo}\"|g" "$cfg"

  python src/emotions/binary/train_binary.py --config "$cfg"
done
