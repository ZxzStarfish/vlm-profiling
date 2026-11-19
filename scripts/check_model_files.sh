#!/usr/bin/env bash
# Verify the pinned model after the user manually uploads it.
set -euo pipefail
model_dir=${1:-"$HOME/projects/vlm-profiling/data/models/Qwen2-VL-2B-Instruct/895c3a49bc3fa70a340399125c650a463535e71c"}
cd "$model_dir"
for name in chat_template.json config.json generation_config.json merges.txt \
  model.safetensors.index.json preprocessor_config.json tokenizer.json \
  tokenizer_config.json vocab.json; do
  test -s "$name" || { printf 'Missing or empty: %s\n' "$name"; exit 1; }
done
printf '%s  %s\n' \
  994ac2b03f97de8bc647d0fe5eba2e4b632b3e28dc03574c29bdfc36cf47e1b9 \
  model-00001-of-00002.safetensors | sha256sum -c -
printf '%s  %s\n' \
  92540d8353c8d226a589a3b179bdb33851c970ee2cc2ac7ba035f79425e7b833 \
  model-00002-of-00002.safetensors | sha256sum -c -
printf 'MODEL_FILES_OK %s\n' "$model_dir"
