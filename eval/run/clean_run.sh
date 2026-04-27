#!/usr/bin/env bash
set -euo pipefail

PRESET="release"

GPJSON_DRIVER_PATH="./build/${PRESET}/apps/gpjson_driver"
CONFIG_PATH="./config/benchmark/tt1_sharemem.toml"

INDEX_BUILDER_TYPE_LIST=(UNCOMBINED SHAREMEM)

# Full file vs 0.5 GiB partition
INDEX_BUILDER_FILE_PARTITION_SIZE_LIST=(0 536870912)

LOG_FOLDER="logs"
LOG_ID="run-example"
LOG_PATH="${LOG_FOLDER}/"$LOG_ID".log"
LOG_SUMMARY_PATH="${LOG_FOLDER}/"$LOG_ID".summary.log"

mkdir -p "$LOG_FOLDER"
echo "*" >"${LOG_FOLDER}/.gitignore"

cmake --build --preset "$PRESET"

# Optional: clear previous log before this benchmark run.
: >"$LOG_PATH"

for INDEX_BUILDER_TYPE in "${INDEX_BUILDER_TYPE_LIST[@]}"; do
  for INDEX_BUILDER_FILE_PARTITION_SIZE in "${INDEX_BUILDER_FILE_PARTITION_SIZE_LIST[@]}"; do
    "$GPJSON_DRIVER_PATH" \
      --config "$CONFIG_PATH" \
      --index-builder.type "$INDEX_BUILDER_TYPE" \
      --index-builder.file-partition-size "$INDEX_BUILDER_FILE_PARTITION_SIZE" \
      2>&1 | tee --append "$LOG_PATH"
  done
done

uv run --project eval/summarize python eval/summarize/main.py "$LOG_PATH" --max-indent-level 3 --relative-to-baseline speedup | tee "$LOG_SUMMARY_PATH"
