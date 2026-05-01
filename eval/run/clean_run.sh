#!/usr/bin/env bash
set -euo pipefail

PRESET="${PRESET:-release}"

GPJSON_DRIVER_PATH="./build/${PRESET}/apps/gpjson_driver"
CONFIG_PATH_LIST=(
  "./config/benchmark/tt1_default.toml"
  "./config/benchmark/tt2_default.toml"
  "./config/benchmark/tt3_default.toml"
  "./config/benchmark/tt4_default.toml"
)

INDEX_BUILDER_TYPE_LIST=(UNCOMBINED SHAREMEM)
INDEX_BUILDER_FILE_PARTITION_SIZE=0
read -r -a SUMMARY_ARGS <<<"${SUMMARY_ARGS:---max-indent-level 3 --relative-to-baseline speedup}"

LOG_FOLDER="logs"
LOG_ID="${LOG_ID:-run-example}"
LOG_PATH="${LOG_FOLDER}/"$LOG_ID".log"
LOG_SUMMARY_PATH="${LOG_FOLDER}/"$LOG_ID".summary.log"

for CONFIG_PATH in "${CONFIG_PATH_LIST[@]}"; do
  if [[ ! -f "$CONFIG_PATH" ]]; then
    echo "Config file not found: $CONFIG_PATH" >&2
    exit 1
  fi
done

sanitize_run_name_component() {
  local value="$1"
  value="${value%.toml}"
  value="${value//[^A-Za-z0-9_]/_}"
  echo "$value"
}

mkdir -p "$LOG_FOLDER"
echo "*" >"${LOG_FOLDER}/.gitignore"

cmake --build --preset "$PRESET"

# Optional: clear previous log before this benchmark run.
: >"$LOG_PATH"

for CONFIG_PATH in "${CONFIG_PATH_LIST[@]}"; do
  CONFIG_NAME="$(sanitize_run_name_component "$(basename "$CONFIG_PATH")")"
  for INDEX_BUILDER_TYPE in "${INDEX_BUILDER_TYPE_LIST[@]}"; do
    RUN_NAME="${CONFIG_NAME}_${INDEX_BUILDER_TYPE}_partition_${INDEX_BUILDER_FILE_PARTITION_SIZE}"
    "$GPJSON_DRIVER_PATH" \
      --config "$CONFIG_PATH" \
      --run-name "$RUN_NAME" \
      --index-builder.type "$INDEX_BUILDER_TYPE" \
      --index-builder.file-partition-size "$INDEX_BUILDER_FILE_PARTITION_SIZE" \
      2>&1 | tee --append "$LOG_PATH"
  done
done

uv run --project eval/summarize python eval/summarize/main.py "$LOG_PATH" "${SUMMARY_ARGS[@]}" | tee "$LOG_SUMMARY_PATH"
