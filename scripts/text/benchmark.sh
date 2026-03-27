#!/bin/bash

# ------------------------------------------------------------------ #
#  Launch all MarianMT benchmark experiments with accelerate.
#
#  Each experiment directory under configs/text/cuda/<ENV>/ must
#  contain:
#    - accelerate.yaml   (accelerate launch config)
#    - model.yaml         (MarianMT model / benchmark config)
#
#  Every experiment is run twice:
#    1. Without profiling  (profile_no.yaml)
#    2. With profiling     (matched profile config for the environment)
#
#  Results are stored in results/text/cuda/<ENV>/{no_profile,profile}/
# ------------------------------------------------------------------ #

set -euo pipefail

ROOT_DIR="$(dirname "$(dirname "$(dirname "$(realpath "$0")")")")"
CONFIGS_DIR="$ROOT_DIR/configs/text/cuda"
PROFILES_DIR="$ROOT_DIR/configs/profile"
RESULTS_DIR="$ROOT_DIR/results/text/cuda"
RUN_SCRIPT="$ROOT_DIR/scripts/text/run.py"

# Map each environment to its matching profile config.
declare -A PROFILE_MAP=(
    ["1xGPU"]="$PROFILES_DIR/profile_gpu.yaml"
    ["2xGPU"]="$PROFILES_DIR/profile_multi_gpu.yaml"
    ["3xGPU"]="$PROFILES_DIR/profile_multi_gpu.yaml"
    ["4xGPU"]="$PROFILES_DIR/profile_multi_gpu.yaml"
    ["4xGPU+BB16"]="$PROFILES_DIR/profile_multi_gpu.yaml"
    ["4xGPU+BB32"]="$PROFILES_DIR/profile_multi_gpu.yaml"
    ["4xGPU+BB64"]="$PROFILES_DIR/profile_multi_gpu.yaml"
    ["4xGPU+BB128"]="$PROFILES_DIR/profile_multi_gpu.yaml"
    ["4xGPU+BB16+BF16"]="$PROFILES_DIR/profile_multi_gpu.yaml"
    ["4xGPU+BB32+BF16"]="$PROFILES_DIR/profile_multi_gpu.yaml"
    ["4xGPU+BB64+BF16"]="$PROFILES_DIR/profile_multi_gpu.yaml"
    ["4xGPU+BB128+BF16"]="$PROFILES_DIR/profile_multi_gpu.yaml"
    ["4xGPU+BB16+BF16+DZ"]="$PROFILES_DIR/profile_multi_gpu.yaml"
    ["4xGPU+BB32+BF16+DZ"]="$PROFILES_DIR/profile_multi_gpu.yaml"
    ["4xGPU+BB64+BF16+DZ"]="$PROFILES_DIR/profile_multi_gpu.yaml"
    ["4xGPU+BB128+BF16+DZ"]="$PROFILES_DIR/profile_multi_gpu.yaml"
)

PROFILE_NO="$PROFILES_DIR/profile_no.yaml"

# ------------------------------------------------------------------ #

run_experiment() {
    local env_name="$1"
    local accel_config="$2"
    local model_config="$3"
    local trace_config="$4"
    local output_dir="$5"
    local label="$6"

    echo "========================================"
    echo "  [$env_name] $label"
    echo "  accelerate : $accel_config"
    echo "  model      : $model_config"
    echo "  trace      : $trace_config"
    echo "  output     : $output_dir"
    echo "========================================"

    accelerate launch --num_cpu_threads_per_process=4 \
        --config_file "$accel_config" \
        "$RUN_SCRIPT" \
        --model "$model_config" \
        --trace "$trace_config" \
        --output "$output_dir"

    echo ""
}

# ------------------------------------------------------------------ #

for env_dir in "$CONFIGS_DIR"/*/; do
    env_name="$(basename "$env_dir")"
    accel_config="$env_dir/accelerate.yaml"
    model_config="$env_dir/model.yaml"

    # Validate that required config files exist.
    if [ ! -f "$accel_config" ]; then
        echo "[SKIP] $env_name: missing accelerate.yaml"
        continue
    fi
    if [ ! -f "$model_config" ]; then
        echo "[SKIP] $env_name: missing model.yaml"
        continue
    fi

    # Resolve the profile config for this environment.
    profile_config="${PROFILE_MAP[$env_name]:-}"
    if [ -z "$profile_config" ]; then
        echo "[WARN] $env_name: no profile mapping, using profile_gpu.yaml"
        profile_config="$PROFILES_DIR/profile_gpu.yaml"
    fi

    # Run 1: no profiling.
    run_experiment \
        "$env_name" \
        "$accel_config" \
        "$model_config" \
        "$PROFILE_NO" \
        "$RESULTS_DIR/$env_name/no_profile" \
        "no profiling"

    # Run 2: with profiling.
    run_experiment \
        "$env_name" \
        "$accel_config" \
        "$model_config" \
        "$profile_config" \
        "$RESULTS_DIR/$env_name/profile" \
        "with profiling"


done

echo ""
echo "All experiments complete. Results in: $RESULTS_DIR"
