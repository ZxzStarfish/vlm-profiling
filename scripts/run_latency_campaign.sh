#!/usr/bin/env bash
set -euo pipefail
cd -- "$(dirname -- "${BASH_SOURCE[0]}")/.."

gpu_index=${1:?Usage: bash scripts/run_latency_campaign.sh GPU_INDEX [all|stages|modules|trace|repro-synthetic|repro-real|repro-smoke] [SESSION]}
mode=${2:-all}
session=${3:-}
case "$mode" in
  all|stages|modules|trace) ;;
  repro-synthetic|repro-real|repro-smoke)
    if [[ ! "$session" =~ ^[a-zA-Z0-9_-]{1,40}$ ]]; then
      printf 'Reproduction requires a session label such as session-a\n' >&2
      exit 2
    fi
    ;;
  *) printf 'Unknown mode: %s\n' "$mode" >&2; exit 2 ;;
esac
if [[ ! "$gpu_index" =~ ^[0-9]+$ ]]; then
  printf 'GPU index must be a number\n' >&2
  exit 2
fi

gpu_uuid=$(nvidia-smi -i "$gpu_index" --query-gpu=uuid --format=csv,noheader)
available_kib=$(df -Pk . | awk 'NR == 2 { print $4 }')
if (( available_kib < 1048576 )); then
  printf 'Need at least 1 GiB free for campaign files\n' >&2
  exit 2
fi
mkdir -p results
if [[ "$mode" == repro-* ]]; then
  run_dir=$(mktemp -d "results/$mode.$session.XXXXXXXX")
  printf '%s\n' "$session" > "$run_dir/session.txt"
else
  run_dir=$(mktemp -d results/campaign.XXXXXXXX)
fi
date -u +'%Y-%m-%dT%H:%M:%SZ' > "$run_dir/started_utc.txt"
printf 'RUN_DIR=%s\n' "$run_dir"

export CUDA_VISIBLE_DEVICES="$gpu_uuid" HF_HUB_OFFLINE=1 PYTHONDONTWRITEBYTECODE=1
common=(.venv/bin/python -u scripts/profile_stages.py --campaign configs/latency_campaign.json --warmup 2)
sha256sum scripts/profile_stages.py scripts/profile_operators.py scripts/run_latency_campaign.sh \
  configs/latency_campaign.json configs/baseline.json data/smoke.png > "$run_dir/input_sha256.txt"
df -h . > "$run_dir/disk.txt"
nvidia-smi -i "$gpu_uuid" --query-gpu=index,uuid,name,driver_version --format=csv > "$run_dir/gpu.csv"

monitor_gpu() {
  printf 'unix_ms,gpu_util_pct,free_mib,sm_clock_mhz,temperature_c,power_w,compute_processes\n'
  while true; do
    local stamp metrics apps
    stamp=$(date +%s%3N) || return
    metrics=$(nvidia-smi -i "$gpu_uuid" --query-gpu=utilization.gpu,memory.free,clocks.sm,temperature.gpu,power.draw --format=csv,noheader,nounits) || return
    apps=$(nvidia-smi --query-compute-apps=gpu_uuid --format=csv,noheader | awk -v gpu="$gpu_uuid" '$1 == gpu {n++} END {print n+0}') || return
    printf '%s,%s,%s\n' "$stamp" "$metrics" "$apps"
    sleep 2
  done
}

run_part() (
  local part=$1 free util apps gpu_free floor=7000
  shift
  read -r free util < <(nvidia-smi -i "$gpu_uuid" --query-gpu=memory.free,utilization.gpu --format=csv,noheader,nounits | tr -d ',')
  apps=$(nvidia-smi --query-compute-apps=gpu_uuid --format=csv,noheader)
  if (( free < floor )) || printf '%s\n' "$apps" | grep -Fxq "$gpu_uuid"; then
    printf 'GPU %s busy before %s: %s MiB free, %s%% utilization\n' "$gpu_index" "$part" "$free" "$util" >&2
    return 2
  fi
  if (( util > 10 )); then
    # An immediate post-run utilization sample can include the last request.
    sleep 2
    read -r gpu_free util < <(nvidia-smi -i "$gpu_uuid" --query-gpu=memory.free,utilization.gpu --format=csv,noheader,nounits | tr -d ',')
    if (( gpu_free < floor || util > 10 )); then
      printf 'GPU %s stayed busy before %s\n' "$gpu_index" "$part" >&2
      return 2
    fi
  fi
  nvidia-smi -i "$gpu_uuid" --query-gpu=timestamp,memory.free,utilization.gpu --format=csv > "$run_dir/${part}_gpu_before.csv"
  if [[ "$mode" == repro-* ]]; then
    monitor_gpu > "$run_dir/gpu_telemetry.csv" 2> "$run_dir/gpu_telemetry.log" &
    sampler_pid=$!
    trap 'kill "$sampler_pid" 2>/dev/null || true; wait "$sampler_pid" 2>/dev/null || true' EXIT
    cat /proc/loadavg > "$run_dir/cpu_load_before.txt"
  fi
  "${common[@]}" "$@" --out "$run_dir/$part.json" 2>&1 | tee "$run_dir/$part.log"
  nvidia-smi -i "$gpu_uuid" --query-gpu=timestamp,memory.free,utilization.gpu --format=csv > "$run_dir/${part}_gpu_after.csv"
  if [[ "$mode" == repro-* ]]; then
    cat /proc/loadavg > "$run_dir/cpu_load_after.txt"
    if [[ $(wc -l < "$run_dir/gpu_telemetry.csv") -lt 2 ]]; then
      printf 'No GPU telemetry samples. Check %s/gpu_telemetry.log\n' "$run_dir" >&2
      return 2
    fi
  fi
)

if [[ "$mode" == all || "$mode" == stages ]]; then
  run_part stages --repeats 7
fi
if [[ "$mode" == all || "$mode" == modules ]]; then
  run_part modules --modules --repeats 3
fi
if [[ "$mode" == all || "$mode" == trace ]]; then
  run_part operators --trace --trace-repeats 2
fi
if [[ "$mode" == repro-synthetic ]]; then
  run_part stages --repeats 7
fi
if [[ "$mode" == repro-real || "$mode" == repro-smoke ]]; then
  sha256sum configs/reproduction_images.json >> "$run_dir/input_sha256.txt"
  real_args=(--image-suite configs/reproduction_images.json)
  if [[ "$mode" == repro-smoke ]]; then
    run_part stages "${real_args[@]}" --case anchor --warmup 1 --repeats 1
  else
    run_part stages "${real_args[@]}" --repeats 7
  fi
fi
printf 'CAMPAIGN_OK %s\n' "$run_dir"
