#!/usr/bin/env bash
set -euo pipefail
cd -- "$(dirname -- "${BASH_SOURCE[0]}")/.."

gpu=${1:?Usage: bash scripts/run_decode_repeats.sh GPU_INDEX}
[[ "$gpu" =~ ^[0-9]+$ ]] || { printf 'GPU index must be numeric\n' >&2; exit 2; }
uuid=$(nvidia-smi -i "$gpu" --query-gpu=uuid --format=csv,noheader)
nsys=${NSYS_BIN:-"$PWD/.cache/nsight/nsight_systems-linux-x86_64-2024.6.2.225-archive/nsight-systems/2024.6.2/target-linux-x64/nsys"}
test -x "$nsys"
test "$(df -Pk . | awk 'NR == 2 {print $4}')" -ge 1048576

check_idle() {
  local free util apps
  read -r free util < <(nvidia-smi -i "$uuid" --query-gpu=memory.free,utilization.gpu --format=csv,noheader,nounits | tr -d ',')
  apps=$(nvidia-smi --query-compute-apps=gpu_uuid --format=csv,noheader)
  if (( free < 7000 || util > 10 )) || grep -Fxq "$uuid" <<< "$apps"; then
    printf 'GPU %s busy: free=%s MiB, utilization=%s%%\n' "$gpu" "$free" "$util" >&2
    return 2
  fi
}

# Compare a CUDA process against this runner's process tree. Persist only aggregate counts/MiB.
is_our_process() {
  local pid=$1 parent
  while (( pid > 1 )); do
    if (( pid == $$ )); then return 0; fi
    parent=$(ps -o ppid= -p "$pid") || return 1
    parent=${parent//[[:space:]]/}
    [[ "$parent" =~ ^[0-9]+$ ]] || return 1
    pid=$parent
  done
  return 1
}

sample_gpu() {
  local stamp metrics processes card pid mib count other other_mib
  printf 'unix_ms,temperature_c,sm_clock_mhz,memory_clock_mhz,used_mib,free_mib,util_pct,power_w,compute_processes,other_compute_processes,other_compute_mib\n'
  while true; do
    stamp=$(date +%s%3N) || return
    metrics=$(nvidia-smi -i "$uuid" --query-gpu=temperature.gpu,clocks.sm,clocks.mem,memory.used,memory.free,utilization.gpu,power.draw --format=csv,noheader,nounits | tr -d ' ') || return
    processes=$(nvidia-smi --query-compute-apps=gpu_uuid,pid,used_gpu_memory --format=csv,noheader,nounits | tr -d ' ') || return
    count=0
    other=0
    other_mib=0
    while IFS=, read -r card pid mib; do
      if [[ "$card" != "$uuid" ]]; then continue; fi
      (( count += 1 ))
      if ! is_our_process "$pid"; then
        (( other += 1 ))
        if [[ "$mib" =~ ^[0-9]+$ ]]; then other_mib=$((other_mib + mib)); fi
      fi
    done <<< "$processes"
    printf '%s,%s,%s,%s,%s\n' "$stamp" "$metrics" "$count" "$other" "$other_mib"
    sleep 1
  done
}

check_idle
out=$(mktemp -d "results/decode-repeat.gpu$gpu.XXXXXXXX")
printf 'RUN_DIR=%s\n' "$out"
export CUDA_VISIBLE_DEVICES="$uuid" HF_HUB_OFFLINE=1 PYTHONDONTWRITEBYTECODE=1
nvidia-smi -i "$uuid" --query-gpu=index,uuid,name,driver_version --format=csv > "$out/gpu.csv"
"$nsys" --version > "$out/nsys_version.txt"
sha256sum scripts/run_decode_repeats.sh scripts/{profile_decode_study,profile_stages,profile_operators,analyze_decode_timeline}.py \
  configs/{baseline,latency_campaign}.json data/smoke.png > "$out/input_sha256.txt"

run_group() (
  local directory="$out/group-$1" monitor
  mkdir "$directory"
  check_idle
  sample_gpu > "$directory/gpu_telemetry.csv" 2> "$directory/gpu_telemetry.log" &
  monitor=$!
  trap 'kill "$monitor" 2>/dev/null || true; wait "$monitor" 2>/dev/null || true' EXIT
  date -u +%FT%TZ > "$directory/started_utc.txt"
  cat /proc/loadavg > "$directory/cpu_load_before.txt"
  .venv/bin/python -u scripts/profile_decode_study.py --mode benchmark --repeats 5 --out "$directory/benchmark.json" 2>&1 | tee "$directory/benchmark.log"
  sleep 2
  check_idle
  "$nsys" profile --trace=cuda,nvtx --sample=none --cpuctxsw=none \
    --capture-range=cudaProfilerApi --capture-range-end=stop -o "$directory/anchor" \
    .venv/bin/python -u scripts/profile_decode_study.py --mode nsight --case anchor --out "$directory/anchor.json" 2>&1 | tee "$directory/anchor.nsys.log"
  "$nsys" export --type sqlite --output "$directory/anchor.sqlite" "$directory/anchor.nsys-rep" > "$directory/export.log" 2>&1
  .venv/bin/python scripts/analyze_decode_timeline.py --trace "$directory/anchor.sqlite" --out "$directory/timeline"
  cat /proc/loadavg > "$directory/cpu_load_after.txt"
  test "$(wc -l < "$directory/gpu_telemetry.csv")" -gt 1
  printf 'REPEAT_GROUP_OK %s\n' "$directory"
)

for group in 1 2 3; do
  run_group "$group"
  sleep 2
done
printf 'DECODE_REPEATS_OK %s\n' "$out"
