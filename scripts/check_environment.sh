#!/usr/bin/env bash
# Read-only inventory plus temporary create/remove probes in the current directory and /mnt/sdb.
# No package installation, model download, GPU workload, or system configuration change.
set -u

section() { printf '\n===== %s =====\n' "$1"; }
run() {
  printf '$'
  printf ' %q' "$@"
  printf '\n'
  "$@" 2>&1 || printf '[WARN] command failed (exit %s)\n' "$?"
}
has() { command -v "$1" >/dev/null 2>&1; }

section "Identity and location"
run id
run pwd
printf 'HOME=%s\n' "${HOME:-<unset>}"
run uname -a

section "GPU and driver"
if has nvidia-smi; then
  run nvidia-smi -L
  run nvidia-smi --query-gpu=index,name,memory.total,memory.used,utilization.gpu,driver_version --format=csv,noheader,nounits
  run nvidia-smi --query-compute-apps=gpu_uuid,pid,used_gpu_memory --format=csv,noheader,nounits
  run nvidia-smi
else
  printf '[WARN] nvidia-smi not found\n'
fi
if has nvcc; then run nvcc --version; else printf 'nvcc: not found (CUDA Wheel does not require it)\n'; fi

section "Python and pip"
if has python3; then
  run python3 --version
  run python3 -c 'import sys,platform; print("executable:",sys.executable); print("architecture:",platform.machine()); print("prefix:",sys.prefix)'
  if has getconf; then run getconf GNU_LIBC_VERSION; fi
  printf 'CUDA_VISIBLE_DEVICES=%s\n' "${CUDA_VISIBLE_DEVICES:-<unset>}"
  run python3 -m pip --version
  if python3 -m venv --help >/dev/null 2>&1; then
    printf 'venv module: available (help check only)\n'
  else
    printf '[WARN] venv module: unavailable\n'
  fi
  if python3 -m ensurepip --version >/dev/null 2>&1; then
    printf 'ensurepip: available\n'
  else
    printf '[WARN] ensurepip: unavailable; use venv --without-pip and pip --python\n'
  fi
  python3 - <<'PY'
try:
    import torch
    print('torch:', torch.__version__)
    print('torch CUDA build:', torch.version.cuda)
    print('torch CUDA available:', torch.cuda.is_available())
    if torch.cuda.is_available():
        print('torch visible GPU count:', torch.cuda.device_count())
except ImportError as exc:
    print('torch: not installed for this python3:', exc)
except Exception as exc:
    print('torch import/CUDA check failed:', type(exc).__name__, str(exc))
PY
else
  printf '[WARN] python3 not found\n'
fi

section "Storage, inode, and temporary write probes"
run df -h . "${HOME:-.}" /mnt/sdb
run df -i . "${HOME:-.}" /mnt/sdb
for target in . /mnt/sdb; do
  if [ ! -d "$target" ]; then
    printf '[WARN] %s: directory missing\n' "$target"
    continue
  fi
  if [ ! -w "$target" ] || [ ! -x "$target" ]; then
    printf '[WARN] %s: no write/traverse permission\n' "$target"
    continue
  fi
  probe=$(mktemp -d "$target/.vlm-probe-XXXXXXXX" 2>&1) || {
    printf '[WARN] %s: temporary directory creation failed: %s\n' "$target" "$probe"
    continue
  }
  printf '%s: temporary directory creation succeeded\n' "$target"
  if ! rmdir -- "$probe"; then printf '[WARN] remove probe manually: %s\n' "$probe"; fi
done
if [ -d /mnt/sdb ]; then run ls -ld /mnt/sdb; fi

section "Profiling tools"
for tool in nsys ncu; do
  if has "$tool"; then
    printf '%s: %s\n' "$tool" "$(command -v "$tool")"
    run "$tool" --version
  else
    printf '%s: not found\n' "$tool"
  fi
done

section "Network (HTTP headers only, <= 8 s per request)"
if has curl; then
  for url in \
    https://pypi.tuna.tsinghua.edu.cn/simple/requests/ \
    https://download.pytorch.org/whl/cu126/torch/ \
    https://huggingface.co/Qwen/Qwen2-VL-2B-Instruct/resolve/main/config.json; do
    code=$(curl -L -I -sS -o /dev/null -w '%{http_code}' --connect-timeout 4 --max-time 8 "$url" 2>&1)
    rc=$?
    printf '%s -> curl_exit=%s http=%s\n' "$url" "$rc" "$code"
  done
else
  printf '[WARN] curl not found; network probes skipped\n'
fi
printf '\nCheck complete. Share the full output for interpretation.\n'
