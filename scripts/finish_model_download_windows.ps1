# Local background helper: wait for verified shards, fetch remaining small files.
$ErrorActionPreference = 'Stop'
$root = (Resolve-Path (Join-Path $PSScriptRoot '..')).Path
$folder = Join-Path $root 'data/models/Qwen2-VL-2B-Instruct/895c3a49bc3fa70a340399125c650a463535e71c'
$ready = Join-Path $root 'data/MODEL_READY.txt'
$log = Join-Path $root 'data/model_finalize.log'
$deadline = (Get-Date).AddHours(24)
try {
  while (-not ((Test-Path (Join-Path $folder 'model-00001-of-00002.safetensors')) -and
                (Test-Path (Join-Path $folder 'model-00002-of-00002.safetensors')))) {
    if ((Get-Date) -ge $deadline) { throw 'Timed out waiting for verified model shards' }
    Start-Sleep -Seconds 30
  }
  & pwsh.exe -NoProfile -File `
    (Join-Path $PSScriptRoot 'download_model_windows.ps1') *> $log
  if ($LASTEXITCODE -ne 0) { throw "Final downloader failed; see $log" }
  Set-Content -LiteralPath $ready -Value $folder -Encoding ascii
} catch {
  Add-Content -LiteralPath $log -Value "ERROR: $_"
}
