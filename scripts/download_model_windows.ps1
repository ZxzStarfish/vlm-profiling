# Download the pinned official Qwen2-VL model into the local workspace.
# Run locally; never on the Linux server.
$ErrorActionPreference = 'Stop'
$repo = 'Qwen/Qwen2-VL-2B-Instruct'
$revision = '895c3a49bc3fa70a340399125c650a463535e71c'
$workspace = (Resolve-Path (Join-Path $PSScriptRoot '..')).Path
$manifestPath = Join-Path $workspace 'data/model_manifest.json'
$manifest = Get-Content -LiteralPath $manifestPath -Raw | ConvertFrom-Json
if ($manifest.sha -ne $revision) { throw 'Model manifest revision mismatch.' }

$files = @(
  'chat_template.json', 'config.json', 'generation_config.json',
  'merges.txt', 'model-00001-of-00002.safetensors',
  'model-00002-of-00002.safetensors', 'model.safetensors.index.json',
  'preprocessor_config.json', 'tokenizer.json', 'tokenizer_config.json',
  'vocab.json'
)
$targetDir = Join-Path $workspace "data/models/Qwen2-VL-2B-Instruct/$revision"
New-Item -ItemType Directory -Force -Path $targetDir | Out-Null

foreach ($name in $files) {
  $entry = @($manifest.siblings | Where-Object { $_.rfilename -eq $name })
  if ($entry.Count -ne 1) { throw "Missing or duplicate manifest entry: $name" }
  $expectedSize = [long]$entry[0].size
  $expectedSha = $entry[0].lfs.sha256
  $target = Join-Path $targetDir $name
  $partial = "$target.part"

  $valid = $false
  if (Test-Path -LiteralPath $target) {
    $valid = (Get-Item -LiteralPath $target).Length -eq $expectedSize
    if ($valid -and $expectedSha) {
      $valid = (Get-FileHash -LiteralPath $target -Algorithm SHA256).Hash -ieq $expectedSha
    }
  }
  if ($valid) {
    Write-Output "SKIP verified: $name"
    continue
  }
  if (Test-Path -LiteralPath $target) { throw "Existing file failed verification: $target" }

  $url = "https://huggingface.co/$repo/resolve/$revision/$name"
  Write-Output "Downloading: $name ($expectedSize bytes)"
  & curl.exe --fail --location --continue-at - --retry 5 --retry-delay 5 --connect-timeout 20 --output $partial $url
  if ($LASTEXITCODE -ne 0) { throw "curl failed for $name with exit $LASTEXITCODE" }
  $actualSize = (Get-Item -LiteralPath $partial).Length
  if ($actualSize -ne $expectedSize) { throw "Size mismatch for $name`: $actualSize vs $expectedSize" }
  if ($expectedSha) {
    $actualSha = (Get-FileHash -LiteralPath $partial -Algorithm SHA256).Hash
    if ($actualSha -ine $expectedSha) { throw "SHA256 mismatch for $name" }
  }
  Move-Item -LiteralPath $partial -Destination $target
  Write-Output "VERIFIED: $name"
}

Write-Output "MODEL_READY: $targetDir"
