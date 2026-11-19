# Local-only, resumable range downloader for the two pinned model shards.
$ErrorActionPreference = 'Stop'
$revision = '895c3a49bc3fa70a340399125c650a463535e71c'
$root = (Resolve-Path (Join-Path $PSScriptRoot '..')).Path
$manifest = Get-Content (Join-Path $root 'data/model_manifest.json') -Raw | ConvertFrom-Json
if ($manifest.sha -ne $revision) { throw 'Manifest revision mismatch' }
$folder = Join-Path $root "data/models/Qwen2-VL-2B-Instruct/$revision"
New-Item -ItemType Directory -Force -Path $folder | Out-Null
$chunkBytes = 32MB
$maxJobs = 12

foreach ($name in @('model-00001-of-00002.safetensors', 'model-00002-of-00002.safetensors')) {
  $entry = @($manifest.siblings | Where-Object { $_.rfilename -eq $name })[0]
  $expected = [long]$entry.size
  $sha = $entry.lfs.sha256
  $target = Join-Path $folder $name
  $partial = "$target.part"
  if (Test-Path $target) {
    if ((Get-Item $target).Length -ne $expected -or
        (Get-FileHash $target -Algorithm SHA256).Hash -ine $sha) {
      throw "Existing target failed verification: $target"
    }
    Write-Output "SKIP verified: $name"
    continue
  }
  $baseLength = if (Test-Path $partial) { (Get-Item $partial).Length } else { 0L }
  if ($baseLength -gt $expected) { throw "Partial too long: $partial" }
  $url = "https://huggingface.co/Qwen/Qwen2-VL-2B-Instruct/resolve/$revision/$name"
  $ranges = [System.Collections.Generic.List[object]]::new()
  for ($start = [long]$baseLength; $start -lt $expected; $start += $chunkBytes) {
    $end = [Math]::Min($start + $chunkBytes - 1, $expected - 1)
    $part = "$target.range.$start"
    $ranges.Add([pscustomobject]@{ Start=$start; End=$end; Path=$part })
  }
  $pending = [System.Collections.Queue]::new()
  foreach ($range in $ranges) {
    $wanted = $range.End - $range.Start + 1
    if (Test-Path $range.Path) {
      if ((Get-Item $range.Path).Length -eq $wanted) { continue }
      if ((Get-Item $range.Path).Length -gt $wanted) {
        throw "Range too long: $($range.Path)"
      }
    }
    $pending.Enqueue($range)
  }
  $jobs = [System.Collections.Generic.List[object]]::new()
  Write-Output "$name`: existing=$baseLength bytes, ranges=$($ranges.Count), pending=$($pending.Count)"
  while ($pending.Count -gt 0 -or $jobs.Count -gt 0) {
    while ($pending.Count -gt 0 -and $jobs.Count -lt $maxJobs) {
      $range = $pending.Dequeue()
      $job = Start-Job -ScriptBlock {
        param($u, $s, $e, $p)
        $wanted = $e - $s + 1
        $attempt = 0
        while ($true) {
          $have = if (Test-Path $p) { (Get-Item $p).Length } else { 0L }
          if ($have -eq $wanted) { break }
          if ($have -gt $wanted) { throw "Range too long: $s-$e" }
          if ($attempt -ge 20) { throw "Too many retries for range $s-$e" }
          $attempt++
          $next = "$p.next"
          if (Test-Path $next) { Remove-Item -LiteralPath $next }
          $offset = $s + $have
          & curl.exe -fLsS --connect-timeout 20 --max-time 120 `
            --range "$offset-$e" --output $next $u 2>$null
          if (Test-Path $next) {
            $source = [System.IO.File]::OpenRead($next)
            $sink = [System.IO.File]::Open($p, [System.IO.FileMode]::Append)
            try { $source.CopyTo($sink) } finally { $source.Dispose(); $sink.Dispose() }
            Remove-Item -LiteralPath $next
          }
          Start-Sleep -Seconds 2
        }
        [pscustomobject]@{ Start=$s; End=$e; Path=$p }
      } -ArgumentList $url, $range.Start, $range.End, $range.Path
      $jobs.Add($job)
    }
    $done = Wait-Job -Job $jobs.ToArray() -Any -Timeout 30
    if ($null -eq $done) { continue }
    $result = Receive-Job -Job $done
    if ($done.State -ne 'Completed') { throw "Range job failed: $($done.ChildJobs[0].JobStateInfo.Reason)" }
    if ((Get-Item $result.Path).Length -ne ($result.End - $result.Start + 1)) {
      throw "Range length mismatch: $($result.Start)-$($result.End)"
    }
    $jobs.Remove($done) | Out-Null
    Remove-Job -Job $done
    Write-Output "Downloaded range $($result.Start)-$($result.End)"
  }
  $stream = [System.IO.File]::Open($partial, [System.IO.FileMode]::Append)
  try {
    foreach ($range in $ranges) {
      $source = [System.IO.File]::OpenRead($range.Path)
      try { $source.CopyTo($stream) } finally { $source.Dispose() }
      Remove-Item -LiteralPath $range.Path
    }
  } finally { $stream.Dispose() }
  if ((Get-Item $partial).Length -ne $expected) { throw "Final size mismatch: $name" }
  if ((Get-FileHash $partial -Algorithm SHA256).Hash -ine $sha) { throw "Final SHA256 mismatch: $name" }
  Move-Item -LiteralPath $partial -Destination $target
  Write-Output "VERIFIED: $name"
}
