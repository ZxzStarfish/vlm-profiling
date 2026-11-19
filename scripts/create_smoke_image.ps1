$ErrorActionPreference = 'Stop'
Add-Type -AssemblyName System.Drawing
$root = (Resolve-Path (Join-Path $PSScriptRoot '..')).Path
$target = Join-Path $root 'data/smoke.png'
$bitmap = [System.Drawing.Bitmap]::new(224, 224)
$graphics = [System.Drawing.Graphics]::FromImage($bitmap)
try {
  $graphics.Clear([System.Drawing.Color]::White)
  $red = [System.Drawing.SolidBrush]::new([System.Drawing.Color]::Red)
  $blue = [System.Drawing.SolidBrush]::new([System.Drawing.Color]::Blue)
  try {
    $graphics.FillRectangle($red, 20, 70, 80, 80)
    $graphics.FillEllipse($blue, 124, 70, 80, 80)
  } finally {
    $red.Dispose()
    $blue.Dispose()
  }
  $bitmap.Save($target, [System.Drawing.Imaging.ImageFormat]::Png)
} finally {
  $graphics.Dispose()
  $bitmap.Dispose()
}
Write-Output $target
