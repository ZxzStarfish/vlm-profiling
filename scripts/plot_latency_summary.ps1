# Run with Windows PowerShell, which provides the .NET Framework charting assembly.
$ErrorActionPreference = 'Stop'
Add-Type -AssemblyName System.Windows.Forms.DataVisualization
Add-Type -AssemblyName System.Drawing

$project = Split-Path -Parent $PSScriptRoot
$stage = @{}
Import-Csv (Join-Path $project 'reports/campaign.dGcY90WE/stages.csv') | ForEach-Object { $stage[$_.id] = $_ }
$decode = @(Import-Csv (Join-Path $project 'reports/decode_repeats/decode_statistics.csv') |
    Where-Object { $_.group -eq 'all' -and $_.measure -in @('event_request_total', 'kernel_trace_total') })

$chart = [System.Windows.Forms.DataVisualization.Charting.Chart]::new()
$chart.Width = 1440
$chart.Height = 960
$chart.BackColor = [Drawing.Color]::White
$colors = @('#1D547E', '#B14D21', '#3E765C')
$mainFont = [Drawing.Font]::new('Segoe UI', 12)
$headingFont = [Drawing.Font]::new('Segoe UI', 16, [Drawing.FontStyle]::Bold)

function Add-Panel($name, $title, $xLabel, $left, $top) {
    $area = $chart.ChartAreas.Add($name)
    $area.Position = [System.Windows.Forms.DataVisualization.Charting.ElementPosition]::new($left, ($top + 9), 48, 39)
    $area.AxisX.Title = $xLabel
    $area.AxisY.Title = 'Latency (ms)'
    $area.AxisY.Minimum = 0
    $area.AxisX.MajorGrid.Enabled = $false
    $area.AxisY.MajorGrid.LineColor = [Drawing.Color]::Gainsboro
    foreach ($axis in $area.AxisX, $area.AxisY) {
        $axis.TitleFont = $mainFont
        $axis.LabelStyle.Font = $mainFont
    }
    $heading = $chart.Titles.Add($title)
    $heading.Position = [System.Windows.Forms.DataVisualization.Charting.ElementPosition]::new($left, $top, 48, 4)
    $heading.Font = $headingFont
    $heading.Alignment = 'MiddleCenter'
    $legend = $chart.Legends.Add($name)
    $legend.DockedToChartArea = $name
    $legend.Position = [System.Windows.Forms.DataVisualization.Charting.ElementPosition]::new(($left + 4), ($top + 4), 40, 4)
    $legend.LegendStyle = 'Row'
    $legend.Alignment = 'Center'
    $legend.Font = $mainFont
    if ($name -ne 'decode') {
        $area.AxisX.Minimum = 0
        $area.AxisX.Maximum = @{image = 500; text = 600; output = 40}[$name]
        $area.AxisX.Interval = @{image = 100; text = 100; output = 8}[$name]
    }
}

function Add-Data($panel, $label, $type, $color, $xValues, $yValues) {
    $series = $chart.Series.Add("$panel/$label")
    $series.LegendText = $label
    $series.Legend = $panel
    $series.ChartArea = $panel
    $series.ChartType = $type
    $series.Color = [Drawing.ColorTranslator]::FromHtml($color)
    $series.BorderWidth = 3
    if ($type -eq 'Line') {
        $series.MarkerStyle = 'Circle'
        $series.MarkerSize = 8
    }
    for ($i = 0; $i -lt $xValues.Count; $i++) {
        if ($type -eq 'Column') {
            [void]$series.Points.AddXY([string]$xValues[$i], [double]$yValues[$i])
        } else {
            [void]$series.Points.AddXY([double]$xValues[$i], [double]$yValues[$i])
        }
    }
}

$image = @('img_224', 'img_336', 'anchor', 'img_672') | ForEach-Object { $stage[$_] }
$text = @('anchor', 'text_16', 'text_48') | ForEach-Object { $stage[$_] }
$output = @('out_4', 'anchor', 'out_32') | ForEach-Object { $stage[$_] }

Add-Panel 'image' 'A. Larger image, longer time to first token' 'Visual tokens' 1 1
Add-Data 'image' 'TTFT' 'Line' $colors[0] $image.visual_tokens $image.ttft_wall_ms
Add-Data 'image' 'ViT' 'Line' $colors[1] $image.visual_tokens $image.vit_stream_ms
Add-Data 'image' 'LLM prefill' 'Line' $colors[2] $image.visual_tokens $image.llm_backbone_prefill_stream_ms

Add-Panel 'text' 'B. Longer prompt raises LLM prefill' 'Nonvisual input tokens' 51 1
Add-Data 'text' 'LLM prefill' 'Line' $colors[2] $text.nonvisual_tokens $text.llm_backbone_prefill_stream_ms
Add-Data 'text' 'ViT' 'Line' $colors[1] $text.nonvisual_tokens $text.vit_stream_ms

Add-Panel 'output' 'C. Longer answer raises total latency' 'Output tokens' 1 51
Add-Data 'output' 'Request latency' 'Line' $colors[2] $output.output_tokens $output.request_wall_ms
Add-Data 'output' 'TTFT' 'Line' $colors[0] $output.output_tokens $output.ttft_wall_ms

Add-Panel 'decode' 'D. Decode ranking depends on timing scope' 'Module (sum over 15 decode steps)' 51 51
$labels = @('Attention', 'MLP')
$events = @('attention', 'mlp') | ForEach-Object {
    $kind = $_
    ($decode | Where-Object { $_.kind -eq $kind -and $_.measure -eq 'event_request_total' }).median_ms
}
$kernels = @('attention', 'mlp') | ForEach-Object {
    $kind = $_
    ($decode | Where-Object { $_.kind -eq $kind -and $_.measure -eq 'kernel_trace_total' }).median_ms
}
Add-Data 'decode' 'CUDA event interval' 'Column' $colors[0] $labels $events
Add-Data 'decode' 'GPU kernel time' 'Column' $colors[1] $labels $kernels

$figureDir = Join-Path $project 'reports/figures'
New-Item -ItemType Directory -Path $figureDir -Force | Out-Null
$figure = Join-Path $figureDir 'latency_summary.png'
$chart.SaveImage($figure, [System.Windows.Forms.DataVisualization.Charting.ChartImageFormat]::Png)
$chart.Dispose()
Write-Output $figure
