[CmdletBinding()]
param(
    [string]$InputPath = "wildboar.jpg",
    [string]$RtspUrl = "rtsp://heobyPublisher:S3curePub!230@k13e106.p.ssafy.io:8554/cctv",
    [int]$Framerate = 5,
    [string]$FfmpegPath = "ffmpeg"
)

if (-not (Get-Command $FfmpegPath -ErrorAction SilentlyContinue)) {
    throw "ffmpeg executable not found. Install FFmpeg and ensure it is on PATH or pass --FfmpegPath"
}
if (-not (Test-Path $InputPath)) {
    throw "Input file '$InputPath' not found"
}

$ext = [System.IO.Path]::GetExtension($InputPath).ToLowerInvariant()
$imageExts = @('.jpg', '.jpeg', '.png', '.bmp', '.gif')

$inputArgs = @('-re')
if ($imageExts -contains $ext) {
    $inputArgs += @('-loop', '1', '-framerate', $Framerate.ToString(), '-i', $InputPath, '-vf', 'scale=1280:720,format=yuv420p')
} else {
    $inputArgs += @('-stream_loop', '-1', '-i', $InputPath)
}

$encodeArgs = @(
    '-c:v', 'libx264',
    '-preset', 'veryfast',
    '-tune', 'zerolatency',
    '-f', 'rtsp',
    '-rtsp_transport', 'tcp',
    $RtspUrl
)

$allArgs = $inputArgs + $encodeArgs
Write-Host "Running: $FfmpegPath $($allArgs -join ' ')"
& $FfmpegPath @allArgs
