#!/usr/bin/env bash
set -euo pipefail

INPUT_PATH="${1:-wildboar.jpg}"
RTSP_URL="${2:-rtsp://heobyPublisher:S3curePub!230@k13e106.p.ssafy.io:8554/cctv}"
FRAMERATE="${FRAMERATE:-5}"
FFMPEG_BIN="${FFMPEG_BIN:-ffmpeg}"

if ! command -v "$FFMPEG_BIN" >/dev/null 2>&1; then
  echo "[publish_rtsp] ffmpeg not found (set FFMPEG_BIN)" >&2
  exit 1
fi
if [ ! -f "$INPUT_PATH" ]; then
  echo "[publish_rtsp] input file '$INPUT_PATH' not found" >&2
  exit 1
fi

EXT_LOWER=$(echo "${INPUT_PATH##*.}" | tr '[:upper:]' '[:lower:]')
if [[ "$EXT_LOWER" =~ ^(jpg|jpeg|png|bmp|gif)$ ]]; then
  INPUT_ARGS=( -re -loop 1 -framerate "$FRAMERATE" -i "$INPUT_PATH" -vf scale=1280:720,format=yuv420p )
else
  INPUT_ARGS=( -re -stream_loop -1 -i "$INPUT_PATH" )
fi

CMD=( "$FFMPEG_BIN" "${INPUT_ARGS[@]}" -c:v libx264 -preset veryfast -tune zerolatency -f rtsp -rtsp_transport tcp "$RTSP_URL" )
echo "[publish_rtsp] Running: ${CMD[*]}"
"${CMD[@]}"
