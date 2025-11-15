#!/bin/bash
# Run ML Inference for Scarecrow 00000000

set -e

echo "🚀 Starting ML Inference for Scarecrow 00000000..."
echo "📍 RTSP URL: rtsp://k13e106.p.ssafy.io:8554/stream/00000000"
echo ""

# Set environment file
export ENV_FILE=.env.scarecrow1

# Activate virtual environment if exists
if [ -d ".venv" ]; then
    echo "✅ Activating virtual environment..."
    source .venv/bin/activate
fi

# Run the pipeline
echo "🎬 Starting pipeline..."
python -m app.runner

# Deactivate venv
deactivate 2>/dev/null || true
