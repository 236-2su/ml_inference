#!/bin/bash
# Run both scarecrows concurrently

set -e

echo "🚀 Starting ML Inference for BOTH scarecrows..."
echo "📍 Scarecrow 1: rtsp://k13e106.p.ssafy.io:8554/stream/00000000"
echo "📍 Scarecrow 2: rtsp://k13e106.p.ssafy.io:8554/stream/99999999"
echo ""

# Activate virtual environment if exists
if [ -d ".venv" ]; then
    echo "✅ Activating virtual environment..."
    source .venv/bin/activate
fi

# Run both in background
echo "🎬 Starting Scarecrow 1 pipeline (background)..."
ENV_FILE=.env.scarecrow1 python -m app.runner > logs/scarecrow1.log 2>&1 &
PID1=$!

echo "🎬 Starting Scarecrow 2 pipeline (background)..."
ENV_FILE=.env.scarecrow2 python -m app.runner > logs/scarecrow2.log 2>&1 &
PID2=$!

echo ""
echo "✅ Both pipelines started!"
echo "   Scarecrow 1 PID: $PID1 (log: logs/scarecrow1.log)"
echo "   Scarecrow 2 PID: $PID2 (log: logs/scarecrow2.log)"
echo ""
echo "📝 To stop:"
echo "   kill $PID1 $PID2"
echo ""
echo "📊 To monitor logs:"
echo "   tail -f logs/scarecrow1.log"
echo "   tail -f logs/scarecrow2.log"
echo ""

# Wait for both processes
wait $PID1 $PID2
