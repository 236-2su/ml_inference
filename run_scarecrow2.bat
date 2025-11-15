@echo off
REM Run ML Inference for Scarecrow 99999999

echo Starting ML Inference for Scarecrow 99999999...
echo RTSP URL: rtsp://k13e106.p.ssafy.io:8554/stream/99999999
echo.

REM Set environment file
set ENV_FILE=.env.scarecrow2

REM Activate virtual environment if exists
if exist .venv\Scripts\activate.bat (
    echo Activating virtual environment...
    call .venv\Scripts\activate.bat
)

REM Run the pipeline
echo Starting pipeline...
python -m app.runner

pause
