# ML Inference Stream Summary

## Overview
- **Goal**: ingest RTSP video from Raspberry Pi, run multi-model inference (YOLO + RTMPose), classify wildlife species, and dispatch alerts through FastAPI.
- **Deployment phases**
  - *Free-tier (m7i-flex.large)*: CPU-only prototype with throttled FPS and selective pose estimation.
  - *GPU cutover (g4dn.xlarge)*: CUDA-enabled pipeline with higher throughput, TensorRT optimizations, and expanded monitoring.

## Processing Pipeline
1. **Stream intake**: mediaMTX pulls RTSP feed, exposes frames to the inference worker.
2. **Detection stage (YOLO)**: detects `person` plus 11 wildlife classes in each frame.
3. **Tracking**: ByteTrack maintains `track_id` continuity with cooldown logic to avoid duplicate alerts.
4. **Pose stage (RTMPose)**: triggered only for `person` tracks; outputs keypoints and posture labels.
5. **Wildlife enrichment**: map YOLO class to `species` and `threat_level`.
6. **Event fusion**: merge detection, pose, species, tracker metadata; attach snapshot when alert-worthy.
7. **Dispatch**: send event JSON to FastAPI `/events`; downstream services fan out notifications.

## Event Schema Highlights
- **Common fields**: `event_id`, `stream_id`, `track_id`, `timestamp_utc`, `bbox`, `confidence`, `inference_latency_ms`, `gpu_enabled`.
- **Human extension**: `pose_label`, `pose_confidence`, `keypoints`, `risk_level`.
- **Wildlife extension**: `species`, `species_confidence`, `threat_level`, optional `behavior_hint`.
- **Media**: optional base64 JPEG or signed URL; include only on alert-level events.

## Resource Plan
- **Free-tier safeguards**: limit YOLO input resolution (720p), cap FPS at 10-15, debounce RTMPose, enable frame dropping when queues overflow.
- **GPU migration tasks**: install NVIDIA drivers, switch to CUDA wheels, re-enable full FPS, benchmark TensorRT, snapshot the instance for rollback.

## Documentation & Assets
- `ml_inference/docs/cctv-streaming.md`: detailed plan and checklist for infrastructure, pipeline, monitoring, and roadmaps.
- `ml_inference/docs/settings.md`: living log for concrete configuration values (stream, mediaMTX, model paths, environment variables, retention).
- `ml_inference/docs/summary.md` (this file): high-level narrative of the inference pipeline and operational assumptions.
- `ml_inference/.env`: local development defaults; copy and edit for each environment.
- `requirements.txt`: Python dependencies for the skeleton (`pip install -r requirements.txt`).
- `docs/README.md`: index pointing to the active documentation set.
- GitHub Actions: `ci.yml` (lint/test) and `deploy.yml` (SSH deploy to EC2; requires secrets `EC2_HOST`, `EC2_USER`, `EC2_SSH_KEY` and an EC2-side deploy key with repo read access).
