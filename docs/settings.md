# Project Settings Log

> Use this file to capture concrete configuration values as the project evolves. Update the entries whenever infrastructure, stream quality, or integration parameters change.
> Local development defaults live in `../.env`; copy to your own `.env` when overriding.

## 1. Stream Source (Raspberry Pi)
- **Camera model**: 
- **Lens / FOV notes**: 
- **Encoder**: (`h264_omx`, `libcamera-vid`, etc.)
- **Resolution**: 
- **Frame rate**: 
- **Bitrate target**: 
- **Key frame interval (GOP)**: 
- **RTSP URL**: 
- **Service watchdog**: (systemd service name, restart policy)
- **Network**: (Wi-Fi SSID or Ethernet path, static IP notes)

## 2. mediaMTX Configuration
- **Pull path name**: 
- **Auth credentials**: (user/pass or token)
- **Reconnect settings**: (`readBufferCount`, `readBufferSize`, backoff strategy)
- **Output profiles enabled**: (RTSP/HLS/WebRTC)
- **Health check endpoint**: 
- **Log level**: 

## 3. Inference Service
- **Model file (.pt) location**: (S3 URI or filesystem path)
- **Model version / hash**: 
- **Framework**: (`ultralytics/yolov8`, `yolov5`, custom)
- **Confidence threshold**: 
- **NMS / tracker settings**: (DeepSORT/ByteTrack params)
- **Batch size / concurrency**: (CPU vs GPU)
- **GPU flags**: (`CUDA_VISIBLE_DEVICES`, `--half`, TensorRT profile)

## 4. FastAPI Integration
- **Endpoint base URL**: 
- **Auth token / header**: 
- **Timeout / retry policy**: 
- **Event payload schema link**: 
- **Health check URL**: 

## 5. Environment Variables
| Key | Purpose | Current value / location |
| --- | --- | --- |
| `MEDIA_RPI_RTSP_URL` | mediaMTX pull source | `rtsp://example.local:8554/stream` (see `.env`) |
| `YOLO_MODEL_PATH` | Path to deployed weights | `models/best.pt` |
| `YOLO_HUMAN_MODEL_PATH` | Path to human-only weights | `models/yolov8x.pt` |
| `YOLO_HUMAN_CONF_THRESHOLD` | Confidence for human detector | `0.35` |
| `YOLO_HUMAN_IOU_THRESHOLD` | IoU for human detector | `0.45` |
| `HUMAN_SKIP_CONF_THRESHOLD` | Confidence above which wildlife model is skipped | `0.7` |
| `YOLO_POSE_MODEL_PATH` | Pose model weights path | `yolov8x-pose.pt` |
| `YOLO_POSE_CONF_THRESHOLD` | Confidence for pose keypoints | `0.35` |
| `POSE_KEYPOINT_CONF_THRESHOLD` | Minimum keypoint score kept for pose heuristics | `0.002` |
| `POSE_LYING_ASPECT_RATIO` | Height/width ratio below which a person is considered lying | `0.65` |
| `POSE_LYING_TORSO_ANGLE_DEG` | Torso angle (deg) below which a person is considered lying | `35` |
| `FASTAPI_ENDPOINT` | REST target for events | `http://localhost:8000/events` |
| `FASTAPI_TOKEN` | Auth secret (stored in SSM/Secrets Manager) | `local-dev-token` (replace in production) |
| `MEDIA_OUTPUT_ROOT` | Local or S3 storage base path | `./artifacts` |
| `GPU_ENABLED` | Feature flag for CUDA mode (`true`/`false`) | `false` (set `true` after GPU cutover) |
| `STREAM_DEFAULT_FPS` | Frame-processing cap | `12` |

## 6. Capture & Retention Policy
- **Snapshot target**: (local path, S3 bucket, prefix)
- **Retention duration**: 
- **Lifecycle policy ID**: 
- **Clip length (when exporting video)**: 
- **Compression settings**: 
- **Manual override procedure**: (how to archive critical events)

## 7. Monitoring & Alerting Thresholds
- **Stream FPS minimum**: 
- **Inference latency (p95)**: 
- **GPU utilization range**: 
- **Disk usage warning level**: 
- **Alert destinations**: (Slack channel, webhook URL, email list)

## 8. Deployment Notes
- **Current EC2 instance ID / type**: 
- **AMI snapshot date**: 
- **Docker image tag in production**: 
- **CI/CD pipeline run URL**: 
- **Pending migration tasks**: 

## 9. Change Log
| Date | Change | Owner | Notes |
| --- | --- | --- | --- |
|  |  |  |  |
|  |  |  |  |

> Tip: keep sensitive values in Parameter Store or Secrets Manager; record only their locations here.
