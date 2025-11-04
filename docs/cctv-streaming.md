# CCTV Streaming & GPU Inference Plan

## 0. Goals and Scope
- Collect the vegetable garden RTSP stream from the Raspberry Pi via mediaMTX and run real-time inference on an AWS GPU server.
- Detect wildlife intrusions and human heatstroke or fall events, then push alerts, metadata, and captured frames to a FastAPI backend.
- Start on an AWS EC2 free-tier instance, but prepare a seamless upgrade path to a g4dn GPU instance once the model (.pt) is ready.

### Phases at a Glance
- **Free-tier (CPU only)**: Stand up core plumbing (mediaMTX pull, CPU-based inference prototype, FastAPI integration, CI/CD, baseline monitoring) without GPU dependencies.
- **GPU Cutover (g4dn)**: Migrate the validated stack onto a GPU instance, enable CUDA acceleration, harden performance monitoring, and finalize alerting.

### Raspberry Pi Stream Settings
- [ ] **Free-tier** Confirm camera resolution, frame rate, and bitrate (for example 1280x720 at 15 FPS, H.264 baseline profile) to balance inference load and network usage.
- [ ] **Free-tier** Configure Raspberry Pi OS to keep the RTSP process alive (systemd service or supervisord) and enable hardware encoding.
- [ ] **Free-tier** Document network path (SSID or Ethernet, router firewall rules, static IP vs DHCP reservation).
- [ ] **GPU cutover** Stress test the stream with multiple hours of continuous feed to check reconnection, latency, and frame drops.
- [ ] **GPU cutover** Capture diagnostic metrics (packet loss, jitter) and feed them into monitoring for troubleshooting.

## 1. High-Level Architecture
- **Input**: Raspberry Pi camera -> RTSP stream.
- **Media layer**: mediaMTX running on the GPU server pulls the RTSP stream; optional HLS/RTMP/WebRTC outputs for monitoring.
- **Inference pipeline**: mediaMTX -> frame decode -> YOLO detection plus object tracking -> event classification.
- **Output**: REST calls to FastAPI (`/events`, `/health`, `/upload`) carrying metadata and captured frames; downstream services (for example Spring Boot) consume those events.
- **Storage and monitoring**: local disk or S3 for logs or snapshots; CloudWatch or Prometheus/Grafana for metrics.

## 2. AWS Infrastructure Preparation
- [ ] **Free-tier** Launch an Ubuntu 22.04 instance (t3.medium or t3.xlarge) with a security group that only opens `TCP 22`, `TCP 80/443`, `TCP 8888` (temporary testing), and `TCP/UDP 554` (RTSP).
- [ ] **Free-tier** Prepare cloud-init, Ansible, or shell scripts to install base packages (`build-essential`, `python3`, `docker`, and so on). Skip NVIDIA packages for now.
- [ ] **Free-tier** Allocate an Elastic IP and decide on domain and TLS rollout (start HTTP; plan for ACM plus ALB or Nginx plus Let's Encrypt).
- [ ] **GPU cutover** Define the upgrade playbook: create an AMI or snapshot, launch a `g4dn.xlarge`, attach the Elastic IP, verify CUDA driver and model performance.

## 3. OS and Runtime Setup
- [ ] **Free-tier** Update Ubuntu, harden SSH, and create deployment users.
- [ ] **Free-tier** Prepare a Python virtual environment (or Conda) with CPU-compatible `torch`, `ultralytics`, `opencv-python`, and supporting libraries.
- [ ] **Free-tier** Target Docker-based deployment: draft `Dockerfile`, `docker-compose.yml`, and optional Kubernetes manifests if scaling is needed later.
- [ ] **Free-tier** Copy `ml_inference/.env` to `.env` per environment and override credentials using AWS SSM/Secrets Manager for shared deployments.
- [ ] **Free-tier** Decide where the trained `.pt` weights will live (S3, EFS, model registry) and script a sync step during deployment.
- [ ] **GPU cutover** Execute the documented CUDA, cuDNN, and NVIDIA Container Toolkit installation steps on the g4dn instance.
- [ ] **GPU cutover** Switch Python packages or Docker image layers to GPU builds (`torch` with CUDA, TensorRT plugins if used).

## 4. mediaMTX Configuration
- [ ] **Free-tier** Configure mediaMTX to pull the Raspberry Pi RTSP URL; enable authentication and reconnection or backoff options.
- [ ] **Free-tier** Decide whether to expose HLS or WebRTC outputs for operators or keep the stream internal to the inference service.
- [ ] **Free-tier** Define how frames move from mediaMTX to the inference module (FFmpeg, GStreamer, or a Go plugin).
- [ ] **Free-tier** Set reconnect delay, read/write timeouts, and jitter buffer parameters to handle Raspberry Pi Wi-Fi drops.
- [ ] **GPU cutover** Load-test concurrent streams and tune buffering or transcoding parameters with GPU acceleration.
- [ ] **GPU cutover** Add health checks (mediaMTX `/v2/paths/list`) and hook alerts into Slack, webhook, or email targets.

## 5. Model and Inference Pipeline
  - **Model choice**: prefer PyTorch YOLOv8 (Ultralytics) or YOLOv5 for strong documentation and community support. Convert to TensorRT or ONNX later if performance tuning is required.
  - **Multi-model pipeline**
    - **Stage 1 (Detection)**: YOLO identifies humans and wildlife species in each frame.
    - **Stage 2a (Pose)**: for `person` detections, crop and run RTMPose to classify posture (standing, crouching, prone, collapsed) with keypoints.
    - **Stage 2b (Wildlife)**: propagate YOLO wildlife classes (11-species model) and map to threat levels.
    - **Fusion**: merge detection, pose, and species metadata into a unified event keyed by tracker ID.
  - **Pipeline steps**
    - [ ] **Free-tier** Pull frames at 10 to 15 FPS and validate the CPU inference loop (lower FPS acceptable during prototyping).
    - [ ] **Free-tier** Integrate an object tracker (ByteTrack recommended) to prove event deduplication logic even without GPU speedups.
    - [ ] **Free-tier** On new or risky detections, capture the frame, package metadata (class, score, bounding box, timestamp), and POST to FastAPI.
    - [ ] **GPU cutover** Reconfigure batching and tracker parameters for higher FPS, and enable mixed-precision or TensorRT optimizations.
    - [ ] **GPU cutover** Measure inference latency, throughput (FPS), and GPU memory; wire metrics into the monitoring stack.
  - **Future enhancements**: TensorRT optimization, multi-stream handling, and a rule engine for complex scenarios.

### Object Tracking Strategy
  - [ ] **Free-tier** Use ByteTrack with IOU 0.45 and detection score 0.3; maintain a store of active `track_id` entries to throttle repeat alerts.
  - [ ] **Free-tier** Define a cooldown window (for example 10 seconds) before treating a returning `track_id` as a new event.
  - [ ] **Free-tier** Persist tracker summaries (enter, exit, duration) for analytics and debugging.
  - [ ] **GPU cutover** Enable appearance embeddings (DeepSORT) if ID switches remain high under higher FPS.
  - [ ] **GPU cutover** Export tracker metrics (active tracks, reassignments) to monitoring for drift detection.

## 6. FastAPI Integration
  - [ ] **Free-tier** Lock in API contracts with the upstream team: endpoints, payload schema, authentication (token-based or internal-only for MVP).
  - [ ] **Free-tier** Define the event JSON schema (object type, probability, bounding boxes, stream ID, capture URL).
  - [ ] **Free-tier** Implement retries or backoff and idempotency keys so duplicate events do not flood the backend.
  - [ ] **Free-tier** Plan an end-to-end integration test: Raspberry Pi -> mediaMTX -> inference -> FastAPI -> downstream service.
  - [ ] **GPU cutover** Perform load and resiliency tests, including burst event scenarios, and validate downstream consumption under higher FPS.

### Event Delivery Flow (FastAPI -> Downstream)
  - [ ] **Free-tier** Draw a sequence diagram or table summarizing hand-offs: inference service -> FastAPI -> message queue or Spring Boot -> notification channel.
  - [ ] **Free-tier** Define acknowledgement contracts (HTTP 200, retry policy) and logging for failed deliveries.
  - [ ] **GPU cutover** Validate latency budgets end-to-end (capture time to final alert) and tune thresholds.

### Event Schema (Recommended)
  - **Common fields**: `event_id` (UUID), `stream_id`, `track_id`, `timestamp_utc`, `frame_index`, `bbox` (x, y, w, h), `confidence`, `inference_latency_ms`, `gpu_enabled`.
  - **Human payload**: `category: "human"`, `pose_label`, `pose_confidence`, `keypoints` (list of x, y, score), `risk_level` (info/warn/critical).
  - **Wildlife payload**: `category: "wildlife"`, `species`, `species_confidence`, `threat_level`, `behavior_hint` (feeding/moving/running if inferred).
  - **Media**: optional `image_jpeg_base64` or signed URL when the event should deliver a snapshot.

## 7. CI/CD and Collaboration
- [ ] **Free-tier** Set up GitHub Actions (or GitLab CI) to lint or test the Python codebase, build Docker images, and push to ECR.
- [ ] **Free-tier** Automate EC2 deployment via SSM Session Manager, CodeDeploy, or a simple SSH-based script triggered on main branch merges.
- [ ] **Free-tier** Keep environment variables and secrets (`MEDIA_RPI_RTSP_URL`, `FASTAPI_ENDPOINT`, `YOLO_MODEL_PATH`, and so on) in SSM Parameter Store or Secrets Manager.
- [ ] **Free-tier** Publish dependency manifest (`requirements.txt`) and ensure CI installs via `pip install -r requirements.txt`.
- [ ] **GPU cutover** Extend pipelines to build GPU-enabled images, run smoke tests with CUDA, and promote artifacts per environment.
- [ ] **GPU cutover** Track model versions with Weights and Biases, DagsHub, or an S3 versioned bucket; capture deployment metadata for rollbacks.

### Storage and Retention
- [ ] **Free-tier** Decide on storage targets for captured frames (local disk path vs S3 bucket) and retention duration.
- [ ] **Free-tier** Implement lifecycle policies (for example S3 expiration after 30 days) and folder naming (stream/date/event-id).
- [ ] **GPU cutover** Benchmark upload times for burst events and ensure bandwidth does not impact inference.
- [ ] **GPU cutover** Add housekeeping jobs or scripts to purge local caches and rotate logs.

## 8. Monitoring and Logging
- [ ] **Free-tier** Monitor CPU utilization, memory, disk, network (CloudWatch Agent or Prometheus Node Exporter).
- [ ] **Free-tier** Track application metrics: stream FPS, inference latency, event counts, tracker health (even if CPU-bound).
- [ ] **Free-tier** Centralize mediaMTX, inference worker, and FastAPI logs in CloudWatch Logs or Loki.
- [ ] **GPU cutover** Add GPU metrics (temperature, utilization, memory) and TensorRT stats; set alert thresholds.
- [ ] **GPU cutover** Wire alerts for stream offline, inference errors, GPU overheating, API failures (Slack, webhook, email).

## 9. Security and Access (MVP Level)
- [ ] **Free-tier** Restrict SSH by IP, limit mediaMTX or FastAPI access to the VPC or VPN, and plan HTTPS enablement before production.
- [ ] **Free-tier** Use an instance IAM role with least privilege for S3, CloudWatch, and Parameter Store access.
- [ ] **Free-tier** Rotate API tokens and keep them outside the repository; enforce OS security updates.
- [ ] **GPU cutover** Re-audit security groups and IAM policies after migration; ensure secrets and certificates are available on the new instance.

## 10. Resource Planning
- **Current target instance**: `m7i-flex.large` (2 vCPU, 8 GiB RAM) for CPU-only prototyping.
- **Free-tier practices**
  - Run YOLO with reduced input size (for example 896x504) and limit processing to 10-15 FPS.
  - Gate RTMPose execution so it triggers only on new `person` detections and at a lower cadence to avoid CPU spikes.
  - Use asynchronous queues; drop excess frames when the processing backlog exceeds N frames.
  - Disable GPU-only dependencies; stick to `torch==CPU build` and avoid installing CUDA packages.
- **GPU cutover plan**
  - Target `g4dn.xlarge`; bake AMI after validating NVIDIA drivers, CUDA, and nvidia-container-toolkit.
  - Switch to CUDA-enabled `torch` wheel and enable TensorRT optimizations if stable.
  - Revisit FPS/resolution caps and re-enable full RTMPose frequency.
  - Document fallback path (revert to AMI snapshot) if GPU tests fail.

## 11. Testing Strategy
- [ ] **Free-tier** Create RTSP simulators (ffmpeg loop, rtsp-simple-server) for CI or local testing when Raspberry Pi is offline.
- [ ] **Free-tier** Add unit and integration tests for frame preprocessing, inference decisions, and FastAPI event handling.
- [ ] **Free-tier** Document manual test cases (Wi-Fi drop, low light, false positives) and expected responses.
- [ ] **GPU cutover** Run load tests with recorded streams to measure FPS under GPU acceleration, including worst-case scenarios.
- [ ] **GPU cutover** Schedule periodic failover drills (mediaMTX restart, FastAPI downtime) and capture recovery metrics.

## 12. Suggested Timeline (1.5 Weeks)
| Days | Focus |
| --- | --- |
| 1 to 2 | Provision free-tier EC2, baseline Ubuntu setup, install mediaMTX, verify stream pull. |
| 3 to 5 | Build inference prototype (stream -> YOLO -> event), finalize FastAPI API contract. |
| 6 to 7 | Implement CI pipeline, baseline monitoring or log shipping, document setup steps. |
| 8 to 9 | Rehearse GPU migration, profile performance, integrate tracker, refine event logic. |
| 10 to 11 | End-to-end test with Raspberry Pi -> FastAPI, validate alert scenarios, backlog cleanup. |

> Next steps after MVP: automate `.pt` weight syncing from the training pipeline, design long-term storage for events (database or data lake), and introduce MLOps tooling (MLflow, Airflow) once the workflow stabilizes.
