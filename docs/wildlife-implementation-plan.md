# Wildlife Pipeline Implementation (Without RTMPose State Machine)

## Checkpoint 1 – Settings & Schema Prep
1. Config knobs
   - .env: add FASTAPI_ENDPOINT, FASTAPI_TOKEN, YOLO_CONF_THRESHOLD, YOLO_IOU_THRESHOLD, INCLUDE_SNAPSHOT.
   - app/config.py: mirror the keys with sane defaults plus validation.
2. Schema alignment
   - Refactor app/event_builder.py to match docs/event-plan.md (no stream_id or gpu_enabled on the wire).
   - Add helper to compute inference_latency_ms via monotonic clock minus frame timestamp.
3. Docs
   - Note in event-plan.md and person-detection-plan.md that wildlife MVP ships first, RTMPose/state machine deferred.

Exit criteria
- Pipeline loads new settings without crashing.
- Updated unit test confirms wildlife event shape only.

## Checkpoint 2 – EventBuilder & Dispatcher Implementation
1. EventBuilder
   - Populate common fields plus species and species_confidence.
   - Accept local EventContext (keeps stream_id and gpu_enabled for logging only).
   - Attach latency metric and optional snapshot placeholder when INCLUDE_SNAPSHOT=true.
2. EventDispatcher
   - Send Authorization: Bearer <FASTAPI_TOKEN>.
   - Apply timeout 5s and exponential backoff (1s -> 5s -> 15s -> 60s -> 120s, max 5 attempts).
   - Log failures with event_id/status; drop 4xx, retry 5xx/network.
3. Retry scaffold
   - Buffer failed events in memory for now and document future Redis/SQS upgrade.

Exit criteria
- Manual POST to real /events succeeds; token errors logged clearly.
- Logs show latency and retry info per batch.

## Checkpoint 3 – YOLO (yolo12s) Integration
1. Model loading
   - Use ultralytics YOLO with YOLO_MODEL_PATH (default yolo12s.pt).
   - Define 11-class label map plus threat levels in the detector module.
2. Inference loop
   - Convert RTSP frame bytes to tensors, run inference, filter by confidence/IoU thresholds.
   - Return Detection objects with label, confidence, bbox.
3. Tracking sanity
   - Keep sequential tracker but ensure unique IDs per detection.
   - Emit debug logs with detection counts for monitoring.

Exit criteria
- Live RTSP stream yields actual wildlife detections (confirmed via logs).
- Event pipeline sends detections end-to-end to the main server.

## Checkpoint 4 – Verification & Documentation
1. Integration smoke
   - Run InferencePipeline.run_once(limit=10) and confirm main server /events returns 202.
   - Inspect main-server logs for species/species_confidence fields.
2. Docs/tests
   - Update docs to state "wildlife MVP ready, RTMPose pending".
   - Add pytest covering EventBuilder output plus Dispatcher retry logic (mock requests.post).
3. Ops notes
   - Document systemd commands, retry queue inspection, health-check steps.

Exit criteria
- CI passes with new tests.
- README/runbook describe current functionality and remaining RTMPose work.

---
Next steps after these checkpoints: integrate RTMPose estimator, implement posture state machine, replace in-memory retry buffer with durable storage.
