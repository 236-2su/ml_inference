# Event Delivery & API Plan

## 1. Scope
- ml_inference pipeline events go to the main server `/events` endpoint.
- Wildlife (yolo12s.pt) is phase 1, RTMPose human events follow.
- Document schema/auth/retry ahead of main-server work.

## 2. Common Payload Fields
| Field | Type | Description |
| ---- | ---- | ---- |
| `event_id` | UUID | Idempotency key (reuse on retries) |
| `category` | `wildlife`\|`human` | Event type |
| `track_id` | int | Tracker-issued ID |
| `timestamp_utc` | ISO8601 str | Frame capture time (UTC) |
| `bbox` | `[x, y, w, h]` | Bounding box in pixels |
| `confidence` | float | YOLO detection confidence |
| `inference_latency_ms` | int | Frame ingest→event build latency |

> `stream_id`와 `gpu_enabled`는 로컬 설정/로그에서만 관리하며 본 서버로 전송하지 않는다.

## 3. Wildlife Event Schema
```
{
  "event_id": "uuid",
  "category": "wildlife",
  "track_id": 42,
  "timestamp_utc": "2025-11-07T02:10:00Z",
  "bbox": [120, 200, 80, 60],
  "confidence": 0.86,
  "inference_latency_ms": 145,
  "species": "boar",
  "species_confidence": 0.82
}
```
- `species`: One of the 11 labels baked into `yolo12s.pt` (document the map).
- `species_confidence`: Class probability from YOLO. Threat level can be added later.

## 4. Human Event Schema
```
{
  "event_id": "uuid",
  "category": "human",
  "track_id": 7,
  "timestamp_utc": "2025-11-07T02:15:30Z",
  "bbox": [90, 150, 70, 150],
  "confidence": 0.74,
  "inference_latency_ms": 190,
  "pose_label": "lying",
  "pose_confidence": 0.81,
  "status": "fall_detected",
  "duration_seconds": 18,
  "image_jpeg_base64": "..."
}
```
- `pose_label`: `standing`, `lying`, `unknown`.
- `status`: State-machine result (`fall_detected`, `heatstroke_watch`, ...).
- `duration_seconds`: Time spent in `lying` status.
- `image_jpeg_base64`: Optional snapshot (compress to ~100KB, respect privacy rules).

## 5. Authentication (Scarecrow Tokens)
- Use `Authorization: Bearer <SCARECROW_TOKEN>`.
- Each scarecrow device gets its own token; send `X-Scarecrow-Id` for traceability.
- Store tokens in AWS SSM/Secrets Manager and document rotation procedures.

## 6. Dispatch & Retry Flow
1. `EventDispatcher` POSTs to `/events` and expects HTTP 202.
2. On 5xx/network errors retry with exponential backoff: 1s → 5s → 15s → 60s → 120s (max 5 attempts).
3. On 401/403 stop immediately and alert operators; 422 indicates payload bug.
4. Persist pending events (file/Redis, later SQS/Kafka) so restarts do not lose data; rely on `event_id` for idempotency.

## 7. Implementation Checklist
1. Extend `Settings` with main-server URL/port (default `http://<ip>:8080/events`), scarecrow token, YOLO/pose thresholds, plus local-only `stream_id`/`gpu_enabled` flags (config only).
2. Replace `Detector` with real `yolo12s.pt` inference + label→threat map.
3. Update `EventBuilder` to emit this schema and compute `inference_latency_ms`.
4. Implement RTMPose + pose classifier + state machine for `pose_label`, `status`, `duration_seconds`.
5. Add snapshot capture/encoding logic guarded by config flags.
6. Enhance `EventDispatcher` with auth header, timeout (5s), retry, structured logging.
7. Build mock RTSP + mock `/events` integration tests covering retries/idempotency.
8. Expose metrics (health, success rate, retry queue length, latency) to dashboards/alerts.

## 8. Open Items
- Document the 11 wildlife classes and threat rules.
- Finalize scarecrow token provisioning and storage.
- Decide snapshot retention/deletion policy and privacy guardrails.
- Revisit `inference_latency_ms` SLA after GPU cutover.
