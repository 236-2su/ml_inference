# Person Detection & Posture Plan (Pretrained YOLO + RTMPose)

## 0. 목표
- **추락/열사병 감지** MVP를 위해 YOLO 기본 가중치로 사람만 우선 감지.
- `person`이 감지되면 RTMPose로 자세를 추정하고, 장기 누워 있음(15분 이상)을 이벤트로 본 서버에 전달.
- 야생동물 모델 학습이 끝나기 전까지, 사람 감지를 빠르게 검증할 수 있는 구조 마련.

## 1. 전체 흐름
1. **스트림 수집**: mediaMTX가 RTSP 스트림을 pull → inference worker로 프레임 전달.
2. **사람 감지 (YOLO)**: MS COCO 등 pretrained YOLO 가중치(`yolov8n.pt` 등)에서 `person` 클래스만 필터.
3. **추적**: ByteTrack/DeepSORT로 `person` 객체 track_id 유지.
4. **포즈 추정 (RTMPose)**: track된 `person` bbox를 crop → RTMPose → posture label & confidence.
5. **상태 머신**:
   - posture가 `standing/unknown → lying`으로 변할 때 낙상 이벤트 트리거.
   - `lying` 상태가 15분 이상 지속되면 열사병 의심 이벤트.
6. **알림 전송**: FastAPI `/events`로 POST → 본 서버 연동.

## 2. YOLO 구성 (사람만 사용)
- **모델**: ultralytics YOLOv8 (pretrained, 우선 최고 성능 `yolov8x.pt` 기준으로 시작).
- **필터링**: `detection.label == 'person'`만 retain.
- **출력**: `bbox`, `confidence`, `frame_id`, `track_id`.
- **환경변수**:
  - `YOLO_MODEL_PATH` → `models/yolov8x.pt`
  - `YOLO_PERSON_CONF` → 0.25 (최소 confidence), `.env`에 추가.
- **파이프라인 반영**:
  - `app/detector.py`: ultralytics 모델 로딩, `person` 클래스만 반환.
  - `app/tracker.py`: ByteTrack/DeepSORT 연동 (추후 GPU 전환 시 동일).

## 3. RTMPose & 자세 분류
- **모델**: RTMPose(mmpose 기반) 또는 open-source pose estimator (lite 버전).
- **입력**: YOLO bbox를 기준으로 crop → 256x192/384x288 등 권장 해상도.
- **출력**: keypoints + skeleton confidence.
- **참고**: OpenMMLab mmpose(https://github.com/open-mmlab/mmpose)에서 RTMPose 모델 및 config, demo 스크립트 확인 가능.
- **자세 라벨링 규칙** (초안):
  - `standing`: 어깨-엉덩이 y좌표 차이 > threshold, torso angle < 30도.
  - `crouching`: 무릎-엉덩이 거리 감소, torso angle 30~60도.
  - `lying`: torso angle > 60도 또는 head-hip y 좌표 차이가 작음.
  - `unknown`: keypoint confidence 낮거나 프레임 드롭.
- **안정화**:
  - 최근 N frame에 대한 majority voting 또는 EMA로 smoothing.
  - keypoint confidence < 0.4일 경우 posture 유지/unknown 처리.

## 4. 상태 머신 & 타이머
- track_id별 상태 저장 (예: Redis, in-memory dict).
- **낙상 로직**:
  - 상태 전이 `standing/unknown → lying`이 빠르게 일어나면 즉시 알림.
  - 이벤트: `type=fallback_fall`, `pose_transition`, `timestamp`.
- **열사병 로직**:
  - `lying` 상태 유지 시간 측정 → 15분 지속 시 `type=heatstroke`.
  - 중간에 posture가 `standing`으로 바뀌면 타이머 리셋.
- **쿨다운**:
  - 동일 track_id에 대해 동일 이벤트는 30분(예시) 동안 1회만 전송.
- **프레임 유실 고려**:
  - track lost → 일정 시간(예: 10초) 이후 상태 초기화.

## 5. FastAPI 이벤트 스키마 (MVP)
```json
{
  "event_id": "uuid",
  "category": "human",
  "track_id": 12,
  "pose_label": "lying",
  "pose_confidence": 0.85,
  "status": "heatstroke",        // or "fall_detected"
  "duration_minutes": 16,        // heatstroke only
  "confidence": 0.87,            // YOLO detection confidence
  "bbox": [x, y, w, h],
  "timestamp_utc": "2025-11-04T05:40:00Z",
  "image_jpeg_base64": "...",    // optional snapshot
  "gpu_enabled": false
}
```

## 6. 테스트 계획
- **Mock 스트림**: ffmpeg loop 영상으로 재생 → YOLO가 person을 감지하는지 확인.
- **문턱값 튜닝**: posture thresholds, lying 지속 시간을 config로 노출.
- **유닛 테스트**:
  - posture classifier → keypoint 값 넣어 expected label 확인.
  - 상태 머신 → lying 시간 누적이 제대로 작동하는지 시뮬레이션.
- **통합 테스트**:
  - `/events`로 FastAPI에 보내고 응답/재시도 확인.
  - 서비스 로그(journalctl -u ml-inference)로 이벤트 발생 여부 체크.

## 7. 후속 과제
- 야생동물 모델(.pt) 접목: YOLO 멀티 클래스 확장 후 threat level 로직 추가.
- Google Vision API 통합: 사람 검증 보조판단 + 얼굴/피사체 자세 인식.
- 데이터 수집: 실제 사람이 넘어지는/누워있는 영상 확보 → threshold 검증 및 RTMPose 튜닝.
- GPU 전환: `torch` CUDA 빌드 설치, RTMPose 최적화(TensorRT 고려).

> 이 계획대로라면 사람 감지 및 자세 판단을 빠르게 MVP 수준으로 구현해 false positive/negative를 줄이고, 추후 야생동물 감지 모델과 통합하기 쉽게 확장할 수 있습니다.
