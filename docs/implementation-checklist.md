# 야생동물 감지 파이프라인 구현 체크리스트

> **목표:** RTSP 스트림으로 12종 야생동물을 실시간 감지하고 FastAPI로 이벤트 전송

## 📋 프로젝트 개요

- **모델:** best.pt (12종 야생동물)
- **클래스:** 족제비, 노루, 청설모, 중대백로, 다람쥐, 멧돼지, 멧돼지열화상, 너구리, 멧토끼, 왜가리, 고라니, 반달가슴곰
- **RTSP 소스:** k13e106.p.ssafy.io:8554/cctv
- **FastAPI Endpoint:** https://k13e106.p.io/dev/api/events
- **개발 환경:** Windows 로컬 → EC2 서버 배포

---

## ✅ Phase 1: 환경 설정 및 모델 확인

### 1.1 개발 환경 구축
- [x] Python 3.10 설치 확인
- [x] requirements.txt 패키지 설치 (torch, ultralytics, opencv 등)
- [x] Visual C++ Redistributable 설치 (Windows DLL 문제 해결)

### 1.2 모델 확인
- [x] best.pt 모델 로드 테스트
- [x] 12종 야생동물 클래스 확인
- [x] .env 파일 수정 (`YOLO_MODEL_PATH=models/best.pt`)

**완료 날짜:** 2025-11-09

---

## 🔄 Phase 2: 코드 수정 및 정리

### 2.1 StreamListener 수정 (RTSP + 웹캠 지원)
- [x] OpenCV VideoCapture 사용하도록 수정
- [x] RTSP URL 연결 로직 구현
- [x] 웹캠(인덱스 0) 연결 로직 구현
- [x] Frame 데이터 타입 변경 (bytes → np.ndarray)
- [x] 재연결 로직 추가 (네트워크 끊김 대응)
- [x] FPS 제한 로직 유지

**파일:** `app/stream_listener.py`

**주요 변경사항:**
```python
# 기존: 더미 프레임 생성
# 수정: cv2.VideoCapture로 실제 영상 읽기
```

### 2.2 runner.py 수정 (야생동물 전용)
- [x] 사람 감지 로직 제거 (PoseEstimator 사용 안 함)
- [x] 모든 detection을 wildlife 이벤트로 처리
- [x] human 관련 분기문 제거
- [x] process_frame() 메서드 단순화

**파일:** `app/runner.py`

**주요 변경사항:**
```python
# 모든 detection → wildlife event
# PoseEstimator 제거
```

### 2.3 Detector 검증
- [x] best.pt 로딩 확인
- [x] 12개 클래스 매핑 확인
- [x] confidence threshold 확인 (0.4)
- [x] IOU threshold 확인 (0.45)

**파일:** `app/detector.py` (수정 불필요, 검증만)
**검증 스크립트:**
```bash
cd ml_inference
py -3 -m app.detector --image path/to/sample.jpg
```

---

## 🧪 Phase 3: 로컬 테스트 (웹캠)

### 3.1 웹캠 연결 테스트
- [ ] StreamListener 웹캠 모드 테스트
- [ ] 프레임 정상 수신 확인
- [ ] FPS 제한 동작 확인

**테스트 명령:**
```bash
cd ml_inference
py -3 -m app.stream_listener --source 0 --limit 5
```
**FFmpeg 송출 대안:** `scripts/publish_rtsp.ps1` (Windows) 또는 `scripts/publish_rtsp.sh` (macOS/Linux)로 로컬 파일을 `rtsp://k13e106.p.ssafy.io:8554/cctv`에 퍼블리시한 뒤 RTSP 주소로 테스트 가능.
**테스트 시도 (2025-11-09):** 로컬 장치 '0' 연결 실패 - OpenCV에서 `Unable to open stream 0` 발생. 실제 웹캠 또는 RTSP URL 필요.
**원격 RTSP 시도 (2025-11-09):** `py -3 -m app.stream_listener --source rtsp://k13e106.p.ssafy.io:8554/cctv --limit 5` 실행 시 `DESCRIBE 404 Not Found` (퍼블리셔 부재)로 실패. FFmpeg 송출 후 재시도 필요.

### 3.2 Detector 추론 테스트
- [ ] 웹캠 프레임으로 추론 테스트
- [x] 동물 인식 확인 (실제 동물 사진/영상 사용)
- [x] Detection 결과 출력 확인

**테스트 방법:** 노트북 웹캠에 동물 사진 비추기
**진행 상황 (2025-11-09):** 로컬 이미지 `wildboar.jpg`로 멧돼지(0.80) 감지 확인. 웹캠 입력은 장비 부재로 미실시.

### 3.3 전체 파이프라인 테스트
- [ ] create_pipeline() 실행
- [ ] 웹캠 → Detector → EventBuilder 흐름 확인
- [ ] 이벤트 JSON 생성 확인
- [ ] 로그 출력 확인

**테스트 명령:**
```bash
cd ml_inference
py -3 app/runner.py
```

---


### 🧪 RTMPose 실험 가이드
- experiments/rtmpose_proto/README.md 참고
- python experiments/rtmpose_proto/run_rtmpose.py --image <person.jpg>
- 최초 실행 시 config/ckpt 자동 다운로드 → YOLOv8x로 bbox 생성 후 RTMPose로 keypoint 출력

## 🌐 Phase 4: FastAPI 연동 테스트

### 4.1 로컬 테스트 (FastAPI 없이)
- [ ] EventDispatcher 로그만 확인
- [ ] 이벤트 JSON 형식 검증
- [ ] retry 로직 동작 확인 (네트워크 오류 시뮬레이션)

### 4.2 FastAPI Endpoint 테스트
- [ ] curl로 endpoint 연결 확인
  ```bash
  curl -X POST https://k13e106.p.io/dev/api/events \
    -H "Content-Type: application/json" \
    -H "Authorization: Bearer local-dev-token" \
    -d '{"test": "connection"}'
  ```
- [ ] 응답 코드 확인 (202 예상)
- [ ] 에러 응답 처리 확인 (4xx, 5xx)

### 4.3 통합 테스트
- [ ] 웹캠으로 전체 파이프라인 실행
- [ ] FastAPI로 이벤트 전송 확인
- [ ] 서버 로그 확인 (이벤트 수신 여부)

---

## 📡 Phase 5: RTSP 스트림 연동

### 5.1 RTSP 연결 테스트
- [ ] k13e106.p.ssafy.io:8554/cctv 연결 확인
- [ ] 프레임 수신 확인
- [ ] 연결 끊김 재연결 테스트

**테스트 명령:**
```bash
cd ml_inference
py -3 -c "from app.stream_listener import StreamListener; \
  url='rtsp://heobyPublisher:S3curePub!230@k13e106.p.ssafy.io:8554/cctv'; \
  listener = StreamListener(url, fps_limit=12); \
  [print(f'Frame {f.index}') for f in listener.once(limit=10)]"
```

### 5.2 RTSP + 추론 테스트
- [ ] RTSP 영상으로 야생동물 감지 테스트
- [ ] 실제 야생동물이 나오는지 확인
- [ ] Detection 로그 확인

### 5.3 장시간 안정성 테스트
- [ ] 10분 이상 연속 실행
- [ ] 메모리 누수 확인
- [ ] 재연결 로직 동작 확인
- [ ] FPS 안정성 확인

---

## 🚀 Phase 6: EC2 서버 배포

### 6.1 코드 정리
- [ ] 불필요한 주석 제거
- [ ] 로그 레벨 조정 (INFO)
- [ ] .env 파일 서버용으로 복사
- [ ] Git commit

### 6.2 EC2 환경 설정
- [ ] Python 3.10 설치
- [ ] requirements.txt 설치
- [ ] best.pt 모델 파일 업로드
- [ ] .env 파일 설정

### 6.3 서버 배포 및 테스트
- [ ] 코드 배포 (git pull 또는 scp)
- [ ] 테스트 실행
- [ ] systemd 서비스 등록
- [ ] 자동 시작 설정

**systemd 서비스 예시:**
```ini
[Unit]
Description=ML Inference Wildlife Detection
After=network.target

[Service]
Type=simple
User=ubuntu
WorkingDirectory=/home/ubuntu/ml_inference
ExecStart=/usr/bin/python3 -m app.runner
Restart=always

[Install]
WantedBy=multi-user.target
```

---

## 🔍 Phase 7: 모니터링 및 최적화

### 7.1 로그 모니터링
- [ ] journalctl로 로그 확인
- [ ] 에러 로그 수집
- [ ] 이벤트 전송 성공률 확인

### 7.2 성능 최적화
- [ ] FPS 조정 (CPU 사용률 고려)
- [ ] confidence threshold 튜닝
- [ ] IOU threshold 튜닝
- [ ] 배치 처리 고려 (필요시)

### 7.3 알림 설정
- [ ] 서비스 다운 알림
- [ ] 에러 누적 알림
- [ ] RTSP 연결 실패 알림

---

## 📝 추가 작업 (나중에)

### 사람 포즈 추정 (RTMPose)
- [ ] RTMPose 라이브러리 설치
- [ ] 포즈 추정 모델 로드
- [ ] 일사병/열사병 상태 머신 구현
- [ ] 작업시간 기반 모드 전환

### 추적 개선
- [ ] ByteTrack 또는 DeepSORT 통합
- [ ] track_id 안정성 개선
- [ ] 중복 이벤트 방지 로직

### 데이터 수집
- [ ] 감지된 야생동물 이미지 저장
- [ ] 모델 재학습용 데이터 수집
- [ ] 오탐/미탐 케이스 분석

---

## 📌 현재 진행 상황

**완료:**
- ✅ 환경 설정
- ✅ 모델 확인 (best.pt - 12종 야생동물)
- ✅ .env 설정
- ✅ StreamListener 수정
- ✅ runner.py 수정

**진행 중:**
- 🔄 테스트 도구 셋업 (StreamListener/Detector CLI 추가)

**대기 중:**
- ⏳ 하드웨어 기반 테스트
- ⏳ 배포

---

## 🐛 트러블슈팅 로그

### 2025-11-09: PyTorch DLL 오류
- **문제:** `OSError: Error loading "c10.dll"`
- **해결:** Visual C++ Redistributable 설치
- **명령:** `winget install Microsoft.VCRedist.2015+.x64`

### 2025-11-09: 모델 클래스 확인
- **문제:** yolo12s.pt는 COCO 모델 (80 클래스)
- **해결:** best.pt가 실제 커스텀 모델 (12종 야생동물)
- **조치:** .env 파일 YOLO_MODEL_PATH 수정

---

## 📞 연락처 / 참고 자료

- **AI Hub 데이터셋:** https://aihub.or.kr/aihubdata/data/view.do?dataSetSn=645
- **Ultralytics 문서:** https://docs.ultralytics.com/
- **OpenCV 문서:** https://docs.opencv.org/
