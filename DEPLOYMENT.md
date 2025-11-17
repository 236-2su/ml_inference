# ML Inference 멀티 스트림 배포 가이드

## 변경 사항 요약

### 1. **status 필드 제거**
- `event_builder.py`에서 `status` 필드를 이벤트 페이로드에서 제거
- 백엔드 API와 형식 일치

### 2. **멀티 스트림 지원**
- 여러 RTSP 스트림을 동시에 처리할 수 있는 `run_multi_pipeline.py` 추가
- 각 스트림은 독립적인 프로세스에서 실행
- 기본 설정: `00000000`, `99999999` 두 개의 스트림

### 3. **URL 경로 수정**
- FASTAPI_ENDPOINT: `https://k13e106.p.ssafy.io/dev/api/ai/events`
- Nginx 라우팅과 일치하도록 수정

---

## 서버 배포 방법

### Step 1: 코드 업로드

```bash
# 로컬에서 Git push
cd C:/Users/lsw01/Desktop/106/ml_inference
git add .
git commit -m "Add multi-stream support and remove status field"
git push

# 서버에서 Pull (SSH 접속)
ssh -i "c:/Users/lsw01/Desktop/106/ml_inference.pem" ubuntu@3.25.104.150

cd /home/ubuntu/ml_inference_deploy
git pull
```

### Step 2: 의존성 업데이트 (필요 시)

```bash
source .venv/bin/activate
pip install -r requirements.txt
```

### Step 3: .env 파일 확인/수정

```bash
nano .env

# 다음 설정이 있는지 확인:
RTSP_STREAMS=00000000,99999999
RTSP_BASE_URL=rtsp://k13e106.p.ssafy.io:8554/stream
FASTAPI_ENDPOINT=https://k13e106.p.ssafy.io/dev/api/ai/events
DEFAULT_FPS=2  # 기존 STREAM_DEFAULT_FPS도 호환 입력으로 동작
USE_MEDIAPIPE_SVM=true
```

### Step 4: systemd 서비스 업데이트

#### **옵션 A: 멀티 스트림 모드 (권장)**

```bash
sudo nano /etc/systemd/system/ml-inference-runner.service
```

다음 내용으로 수정:

```ini
[Unit]
Description=ML Inference Multi-Stream Pipeline Runner
After=network.target

[Service]
Type=simple
WorkingDirectory=/home/ubuntu/ml_inference_deploy
Environment=PYTHONUNBUFFERED=1
Environment=PYTHONPATH=/home/ubuntu/ml_inference_deploy
ExecStart=/home/ubuntu/ml_inference_deploy/.venv/bin/python /home/ubuntu/ml_inference_deploy/run_multi_pipeline.py
Restart=on-failure
RestartSec=10
User=ubuntu

[Install]
WantedBy=multi-user.target
```

#### **옵션 B: 단일 스트림 모드**

기존 설정 유지 (`run_pipeline.py` 사용)

### Step 5: 서비스 재시작

```bash
# systemd 설정 리로드
sudo systemctl daemon-reload

# 서비스 재시작
sudo systemctl restart ml-inference-runner.service

# 상태 확인
sudo systemctl status ml-inference-runner.service

# 로그 확인 (실시간)
journalctl -u ml-inference-runner.service -f
```

---

## RTSP 스트림 테스트

### 1. mediaMTX에서 스트림 확인

```bash
# 백엔드 서버 SSH 접속
ssh -i "c:/Users/lsw01/Desktop/106/K13E106T.pem" ubuntu@k13e106.p.ssafy.io

# 활성 스트림 확인
curl http://localhost:9997/v3/paths/list | jq

# 특정 스트림 테스트
ffprobe rtsp://localhost:8554/stream/00000000
ffprobe rtsp://localhost:8554/stream/99999999
```

### 2. RTSP 스트림 Publish 방법

Raspberry Pi나 카메라에서 mediaMTX로 스트림을 publish해야 합니다:

```bash
# FFmpeg를 사용한 테스트 스트림 publish 예제
ffmpeg -re -f lavfi -i testsrc=size=640x480:rate=10 \
  -f lavfi -i sine=frequency=1000 \
  -c:v libx264 -preset ultrafast -c:a aac \
  -f rtsp rtsp://k13e106.p.ssafy.io:8554/stream/00000000
```

### 3. ML Inference 연결 테스트

```bash
# ML inference 서버에서
cd /home/ubuntu/ml_inference_deploy

# 단일 스트림 테스트
source .venv/bin/activate
python run_pipeline.py

# 멀티 스트림 테스트
python run_multi_pipeline.py
```

---

## 로그 확인

### ML Inference Runner 로그

```bash
# 최근 50줄
journalctl -u ml-inference-runner.service -n 50

# 실시간 모니터링
journalctl -u ml-inference-runner.service -f

# 에러만 필터링
journalctl -u ml-inference-runner.service -p err -n 50
```

### FastAPI 서버 로그

```bash
journalctl -u ml-inference.service -f
```

### 백엔드 로그

```bash
# 백엔드 서버에서
docker logs heoby-dev-backend-1 -f --tail 100
```

---

## 트러블슈팅

### 문제 1: RTSP 연결 실패

**에러:**
```
RuntimeError: Unable to open RTSP stream rtsp://...
```

**해결:**
1. mediaMTX에 해당 스트림이 publish되고 있는지 확인
2. 네트워크 연결 확인 (방화벽, 포트 8554)
3. RTSP URL 경로 확인

### 문제 2: 백엔드 API 401 Unauthorized

**에러:**
```
{"error":"UNAUTHORIZED","message":"로그인이 필요합니다."}
```

**해결:**
1. FASTAPI_ENDPOINT가 `/dev/api/ai/events`로 끝나는지 확인
2. 백엔드 SecurityConfig에서 `/ai/**` permitAll 확인

### 문제 3: 멀티 프로세스 시작 안 됨

**해결:**
```bash
# Python multiprocessing 디버깅
python -c "import multiprocessing; multiprocessing.set_start_method('spawn'); print('OK')"

# 환경 변수 확인
cat .env | grep RTSP_STREAMS
```

---

## 모니터링

### 프로세스 확인

```bash
# ML inference 프로세스
ps aux | grep python | grep -E 'runner|pipeline'

# 리소스 사용량
htop
```

### 네트워크 확인

```bash
# RTSP 연결 확인
netstat -tunlp | grep 8554

# 백엔드 API 연결
netstat -tunlp | grep 8181
```

---

## 성능 최적화

현재 설정:
- **FPS**: 2 (매우 낮음, 배터리/네트워크 절약)
- **Pose Estimator**: MediaPipe + SVM (경량)
- **Snapshot**: 활성화 (이미지 포함)
- **Event Filter**: 활성화 (중복 제거)

성능 향상이 필요하면:
```bash
# .env 파일 수정
DEFAULT_FPS=5  # FPS 증가 (STREAM_DEFAULT_FPS도 동일하게 동작)
GPU_ENABLED=true      # GPU 사용 (CUDA 설치 필요)
INCLUDE_SNAPSHOT=false  # 스냅샷 비활성화 (네트워크 절약)
```

---

## 롤백 방법

```bash
# Git 이전 커밋으로 되돌리기
git log --oneline -5
git checkout <commit-hash>

# 서비스 재시작
sudo systemctl restart ml-inference-runner.service
```
