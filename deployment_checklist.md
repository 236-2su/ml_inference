# 서버 배포 체크리스트

## ⚠️ 배포 전 필수 확인 사항

### 1. 모델 파일 확인
- [x] `models/svm_model_sit_stand_lie.pkl` - ✅ 존재 (139KB)
- [x] `models/best.pt` - ✅ 존재 (19MB)
- [ ] `models/yolov8x.pt` - ❌ **누락!** (사람 감지용 YOLO 모델)
  - **해결 방법**:
    - 옵션 1: yolov8x.pt 다운로드 필요
    - 옵션 2: `.env`에서 `YOLO_HUMAN_MODEL_PATH=None` 설정 (사람 감지 비활성화)
    - 옵션 3: best.pt를 사람 감지에도 사용 (추천 ❌)

### 2. .env 파일 설정 확인
- [x] RTSP 스트림 URL 설정됨: `rtsp://k13e106.p.ssafy.io:8554/cctv`
- [x] FastAPI 엔드포인트 설정됨: `https://k13e106.p.ssafy.io/dev/api/events`
- [ ] **FASTAPI_TOKEN 변경 필요!** - 현재: `local-dev-token` (보안 취약)
  - 실제 토큰으로 변경하세요
- [x] MediaPipe + SVM 활성화됨: `USE_MEDIAPIPE_SVM=true`
- [x] GPU 설정: `GPU_ENABLED=false` (서버 GPU 있으면 true로 변경)

### 3. 의존성 패키지
- [x] requirements.txt 준비됨
- [x] venv 설치 완료
- [ ] **서버에서 재설치 필요** (Windows venv는 Linux에서 안됨)

### 4. 불필요한 파일 제거
- [ ] 테스트 비디오 파일 (test*.mp4) - 용량 절약
- [ ] 테스트 스크립트:
  - `quick_test.py`
  - `test_all_7_videos.py`
  - `test_json_output.py`
  - `test_streaming_multiperson.py`

### 5. 서버 환경 확인
- [ ] Python 3.10 이상 설치됨?
- [ ] 네트워크: RTSP 스트림 접근 가능?
- [ ] 네트워크: FastAPI 엔드포인트 접근 가능?
- [ ] CPU/메모리: 최소 2 vCPU, 8GB RAM
- [ ] 권한: 포트 및 파일 시스템 접근 권한

---

## 📦 배포 방법

### 방법 1: 수동 배포 (간단)

```bash
# 1. ml_inference 폴더를 서버로 복사
scp -r ml_inference/ user@server:/path/to/app/

# 2. 서버에서 실행
ssh user@server
cd /path/to/app/ml_inference

# 3. Python 가상환경 생성
python3 -m venv venv
source venv/bin/activate  # Linux
# 또는
source venv/Scripts/activate  # Windows

# 4. 의존성 설치
pip install --upgrade pip
pip install -r requirements.txt

# 5. yolov8x.pt 다운로드 (필요시)
# Ultralytics가 자동으로 다운로드하지만, 수동으로도 가능:
# wget https://github.com/ultralytics/assets/releases/download/v0.0.0/yolov8x.pt
# mv yolov8x.pt models/

# 6. 테스트 실행
python -m app.runner --help

# 7. 실제 실행
python -m app.runner
```

### 방법 2: systemd 서비스 (프로덕션, Linux)

서버에 `/etc/systemd/system/ml-inference.service` 파일 생성:

```ini
[Unit]
Description=ML Inference Pipeline - MediaPipe + SVM Pose Detection
After=network.target

[Service]
Type=simple
User=your-user
WorkingDirectory=/path/to/ml_inference
Environment="PATH=/path/to/ml_inference/venv/bin"
ExecStart=/path/to/ml_inference/venv/bin/python -m app.runner
Restart=always
RestartSec=10

[Install]
WantedBy=multi-user.target
```

실행:
```bash
sudo systemctl daemon-reload
sudo systemctl enable ml-inference
sudo systemctl start ml-inference
sudo systemctl status ml-inference

# 로그 확인
sudo journalctl -u ml-inference -f
```

### 방법 3: Docker (권장, 어디서나 동일)

`Dockerfile` 생성 (ml_inference 폴더에):

```dockerfile
FROM python:3.10-slim

WORKDIR /app

# 시스템 의존성 설치
RUN apt-get update && apt-get install -y \
    libgl1-mesa-glx \
    libglib2.0-0 \
    && rm -rf /var/lib/apt/lists/*

# Python 의존성 설치
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# 애플리케이션 복사
COPY . .

# 모델 파일 확인 (빌드 시 에러 방지)
RUN ls -la models/

# 실행
CMD ["python", "-m", "app.runner"]
```

`docker-compose.yml` 생성:

```yaml
version: '3.8'

services:
  ml-inference:
    build: .
    container_name: ml-inference
    restart: always
    env_file:
      - .env
    volumes:
      - ./models:/app/models:ro
      - ./artifacts:/app/artifacts
    network_mode: host
    logging:
      driver: "json-file"
      options:
        max-size: "10m"
        max-file: "3"
```

실행:
```bash
docker-compose up -d
docker-compose logs -f
```

---

## 🔧 배포 후 확인 사항

### 1. 서비스 작동 확인
```bash
# 프로세스 실행 중인지 확인
ps aux | grep runner

# 로그 확인
tail -f /var/log/ml-inference.log  # 또는 systemd/docker 로그
```

### 2. 네트워크 연결 확인
```bash
# RTSP 스트림 접근 가능한지
ffmpeg -i rtsp://k13e106.p.ssafy.io:8554/cctv -frames:v 1 test.jpg

# FastAPI 엔드포인트 접근 가능한지
curl -X POST https://k13e106.p.ssafy.io/dev/api/events \
  -H "Authorization: Bearer your-token" \
  -H "Content-Type: application/json" \
  -d '{"test": true}'
```

### 3. 이벤트 전송 확인
- FastAPI 서버에서 이벤트가 수신되는지 확인
- JSON 형식이 올바른지 확인
- 자세 분류가 정확한지 확인 (sitting/standing/lying)

---

## ⚡ 성능 최적화 (선택사항)

### CPU 서버 (현재 설정)
```bash
# .env
GPU_ENABLED=false
STREAM_DEFAULT_FPS=12  # 12 FPS로 제한
```

### GPU 서버 (성능 향상)
```bash
# .env
GPU_ENABLED=true
STREAM_DEFAULT_FPS=30  # 30 FPS로 증가

# PyTorch GPU 버전 설치
pip install torch torchvision --index-url https://download.pytorch.org/whl/cu118
```

---

## 🚨 문제 해결

### 문제 1: YOLO 모델 다운로드 실패
```bash
# 수동 다운로드
cd models
wget https://github.com/ultralytics/assets/releases/download/v8.0.0/yolov8x.pt
```

### 문제 2: RTSP 스트림 연결 실패
- 방화벽 확인
- 네트워크 접근 권한 확인
- RTSP URL 정확한지 확인

### 문제 3: FastAPI 이벤트 전송 실패
- API 엔드포인트 URL 확인
- 토큰 유효성 확인
- 서버 로그 확인

### 문제 4: 메모리 부족
```bash
# FPS 낮추기
STREAM_DEFAULT_FPS=8

# 스냅샷 비활성화
INCLUDE_SNAPSHOT=false
```

---

## 📊 모니터링

### 리소스 사용량 확인
```bash
# CPU/메모리 사용량
top -p $(pgrep -f runner)

# 디스크 사용량
df -h
du -sh ml_inference/
```

### 로그 모니터링
```bash
# Systemd
sudo journalctl -u ml-inference -f --since "1 hour ago"

# Docker
docker-compose logs -f --tail=100
```

---

## ✅ 최종 배포 체크리스트

배포 전 이것들을 확인하세요:

- [ ] yolov8x.pt 모델 다운로드 완료 (또는 .env 수정)
- [ ] FASTAPI_TOKEN을 실제 토큰으로 변경
- [ ] RTSP URL이 서버에서 접근 가능한지 확인
- [ ] FastAPI 엔드포인트가 서버에서 접근 가능한지 확인
- [ ] 테스트 파일 제거 (용량 절약)
- [ ] venv 제거 (서버에서 재생성)
- [ ] .env 파일 보안 설정 (chmod 600 .env)
- [ ] 서버 Python 버전 확인 (3.10+)
- [ ] 배포 방법 선택 (수동/systemd/Docker)
- [ ] 모니터링 설정 (로그, 알람)

배포 준비 완료! 🚀
