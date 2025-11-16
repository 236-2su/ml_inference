# RTSP 스트림 설정 가이드

## 현재 상황

**문제:** mediaMTX 서버에 활성 RTSP 스트림이 없습니다.
- `/stream/00000000` - ❌ Not published
- `/stream/99999999` - ❌ Not published

**필요한 작업:** Raspberry Pi 또는 카메라에서 RTSP 스트림을 mediaMTX로 publish해야 합니다.

---

## 1. 테스트 스트림 생성 (개발/테스트용)

백엔드 서버에서 FFmpeg를 사용해 테스트 스트림을 만들 수 있습니다:

### 설치 (필요시)

```bash
ssh -i "c:/Users/lsw01/Desktop/106/K13E106T.pem" ubuntu@k13e106.p.ssafy.io

# FFmpeg 설치 확인
ffmpeg -version || sudo apt install ffmpeg -y
```

### 테스트 스트림 생성

#### Stream 1: 00000000 (컬러 테스트 패턴)

```bash
# 백그라운드에서 실행
nohup ffmpeg -re -f lavfi -i testsrc=size=640x480:rate=2 \
  -c:v libx264 -preset ultrafast -tune zerolatency \
  -b:v 500k -g 20 \
  -f rtsp rtsp://localhost:8554/stream/00000000 \
  > /tmp/stream_00000000.log 2>&1 &

echo $! > /tmp/stream_00000000.pid
```

#### Stream 2: 99999999 (다른 패턴)

```bash
nohup ffmpeg -re -f lavfi -i rgbtestsrc=size=640x480:rate=2 \
  -c:v libx264 -preset ultrafast -tune zerolatency \
  -b:v 500k -g 20 \
  -f rtsp rtsp://localhost:8554/stream/99999999 \
  > /tmp/stream_99999999.log 2>&1 &

echo $! > /tmp/stream_99999999.pid
```

### 스트림 상태 확인

```bash
# 프로세스 확인
ps aux | grep ffmpeg | grep stream

# 로그 확인
tail -f /tmp/stream_00000000.log
tail -f /tmp/stream_99999999.log

# RTSP 스트림 확인
ffprobe rtsp://localhost:8554/stream/00000000
ffprobe rtsp://localhost:8554/stream/99999999
```

### 스트림 중지

```bash
# PID 사용
kill $(cat /tmp/stream_00000000.pid)
kill $(cat /tmp/stream_99999999.pid)

# 또는 직접 종료
pkill -f "ffmpeg.*stream/00000000"
pkill -f "ffmpeg.*stream/99999999"
```

---

## 2. Raspberry Pi에서 RTSP Publish

Raspberry Pi에 카메라가 연결되어 있다면:

### 설치

```bash
# Raspberry Pi에서
sudo apt update
sudo apt install ffmpeg v4l-utils -y
```

### 카메라 확인

```bash
# 연결된 카메라 리스트
v4l2-ctl --list-devices

# 카메라 정보 확인 (보통 /dev/video0)
v4l2-ctl --device=/dev/video0 --all
```

### RTSP Publish 스크립트

`/home/pi/publish_rtsp.sh` 파일 생성:

```bash
#!/bin/bash

# 설정
SERIAL_NUMBER="00000000"  # 또는 99999999
RTSP_SERVER="k13e106.p.ssafy.io:8554"
VIDEO_DEVICE="/dev/video0"
FPS=2

# 카메라에서 RTSP로 스트리밍
ffmpeg -f v4l2 -framerate $FPS -video_size 640x480 -i $VIDEO_DEVICE \
  -c:v libx264 -preset ultrafast -tune zerolatency \
  -b:v 500k -g 10 -pix_fmt yuv420p \
  -f rtsp rtsp://$RTSP_SERVER/stream/$SERIAL_NUMBER
```

### 실행

```bash
chmod +x /home/pi/publish_rtsp.sh
nohup /home/pi/publish_rtsp.sh > /tmp/rtsp_publish.log 2>&1 &
```

### systemd 서비스로 등록 (자동 시작)

`/etc/systemd/system/rtsp-publisher.service` 생성:

```ini
[Unit]
Description=RTSP Camera Publisher
After=network.target

[Service]
Type=simple
User=pi
WorkingDirectory=/home/pi
ExecStart=/home/pi/publish_rtsp.sh
Restart=always
RestartSec=10

[Install]
WantedBy=multi-user.target
```

```bash
sudo systemctl enable rtsp-publisher.service
sudo systemctl start rtsp-publisher.service
sudo systemctl status rtsp-publisher.service
```

---

## 3. IP 카메라 Re-streaming

IP 카메라의 RTSP를 mediaMTX로 re-streaming:

```bash
# IP 카메라 RTSP를 mediaMTX로 전달
ffmpeg -rtsp_transport tcp \
  -i rtsp://admin:password@192.168.1.100:554/stream \
  -c copy \
  -f rtsp rtsp://k13e106.p.ssafy.io:8554/stream/00000000
```

---

## 4. mediaMTX 설정 확인

### mediaMTX 로그 확인

```bash
docker logs heoby-mediamtx -f
```

### mediaMTX API로 스트림 확인

```bash
# 활성 경로 리스트
curl http://localhost:9997/v3/paths/list | jq

# 특정 경로 정보
curl http://localhost:9997/v3/paths/get/stream/00000000 | jq
```

---

## 5. 빠른 테스트 (백엔드 서버에서)

```bash
# SSH 접속
ssh -i "c:/Users/lsw01/Desktop/106/K13E106T.pem" ubuntu@k13e106.p.ssafy.io

# 두 개의 테스트 스트림 생성 (한 번에 실행)
cat << 'EOF' > /tmp/start_test_streams.sh
#!/bin/bash

# Stream 00000000
nohup ffmpeg -re -f lavfi -i testsrc=size=640x480:rate=2 \
  -c:v libx264 -preset ultrafast -tune zerolatency -b:v 500k -g 20 \
  -f rtsp rtsp://localhost:8554/stream/00000000 \
  > /tmp/stream_00000000.log 2>&1 &
echo $! > /tmp/stream_00000000.pid

# Stream 99999999
sleep 2
nohup ffmpeg -re -f lavfi -i rgbtestsrc=size=640x480:rate=2 \
  -c:v libx264 -preset ultrafast -tune zerolatency -b:v 500k -g 20 \
  -f rtsp rtsp://localhost:8554/stream/99999999 \
  > /tmp/stream_99999999.log 2>&1 &
echo $! > /tmp/stream_99999999.pid

echo "Test streams started!"
echo "Stream 00000000 PID: $(cat /tmp/stream_00000000.pid)"
echo "Stream 99999999 PID: $(cat /tmp/stream_99999999.pid)"
EOF

chmod +x /tmp/start_test_streams.sh
/tmp/start_test_streams.sh

# 스트림 확인 (5초 대기 후)
sleep 5
ffprobe rtsp://localhost:8554/stream/00000000 2>&1 | grep Stream
ffprobe rtsp://localhost:8554/stream/99999999 2>&1 | grep Stream
```

### 테스트 스트림 중지

```bash
cat << 'EOF' > /tmp/stop_test_streams.sh
#!/bin/bash
if [ -f /tmp/stream_00000000.pid ]; then
    kill $(cat /tmp/stream_00000000.pid) 2>/dev/null
    rm /tmp/stream_00000000.pid
fi
if [ -f /tmp/stream_99999999.pid ]; then
    kill $(cat /tmp/stream_99999999.pid) 2>/dev/null
    rm /tmp/stream_99999999.pid
fi
echo "Test streams stopped!"
EOF

chmod +x /tmp/stop_test_streams.sh
/tmp/stop_test_streams.sh
```

---

## 6. ML Inference 연결 테스트

스트림이 활성화되면:

```bash
# ML inference 서버에서
ssh -i "c:/Users/lsw01/Desktop/106/ml_inference.pem" ubuntu@3.25.104.150

cd /home/ubuntu/ml_inference_deploy
source .venv/bin/activate

# 멀티 스트림 테스트
python run_multi_pipeline.py

# 로그 확인
# [00000000]와 [99999999] 로그가 모두 보여야 함
```

---

## 트러블슈팅

### 문제: "Connection refused"

**원인:** mediaMTX가 실행되지 않음

**해결:**
```bash
docker ps | grep mediamtx
docker restart heoby-mediamtx
```

### 문제: "404 Not Found"

**원인:** 스트림이 publish되지 않음

**해결:** 위의 테스트 스트림 생성 방법 사용

### 문제: "Connection timeout"

**원인:** 방화벽/네트워크 문제

**해결:**
```bash
# 포트 8554 확인
sudo netstat -tunlp | grep 8554

# 방화벽 확인 (필요시)
sudo ufw status
sudo ufw allow 8554/tcp
```
