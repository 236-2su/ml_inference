# Multi-Stream RTSP Configuration Guide

## Overview

This ml_inference setup now supports **automatic serial number extraction from RTSP URLs** and can process **multiple streams simultaneously**.

---

## 🔑 Key Features

### 1. Automatic Serial Number Extraction

**Serial numbers are automatically extracted from RTSP URL paths:**

```
rtsp://k13e106.p.ssafy.io:8554/stream/00000000 → Serial: "00000000"
rtsp://k13e106.p.ssafy.io:8554/stream/99999999 → Serial: "99999999"
```

**No need to manually set `SCARECROW_SERIAL_NUMBER`!**

### 2. Multiple Configuration Files

Three environment files are provided:

- `.env` - Default (Scarecrow 00000000)
- `.env.scarecrow1` - Scarecrow 00000000
- `.env.scarecrow2` - Scarecrow 99999999

---

## 🚀 Running Single Stream

### Option 1: Default Configuration

```bash
# Uses .env file (Scarecrow 00000000)
python -m app.runner
```

### Option 2: Using Shell Scripts (Linux/Mac)

```bash
# Scarecrow 1
chmod +x run_scarecrow1.sh
./run_scarecrow1.sh

# Scarecrow 2
chmod +x run_scarecrow2.sh
./run_scarecrow2.sh
```

### Option 3: Using Batch Files (Windows)

```cmd
# Scarecrow 1
run_scarecrow1.bat

# Scarecrow 2
run_scarecrow2.bat
```

---

## 🔀 Running Multiple Streams Simultaneously

### Option 1: Manual (Two Terminals)

**Terminal 1:**
```bash
ENV_FILE=.env.scarecrow1 python -m app.runner
```

**Terminal 2:**
```bash
ENV_FILE=.env.scarecrow2 python -m app.runner
```

### Option 2: Background Script (Linux/Mac)

```bash
chmod +x run_both.sh
./run_both.sh

# Output:
# ✅ Both pipelines started!
#    Scarecrow 1 PID: 12345 (log: logs/scarecrow1.log)
#    Scarecrow 2 PID: 12346 (log: logs/scarecrow2.log)

# Monitor logs:
tail -f logs/scarecrow1.log
tail -f logs/scarecrow2.log

# Stop both:
kill 12345 12346
```

### Option 3: Systemd Services (Production Linux)

Create service files:

**`/etc/systemd/system/ml-inference-00000000.service`:**
```ini
[Unit]
Description=ML Inference for Scarecrow 00000000
After=network.target

[Service]
Type=simple
User=ubuntu
WorkingDirectory=/home/ubuntu/ml_inference
Environment="ENV_FILE=.env.scarecrow1"
ExecStart=/home/ubuntu/ml_inference/.venv/bin/python -m app.runner
Restart=always
RestartSec=10

[Install]
WantedBy=multi-user.target
```

**`/etc/systemd/system/ml-inference-99999999.service`:**
```ini
[Unit]
Description=ML Inference for Scarecrow 99999999
After=network.target

[Service]
Type=simple
User=ubuntu
WorkingDirectory=/home/ubuntu/ml_inference
Environment="ENV_FILE=.env.scarecrow2"
ExecStart=/home/ubuntu/ml_inference/.venv/bin/python -m app.runner
Restart=always
RestartSec=10

[Install]
WantedBy=multi-user.target
```

**Start services:**
```bash
sudo systemctl daemon-reload
sudo systemctl enable ml-inference-00000000
sudo systemctl enable ml-inference-99999999
sudo systemctl start ml-inference-00000000
sudo systemctl start ml-inference-99999999

# Check status:
sudo systemctl status ml-inference-00000000
sudo systemctl status ml-inference-99999999

# View logs:
sudo journalctl -u ml-inference-00000000 -f
sudo journalctl -u ml-inference-99999999 -f
```

---

## 📝 Configuration Files

### `.env.scarecrow1` (Scarecrow 00000000)

```env
MEDIA_RPI_RTSP_URL=rtsp://k13e106.p.ssafy.io:8554/stream/00000000
FASTAPI_ENDPOINT=https://k13e106.p.ssafy.io/dev/api/events
INCLUDE_SNAPSHOT=true
DEFAULT_FPS=0.5  # STREAM_DEFAULT_FPS도 입력 가능
LISTENER_FPS=12  # 기본 RTSP 읽기 속도
# ... other settings
```

### `.env.scarecrow2` (Scarecrow 99999999)

```env
MEDIA_RPI_RTSP_URL=rtsp://k13e106.p.ssafy.io:8554/stream/99999999
FASTAPI_ENDPOINT=https://k13e106.p.ssafy.io/dev/api/events
INCLUDE_SNAPSHOT=true
DEFAULT_FPS=0.5  # STREAM_DEFAULT_FPS도 입력 가능
LISTENER_FPS=12
# ... other settings
```

---

## ⚙️ Manual Serial Number Override

If you need to manually set the serial number (rare cases):

```env
# In .env file:
MEDIA_RPI_RTSP_URL=rtsp://k13e106.p.ssafy.io:8554/stream/custom_path
SCARECROW_SERIAL_NUMBER=00000000  # Manual override
```

**Priority:** Manual `SCARECROW_SERIAL_NUMBER` > Extracted from URL

---

## 🔍 Verification

### Check Serial Number Extraction

```python
from app.runner import extract_serial_from_rtsp

# Test extraction
url1 = "rtsp://k13e106.p.ssafy.io:8554/stream/00000000"
print(extract_serial_from_rtsp(url1))  # Output: "00000000"

url2 = "rtsp://k13e106.p.ssafy.io:8554/stream/99999999"
print(extract_serial_from_rtsp(url2))  # Output: "99999999"
```

### Check Events Sent to HEOBY

```bash
# Watch logs for serial numbers
tail -f logs/scarecrow1.log | grep "stream_id"
tail -f logs/scarecrow2.log | grep "stream_id"

# Expected output:
# {"stream_id": "00000000", "category": "human", ...}
# {"stream_id": "99999999", "category": "wildlife", ...}
```

---

## ⚠️ Prerequisites

### 1. HEOBY Database

Ensure both scarecrows are registered in HEOBY DB:

```sql
INSERT INTO Scarecrow (crow_uuid, serial_number, crow_name, ...)
VALUES ('uuid-1', '00000000', 'Scarecrow 1', ...);

INSERT INTO Scarecrow (crow_uuid, serial_number, crow_name, ...)
VALUES ('uuid-2', '99999999', 'Scarecrow 2', ...);
```

### 2. RTSP Streams

Verify RTSP streams are accessible:

```bash
# Test Scarecrow 1
ffplay rtsp://k13e106.p.ssafy.io:8554/stream/00000000

# Test Scarecrow 2
ffplay rtsp://k13e106.p.ssafy.io:8554/stream/99999999
```

### 3. System Resources

**Running two streams requires:**
- CPU: 6-8 cores (or 4 cores with GPU)
- RAM: 4GB minimum
- GPU: 8GB VRAM (optional, 3× faster)
- Network: 10 Mbps upload

---

## 📊 Resource Usage (2 Streams)

| Resource | Single | Dual | Notes |
|----------|--------|------|-------|
| Memory | 1.1GB | 2.2GB | Models loaded twice |
| CPU | 15-30% | 30-60% | Parallel processing |
| GPU VRAM | 3GB | 6GB | If GPU enabled |
| Network | 1.8 Mbps | 3.6 Mbps | With snapshots |

---

## 🐛 Troubleshooting

### Issue: "Unknown serialNumber" Error

**Symptom:** HEOBY returns 500 error
```
IllegalArgumentException: Unknown serialNumber: 00000000
```

**Solution:** Register scarecrow in HEOBY DB (see Prerequisites #1)

---

### Issue: Serial Number Not Extracted

**Symptom:** Events show `stream_id: "unknown"`

**Solution:** Check RTSP URL format:
```env
# ✅ Correct:
MEDIA_RPI_RTSP_URL=rtsp://host:port/stream/00000000

# ❌ Wrong:
MEDIA_RPI_RTSP_URL=rtsp://host:port/cctv  # No serial in path
```

**Workaround:** Use manual override:
```env
MEDIA_RPI_RTSP_URL=rtsp://host:port/cctv
SCARECROW_SERIAL_NUMBER=00000000
```

---

### Issue: RTSP Connection Failed

**Symptom:** "Unable to open RTSP stream"

**Solution:**
1. Test RTSP URL: `ffplay <RTSP_URL>`
2. Check network/firewall (port 8554)
3. Verify mediaMTX is running
4. Check RTSP credentials (if any)

---

### Issue: High Memory Usage

**Symptom:** System runs out of memory with 2 streams

**Solutions:**
1. **Enable GPU:** `GPU_ENABLED=true` (reduces CPU memory)
2. **Reduce inference FPS:** `DEFAULT_FPS=0.5` (or lower) – STREAM_DEFAULT_FPS도 동작
3. **Disable snapshots:** `INCLUDE_SNAPSHOT=false` (saves bandwidth)
4. **Use smaller models:** Switch to lighter YOLO models

---

## 🎯 Best Practices

### 1. Log Management

Create logs directory:
```bash
mkdir -p logs
```

Rotate logs:
```bash
# logrotate config
/home/ubuntu/ml_inference/logs/*.log {
    daily
    rotate 7
    compress
    delaycompress
    missingok
    notifempty
}
```

### 2. Monitoring

```bash
# Check if processes are running
ps aux | grep "app.runner"

# Monitor resource usage
htop -p $(pgrep -f "app.runner")

# Watch event dispatch
tail -f logs/*.log | grep "Event dispatched"
```

### 3. Error Handling

Enable auto-restart with systemd (see Systemd Services section)

---

## 📚 Related Documentation

- [Integration Flow](docs/integration-flow.md) - End-to-end data flow
- [Event Schema](docs/event-plan.md) - Event payload structure
- [Deployment Checklist](deployment_checklist.md) - Production deployment

---

## 🤝 Support

For issues or questions:
1. Check HEOBY backend logs
2. Check ml_inference logs
3. Verify RTSP stream availability
4. Confirm DB scarecrow registration

---

**Version:** 2.0 (Multi-stream support)
**Last Updated:** 2025-11-15
