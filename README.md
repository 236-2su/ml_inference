## ml_inference

실시간 RTSP 스트림에서 사람/야생동물을 분리 감지하고, RTMPose ONNX 모델로 사람의 포즈(서 있음/앉음/쭈그림/누움/미확인)와 상태 머신 이벤트(낙상, 열사병 주의/경보)를 생성해 FastAPI 서버로 전송하는 파이프라인입니다.

### 구성 요소
- `app/detector.py` – Ultralytics YOLO 기반 사람/야생동물 감지
- `app/pose_estimator.py` – ONNXRuntime + RTMPose 추론, `app/pose_classifier.py`에 위임된 휴리스틱으로 포즈 라벨링
- `app/pose_state_machine.py` – 트래킹 ID마다 낙상/열사병 상태를 계산
- `app/runner.py` – StreamListener → Detector → Tracker → PoseEstimator → EventBuilder → EventDispatcher 순서로 파이프라인 실행
- pp/event_filter.py – Duplicate track events are filtered before dispatch
- `fastapi_app/main.py` – 수신 측 FastAPI 예제
- `scripts/rtmpose_image_eval.py` – 정적 이미지에서 포즈 추론 결과를 확인하는 도구

### 환경 변수(.env)
`app/config.py`와 `.env`가 동일한 키를 공유합니다.

```
YOLO_MODEL_PATH=models/best.pt
YOLO_HUMAN_MODEL_PATH=models/yolov8x.pt
YOLO_POSE_MODEL_PATH=models/rtmpose_body2d/<...>/model.onnx
POSE_KEYPOINT_CONF_THRESHOLD=0.002
POSE_LYING_ASPECT_RATIO=0.65
POSE_LYING_TORSO_ANGLE_DEG=35
POSE_HEATSTROKE_WATCH_SECONDS=300
POSE_HEATSTROKE_ALERT_SECONDS=900
POSE_STATE_IDLE_TTL_SECONDS=120
POSE_STATE_MIN_CONFIDENCE=0.35
```

RTSP/FastAPI 엔드포인트, 토큰 등 나머지 항목도 `.env`를 통해 주입합니다. 운영 환경에서는 Secrets Manager/SSM을 사용하는 것을 권장합니다.

### 로컬 개발
1. Python 3.10+ 가상환경 생성  
   ```bash
   python -m venv .venv
   source .venv/bin/activate
   pip install --upgrade pip
   pip install -r requirements.txt
   ```
2. 환경 변수 준비: `.env` 복사 후 값 채우기.
3. 유닛 테스트 실행: `pytest`  
4. 샘플 이미지 평가:  
   ```bash
   python scripts/rtmpose_image_eval.py ../lying.jpg ../pose_test1.jpg
   ```

### CI
`./github/workflows/ci.yml`
- main 브랜치에 push/PR 시 실행
- Python 3.10/3.12 매트릭스에서 의존성 설치, pytest, FastAPI import 스모크 테스트
- pip 캐시를 사용해 반복 빌드 최적화

### CD
`./github/workflows/deploy.yml`
- main에 push되면 CI 통과 후 실행
- appleboy/ssh-action으로 EC2에 접속, `/home/ubuntu/ml_inference_deploy` (또는 `secrets.EC2_APP_DIR`)에 코드를 배포
- 신규 가상환경 생성, `pip install -r requirements.txt`, `pytest` 재실행, `ml-inference` systemd 서비스를 재시작
- 필요 Secrets  
  - `EC2_HOST`, `EC2_USER`, `EC2_SSH_KEY` (private key)  
  - 선택: `EC2_APP_DIR` (기본값 `/home/ubuntu/ml_inference_deploy`)

### 서버 수동 배포(비상 시)
```bash
ssh -i <pem> ubuntu@<EC2>
cd /home/ubuntu/ml_inference_deploy
git pull origin main
source .venv/bin/activate
pip install -r requirements.txt
pytest
sudo systemctl restart ml-inference
```

### 트러블슈팅
- **onnxruntime GPU 경고**: CPU 전용 인스턴스에서 발생하는 `/sys/class/drm/...` 경고는 무시해도 됩니다.
- **포즈 분류 오차**: `.env`의 `POSE_KEYPOINT_CONF_THRESHOLD`, `POSE_LYING_ASPECT_RATIO` 등을 조정한 뒤 `scripts/rtmpose_image_eval.py`로 빠르게 검증하세요.
- **CI/CD 실패**: Actions 로그를 참고하고, 서버 측 `/var/log/syslog` 및 `journalctl -u ml-inference`에서 runtime 로그를 확인하세요.

