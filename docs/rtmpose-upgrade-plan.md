## RTMPose 업그레이드 실행 계획

### 1. 현재 상태 요약
- **감지 파이프라인**: Python 3.12.3 + Ultralytics YOLO 조합으로 사람/야생동물 이중 감지 및 이벤트 전송이 정상 작동 중.
- **포즈 추론**: `models/rtmpose_body2d/.../model.onnx`를 ONNXRuntime으로 구동. `app/pose_classifier.py`의 휴리스틱이 `standing/sitting/crouching/lying/unknown` 다섯 클래스를 반환.
- **상태 머신**: `app/pose_state_machine.PoseStateMachine`이 트래킹 ID별로 낙상(`fall_detected`)·열사병 주의(`heatstroke_watch`, 5분)·열사병 경보(`heatstroke_alert`, 15분)를 단발성으로 생성.
- **서버 배포**: `/home/ubuntu/ml_inference_deploy`에 최신 코드와 .venv가 설치되어 있으며, systemd 서비스 `ml-inference`가 해당 경로를 바라보도록 구성되어 있음.

### 2. 남은 과제
| 구분 | 작업 | 비고 |
| --- | --- | --- |
| 파라미터 튜닝 | `.env`의 `POSE_*` 값을 현장 데이터에 맞춰 보정 | lying → sitting 오분류 사례 개선 |
| 추가 포즈 | 필요 시 `kneeling`, `running` 등 커스텀 라벨 추가 | `PoseClassifierConfig` 확장 |
| 상태 로깅 | FastAPI 이벤트에 status history(진입/해제 시각) 기록 | `EventBuilder` 확장 |
| 모니터링 | systemd 로그 + CloudWatch/Prometheus 연동 | 장애 시 조기 탐지 |

### 3. 개발/테스트 절차
1. **로컬**
   - Python 3.10+ 가상환경 구성, `pip install -r requirements.txt`
   - `pytest`로 단위 테스트 실행
   - `python scripts/rtmpose_image_eval.py ../lying.jpg ...`로 휴리스틱 빠르게 확인
2. **CI (GitHub Actions)**
   - `ci.yml`이 main 브랜치 push/PR마다 Python 3.10/3.12에서 pytest 및 FastAPI import 체크 수행
3. **CD**
   - `deploy.yml`이 동일 테스트 후 appleboy/ssh-action으로 EC2(`secrets.EC2_HOST`)에 접속, `/home/ubuntu/ml_inference_deploy`에 pull + pip install + pytest + `sudo systemctl restart ml-inference`

### 4. 운영 시나리오
1. **낙상 감지**: standing/sitting 등 → lying 전환이 일정 confidence 이상일 때 `fall_detected` 이벤트 즉시 전송.
2. **열사병 감시**: lying 지속 시간이 300초에 도달하면 `heatstroke_watch`, 900초에 도달하면 `heatstroke_alert`로 승급. 각 상태는 최초 1회만 송출.
3. **트래킹 만료**: `POSE_STATE_IDLE_TTL_SECONDS`(기본 120초) 동안 프레임이 유입되지 않으면 상태 머신이 자동으로 정리.

### 5. 운영 가이드
- **배포**: main으로 PR 머지 → Actions가 CI/CD 실행 → 성공 시 systemd 서비스 자동 재시작. 실패 시 Actions 로그 확인 후 수동으로 `/home/ubuntu/ml_inference_deploy`에서 `git pull && pip install -r requirements.txt && pytest && sudo systemctl restart ml-inference`.
- **롤백**: `git revert` 또는 특정 커밋으로 `git reset --hard <commit>` 실행 후 동일 절차로 배포.
- **로그 확인**: `journalctl -u ml-inference -f` 로 런타임 로그 확인, FastAPI 수신 서버 로그로 최종 이벤트 확인.

### 6. 문서/지식 공유
- 개발/운영 절차는 `README.md`에 기록.
- 파라미터 조정 결과, 현장 피드백은 이 문서를 업데이트하여 아카이브.
