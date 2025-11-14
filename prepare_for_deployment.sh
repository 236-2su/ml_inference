#!/bin/bash
# ml_inference 서버 배포 준비 스크립트

echo "================================"
echo "ML Inference 배포 준비 시작"
echo "================================"

# 1. 불필요한 파일 제거
echo ""
echo "[1/5] 테스트 파일 제거 중..."
rm -f quick_test.py test_all_7_videos.py test_json_output.py test_streaming_multiperson.py
rm -f deployment_checklist.md
echo "✓ 테스트 파일 제거 완료"

# 2. venv 제거 (서버에서 재생성)
echo ""
echo "[2/5] 가상환경 제거 중..."
rm -rf venv/
echo "✓ venv 폴더 제거 완료 (서버에서 재생성 필요)"

# 3. 모델 파일 확인
echo ""
echo "[3/5] 필수 모델 파일 확인 중..."
if [ -f "models/svm_model_sit_stand_lie.pkl" ]; then
    echo "✓ SVM 모델 존재: models/svm_model_sit_stand_lie.pkl"
else
    echo "✗ 오류: SVM 모델이 없습니다!"
    exit 1
fi

if [ -f "models/yolov8x.pt" ]; then
    echo "✓ YOLO 사람 감지 모델 존재: models/yolov8x.pt"
else
    echo "⚠ 경고: models/yolov8x.pt 없음"
    echo "  → 서버에서 첫 실행 시 자동 다운로드됩니다"
    echo "  → 또는 .env에서 YOLO_HUMAN_MODEL_PATH 수정 필요"
fi

# 4. .env 파일 확인
echo ""
echo "[4/5] 설정 파일 확인 중..."
if grep -q "FASTAPI_TOKEN=local-dev-token" .env; then
    echo "⚠ 경고: FASTAPI_TOKEN이 기본값입니다"
    echo "  → 프로덕션 토큰으로 변경하세요!"
else
    echo "✓ FASTAPI_TOKEN 설정됨"
fi

if grep -q "USE_MEDIAPIPE_SVM=true" .env; then
    echo "✓ MediaPipe + SVM 활성화됨"
else
    echo "⚠ 경고: MediaPipe + SVM이 비활성화되어 있습니다"
fi

# 5. 배포 아카이브 생성
echo ""
echo "[5/5] 배포 아카이브 생성 중..."
cd ..
tar -czf ml_inference_deploy.tar.gz ml_inference/
echo "✓ 배포 파일 생성: ml_inference_deploy.tar.gz"

# 완료
echo ""
echo "================================"
echo "배포 준비 완료!"
echo "================================"
echo ""
echo "다음 명령으로 서버에 업로드하세요:"
echo "  scp ml_inference_deploy.tar.gz user@server:/path/to/"
echo ""
echo "서버에서 실행:"
echo "  tar -xzf ml_inference_deploy.tar.gz"
echo "  cd ml_inference"
echo "  python3 -m venv venv"
echo "  source venv/bin/activate"
echo "  pip install -r requirements.txt"
echo "  python -m app.runner"
echo ""
