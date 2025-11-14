"""Test YOLO model loading and detection."""

import sys
from pathlib import Path

try:
    from ultralytics import YOLO
except ImportError:
    print("ERROR: ultralytics not installed")
    sys.exit(1)

print("="*80)
print("YOLO 모델 테스트")
print("="*80)

# Test yolo12s.pt for human detection
print("\n[1/2] yolo12s.pt (사람 감지) 로딩 중...")
try:
    human_model = YOLO('models/yolo12s.pt')
    print(f"✓ 로딩 성공!")
    print(f"  - 모델 타입: {human_model.model.yaml.get('type', 'N/A')}")
    print(f"  - 클래스 수: {len(human_model.names)}")
    print(f"  - 주요 클래스: {list(human_model.names.values())[:5]}...")

    # Check if 'person' class exists
    person_classes = [k for k, v in human_model.names.items() if 'person' in v.lower() or 'human' in v.lower()]
    if person_classes:
        print(f"✓ 사람 감지 클래스 발견: {[human_model.names[k] for k in person_classes]}")
    else:
        print("⚠ 경고: 사람 감지 클래스를 찾을 수 없습니다")

except Exception as e:
    print(f"✗ 로딩 실패: {e}")
    sys.exit(1)

# Test best.pt for wildlife detection
print("\n[2/2] best.pt (동물 감지) 로딩 중...")
try:
    wildlife_model = YOLO('models/best.pt')
    print(f"✓ 로딩 성공!")
    print(f"  - 모델 타입: {wildlife_model.model.yaml.get('type', 'N/A')}")
    print(f"  - 클래스 수: {len(wildlife_model.names)}")
    print(f"  - 클래스: {list(wildlife_model.names.values())}")
except Exception as e:
    print(f"✗ 로딩 실패: {e}")

# Test on an image if available
print("\n" + "="*80)
print("테스트 이미지로 감지 테스트")
print("="*80)

test_image = "../test1.mp4"  # Will extract first frame
if Path(test_image).exists():
    print(f"\n영상에서 첫 프레임 추출 및 테스트: {test_image}")

    try:
        import cv2
        cap = cv2.VideoCapture(test_image)
        ret, frame = cap.read()
        cap.release()

        if ret:
            # Save first frame
            cv2.imwrite('test_frame.jpg', frame)

            # Detect humans
            results = human_model('test_frame.jpg', verbose=False)
            detections = results[0].boxes

            print(f"\n사람 감지 결과:")
            print(f"  - 감지된 객체 수: {len(detections)}")

            if len(detections) > 0:
                for i, box in enumerate(detections):
                    cls = int(box.cls[0])
                    conf = float(box.conf[0])
                    label = human_model.names[cls]
                    print(f"  - 객체 {i+1}: {label} (신뢰도: {conf:.2f})")
            else:
                print("  - 감지된 사람 없음 (영상에 사람이 없거나 신뢰도가 낮을 수 있음)")

    except Exception as e:
        print(f"✗ 테스트 실패: {e}")
else:
    print(f"테스트 영상 없음: {test_image}")

print("\n" + "="*80)
print("모델 테스트 완료!")
print("="*80)
print("\n다음 단계:")
print("  1. yolo12s.pt가 사람을 감지할 수 있으면 → 배포 준비 완료! ✅")
print("  2. yolo12s.pt가 사람을 감지하지 못하면 → yolov8n.pt 등 다른 모델 필요")
print("")
