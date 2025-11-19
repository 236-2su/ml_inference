AI 기반 야생동물 탐지 및 안전 모니터링 시스템

이 프로젝트는 실시간 비디오 스트림을 분석하여 야생동물(멧돼지, 고라니 등)을 탐지하고, 사람의 포즈를 추정하여 위급 상황(낙상, 열사병 등)을 모니터링하는 AI 추론 시스템입니다.

## 기술 아키텍처 (Technical Architecture)

### 1. 야생동물 인식 (Wildlife Recognition with YOLO12)

본 시스템은 **YOLO12** 모델을 활용하여 비디오 프레임 내의 객체를 실시간으로 탐지합니다.

- **구현**: `Detector` 클래스는 `ultralytics` 라이브러리를 기반으로 동작하며, 사전 학습된 YOLO 모델을 로드하여 추론을 수행합니다.
- **프로세스**:
  1. 입력된 프레임은 모델이 처리할 수 있는 포맷으로 전처리됩니다.
  2. YOLO12 모델이 객체의 위치(Bounding Box)와 클래스(종류), 신뢰도(Confidence)를 예측합니다.
  3. 탐지된 객체 중 야생동물로 분류된 객체에 대해서만 이벤트를 생성하며, 설정된 신뢰도 임계값(Threshold) 미만의 탐지는 필터링하여 오탐지를 줄입니다.

### 2. 포즈 추정 및 행동 분류 (Pose Estimation & SVM Classification)

본 시스템은 **MediaPipe Pose**와 **SVM(Support Vector Machine)**을 결합한 하이브리드 방식을 사용하여 실시간으로 행동을 분석합니다. (RTMPose는 사용하지 않습니다.)

- **MediaPipe Pose**:

  - 검출된 사람 영역(ROI)에서 **33개의 신체 랜드마크**를 추출합니다.
  - 각 랜드마크는 `(x, y)` 좌표를 가지며, 이미지 크기에 대해 0.0~1.0으로 정규화된 값을 사용합니다.
- **SVM 분류기 (SVM Classifier)**:

  - **입력 (Input)**: 33개 랜드마크의 x좌표와 y좌표를 이어 붙인 **66차원 벡터** (`[x1, ..., x33, y1, ..., y33]`)를 입력으로 받습니다.
  - **추론 (Inference)**: 학습된 SVM 모델(`svm_model_sit_stand_lie.pkl`)이 입력 벡터와 초평면(Hyperplane) 사이의 거리를 계산합니다.
  - **결과 (Output)**: `decision_function`을 통해 계산된 점수를 기반으로 **'서 있음(standing)'**, **'앉아 있음(sitting)'**, **'누워 있음(lying)'** 중 가장 확률이 높은 클래스를 선택합니다.
  - **신뢰도 (Confidence)**: 결정 경계로부터의 거리를 확률로 변환하여 신뢰도 점수를 산출합니다.
- **위급 상황 감지**: '누워 있음' 상태가 설정된 임계 시간(예: 5분) 이상 지속될 경우, 이를 위급 상황으로 판단하여 알림을 전송합니다.

### 3. 실시간 비디오 스트리밍 (Real-time Video Streaming)

RTSP 프로토콜을 통해 CCTV나 IP 카메라로부터 비디오 데이터를 안정적으로 수신합니다.

- **StreamListener**: OpenCV(`cv2.VideoCapture`)를 사용하여 스트림을 연결하고 프레임을 캡처합니다.
- **최적화**: 실시간성을 보장하기 위해 수신 버퍼 크기를 최소화하여 지연(Latency)을 줄이고, 항상 최신 프레임을 처리하도록 설계되었습니다.
- **재연결 로직**: 네트워크 불안정으로 인해 스트림이 끊길 경우, 지수 백오프(Exponential Backoff) 전략을 사용하여 자동으로 재연결을 시도합니다.

### 4. 데이터 송수신 및 AI 추론 서버 (Data Transmission & Inference Server)

분석된 메타데이터와 이벤트 정보는 중앙 AI 추론 서버로 실시간 전송됩니다.

- **EventDispatcher**: 추론 결과(탐지된 객체, 포즈 상태, 위험도 등)를 JSON 포맷으로 구조화합니다.
- **통신 프로토콜**: HTTP POST 요청을 통해 FastAPI 기반의 백엔드 서버 엔드포인트로 데이터를 전송합니다.
- **안정성 확보**: 서버 통신 실패 시, 데이터를 유실하지 않기 위해 재시도 로직이 동작하며, 네트워크 상태에 따라 전송 주기를 조절합니다.

## 학습 및 데이터 전처리 (Training & Preprocessing)

본 프로젝트는 `S13P31E106` 폴더 내의 스크립트를 사용하여 데이터 전처리 및 모델 학습을 수행했습니다.

### 1. 데이터 전처리 (Preprocessing)

AI Hub의 야생동물 데이터셋을 YOLO 학습에 적합한 포맷으로 변환하는 2단계 프로세스를 거칩니다.

#### 1단계: AI Hub → COCO 포맷 변환

- **원본 데이터**: AI Hub에서 제공하는 개별 이미지 파일과 각 이미지별 레이블 JSON 파일
- **변환 과정**:
  - 각 이미지의 레이블 JSON 파일을 파싱하여 COCO 형식의 단일 JSON 파일로 통합합니다.
  - `images`, `annotations`, `categories` 필드를 가진 표준 COCO 포맷을 생성합니다.

#### 2단계: COCO → YOLO 포맷 변환

- **스크립트**: `src/data/convert_aihub_to_yolo.py`
- **기능**:
  - **포맷 변환**: COCO 스타일의 JSON 어노테이션을 파싱하여 YOLO 포맷(`class x_center y_center width height`)의 텍스트 파일로 변환합니다.
  - **데이터 분할**: 전체 데이터를 Train(90%), Validation(10%), Test 셋으로 무작위 분할합니다.
  - **정규화**: 이미지 크기에 맞춰 바운딩 박스 좌표를 0~1 사이 값으로 정규화합니다.
  - **디렉토리 구조화**: `images/`와 `labels/` 폴더 아래에 학습/검증/테스트 셋을 체계적으로 정리하고, 학습 설정 파일인 `data.yaml`을 자동 생성합니다.

### 2. 모델 학습 (Model Training)

#### A. 야생동물 객체 탐지 (YOLO12)

- **스크립트**: `src/cli.py` (Task: `train_detection`)
- **프레임워크**: `ultralytics`
- **절차**:
  - 전처리된 `data.yaml`을 로드합니다.
  - 사전 학습된 `yolo12n.pt` (또는 `yolo11s.pt`) 모델을 기반으로 파인 튜닝(Fine-tuning)을 수행합니다.
  - 학습된 모델은 `best.pt`로 저장되며, 이를 추론 서버의 `Detector`에서 사용합니다.

#### B. 행동 인식 (Action Recognition - TCN)

- **스크립트**: `src/training/train_tcn.py`
- **모델**: `FallTemporalCNN` (Temporal Convolutional Network)
- **절차**:
  - **입력**: 포즈 추정 모델에서 추출한 시계열 키포인트 데이터(Clip).
  - **학습**: 낙상(Fall)과 같은 특정 행동 패턴을 학습합니다. `BalancedBCEWithLogits` 손실 함수를 사용하여 데이터 불균형을 해소하고, AdamW 옵티마이저로 최적화합니다.
  - **결과**: 행동 분류 결과는 위급 상황 판단 로직의 핵심 입력으로 사용됩니다.
