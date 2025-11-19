# AI Project Presentation Instructions

This document provides a step-by-step guide to creating the remaining slides for your AI project presentation. Based on the technical architecture and codebase analysis, here are the recommended slides to follow your existing "Resource Data" and "Preprocessing" slides.

---

## Slide 3: Wildlife Detection Model Training (YOLO12)

**Goal**: Explain how the object detection model was trained.

* **Title**: Wildlife Object Detection with YOLO12
* **Key Content**:
  * **Model Selection**: Used **YOLO12** (Fine-tuned from `yolo12n.pt` or `yolo11s.pt`).
  * **Framework**: `Ultralytics` library.
  * **Training Process**:
    * Input: Preprocessed YOLO format data (`data.yaml`).
    * Task: `train_detection` via `src/cli.py`.
    * Optimization: Fine-tuning pre-trained weights for specific wildlife classes (Boar, Elk, etc.).
* **Visual Suggestions**:
  * Screenshot of the training log or loss curves (if available from TensorBoard/MLflow).
  * Example image showing a bounding box detection with confidence score.
* **Speaker Notes**: "We chose YOLO12 for its balance of speed and accuracy. We fine-tuned a pre-trained model using our converted dataset to specifically recognize wildlife in our target environment."

---

## Slide 4: Human Pose Estimation Strategy

**Goal**: Detail the hybrid approach for detecting human actions.

* **Title**: Hybrid Pose Estimation: MediaPipe + SVM
* **Key Content**:
  * **Why Hybrid?**: Lightweight and fast compared to heavy deep learning models for simple state classification.
  * **Step 1: Feature Extraction**:
    * Used **MediaPipe Pose** to extract **33 Body Landmarks**.
    * Keypoints are normalized to (0~1) range.
  * **Step 2: Classification (SVM)**:
    * **Input**: Flattened **66-dimensional vector** (33 points × x, y coordinates).
    * **Model**: Support Vector Machine (SVM) with a linear/RBF kernel.
    * **Classes**: Standing, Sitting, Lying.
* **Visual Suggestions**:
  * Diagram: `Input Image` -> `MediaPipe Skeleton` -> `[x1, y1, ... x33, y33]` -> `SVM` -> `Class`.
  * Code snippet showing the 66-dim vector construction.
* **Speaker Notes**: "Instead of using a heavy end-to-end model, we used a hybrid approach. MediaPipe gives us the skeleton, and a lightweight SVM classifies the posture based on the geometry of these points."

---

## Slide 5: Advanced Action Recognition (Fall Detection)

**Goal**: Explain how critical events like falling are detected over time.

* **Title**: Temporal Action Recognition (TCN)
* **Key Content**:
  * **Problem**: Static pose isn't enough to detect a "fall" (it's a sequence of movements).
  * **Solution**: **Temporal Convolutional Network (TCN)**.
  * **Input**: A sequence (clip) of pose keypoints over time.
  * **Training**:
    * Script: `src/training/train_tcn.py`.
    * Loss Function: `BalancedBCEWithLogits` (handles class imbalance).
  * **Output**: Probability of "Fall" event.
* **Visual Suggestions**:
  * A timeline graphic showing a person transitioning from standing to lying.
  * Architecture diagram of a simple TCN (1D Convolutions over time).
* **Speaker Notes**: "To distinguish between simply 'lying down' and 'falling', we analyze the temporal sequence of poses using a TCN. This allows us to capture the sudden change in posture characteristic of a fall."

---

## Slide 6: Real-time Inference Architecture

**Goal**: Show how the system runs live.

* **Title**: Real-time Inference Pipeline
* **Key Content**:
  * **Streaming**: RTSP Stream ingestion via OpenCV (`StreamListener`).
    * *Optimization*: Buffer size set to 1 to minimize latency.
  * **Pipeline Steps**:
    1. **Frame Capture**: Get latest image.
    2. **Detection**: YOLO12 scans for animals.
    3. **Pose**: If human detected -> MediaPipe + SVM.
    4. **Event Logic**: Check conditions (e.g., "Lying down > 5 mins").
* **Visual Suggestions**:
  * Flowchart: `Camera` -> `StreamListener` -> `Detector/PoseEstimator` -> `EventBuilder` -> `Dispatcher`.
* **Speaker Notes**: "Real-time performance is critical. We optimized the video buffer to ensure we are always processing the latest frame, minimizing the delay between an event happening and it being detected."

---

## Slide 7: System Integration & Alerting

**Goal**: Explain how the AI talks to the backend.

* **Title**: Data Transmission & Server Integration
* **Key Content**:
  * **Event Dispatcher**:
    * Formats results into JSON.
    * Sends via **HTTP POST** to FastAPI backend.
  * **Reliability**:
    * Implements **Exponential Backoff** retry logic.
    * Ensures data isn't lost during network blips.
* **Visual Suggestions**:
  * JSON snippet example (showing `class_id`, `confidence`, `timestamp`).
  * Icon diagram: `Edge Device (AI)` --(HTTP)--> `Cloud Server`.
* **Speaker Notes**: "The AI doesn't just detect; it communicates. We built a robust dispatcher that sends structured JSON data to our backend, ensuring that every critical alert reaches the server even if the network fluctuates."

---

## Slide 8: Conclusion

**Goal**: Wrap up the project impact.

* **Title**: Project Impact & Future Work
* **Key Content**:
  * **Summary**: A comprehensive safety system for both wildlife and humans.
  * **Key Tech**: YOLO12 (SOTA Detection), Hybrid Pose (Efficiency), TCN (Temporal Context).
  * **Future Improvements**:
    * Deploying on edge devices (Jetson Nano/Orin).
    * Improving night-time detection with thermal imagery.
