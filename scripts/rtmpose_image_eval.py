from __future__ import annotations

import argparse
from datetime import datetime, timezone
from pathlib import Path
from typing import Iterable, List

import cv2
from ultralytics import YOLO

import sys

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from app.detector import Detection
from app.pose_estimator import PoseEstimator
from app.stream_listener import Frame


def format_bbox_xyxy(bbox: tuple[int, int, int, int]) -> str:
    x1, y1, x2, y2 = bbox
    return f"({x1},{y1})-({x2},{y2})"


def detect_people(image, detector: YOLO, min_conf: float) -> List[Detection]:
    results = detector.predict(image, conf=min_conf, classes=[0], verbose=False)
    detections: List[Detection] = []
    if not results:
        return detections
    boxes = results[0].boxes
    if boxes is None or boxes.data is None:
        return detections
    for xyxy, conf in zip(boxes.xyxy.cpu().numpy(), boxes.conf.cpu().numpy()):
        x1, y1, x2, y2 = xyxy.astype(int)
        bbox = (x1, y1, x2 - x1, y2 - y1)
        detections.append(Detection(label="human", confidence=float(conf), bbox=bbox))
    return detections


def main(image_paths: Iterable[Path], model_path: Path, yolo_model: Path, min_conf: float) -> None:
    pose_estimator = PoseEstimator(
        model_path=model_path.as_posix(),
        keypoint_conf_threshold=0.002,
        lying_aspect_ratio=0.65,
        lying_torso_angle_deg=35.0,
    )
    detector = YOLO(yolo_model.as_posix())
    for idx, image_path in enumerate(image_paths):
        print(f"\n[{image_path.name}] {image_path}")
        image = cv2.imread(image_path.as_posix())
        if image is None:
            print("  - file missing")
            continue
        detections = detect_people(image, detector, min_conf=min_conf)
        if not detections:
            print("  - no people detected")
            continue
        frame = Frame(index=idx, timestamp=datetime.now(tz=timezone.utc), image=image)
        for det_idx, detection in enumerate(detections, start=1):
            pose = pose_estimator.infer(frame, detection)
            x, y, w, h = detection.bbox
            bbox_xyxy = (x, y, x + w, y + h)
            print(
                f"  Person {det_idx}: bbox={format_bbox_xyxy(bbox_xyxy)} det_conf={detection.confidence:.3f} "
                f"pose={pose.label} pose_conf={pose.confidence:.3f}"
            )


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Evaluate the PoseEstimator on sample images.")
    parser.add_argument(
        "--model",
        type=Path,
        default=Path("models/rtmpose_body2d/job_jp0l2r8np_optimized_onnx/model.onnx"),
        help="Path to the RTMPose ONNX model file.",
    )
    parser.add_argument(
        "--yolo-model",
        type=Path,
        default=Path("models/yolov8x.pt"),
        help="YOLO weights used to find people before pose estimation.",
    )
    parser.add_argument(
        "--conf",
        type=float,
        default=0.35,
        help="Minimum confidence for YOLO person detections.",
    )
    parser.add_argument(
        "images",
        nargs="*",
        type=Path,
        default=[
            Path("agriculture.jpg"),
            Path("lying.jpg"),
            Path("pose_test1.jpg"),
            Path("pose_test2.jpg"),
            Path("test_test.jpg"),
        ],
    )
    args = parser.parse_args()
    main(args.images, args.model, args.yolo_model, args.conf)
