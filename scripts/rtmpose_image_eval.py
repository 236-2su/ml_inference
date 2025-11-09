from __future__ import annotations

import argparse
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable, List, Sequence, Tuple

import cv2
import numpy as np
import onnxruntime as ort
from ultralytics import YOLO

import sys

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from app.pose_estimator import classify_pose_from_keypoints


MEAN = np.array([123.675, 116.28, 103.53], dtype=np.float32)
STD = np.array([58.395, 57.12, 57.375], dtype=np.float32)
INPUT_SIZE = (256, 192)  # (height, width)
SIMCC_SPLIT_RATIO = 2.0
POSE_MIN_CONF = 0.002


@dataclass
class PoseOutput:
    pose_label: str
    pose_confidence: float
    bbox: Tuple[int, int, int, int]
    detection_conf: float


class RTMPoseRunner:
    def __init__(self, model_path: Path) -> None:
        model_path = model_path.resolve()
        self.session = ort.InferenceSession(model_path.as_posix(), providers=["CPUExecutionProvider"])
        self.input_name = self.session.get_inputs()[0].name
        self.detector = YOLO("ml_inference/models/yolov8x.pt")

    def run_on_image(self, image_path: Path) -> List[PoseOutput]:
        image = cv2.imread(str(image_path))
        if image is None:
            raise FileNotFoundError(f"Failed to load image {image_path}")
        detections = self._detect_persons(image)
        outputs: List[PoseOutput] = []
        for bbox, conf in detections:
            crops, meta = self._crop_and_preprocess(image, bbox)
            preds = self.session.run(None, {self.input_name: crops})  # type: ignore[arg-type]
            keypoints = self._decode_simcc(preds, meta)
            pose_label, pose_conf = classify_pose_from_keypoints(
                keypoints,
                min_conf=POSE_MIN_CONF,
                lying_aspect_ratio=0.65,
                lying_torso_angle_deg=35,
            )
            outputs.append(PoseOutput(pose_label, pose_conf, bbox, conf))
        return outputs

    def _detect_persons(self, image: np.ndarray, min_conf: float = 0.35) -> List[Tuple[Tuple[int, int, int, int], float]]:
        results = self.detector.predict(image, conf=min_conf, classes=[0], verbose=False)
        detections: List[Tuple[Tuple[int, int, int, int], float]] = []
        if not results:
            return detections
        boxes = results[0].boxes
        if boxes is None or boxes.data is None:
            return detections
        for xyxy, conf in zip(boxes.xyxy.cpu().numpy(), boxes.conf.cpu().numpy()):
            x1, y1, x2, y2 = xyxy.astype(int)
            detections.append(((x1, y1, x2, y2), float(conf)))
        return detections

    def _crop_and_preprocess(self, image: np.ndarray, bbox: Tuple[int, int, int, int]) -> Tuple[np.ndarray, dict]:
        h, w = image.shape[:2]
        x1, y1, x2, y2 = bbox
        pad_w = int((x2 - x1) * 0.1)
        pad_h = int((y2 - y1) * 0.1)
        x1 = np.clip(x1 - pad_w, 0, w - 1)
        y1 = np.clip(y1 - pad_h, 0, h - 1)
        x2 = np.clip(x2 + pad_w, 0, w - 1)
        y2 = np.clip(y2 + pad_h, 0, h - 1)
        crop = image[y1:y2, x1:x2]
        resized = cv2.resize(crop, (INPUT_SIZE[1], INPUT_SIZE[0]), interpolation=cv2.INTER_LINEAR)
        tensor = resized.astype(np.float32)
        tensor = (tensor - MEAN) / STD
        tensor = np.transpose(tensor, (2, 0, 1))[None, ...]
        meta = {
            "bbox": (x1, y1, x2, y2),
            "scale": (crop.shape[1] / INPUT_SIZE[1], crop.shape[0] / INPUT_SIZE[0]),
        }
        return tensor, meta

    def _decode_simcc(self, outputs: Sequence[np.ndarray], meta: dict) -> np.ndarray:
        pred_x, pred_y = outputs
        pred_x = np.squeeze(pred_x, axis=0)
        pred_y = np.squeeze(pred_y, axis=0)

        x_probs = self._softmax(pred_x, axis=-1)
        y_probs = self._softmax(pred_y, axis=-1)

        x_idx = np.argmax(x_probs, axis=-1)
        y_idx = np.argmax(y_probs, axis=-1)

        x_conf = np.take_along_axis(x_probs, x_idx[..., None], axis=-1).squeeze(-1)
        y_conf = np.take_along_axis(y_probs, y_idx[..., None], axis=-1).squeeze(-1)
        conf = (x_conf + y_conf) / 2.0

        coords = np.stack(
            [
                x_idx.astype(np.float32) / SIMCC_SPLIT_RATIO,
                y_idx.astype(np.float32) / SIMCC_SPLIT_RATIO,
                conf.astype(np.float32),
            ],
            axis=-1,
        )
        scale_x, scale_y = meta["scale"]
        coords[:, 0] = coords[:, 0] * scale_x + meta["bbox"][0]
        coords[:, 1] = coords[:, 1] * scale_y + meta["bbox"][1]
        return coords

    @staticmethod
    def _softmax(arr: np.ndarray, axis: int = -1) -> np.ndarray:
        arr = arr - np.max(arr, axis=axis, keepdims=True)
        exp = np.exp(arr)
        return exp / np.sum(exp, axis=axis, keepdims=True)


def format_bbox(bbox: Tuple[int, int, int, int]) -> str:
    x1, y1, x2, y2 = bbox
    return f"({x1},{y1})-({x2},{y2})"


def main(image_paths: Iterable[Path], model_path: Path) -> None:
    runner = RTMPoseRunner(model_path)
    for image_path in image_paths:
        print(f"\n[{image_path.name}] {image_path}")
        if not image_path.exists():
            print("  - file missing")
            continue
        outputs = runner.run_on_image(image_path)
        if not outputs:
            print("  - no people detected")
            continue
        for idx, pose in enumerate(outputs, start=1):
            print(
                f"  Person {idx}: bbox={format_bbox(pose.bbox)} "
                f"det_conf={pose.detection_conf:.3f} pose={pose.pose_label} pose_conf={pose.pose_confidence:.3f}"
            )


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Evaluate RTMPose ONNX on sample images.")
    parser.add_argument(
        "--model",
        type=Path,
        default=Path("models/rtmpose_body2d/job_jp0l2r8np_optimized_onnx/model.onnx"),
        help="Path to the RTMPose ONNX model file.",
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
    main(args.images, args.model)
