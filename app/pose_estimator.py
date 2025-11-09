"""RTMPose ONNX-based pose estimator that runs inference and delegates labeling."""

from __future__ import annotations

import logging
from dataclasses import dataclass
from typing import List, Optional, Sequence

import numpy as np

from .detector import Detection
from .pose_classifier import PoseClassifierConfig, classify_pose_from_keypoints, label_pose
from .stream_listener import Frame

try:  # pragma: no cover - runtime dependency
    import cv2  # type: ignore
except ImportError:  # pragma: no cover
    cv2 = None  # type: ignore

try:  # pragma: no cover - runtime dependency
    import onnxruntime as ort  # type: ignore
except ImportError:  # pragma: no cover
    ort = None  # type: ignore

log = logging.getLogger(__name__)

INPUT_SIZE = (256, 192)  # (height, width)
MEAN = np.array([123.675, 116.28, 103.53], dtype=np.float32)
STD = np.array([58.395, 57.12, 57.375], dtype=np.float32)
SIMCC_SPLIT_RATIO = 2.0
PAD_RATIO = 0.1

default_keypoints: List[tuple[float, float, float]] = []


@dataclass
class PoseResult:
    label: str
    confidence: float
    keypoints: List[tuple[float, float, float]]




class PoseEstimator:
    """RTMPose-backed estimator that annotates human detections with pose labels."""

    def __init__(
        self,
        model_path: str | None = None,
        conf_threshold: float = 0.35,
        keypoint_conf_threshold: float = 0.3,
        lying_aspect_ratio: float = 0.65,
        lying_torso_angle_deg: float = 35.0,
    ) -> None:
        self.model_path = model_path
        self.conf_threshold = conf_threshold
        self.classifier_config = PoseClassifierConfig(
            min_keypoint_conf=keypoint_conf_threshold,
            lying_aspect_ratio=lying_aspect_ratio,
            lying_torso_angle_deg=lying_torso_angle_deg,
        )
        self._session: Optional["ort.InferenceSession"] = None
        self._input_name: Optional[str] = None
        self._load_model()

    def infer(self, frame: Frame, detection: Detection) -> PoseResult:
        if detection.confidence < self.conf_threshold:
            return PoseResult(label="unknown", confidence=0.0, keypoints=default_keypoints)
        if not self._session or not self._input_name:
            return PoseResult(label="unknown", confidence=0.0, keypoints=default_keypoints)
        image = self._frame_to_ndarray(frame)
        if image is None or cv2 is None:
            return PoseResult(label="unknown", confidence=0.0, keypoints=default_keypoints)

        prepared = self._prepare_input(image, detection.bbox)
        if prepared is None:
            return PoseResult(label="unknown", confidence=0.0, keypoints=default_keypoints)
        tensor, meta = prepared
        outputs = self._session.run(None, {self._input_name: tensor})  # type: ignore[arg-type]
        keypoints = self._decode_simcc(outputs, meta)
        if keypoints.size == 0:
            return PoseResult(label="unknown", confidence=0.0, keypoints=default_keypoints)

        label, score = classify_pose_from_keypoints(keypoints, config=self.classifier_config)
        keypoints_list = [(float(k[0]), float(k[1]), float(k[2])) for k in keypoints]
        return PoseResult(label=label, confidence=score, keypoints=keypoints_list)

    def _load_model(self) -> None:
        if not self.model_path:
            log.info("Pose estimator disabled (model path missing).")
            return
        if ort is None:
            log.warning("onnxruntime not installed; pose estimator disabled.")
            return
        try:
            session = ort.InferenceSession(self.model_path, providers=["CPUExecutionProvider"])
            self._session = session
            self._input_name = session.get_inputs()[0].name
            log.info("Loaded RTMPose model %s", self.model_path)
        except FileNotFoundError:
            log.warning("Pose model %s not found; pose estimator disabled.", self.model_path)
        except Exception:  # pragma: no cover - diagnostics only
            log.exception("Failed to load pose model %s", self.model_path)

    def _frame_to_ndarray(self, frame: Frame) -> Optional[np.ndarray]:
        if frame.image is None:
            return None
        if isinstance(frame.image, np.ndarray):
            return frame.image
        if isinstance(frame.image, bytes):
            if cv2 is None:
                return None
            np_buffer = np.frombuffer(frame.image, dtype=np.uint8)
            decoded = cv2.imdecode(np_buffer, cv2.IMREAD_COLOR)
            return decoded
        return None

    def _prepare_input(
        self, image: np.ndarray, bbox: tuple[int, int, int, int]
    ) -> Optional[tuple[np.ndarray, dict]]:
        h, w = image.shape[:2]
        x, y, bw, bh = bbox
        pad_w = int(bw * PAD_RATIO)
        pad_h = int(bh * PAD_RATIO)
        x1 = int(np.clip(x - pad_w, 0, w - 1))
        y1 = int(np.clip(y - pad_h, 0, h - 1))
        x2 = int(np.clip(x + bw + pad_w, 0, w - 1))
        y2 = int(np.clip(y + bh + pad_h, 0, h - 1))
        if x2 <= x1 or y2 <= y1:
            return None
        crop = image[y1:y2, x1:x2]
        if crop.size == 0:
            return None
        resized = cv2.resize(crop, (INPUT_SIZE[1], INPUT_SIZE[0]), interpolation=cv2.INTER_LINEAR)
        tensor = resized.astype(np.float32)
        tensor = (tensor - MEAN) / STD
        tensor = np.transpose(tensor, (2, 0, 1))[None, ...]
        meta = {
            "bbox": (x1, y1, x2, y2),
            "scale_x": crop.shape[1] / INPUT_SIZE[1],
            "scale_y": crop.shape[0] / INPUT_SIZE[0],
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
        coords[:, 0] = coords[:, 0] * meta["scale_x"] + meta["bbox"][0]
        coords[:, 1] = coords[:, 1] * meta["scale_y"] + meta["bbox"][1]
        return coords

    @staticmethod
    def _softmax(arr: np.ndarray, axis: int = -1) -> np.ndarray:
        arr = arr - np.max(arr, axis=axis, keepdims=True)
        exp = np.exp(arr)
        return exp / np.sum(exp, axis=axis, keepdims=True)
