"""YOLO-based detector built with ultralytics."""

from __future__ import annotations

import logging
import time
from dataclasses import dataclass
from typing import List, Optional, Set

import numpy as np

try:
    import cv2  # type: ignore
except ImportError as exc:  # pragma: no cover - runtime dependency
    raise ImportError(
        "opencv-python-headless is required for the detector. "
        "Install it via `pip install opencv-python-headless`."
    ) from exc

try:
    from ultralytics import YOLO  # type: ignore
except ImportError:  # pragma: no cover - optional at test time
    YOLO = None  # type: ignore

from .stream_listener import Frame

log = logging.getLogger(__name__)


@dataclass
class Detection:
    label: str
    confidence: float
    bbox: tuple[int, int, int, int]  # x, y, w, h


class Detector:
    def __init__(
        self,
        model_path: str,
        conf_threshold: float,
        iou_threshold: float,
        allowed_labels: Optional[Set[str]] = None,
        device: Optional[str] = None,
    ) -> None:
        self.model_path = model_path
        self.conf_threshold = conf_threshold
        self.iou_threshold = iou_threshold
        self.allowed_labels = {label.lower() for label in allowed_labels} if allowed_labels else None
        if YOLO is None:  # pragma: no cover - exercised in unit tests
            raise ImportError(
                "ultralytics package is required for the detector. "
                "Install it via `pip install ultralytics`."
            )

        # Auto-detect device if not specified
        if device is None:
            try:
                import torch
                device = "cuda:0" if torch.cuda.is_available() else "cpu"
            except ImportError:
                device = "cpu"

        self.device = device
        self.model = YOLO(model_path)
        # Move model to GPU if available
        if hasattr(self.model, 'to'):
            self.model.to(device)

        self.label_map = {int(k): v for k, v in self.model.names.items()}
        log.info(
            "Loaded YOLO model %s with %s classes on device: %s",
            model_path,
            len(self.label_map),
            device,
        )

    def detect(self, frame: Frame) -> List[Detection]:
        """Run YOLO inference on the provided frame."""
        inference_start = time.perf_counter()

        image = self._prepare_image(frame)
        if image is None:
            return []

        # Time the actual model inference
        predict_start = time.perf_counter()
        results = self.model.predict(
            source=image,
            conf=self.conf_threshold,
            iou=self.iou_threshold,
            verbose=False,
            device=self.device,
        )
        predict_time_ms = (time.perf_counter() - predict_start) * 1000

        if not results:
            log.debug("YOLO inference took %.2f ms (no detections)", predict_time_ms)
            return []

        detections: List[Detection] = []
        first = results[0]
        boxes = getattr(first, "boxes", None)
        if not boxes:
            log.debug("YOLO inference took %.2f ms (no boxes)", predict_time_ms)
            return detections

        for box in boxes:
            cls_idx = int(box.cls.item())
            label = self.label_map.get(cls_idx, str(cls_idx))
            if self.allowed_labels and label.lower() not in self.allowed_labels:
                continue
            conf = float(box.conf.item())
            xywh = box.xywh[0].tolist()
            x, y, w, h = (int(value) for value in xywh)
            detections.append(Detection(label=label, confidence=conf, bbox=(x, y, w, h)))

        total_time_ms = (time.perf_counter() - inference_start) * 1000
        log.info(
            "YOLO detection complete: model=%s, inference_time=%.2f ms, total_time=%.2f ms, detections=%d",
            self.model_path.split('/')[-1],
            predict_time_ms,
            total_time_ms,
            len(detections)
        )

        return detections

    def _prepare_image(self, frame: Frame) -> Optional[np.ndarray]:
        """Decode frame payload into an array YOLO understands."""
        if frame.image is None:
            log.debug("Frame %s has no image payload; skipping detection", frame.index)
            return None

        if isinstance(frame.image, np.ndarray):
            return frame.image

        if isinstance(frame.image, bytes):
            np_buffer = np.frombuffer(frame.image, dtype=np.uint8)
            image = cv2.imdecode(np_buffer, cv2.IMREAD_COLOR)
            if image is None:
                log.warning("Unable to decode frame %s; skipping", frame.index)
            return image

        log.warning("Unsupported frame image type %s", type(frame.image))
        return None
