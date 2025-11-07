"""YOLO-based detector built with ultralytics."""

from __future__ import annotations

import logging
from dataclasses import dataclass
from typing import List, Optional

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
except ImportError as exc:  # pragma: no cover - runtime dependency
    raise ImportError(
        "ultralytics package is required for the detector. "
        "Install it via `pip install ultralytics`."
    ) from exc

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
    ) -> None:
        self.model_path = model_path
        self.conf_threshold = conf_threshold
        self.iou_threshold = iou_threshold
        self.model = YOLO(model_path)
        self.label_map = {int(k): v for k, v in self.model.names.items()}
        log.info(
            "Loaded YOLO model %s with %s classes",
            model_path,
            len(self.label_map),
        )

    def detect(self, frame: Frame) -> List[Detection]:
        """Run YOLO inference on the provided frame."""
        image = self._prepare_image(frame)
        if image is None:
            return []

        results = self.model.predict(
            source=image,
            conf=self.conf_threshold,
            iou=self.iou_threshold,
            verbose=False,
        )
        if not results:
            return []

        detections: List[Detection] = []
        first = results[0]
        boxes = getattr(first, "boxes", None)
        if not boxes:
            return detections

        for box in boxes:
            cls_idx = int(box.cls.item())
            label = self.label_map.get(cls_idx, str(cls_idx))
            conf = float(box.conf.item())
            xywh = box.xywh[0].tolist()
            x, y, w, h = (int(value) for value in xywh)
            detections.append(Detection(label=label, confidence=conf, bbox=(x, y, w, h)))
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
