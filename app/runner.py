"""Pipeline entrypoint wiring all components together."""

from __future__ import annotations

import base64
import logging
import time
from typing import List, Optional

import numpy as np

try:
    import cv2
except ImportError as exc:  # pragma: no cover - runtime dependency
    raise ImportError(
        "opencv-python-headless is required for snapshot support. "
        "Install it via `pip install opencv-python-headless`."
    ) from exc

from .config import Settings, get_settings
from .detector import Detector
from .dispatcher import EventDispatcher
from .event_builder import EventBuilder, EventContext
from .stream_listener import Frame, StreamListener
from .tracker import Tracker

log = logging.getLogger(__name__)


class InferencePipeline:
    def __init__(
        self,
        settings: Settings,
        listener: StreamListener,
        detector: Detector,
        tracker: Tracker,
        event_builder: EventBuilder,
        dispatcher: EventDispatcher,
    ) -> None:
        self.settings = settings
        self.listener = listener
        self.detector = detector
        self.tracker = tracker
        self.event_builder = event_builder
        self.dispatcher = dispatcher

    def process_frame(self, frame: Frame) -> List[dict]:
        processing_started = time.perf_counter()
        snapshot_b64 = self._maybe_encode_snapshot(frame)

        detections = self.detector.detect(frame)
        tracks = self.tracker.update(frame, detections)
        ctx = EventContext(
            stream_id=self.settings.media_rpi_rtsp_url,
            gpu_enabled=self.settings.gpu_enabled,
        )
        events: List[dict] = []
        for track in tracks:
            latency_ms = int((time.perf_counter() - processing_started) * 1000)
            events.append(
                self.event_builder.build_wildlife_event(
                    ctx=ctx,
                    track=track,
                    frame_timestamp=frame.timestamp,
                    inference_latency_ms=latency_ms,
                    snapshot_b64=snapshot_b64,
                )
            )
        return events

    def run_once(self, limit: int = 10) -> None:
        for frame in self.listener.once(limit=limit):
            events = self.process_frame(frame)
            if events:
                self.dispatcher.send_batch(events)

    def run_forever(self) -> None:
        for frame in self.listener.frames():
            events = self.process_frame(frame)
            if events:
                self.dispatcher.send_batch(events)

    def _maybe_encode_snapshot(self, frame: Frame) -> Optional[str]:
        if not self.settings.include_snapshot or frame.image is None:
            return None
        if isinstance(frame.image, bytes):
            return base64.b64encode(frame.image).decode("ascii")
        if isinstance(frame.image, np.ndarray):
            success, buffer = cv2.imencode(".jpg", frame.image)
            if not success:
                log.warning("Snapshot encoding failed for frame %s", frame.index)
                return None
            return base64.b64encode(buffer.tobytes()).decode("ascii")
        log.debug("Snapshot capture skipped for frame %s (unsupported type)", frame.index)
        return None


def create_pipeline(settings: Settings | None = None) -> InferencePipeline:
    settings = settings or get_settings()
    listener = StreamListener(settings.media_rpi_rtsp_url, fps_limit=settings.default_fps)
    detector = Detector(
        settings.yolo_model_path,
        conf_threshold=settings.yolo_conf_threshold,
        iou_threshold=settings.yolo_iou_threshold,
    )
    tracker = Tracker()
    event_builder = EventBuilder()
    dispatcher = EventDispatcher(settings)
    return InferencePipeline(
        settings=settings,
        listener=listener,
        detector=detector,
        tracker=tracker,
        event_builder=event_builder,
        dispatcher=dispatcher,
    )


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    pipeline = create_pipeline()
    pipeline.run_once(limit=5)
