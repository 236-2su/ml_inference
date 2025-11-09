"""Pipeline entrypoint wiring all components together."""

from __future__ import annotations

import base64
import logging
import time
from typing import Iterable, List, Optional

import numpy as np

try:  # pragma: no cover - runtime dependency
    import cv2
except ImportError as exc:  # pragma: no cover
    raise ImportError(
        "opencv-python-headless is required for snapshot encoding. "
        "Install it via `pip install opencv-python-headless`."
    ) from exc

from .config import Settings, get_settings
from .detector import Detection, Detector
from .dispatcher import EventDispatcher
from .event_builder import EventBuilder, EventContext
from .pose_estimator import PoseEstimator
from .stream_listener import Frame, StreamListener
from .tracker import Tracker

log = logging.getLogger(__name__)


class InferencePipeline:
    def __init__(
        self,
        settings: Settings,
        listener: StreamListener,
        wildlife_detector: Detector,
        human_detector: Optional[Detector],
        tracker: Tracker,
        pose_estimator: PoseEstimator,
        event_builder: EventBuilder,
        dispatcher: EventDispatcher,
    ) -> None:
        self.settings = settings
        self.listener = listener
        self.wildlife_detector = wildlife_detector
        self.human_detector = human_detector
        self.tracker = tracker
        self.pose_estimator = pose_estimator
        self.event_builder = event_builder
        self.dispatcher = dispatcher

    def process_frame(self, frame: Frame) -> List[dict]:
        processing_started = time.perf_counter()
        snapshot_b64 = self._maybe_encode_snapshot(frame)

        detections: List[Detection] = []
        skip_wildlife = False
        if self.human_detector:
            human_detections = self.human_detector.detect(frame)
            if any(
                detection.confidence >= self.settings.human_skip_conf_threshold
                for detection in human_detections
            ):
                skip_wildlife = True
            for detection in human_detections:
                detection.label = "human"
            detections.extend(human_detections)

        if not skip_wildlife:
            wildlife_detections = self.wildlife_detector.detect(frame)
            detections.extend(wildlife_detections)

        tracks = self.tracker.update(frame, detections)
        ctx = EventContext(
            stream_id=self.settings.media_rpi_rtsp_url,
            gpu_enabled=self.settings.gpu_enabled,
        )
        events: List[dict] = []
        for track in tracks:
            detection = track.detection
            latency_ms = int((time.perf_counter() - processing_started) * 1000)
            if detection.label == "human":
                pose = self.pose_estimator.infer(frame, detection)
                events.append(
                    self.event_builder.build_human_event(
                        ctx=ctx,
                        track=track,
                        frame_timestamp=frame.timestamp,
                        pose=pose,
                        inference_latency_ms=latency_ms,
                        snapshot_b64=snapshot_b64,
                    )
                )
            else:
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
        if not self.settings.include_snapshot:
            return None
        if isinstance(frame.image, bytes):
            return base64.b64encode(frame.image).decode("ascii")
        if isinstance(frame.image, np.ndarray):
            success, buffer = cv2.imencode(".jpg", frame.image)
            if not success:
                log.warning("Failed to encode snapshot for frame %s", frame.index)
                return None
            return base64.b64encode(buffer.tobytes()).decode("ascii")
        log.debug("Snapshot capture skipped for frame %s (unsupported type)", frame.index)
        return None


def create_pipeline(settings: Settings | None = None) -> InferencePipeline:
    settings = settings or get_settings()
    listener = StreamListener(settings.media_rpi_rtsp_url, fps_limit=settings.default_fps)
    wildlife_detector = Detector(
        settings.yolo_model_path,
        conf_threshold=settings.yolo_conf_threshold,
        iou_threshold=settings.yolo_iou_threshold,
    )
    human_detector: Optional[Detector] = None
    if settings.yolo_human_model_path:
        human_detector = Detector(
            settings.yolo_human_model_path,
            conf_threshold=settings.yolo_human_conf_threshold,
            iou_threshold=settings.yolo_human_iou_threshold,
            allowed_labels={"human", "person"},
        )
    tracker = Tracker()
    pose_estimator = PoseEstimator(
        model_path=settings.yolo_pose_model_path,
        conf_threshold=settings.yolo_pose_conf_threshold,
    )
    event_builder = EventBuilder(pose_estimator=pose_estimator)
    dispatcher = EventDispatcher(settings)
    return InferencePipeline(
        settings=settings,
        listener=listener,
        wildlife_detector=wildlife_detector,
        human_detector=human_detector,
        tracker=tracker,
        pose_estimator=pose_estimator,
        event_builder=event_builder,
        dispatcher=dispatcher,
    )


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    pipeline = create_pipeline()
    pipeline.run_once(limit=5)
