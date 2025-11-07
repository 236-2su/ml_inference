"""Pipeline entrypoint wiring all components together."""

from __future__ import annotations

import base64
import logging
import time
from typing import Iterable, List, Optional

from .config import Settings, get_settings
from .detector import Detector
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
        detector: Detector,
        tracker: Tracker,
        pose_estimator: PoseEstimator,
        event_builder: EventBuilder,
        dispatcher: EventDispatcher,
    ) -> None:
        self.settings = settings
        self.listener = listener
        self.detector = detector
        self.tracker = tracker
        self.pose_estimator = pose_estimator
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
    pose_estimator = PoseEstimator()
    event_builder = EventBuilder(pose_estimator=pose_estimator)
    dispatcher = EventDispatcher(settings)
    return InferencePipeline(
        settings=settings,
        listener=listener,
        detector=detector,
        tracker=tracker,
        pose_estimator=pose_estimator,
        event_builder=event_builder,
        dispatcher=dispatcher,
    )


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    pipeline = create_pipeline()
    pipeline.run_once(limit=5)
