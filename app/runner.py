"""Pipeline entrypoint wiring all skeleton components together."""

from __future__ import annotations

import logging
from typing import Iterable, List

from .config import Settings, get_settings
from .detector import Detector, Detection
from .dispatcher import EventDispatcher
from .event_builder import EventBuilder, EventContext
from .pose_estimator import PoseEstimator
from .stream_listener import StreamListener
from .tracker import Track, Tracker

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

    def process_frame(self, frame) -> List[dict]:
        detections = self.detector.detect(frame)
        tracks = self.tracker.update(frame, detections)
        ctx = EventContext(
            stream_id=self.settings.media_rpi_rtsp_url,
            gpu_enabled=self.settings.gpu_enabled,
        )
        events: List[dict] = []
        for track in tracks:
            detection = track.detection
            if detection.label == "human":
                pose = self.pose_estimator.infer(frame, detection)
                events.append(
                    self.event_builder.build_human_event(ctx, track, frame.timestamp, pose)
                )
            else:
                events.append(
                    self.event_builder.build_wildlife_event(ctx, track, frame.timestamp)
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


def create_pipeline(settings: Settings | None = None) -> InferencePipeline:
    settings = settings or get_settings()
    listener = StreamListener(settings.media_rpi_rtsp_url, fps_limit=settings.default_fps)
    detector = Detector(settings.yolo_model_path)
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
