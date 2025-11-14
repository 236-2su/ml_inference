"""이벤트 필터링: 변화가 있을 때만 전송"""

from dataclasses import dataclass
from typing import Dict, Optional


@dataclass
class TrackState:
    """각 track의 이전 상태 저장"""
    category: str = "unknown"  # human or wildlife
    pose_label: str = "unknown"  # sitting/standing/lying (human only)
    species: str = "unknown"  # 동물 종류 (wildlife only)
    bbox: tuple = (0, 0, 0, 0)
    is_present: bool = False


class EventFilter:
    """변화가 있을 때만 이벤트 전송하도록 필터링"""

    def __init__(
        self,
        enable_pose_change: bool = True,
        enable_presence_change: bool = True,
        enable_important_status: bool = True,
        enable_position_change: bool = False,
        position_threshold: int = 100,
    ):
        """
        Args:
            enable_pose_change: 자세 변화 감지 (sitting→standing)
            enable_presence_change: 등장/사라짐 감지
            enable_important_status: 중요 상태 항상 전송 (fall, heatstroke)
            enable_position_change: 위치 변화 감지
            position_threshold: 위치 변화 임계값 (픽셀)
        """
        self.enable_pose_change = enable_pose_change
        self.enable_presence_change = enable_presence_change
        self.enable_important_status = enable_important_status
        self.enable_position_change = enable_position_change
        self.position_threshold = position_threshold

        self._track_states: Dict[int, TrackState] = {}

    def should_send_event(self, event: dict) -> bool:
        """이벤트를 전송해야 하는지 판단

        Args:
            event: 이벤트 딕셔너리

        Returns:
            True: 전송해야 함 (변화 있음)
            False: 전송 안 함 (변화 없음)
        """
        track_id = event.get("track_id")
        if track_id is None:
            return True  # track_id 없으면 항상 전송

        # 1. 중요 상태는 항상 전송
        if self.enable_important_status:
            status = event.get("status")
            if status in ["fall_detected", "heatstroke_watch", "heatstroke_alert"]:
                return True

        # 이전 상태 가져오기
        prev_state = self._track_states.get(track_id)

        # 2. 새로 등장한 사람 (첫 감지)
        if prev_state is None:
            if self.enable_presence_change:
                self._update_state(track_id, event)
                return True  # 첫 등장 → 전송
            else:
                self._update_state(track_id, event)
                return False

        # 3. 자세 변화 감지
        if self.enable_pose_change:
            current_pose = event.get("pose_label", "unknown")
            if current_pose != prev_state.pose_label:
                self._update_state(track_id, event)
                return True  # 자세 변화 → 전송

        # 4. 위치 변화 감지 (선택적)
        if self.enable_position_change:
            current_bbox = event.get("bbox", (0, 0, 0, 0))
            if self._has_position_changed(prev_state.bbox, current_bbox):
                self._update_state(track_id, event)
                return True  # 위치 크게 이동 → 전송

        # 상태 업데이트 (전송은 안 함)
        self._update_state(track_id, event)
        return False  # 변화 없음 → 전송 안 함

    def mark_track_disappeared(self, track_id: int) -> bool:
        """사람이 사라졌을 때 호출

        Args:
            track_id: 사라진 track ID

        Returns:
            True: 사라짐 이벤트 전송해야 함
            False: 전송 안 함
        """
        if not self.enable_presence_change:
            return False

        if track_id in self._track_states:
            del self._track_states[track_id]
            return True  # 사라짐 → 전송

        return False

    def _update_state(self, track_id: int, event: dict):
        """track 상태 업데이트"""
        self._track_states[track_id] = TrackState(
            pose_label=event.get("pose_label", "unknown"),
            bbox=event.get("bbox", (0, 0, 0, 0)),
            is_present=True,
        )

    def _has_position_changed(
        self, prev_bbox: tuple, current_bbox: tuple
    ) -> bool:
        """위치가 크게 변했는지 확인"""
        if not prev_bbox or not current_bbox:
            return False

        # 중심점 계산
        prev_x = prev_bbox[0] + prev_bbox[2] / 2
        prev_y = prev_bbox[1] + prev_bbox[3] / 2
        curr_x = current_bbox[0] + current_bbox[2] / 2
        curr_y = current_bbox[1] + current_bbox[3] / 2

        # 유클리드 거리
        distance = ((curr_x - prev_x) ** 2 + (curr_y - prev_y) ** 2) ** 0.5

        return distance > self.position_threshold

    def get_stats(self) -> dict:
        """필터링 통계 반환"""
        return {
            "active_tracks": len(self._track_states),
            "tracked_people": list(self._track_states.keys()),
        }
