from app.event_filter import EventFilter


def _base_event(track_id: int = 1, category: str = "human", pose_label: str = "standing"):
    return {
        "track_id": track_id,
        "category": category,
        "pose_label": pose_label,
        "bbox": (0, 0, 10, 10),
    }


def test_presence_change_reports_first_event():
    event_filter = EventFilter()
    assert event_filter.should_send_event(_base_event()) is True
    assert event_filter.should_send_event(_base_event()) is False


def test_pose_change_detected_even_without_presence_events():
    event_filter = EventFilter(enable_presence_change=False)
    assert event_filter.should_send_event(_base_event()) is False
    changed = _base_event(pose_label="lying")
    assert event_filter.should_send_event(changed) is True


def test_wildlife_species_change_is_forwarded():
    event_filter = EventFilter(enable_presence_change=False)
    fox = _base_event(category="wildlife")
    fox["species"] = "fox"
    hog = _base_event(category="wildlife")
    hog["species"] = "hog"
    assert event_filter.should_send_event(fox) is False
    assert event_filter.should_send_event(fox) is False
    assert event_filter.should_send_event(hog) is True


def test_status_events_always_pass():
    event_filter = EventFilter(enable_presence_change=False)
    fall_event = _base_event()
    fall_event["status"] = "fall_detected"
    assert event_filter.should_send_event(fall_event) is True


def test_position_change_threshold():
    event_filter = EventFilter(enable_position_change=True, position_threshold=50)
    assert event_filter.should_send_event(_base_event()) is True
    no_move = _base_event()
    no_move["bbox"] = (5, 5, 10, 10)
    assert event_filter.should_send_event(no_move) is False
    moved = _base_event()
    moved["bbox"] = (500, 500, 10, 10)
    assert event_filter.should_send_event(moved) is True


def test_prune_drops_state():
    event_filter = EventFilter()
    assert event_filter.should_send_event(_base_event(track_id=7)) is True
    event_filter.prune([])
    assert event_filter.should_send_event(_base_event(track_id=7)) is True
