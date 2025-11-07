"""Send events to FastAPI endpoint with auth and retries."""

from __future__ import annotations

import logging
import time
from typing import Dict, Iterable, List, Optional

import requests

from .config import Settings

log = logging.getLogger(__name__)


class EventDispatcher:
    _BACKOFF_SCHEDULE = (1, 5, 15, 60, 120)

    def __init__(self, settings: Settings, session: Optional[requests.Session] = None) -> None:
        self.settings = settings
        self.session = session or requests.Session()
        self.failed_events: List[Dict[str, object]] = []

    def send_batch(self, events: Iterable[Dict[str, object]]) -> None:
        headers = {"Content-Type": "application/json"}
        if self.settings.fastapi_token:
            headers["Authorization"] = f"Bearer {self.settings.fastapi_token}"
        for event in events:
            self._post_event(event, headers)

    def _post_event(
        self,
        event: Dict[str, object],
        headers: Dict[str, str],
        timeout: float = 5.0,
    ) -> None:
        event_id = event.get("event_id")
        for attempt, backoff in enumerate(self._BACKOFF_SCHEDULE, start=1):
            try:
                response = self.session.post(
                    str(self.settings.fastapi_endpoint),
                    json=event,
                    headers=headers,
                    timeout=timeout,
                )
                if response.status_code >= 500 or response.status_code == 429:
                    raise requests.HTTPError(
                        f"Server error {response.status_code}", response=response
                    )
                if 400 <= response.status_code < 500 and response.status_code not in (429,):
                    log.error(
                        "Dropping event %s due to client error (%s): %s",
                        event_id,
                        response.status_code,
                        response.text,
                    )
                    self.failed_events.append({"event": event, "status": response.status_code})
                    return
                log.info("Event dispatched (id=%s status=%s)", event_id, response.status_code)
                return
            except requests.RequestException as exc:
                is_last_attempt = attempt == len(self._BACKOFF_SCHEDULE)
                log.warning(
                    "Dispatch attempt %s/%s for %s failed: %s",
                    attempt,
                    len(self._BACKOFF_SCHEDULE),
                    event_id,
                    exc,
                )
                if is_last_attempt:
                    log.error("Giving up on event %s after retries", event_id)
                    self.failed_events.append({"event": event, "error": str(exc)})
                    return
                time.sleep(backoff)
