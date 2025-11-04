"""Send events to FastAPI endpoint."""

from __future__ import annotations

import logging
from typing import Dict, Iterable, Optional

import requests

from .config import Settings

log = logging.getLogger(__name__)


class EventDispatcher:
    def __init__(self, settings: Settings) -> None:
        self.settings = settings

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
        try:
            response = requests.post(
                str(self.settings.fastapi_endpoint),
                json=event,
                headers=headers,
                timeout=timeout,
            )
            response.raise_for_status()
            log.info("Event dispatched (id=%s)", event.get("event_id"))
        except requests.RequestException as exc:
            log.error("Failed to dispatch event: %s", exc, exc_info=True)
            # In a real implementation, buffer for retry or push to a queue.
