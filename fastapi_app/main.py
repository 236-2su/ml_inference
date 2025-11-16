"\"\"\"Minimal FastAPI app to receive inference events.\"\"\""

from __future__ import annotations

from typing import Any, Dict, List, Optional

from fastapi import FastAPI, Header, HTTPException, status
from pydantic import BaseModel

app = FastAPI(title="Inference Event Receiver", version="0.1.0")


class EventPayload(BaseModel):
    event_id: str
    category: str
    stream_id: str
    track_id: int
    timestamp_utc: str
    bbox: List[int]
    confidence: Optional[float] = None
    pose_label: Optional[str] = None
    pose_confidence: Optional[float] = None
    keypoints: Optional[List[List[float]]] = None
    species: Optional[str] = None
    species_confidence: Optional[float] = None
    threat_level: Optional[str] = None
    gpu_enabled: bool = False


_EVENT_STORE: List[Dict[str, Any]] = []


@app.get("/health")
def health() -> Dict[str, str]:
    return {"status": "ok"}


@app.post("/events", status_code=status.HTTP_202_ACCEPTED)
def ingest_event(
    payload: EventPayload,
) -> Dict[str, str]:
    # 백엔드에서 permit 열어놨으므로 인증 체크 제거
    _EVENT_STORE.append(payload.dict())
    return {"status": "queued", "event_id": payload.event_id}
