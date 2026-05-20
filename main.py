import os
import time
import uuid
from typing import Any, Dict, List, Optional

import httpx
from fastapi import FastAPI, HTTPException, Request
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse, StreamingResponse
from pydantic import BaseModel, Field

from baseline import baseline_reply, extract_text
from bdi import BDIState
from embeddings import embed_texts, novelty_score
from pulse_feed import PulseEvent, PulseFeed
from refusal import detect_refusal

APP_NAME = "pulse-proxy"
UPSTREAM_CHAT_URL = os.getenv("PULSE_UPSTREAM_CHAT_URL", "https://api.openai.com/v1/chat/completions")
UPSTREAM_API_KEY = os.getenv("OPENAI_API_KEY") or os.getenv("PULSE_UPSTREAM_API_KEY")
UPSTREAM_TIMEOUT = float(os.getenv("PULSE_UPSTREAM_TIMEOUT", "120"))

app = FastAPI(title=APP_NAME, version="0.1.0")
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

feed = PulseFeed(max_events=int(os.getenv("PULSE_FEED_LIMIT", "500")))
bdi_state = BDIState()


class ChatProxyRequest(BaseModel):
    model: str = Field(default="gpt-4o-mini")
    messages: List[Dict[str, Any]] = Field(default_factory=list)
    temperature: Optional[float] = None
    stream: bool = False

    class Config:
        extra = "allow"


def _auth_headers(request: Request) -> Dict[str, str]:
    headers = {"content-type": "application/json"}
    incoming_auth = request.headers.get("authorization")
    if incoming_auth:
        headers["authorization"] = incoming_auth
    elif UPSTREAM_API_KEY:
        headers["authorization"] = f"Bearer {UPSTREAM_API_KEY}"
    return headers


def _messages_text(messages: List[Dict[str, Any]]) -> str:
    parts: List[str] = []
    for message in messages:
        role = message.get("role", "unknown")
        parts.append(f"{role}: {extract_text(message.get('content', ''))}")
    return "\n".join(parts).strip()


def _record_input(payload: ChatProxyRequest) -> Dict[str, Any]:
    prompt_text = _messages_text(payload.messages)
    vectors = embed_texts([prompt_text]) if prompt_text else []
    novelty = novelty_score(prompt_text, feed.recent_texts(limit=25)) if prompt_text else 0.0
    bdi = bdi_state.observe(payload.messages)
    pulse = {
        "request_id": str(uuid.uuid4()),
        "model": payload.model,
        "prompt_chars": len(prompt_text),
        "embedding_dim": len(vectors[0]) if vectors else 0,
        "novelty": novelty,
        "bdi": bdi,
    }
    feed.add(PulseEvent(kind="input", text=prompt_text, data=pulse))
    return pulse


def _record_output(request_id: str, response_json: Dict[str, Any]) -> None:
    text = ""
    choices = response_json.get("choices") or []
    if choices:
        message = choices[0].get("message") or {}
        text = extract_text(message.get("content", ""))
    refusal = detect_refusal(text)
    feed.add(
        PulseEvent(
            kind="output",
            text=text,
            data={"request_id": request_id, "refusal": refusal, "response_chars": len(text)},
        )
    )


def _fallback_completion(payload: ChatProxyRequest, pulse: Dict[str, Any]) -> Dict[str, Any]:
    content = baseline_reply(payload.messages)
    now = int(time.time())
    response = {
        "id": f"chatcmpl-pulse-{pulse['request_id']}",
        "object": "chat.completion",
        "created": now,
        "model": payload.model,
        "choices": [
            {
                "index": 0,
                "message": {"role": "assistant", "content": content},
                "finish_reason": "stop",
            }
        ],
        "usage": {"prompt_tokens": 0, "completion_tokens": 0, "total_tokens": 0},
        "pulse": {**pulse, "mode": "baseline"},
    }
    _record_output(pulse["request_id"], response)
    return response


@app.get("/health")
def health() -> Dict[str, Any]:
    return {
        "ok": True,
        "service": APP_NAME,
        "upstream_configured": bool(UPSTREAM_API_KEY),
        "events": feed.count,
    }


@app.get("/pulse")
def pulse(limit: int = 50) -> Dict[str, Any]:
    return {"events": feed.list(limit=limit)}


@app.get("/events")
def events() -> Dict[str, Any]:
    return {"events": feed.list(limit=50)}


@app.get("/pulse/bdi")
def pulse_bdi() -> Dict[str, Any]:
    return bdi_state.snapshot()


@app.post("/v1/chat/completions")
async def chat_completions(payload: ChatProxyRequest, request: Request):
    pulse = _record_input(payload)

    if not UPSTREAM_API_KEY and not request.headers.get("authorization"):
        return JSONResponse(_fallback_completion(payload, pulse))

    body = payload.model_dump(exclude_none=True)
    headers = _auth_headers(request)

    if payload.stream:
        async def stream_upstream():
            async with httpx.AsyncClient(timeout=UPSTREAM_TIMEOUT) as client:
                async with client.stream("POST", UPSTREAM_CHAT_URL, headers=headers, json=body) as upstream:
                    if upstream.status_code >= 400:
                        detail = await upstream.aread()
                        raise HTTPException(status_code=upstream.status_code, detail=detail.decode("utf-8", "ignore"))
                    async for chunk in upstream.aiter_bytes():
                        yield chunk
            feed.add(PulseEvent(kind="stream_end", text="", data={"request_id": pulse["request_id"]}))

        return StreamingResponse(stream_upstream(), media_type="text/event-stream")

    async with httpx.AsyncClient(timeout=UPSTREAM_TIMEOUT) as client:
        upstream = await client.post(UPSTREAM_CHAT_URL, headers=headers, json=body)

    if upstream.status_code >= 400:
        raise HTTPException(status_code=upstream.status_code, detail=upstream.text)

    response_json = upstream.json()
    response_json["pulse"] = {**pulse, "mode": "proxy"}
    _record_output(pulse["request_id"], response_json)
    return JSONResponse(response_json)


@app.post("/pulse/reset")
def reset() -> Dict[str, Any]:
    feed.clear()
    bdi_state.reset()
    return {"ok": True}
