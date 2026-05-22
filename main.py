import os
import re
import time
import uuid
from decimal import Decimal, InvalidOperation
from typing import Any, Dict, List, Optional

import httpx
from fastapi import FastAPI, HTTPException, Request
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse, StreamingResponse
from pydantic import BaseModel, Field

from audit import supabase_audit_client
from baseline import baseline_reply, extract_text
from bdi import BDIState
from circuit_breaker import CircuitBreakerMiddleware
from embeddings import embed_texts, novelty_score
from pulse_feed import PulseEvent, PulseFeed
from refusal import detect_refusal

APP_NAME = "pulse-proxy"
UPSTREAM_CHAT_URL = os.getenv("PULSE_UPSTREAM_CHAT_URL", "https://api.openai.com/v1/chat/completions")
UPSTREAM_API_KEY = os.getenv("OPENAI_API_KEY") or os.getenv("PULSE_UPSTREAM_API_KEY")
UPSTREAM_TIMEOUT = float(os.getenv("PULSE_UPSTREAM_TIMEOUT", "120"))
FINANCIAL_TRANSACTION_LIMIT = Decimal("5000")
FINANCIAL_KEYWORDS = {
    "budget",
    "charge",
    "contract",
    "cost",
    "expense",
    "financial",
    "invoice",
    "order",
    "pay",
    "payment",
    "procure",
    "procurement",
    "purchase",
    "quote",
    "requisition",
    "spend",
    "transaction",
    "vendor",
}
FINANCIAL_FIELD_HINTS = {
    "amount",
    "budget",
    "charge",
    "cost",
    "expense",
    "invoice",
    "payment",
    "price",
    "purchase",
    "quote",
    "spend",
    "total",
    "transaction",
    "value",
}
AMOUNT_PATTERN = re.compile(
    r"(?:\$|usd\s*)\s*(\d[\d,]*(?:\.\d+)?)|(\d[\d,]*(?:\.\d+)?)\s*(?:usd|dollars?)",
    re.IGNORECASE,
)

app = FastAPI(title=APP_NAME, version="0.1.0")
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=False,
    allow_methods=["*"],
    allow_headers=["*"],
)

feed = PulseFeed(max_events=int(os.getenv("PULSE_FEED_LIMIT", "500")))
bdi_state = BDIState()
circuit_breaker = CircuitBreakerMiddleware()


def _emit_circuit_breaker_event(event: object) -> None:
    if not hasattr(event, "state"):
        return

    feed.add(
        PulseEvent(
            kind="governance",
            text="kill-switch",
            data={
                "state": getattr(event, "state", "unknown"),
                "reason": getattr(event, "reason", "unknown"),
                "bdi_score": getattr(event, "bdi_score", 0.0),
                "entropy": getattr(event, "entropy", 0.0),
                "timestamp": getattr(event, "timestamp", ""),
            },
        )
    )


circuit_breaker.register_listener(_emit_circuit_breaker_event)


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


def _to_decimal(value: Any) -> Optional[Decimal]:
    if isinstance(value, bool):
        return None
    if isinstance(value, (int, float, Decimal)):
        return Decimal(str(value))
    if isinstance(value, str):
        normalized = value.replace(",", "").strip()
        if not normalized:
            return None
        try:
            return Decimal(normalized)
        except InvalidOperation:
            return None
    return None


def _financial_amounts_from_payload(value: Any, key_path: str = "") -> List[Decimal]:
    amounts: List[Decimal] = []
    lowered_key = key_path.lower()
    has_financial_hint = any(hint in lowered_key for hint in FINANCIAL_FIELD_HINTS)

    if isinstance(value, dict):
        for key, item in value.items():
            next_key = f"{key_path}.{key}" if key_path else str(key)
            amounts.extend(_financial_amounts_from_payload(item, next_key))
    elif isinstance(value, list):
        for item in value:
            amounts.extend(_financial_amounts_from_payload(item, key_path))
    elif has_financial_hint:
        amount = _to_decimal(value)
        if amount is not None:
            amounts.append(amount)

    return amounts


def _financial_amounts_from_text(text: str) -> List[Decimal]:
    amounts: List[Decimal] = []
    for match in AMOUNT_PATTERN.finditer(text):
        raw_amount = match.group(1) or match.group(2)
        amount = _to_decimal(raw_amount)
        if amount is not None:
            amounts.append(amount)
    return amounts


def _is_financial_text(text: str) -> bool:
    lowered = text.lower()
    return any(keyword in lowered for keyword in FINANCIAL_KEYWORDS)


def _calculate_shannon_entropy(text: str) -> float:
    text = text or ""
    if not text:
        return 0.0
    frequencies: dict[str, int] = {}
    for symbol in text:
        frequencies[symbol] = frequencies.get(symbol, 0) + 1

    entropy = 0.0
    length = len(text)
    for count in frequencies.values():
        probability = count / length
        entropy -= probability * (probability and __import__("math").log2(probability))

    max_entropy = __import__("math").log2(len(frequencies)) if frequencies else 1.0
    normalized_entropy = entropy / max_entropy if max_entropy > 0 else 0.0
    return min(1.0, max(0.0, normalized_entropy))


def _normalized_bdi_score(bdi_snapshot: dict[str, object]) -> float:
    beliefs = bdi_snapshot.get("beliefs") or {}
    desires = bdi_snapshot.get("desires") or []
    intentions = bdi_snapshot.get("intentions") or []
    turns = int(bdi_snapshot.get("turns") or 0)

    belief_score = min(1.0, len(beliefs) / 10.0)
    desire_score = min(1.0, len(desires) / 8.0)
    intention_score = min(1.0, len(intentions) / 5.0)
    turn_score = min(1.0, turns / 20.0)

    score = (belief_score * 0.25) + (desire_score * 0.25) + (intention_score * 0.3) + (turn_score * 0.2)
    return min(1.0, max(0.0, score))


def _should_block_financial_transaction(payload: ChatProxyRequest) -> bool:
    payload_dict = payload.model_dump(exclude_none=True)
    prompt_text = _messages_text(payload.messages)
    amounts = _financial_amounts_from_payload(payload_dict)

    if _is_financial_text(prompt_text):
        amounts.extend(_financial_amounts_from_text(prompt_text))

    return any(amount > FINANCIAL_TRANSACTION_LIMIT for amount in amounts)


def _enforce_financial_circuit_breaker(payload: ChatProxyRequest) -> None:
    if _should_block_financial_transaction(payload):
        raise HTTPException(
            status_code=403,
            detail="Forbidden: financial transactions over $5,000 are blocked by the governance circuit breaker.",
        )


def _record_input(payload: ChatProxyRequest) -> Dict[str, Any]:
    prompt_text = _messages_text(payload.messages)
    vectors = embed_texts([prompt_text]) if prompt_text else []
    novelty = novelty_score(prompt_text, feed.recent_texts(limit=25)) if prompt_text else 0.0
    bdi = bdi_state.observe(payload.messages)
    bdi_score = _normalized_bdi_score(bdi)
    pulse = {
        "request_id": str(uuid.uuid4()),
        "model": payload.model,
        "prompt_chars": len(prompt_text),
        "embedding_dim": len(vectors[0]) if vectors else 0,
        "novelty": novelty,
        "bdi": bdi,
        "bdi_score": bdi_score,
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


def _extra_payload_fields(payload: ChatProxyRequest) -> Dict[str, Any]:
    return getattr(payload, "model_extra", None) or {}


def _agent_id(payload: ChatProxyRequest, request: Request) -> Optional[str]:
    extra_fields = _extra_payload_fields(payload)
    value = extra_fields.get("agent_id") or request.headers.get("x-agent-id")
    return str(value) if value is not None else None


def _bdi_score(payload: ChatProxyRequest, pulse: Dict[str, Any]) -> float:
    extra_fields = _extra_payload_fields(payload)
    if extra_fields.get("bdi_score") is not None:
        try:
            return float(extra_fields["bdi_score"])
        except (TypeError, ValueError):
            pass

    bdi = pulse.get("bdi") or {}
    beliefs = bdi.get("beliefs") or {}
    desires = bdi.get("desires") or []
    intentions = bdi.get("intentions") or []
    turns = bdi.get("turns") or 0
    return float(len(beliefs) + len(desires) + len(intentions) + turns)


async def _insert_audit_log(
    payload: ChatProxyRequest,
    request: Request,
    pulse: Dict[str, Any],
    response_payload: Dict[str, Any],
) -> None:
    await supabase_audit_client.insert_audit_log(
        agent_id=_agent_id(payload, request),
        request_payload=payload.model_dump(exclude_none=True),
        response_payload=response_payload,
        bdi_score=_bdi_score(payload, pulse),
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
        "circuit_breaker": {
            "state": circuit_breaker.state.value,
            "last_event": circuit_breaker.last_event.to_dict() if circuit_breaker.last_event is not None else None,
        },
    }


@app.get("/circuit-breaker/status")
def circuit_breaker_status() -> dict[str, object]:
    return circuit_breaker.status()


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
    _enforce_financial_circuit_breaker(payload)
    pulse = _record_input(payload)
    entropy = _calculate_shannon_entropy(_messages_text(payload.messages))
    pulse["entropy"] = entropy

    evaluation = circuit_breaker.evaluate(bdi_score=pulse["bdi_score"], entropy=entropy)
    if evaluation["action"] != "proceed":
        response_payload = {
            "status_code": 423 if evaluation["action"] == "kill" else 409,
            "error": evaluation["reason"],
            "pulse": {**pulse, "governance": evaluation},
        }
        await _insert_audit_log(payload, request, pulse, response_payload)
        raise HTTPException(status_code=response_payload["status_code"], detail=evaluation["reason"])

    if not UPSTREAM_API_KEY and not request.headers.get("authorization"):
        response_json = _fallback_completion(payload, pulse)
        await _insert_audit_log(payload, request, pulse, response_json)
        return JSONResponse(response_json)

    body = payload.model_dump(exclude_none=True)
    headers = _auth_headers(request)

    if payload.stream:
        async def stream_upstream():
            response_payload: Dict[str, Any] = {"stream": True, "chunks": 0, "status_code": 200}
            try:
                async with httpx.AsyncClient(timeout=UPSTREAM_TIMEOUT) as client:
                    async with client.stream("POST", UPSTREAM_CHAT_URL, headers=headers, json=body) as upstream:
                        response_payload["status_code"] = upstream.status_code
                        if upstream.status_code >= 400:
                            detail = await upstream.aread()
                            response_payload["error"] = detail.decode("utf-8", "ignore")
                            await _insert_audit_log(payload, request, pulse, response_payload)
                            raise HTTPException(status_code=upstream.status_code, detail=response_payload["error"])
                        async for chunk in upstream.aiter_bytes():
                            response_payload["chunks"] += 1
                            yield chunk
            except httpx.HTTPError as exc:
                response_payload["status_code"] = 502
                response_payload["error"] = str(exc)
                await _insert_audit_log(payload, request, pulse, response_payload)
                raise HTTPException(status_code=502, detail=str(exc))
            feed.add(PulseEvent(kind="stream_end", text="", data={"request_id": pulse["request_id"]}))
            await _insert_audit_log(payload, request, pulse, response_payload)

        return StreamingResponse(stream_upstream(), media_type="text/event-stream")

    try:
        async with httpx.AsyncClient(timeout=UPSTREAM_TIMEOUT) as client:
            upstream = await client.post(UPSTREAM_CHAT_URL, headers=headers, json=body)
    except httpx.HTTPError as exc:
        await _insert_audit_log(
            payload,
            request,
            pulse,
            {"status_code": 502, "error": str(exc)},
        )
        raise HTTPException(status_code=502, detail=str(exc))

    if upstream.status_code >= 400:
        await _insert_audit_log(
            payload,
            request,
            pulse,
            {"status_code": upstream.status_code, "error": upstream.text},
        )
        raise HTTPException(status_code=upstream.status_code, detail=upstream.text)

    response_json = upstream.json()
    response_json["pulse"] = {**pulse, "mode": "proxy"}
    _record_output(pulse["request_id"], response_json)
    await _insert_audit_log(payload, request, pulse, response_json)
    return JSONResponse(response_json)


@app.post("/pulse/reset")
def reset() -> Dict[str, Any]:
    feed.clear()
    bdi_state.reset()
    return {"ok": True}
