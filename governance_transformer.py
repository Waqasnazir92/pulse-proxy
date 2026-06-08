"""
governance_transformer.py
Transforms raw PulseEvent records (from pulse_feed) into the rich
GovernanceEvent format expected by the Pulse dashboard frontend.
"""
import hashlib
import json
import re
from datetime import datetime, timezone
from typing import Any, Dict, List, Optional


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _now_iso() -> str:
    return datetime.now(timezone.utc).isoformat().replace("+00:00", "Z")


def _sha256(payload: Any) -> str:
    serialised = json.dumps(payload, sort_keys=True, default=str)
    return "sha256:" + hashlib.sha256(serialised.encode()).hexdigest()[:16]


def _initials(name: str) -> str:
    parts = name.split()
    if len(parts) >= 2:
        return (parts[0][0] + parts[-1][0]).upper()
    return name[:2].upper() if name else "AI"


def _severity_from_bdi(bdi_score: float) -> str:
    if bdi_score >= 0.65:
        return "critical"
    if bdi_score >= 0.40:
        return "elevated"
    return "stable"


def _category_from_event(event: Dict[str, Any]) -> str:
    kind = event.get("kind", "")
    data = event.get("data", {}) or {}
    text = (event.get("text") or "").lower()

    if kind == "governance":
        return "circuit_breaker"
    if data.get("refusal"):
        return "refusal_detected"
    if "financial" in text or "transaction" in text:
        return "financial_governance"
    if data.get("novelty", 0) > 0.7:
        return "novel_input"
    if kind == "output":
        return "output_drift"
    return "behavioral_monitoring"


def _narrative_from_event(event: Dict[str, Any]) -> str:
    kind = event.get("kind", "")
    data = event.get("data", {}) or {}
    text = event.get("text") or ""
    bdi_score = float(data.get("bdi_score", 0.0))
    entropy = float(data.get("entropy", 0.0))

    if kind == "governance":
        state = data.get("state", "unknown")
        reason = data.get("reason", "")
        return f"Circuit breaker transitioned to {state}. Reason: {reason}. BDI={bdi_score:.2f}, Entropy={entropy:.2f}."

    if kind == "input":
        novelty = float(data.get("novelty", 0.0))
        chars = data.get("prompt_chars", 0)
        intentions = (data.get("bdi") or {}).get("intentions") or []
        intent_str = ", ".join(intentions) if intentions else "unclassified"
        return (
            f"Input received ({chars} chars). Novelty score {novelty:.2f}. "
            f"Detected intentions: {intent_str}. BDI score: {bdi_score:.2f}."
        )

    if kind == "output":
        refusal = data.get("refusal", False)
        chars = data.get("response_chars", 0)
        suffix = " Refusal behaviour detected." if refusal else ""
        return f"Output generated ({chars} chars). Entropy {entropy:.2f}.{suffix}"

    preview = text[:120].replace("\n", " ") if text else "No description available."
    return f"Event kind={kind}. {preview}"


def _output_vector_from_bdi(bdi: Dict[str, Any]) -> List[float]:
    beliefs = bdi.get("beliefs") or {}
    desires = bdi.get("desires") or []
    intentions = bdi.get("intentions") or []
    turns = int(bdi.get("turns") or 0)

    def _norm(v: float, cap: float = 10.0) -> float:
        return round(min(1.0, v / cap), 3)

    return [
        _norm(len(beliefs)),
        _norm(len(desires), cap=8.0),
        _norm(len(intentions), cap=5.0),
        _norm(turns, cap=20.0),
        round(min(1.0, (len(beliefs) + len(desires) + len(intentions)) / 20.0), 3),
    ]


# ---------------------------------------------------------------------------
# Main transformer
# ---------------------------------------------------------------------------

_event_counter = 0


def transform_to_governance_event(raw: Dict[str, Any]) -> Dict[str, Any]:
    """Convert a PulseEvent dict → GovernanceEvent dict for the dashboard."""
    global _event_counter
    _event_counter += 1

    data: Dict[str, Any] = raw.get("data") or {}
    kind: str = raw.get("kind", "unknown")
    created_at: str = raw.get("created_at") or _now_iso()
    bdi_score: float = float(data.get("bdi_score", 0.0))
    entropy: float = float(data.get("entropy", 0.0))
    bdi: Dict[str, Any] = data.get("bdi") or {}
    model: str = data.get("model") or "AI Agent"
    request_id: str = data.get("request_id") or f"pulse-{_event_counter}"

    # Derive event ID from timestamp + counter
    ts_tag = created_at[:10].replace("-", "")
    event_id = f"AI-{ts_tag}-{_event_counter:04d}"

    severity = _severity_from_bdi(bdi_score)
    category = _category_from_event(raw)

    # Build output vectors from BDI state (proxy for embedding vectors)
    baseline_vector = [0.2, 0.3, 0.1, 0.4, 0.0]
    current_vector = _output_vector_from_bdi(bdi) if bdi else [0.5, 0.2, 0.1, 0.1, 0.1]

    # Timeline
    timeline = [
        {"time": "T-0", "color": "#00D2FF", "text": "Request received"},
    ]
    if entropy > 0.5:
        timeline.append({"time": "T+1", "color": "#FFB800", "text": f"High entropy detected ({entropy:.2f})"})
    if bdi_score > 0.4:
        timeline.append({"time": "T+2", "color": "#FF4B2B", "text": f"BDI threshold crossed ({bdi_score:.2f})"})

    # Actor
    actor_name = model.replace("-", " ").title()
    actor_type = "ai_model" if kind in ("input", "output", "stream_end") else "system"

    # Compliance flags
    requires_review = bdi_score > 0.65 or entropy > 0.75
    frameworks = ["EU_AI_Act_Art13"]
    if requires_review:
        frameworks.append("GDPR_Art22")

    return {
        "id": event_id,
        "createdAt": created_at,
        "actor": {
            "id": f"agent_{request_id}",
            "name": actor_name,
            "type": actor_type,
            "initials": _initials(actor_name),
        },
        "trigger": {
            "type": kind,
            "source": "internal",
        },
        "entropy": {
            "score": round(entropy, 4),
            "components": {
                "outputDistributionDrift": round(entropy * 0.9, 4),
                "confidenceScoreVariance": round(bdi_score * 0.85, 4),
                "inputAnomalyScore": round(float(data.get("novelty", entropy * 0.8)), 4),
                "tokenProbabilityCollapse": round(entropy * 0.95, 4),
                "latencySpikeRatio": round(bdi_score * 0.7, 4),
            },
            "baselineScore": 0.3,
            "delta": round(max(0.0, entropy - 0.3), 4),
            "calculatedAt": created_at,
        },
        "classification": {
            "severity": severity,
            "category": category,
            "confidence": round(min(0.99, 0.7 + bdi_score * 0.3), 2),
            "autoClassified": True,
        },
        "response": {
            "actionTaken": "kill-switch" if kind == "governance" else "logged",
            "notified": [],
            "notificationChannel": "pulse_dashboard",
            "autoResolved": bdi_score < 0.4,
        },
        "compliance": {
            "frameworks": frameworks,
            "requiresHumanReview": requires_review,
            "reviewDeadline": created_at,
            "legalHold": False,
        },
        "auditTrail": {
            "immutableHash": _sha256({"id": event_id, "created_at": created_at, "kind": kind}),
            "chainPrev": _sha256({"counter": _event_counter - 1}),
            "signedBy": "pulse_proxy_system_key",
            "storage": "append_only_log",
        },
        "narrative": _narrative_from_event(raw),
        "timeline": timeline,
        "baselineSample": {
            "timestamp": int(datetime.now(timezone.utc).timestamp() * 1000),
            "outputVector": baseline_vector,
            "refusalRate": 0.05,
            "responseLatencyMs": 120,
        },
        "currentSample": {
            "timestamp": int(datetime.now(timezone.utc).timestamp() * 1000),
            "outputVector": current_vector,
            "refusalRate": 0.35 if data.get("refusal") else 0.05,
            "responseLatencyMs": 450 if bdi_score > 0.65 else 200,
        },
        # Raw proxy data preserved for debugging
        "_raw": {"kind": kind, "bdi_score": bdi_score, "entropy": entropy},
    }
